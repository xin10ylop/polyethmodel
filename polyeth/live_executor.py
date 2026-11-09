"""Live execution layer for the paper trader's decisions on Polymarket (official Python SDK `polymarket-client`).

The paper trader keeps deciding and keeps its own simulated book; when LIVE=1 it also hands every decision to this
executor, which places the real resting orders and tracks the real fills:
  A : BUY limit at 0.50 on the model's side at T-45, cancelled at T-40 if unfilled; on fill a SELL limit at 0.85 (GTC);
      whatever is unsold at the close is held to settlement and redeemed.
  B : BUY limit at the last print at T+1.6, cancelled after 10 s; on fill SELL limit at 0.85; else settle + redeem.
  D : BUY limit at the last print at T+90, cancelled after 10 s; no sell; settle + redeem.
Sizing: STAKE_USD per trade at the entry price, rounded to the market's size precision and at least MIN_SHARES.
Risk: MAX_OPEN positions, DAILY_LOSS_LIMIT (stop opening new trades for the UTC day once realised P&L is below it),
and a kill file (touch /home/user/paper/STOP) that cancels all open orders and stops new ones.
DRY_RUN=1 logs every order it would place without touching the account.
Fees: every order is post-only (maker), so no taker fee is ever paid; an order that would cross the book is not placed
(ALLOW_TAKER=1 re-sends it crossing and books the fee). The tally reports fees paid, which should read $0.00.

Environment: POLYMARKET_PRIVATE_KEY (signer key), POLYMARKET_WALLET_ADDRESS (account wallet), and for redemption
POLYMARKET_RELAYER_API_KEY + POLYMARKET_RELAYER_API_KEY_ADDRESS (polymarket.com -> Settings -> API Keys -> Relayer).
Everything is logged to PAPER_DIR/live_orders.jsonl and live_trades.jsonl, and live.log.
"""
import os, json, time, math, datetime as dt, threading, traceback

STAKE_USD = float(os.environ.get("STAKE_USD", "10"))
MIN_SHARES = 5.0
MAX_OPEN = int(os.environ.get("MAX_OPEN", "4"))
DAILY_LOSS_LIMIT = float(os.environ.get("DAILY_LOSS_LIMIT", "-100"))
DRY_RUN = os.environ.get("DRY_RUN", "1") == "1"
ALLOW_TAKER = os.environ.get("ALLOW_TAKER", "0") == "1"   # default: post-only everywhere, a fee can never be paid
CANCEL_ALL = os.environ.get("CANCEL_ALL_ON_START", "0") == "1"   # 1: wipe every resting order of the account at start / stop (only when no other bot shares the wallet)
PAPER = os.environ.get("PAPER_DIR") or os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "paper_eth")
STOP_FILE = os.path.join(PAPER, "STOP")
CFG_TP = {"A": 0.80, "B": 0.85, "D": None}   # sell limit per strategy (mirrors paper_trader CFG_*; used only when re-posting after a restart)
LOG = open(os.path.join(PAPER, "live.log"), "a")


def log(*a):
    s = f"{dt.datetime.now(dt.UTC):%H:%M:%S} [live{'/dry' if DRY_RUN else ''}] " + " ".join(str(x) for x in a)
    print(s, flush=True); LOG.write(s + "\n"); LOG.flush()


def taker_fee(price, size):
    """Polymarket taker fee on the 5-minute markets: 0.07 * p * (1 - p) per share (feeRateBps 700, exponent 1); makers pay nothing."""
    return 0.07 * price * (1.0 - price) * size


class SDK:
    """Thin wrapper over the official SDK so the executor can be exercised without it (DRY_RUN) and so that
    method names live in one place. Fill in from docs.polymarket.com/trading/manage-orders if the SDK differs."""
    def __init__(self):
        self.client = None
        if DRY_RUN: return
        from polymarket import SecureClient, RelayerApiKey   # pip install polymarket-client
        kw = dict(private_key=os.environ["POLYMARKET_PRIVATE_KEY"], wallet=os.environ["POLYMARKET_WALLET_ADDRESS"])
        if os.environ.get("POLYMARKET_RELAYER_API_KEY"):
            kw["api_key"] = RelayerApiKey(key=os.environ["POLYMARKET_RELAYER_API_KEY"], address=os.environ["POLYMARKET_RELAYER_API_KEY_ADDRESS"])
        self.client = SecureClient.create(**kw)
        log("sdk ready: wallet", os.environ["POLYMARKET_WALLET_ADDRESS"], "type", getattr(self.client, "wallet_type", "?"))
        try: self.client.setup_trading_approvals(); log("trading approvals ok (pUSD + outcome tokens)")
        except Exception as e: log("setup_trading_approvals:", repr(e)[:160])

    def limit(self, token_id, side, price, size, post_only=False):
        if DRY_RUN: return dict(order_id=f"dry-{int(time.time()*1000)}", status="live")
        r = self.client.place_limit_order(asset_id=token_id, side=side, price=f"{price:.2f}", size=f"{size:.2f}", post_only=post_only)
        if not getattr(r, "ok", True): raise RuntimeError(f"order rejected: {getattr(r, 'code', '')} {getattr(r, 'message', '')}")
        return dict(order_id=r.order_id, status=r.status, trade_ids=list(getattr(r, "trade_ids", []) or []))

    def cancel(self, order_id):
        if DRY_RUN: return True
        return self.client.cancel_order(order_id=order_id)

    def cancel_all(self):
        if DRY_RUN: return True
        return self.client.cancel_all()

    def status(self, order_id):
        """-> (status, size_matched). Field names per manage-orders docs; adjust there if the SDK names differ."""
        if DRY_RUN: return "live", 0.0
        o = self.client.get_order(order_id=order_id)
        return getattr(o, "status", "?"), float(getattr(o, "size_matched", 0) or 0)

    def redeem(self, condition_id):
        if DRY_RUN: return "dry"
        tx = self.client.redeem_positions(condition_id=condition_id); out = tx.wait()
        return getattr(out, "transaction_hash", "?")


class LiveExecutor:
    def __init__(self):
        self.sdk = SDK(); self.lock = threading.RLock()      # re-entrant: the SIGTERM handler runs in the main thread, which may already hold it inside poll()
        self.orders = {}          # order_id -> dict(strategy, T, side, token, price, size, kind buy|sell, cancel_at, filled, cancel_sent, err)
        self.positions = {}       # (strategy, T) -> dict(side, token, cond, size, entry, sell_orders, resting, sold, pnl, settled, redeemed, next_check)
        self.day = None; self.day_pnl = 0.0; self.halted = False
        self.fo = open(os.path.join(PAPER, "live_orders.jsonl"), "a"); self.ft = open(os.path.join(PAPER, "live_trades.jsonl"), "a")
        self.pos_file = os.path.join(PAPER, "live_positions.json"); self.redeem_q = []; self._redeem_thread = None
        log(f"executor up: stake ${STAKE_USD}, max open {MAX_OPEN}, daily loss limit ${DAILY_LOSS_LIMIT}, dry_run={DRY_RUN}")
        if not DRY_RUN:
            if not os.environ.get("POLYMARKET_RELAYER_API_KEY"): log("WARNING: no POLYMARKET_RELAYER_API_KEY: winning positions held to settlement will NOT be redeemed automatically")
            if CANCEL_ALL:
                try: self.sdk.cancel_all(); log("startup: cancelled every resting order of the account (CANCEL_ALL_ON_START=1)")
                except Exception as e: log(f"startup cancel_all FAILED: {repr(e)[:120]} - check the Polymarket UI for resting orders")
        self._load_state()

    # ---------------------------------------------------------------- persistence (a restart must not forget shares it holds or the day's loss)
    def _save_state(self):
        try:
            tmp = self.pos_file + ".tmp"
            json.dump({"day": self.day, "day_pnl": self.day_pnl, "halted": self.halted, "redeem_q": self.redeem_q, "orders": self.orders,
                       "positions": {f"{k[0]}|{k[1]}": p for k, p in self.positions.items() if not (p["settled"] and p.get("redeemed", True))}}, open(tmp, "w")); os.replace(tmp, self.pos_file)
        except Exception as e: log(f"save state failed: {repr(e)[:100]}")

    def _load_state(self):
        if not os.path.exists(self.pos_file): return
        try: saved = json.load(open(self.pos_file))
        except Exception as e: log(f"load state failed: {repr(e)[:100]}"); return
        today = dt.datetime.now(dt.UTC).date().isoformat()
        if saved.get("day") == today:
            self.day, self.day_pnl, self.halted = today, float(saved.get("day_pnl", 0.0)), bool(saved.get("halted", False))
            log(f"restored today's realised P&L {self.day_pnl:+.2f}" + (" (daily loss limit already reached: no new trades today)" if self.halted else ""))
        self.redeem_q = list(saved.get("redeem_q", [])); now = time.time()
        # orders this bot left on the book (crash / restart): cancel exactly those (another bot may share the wallet), keep the BUYs tracked so that
        # a fill that happened while the bot was down is picked up by poll() and booked; the SELLs are re-posted below for the positions still open
        for oid, o in (saved.get("orders") or {}).items():
            if DRY_RUN: continue
            try: self.sdk.cancel(oid)
            except Exception as e: log(f"startup: cancel of own order {oid} failed ({repr(e)[:80]})")
            if o.get("kind") == "buy": o["cancel_sent"] = True; o["cancel_at"] = 0; self.orders[oid] = o
        sold_adj = {}
        for oid, o in (saved.get("orders") or {}).items():          # a take-profit that filled while the bot was down: book it on the restored position
            if DRY_RUN or o.get("kind") != "sell": continue
            try:
                _, matched = self.sdk.status(oid); new = float(matched) - float(o.get("filled", 0.0))
                if new > 1e-9: k = f"{o['strategy']}|{o['T']}"; sold_adj[k] = sold_adj.get(k, 0.0) + new; sold_adj[k + "|px"] = o["price"]
            except Exception as e: log(f"startup: status of own sell {oid} failed ({repr(e)[:80]}) - if it filled while the bot was down the tally is off by that sale")
        if saved.get("orders"): log(f"startup: cancelled {len(saved['orders'])} order(s) left by the previous run; {len(self.orders)} buy(s) kept under watch for fills made while the bot was down")
        for key, p in saved.get("positions", {}).items():
            st, T = key.split("|"); T = int(T); p["sell_orders"] = []; p["resting"] = 0.0; p["next_check"] = 0.0; p.setdefault("redeemed", True); p.setdefault("tp", CFG_TP.get(st)); p["next_sell_try"] = 0.0
            if key in sold_adj:
                new = sold_adj[key]; pnl = (sold_adj[key + "|px"] - p["entry"]) * new; p["sold"] += new; p["pnl"] += pnl; self._book(pnl)
                self._rec(self.ft, event="fill", strategy=st, T=T, side=p["side"], price=sold_adj[key + "|px"], size=new, kind="sell", fee=0.0, pnl=round(pnl, 4), note="filled while the bot was down")
                log(f"[{st}] take-profit filled while the bot was down: SOLD @{sold_adj[key + '|px']:.2f} x{new:.2f} pnl {pnl:+.2f}")
                if p["sold"] >= p["size"] - 1e-6: p["settled"] = True
            self.positions[(st, T)] = p; held = p["size"] - p["sold"]
            if p["settled"]: continue
            log(f"[{st}] restored position from a previous run: {p['side']} x{held:.2f} @{p['entry']:.2f} (window {dt.datetime.fromtimestamp(T, dt.UTC):%H:%M})")
            if now < T + 300: self._post_sell(st, T, p, CFG_TP.get(st))
        if self.redeem_q: log(f"{len(self.redeem_q)} redemption(s) pending from a previous run"); self._start_redeemer()
        self._save_state()

    # ---------------------------------------------------------------- helpers
    def _rec(self, f, **d):
        d["ts"] = round(time.time(), 2); f.write(json.dumps(d) + "\n"); f.flush()

    def _roll_day(self):
        d = dt.datetime.now(dt.UTC).date().isoformat()
        if d != self.day: self.day, self.day_pnl, self.halted = d, 0.0, False; self._save_state()

    def _book(self, pnl):
        self._roll_day(); self.day_pnl += pnl

    def _can_open(self, strategy):
        self._roll_day()
        if os.path.exists(STOP_FILE): log("STOP file present: no new orders"); return False
        if self.halted: return False
        if self.day_pnl <= DAILY_LOSS_LIMIT: self.halted = True; self._save_state(); log(f"daily loss limit reached ({self.day_pnl:+.2f}): no new trades today"); return False
        open_pos = len([p for p in self.positions.values() if not p["settled"]]) + len([o for o in self.orders.values() if o["kind"] == "buy" and not o["filled"] and not o.get("cancel_sent")])
        if open_pos >= MAX_OPEN: log(f"max open positions ({MAX_OPEN}) reached: skip {strategy}"); return False
        return True

    @staticmethod
    def _size(price):
        return max(MIN_SHARES, math.floor(STAKE_USD / price * 100) / 100)

    @staticmethod
    def _floor2(x):
        return math.floor(x * 100 + 1e-9) / 100

    def _place(self, token, side, price, size):
        """Every order is post-only: it rests on the book and can only ever be filled as a maker (no fee). If the exchange
        rejects it because the price would cross the book, the order is NOT placed (a BUY is skipped, a SELL falls back to
        holding to settlement) unless ALLOW_TAKER=1, in which case the same limit is re-sent crossing and the taker fee
        0.07 p(1-p) per share is booked on the fill. Returns (order, taker_flag)."""
        try:
            return self.sdk.limit(token, side, price, size, post_only=True), False
        except Exception as e:
            if not ALLOW_TAKER:
                log(f"post-only {side} @{price:.2f} rejected ({repr(e)[:100]}); not placed (ALLOW_TAKER=0: never pay a fee)"); raise
            log(f"post-only {side} @{price:.2f} rejected ({repr(e)[:100]}); placing as a crossing limit (ALLOW_TAKER=1)")
            return self.sdk.limit(token, side, price, size, post_only=False), True

    def _cancel(self, oid, o, reason=""):
        """Send the cancel but keep tracking the order until the exchange confirms it is gone (cancelled or filled)."""
        try: self.sdk.cancel(oid); o["cancel_sent"] = True
        except Exception as e:
            o["cancel_sent"] = True; o["cancel_err"] = o.get("cancel_err", 0) + 1
            if o["cancel_err"] in (1, 5, 50): log(f"cancel {oid} failed x{o['cancel_err']} ({reason}): {repr(e)[:100]} - still tracked, will retry")

    # ---------------------------------------------------------------- decisions from the paper trader
    def buy(self, strategy, T, side, price, cancel_at, token_up, token_down, condition_id, tp):
        """Rest a BUY limit for `strategy` on `side` at `price`; cancel at epoch `cancel_at` if unfilled; sell limit `tp` after the fill (None = hold)."""
        with self.lock:
            if not self._can_open(strategy): return None
            if cancel_at is not None and time.time() > cancel_at - 1.0:
                log(f"[{strategy}] BUY {side} NOT placed: {cancel_at - time.time():+.1f}s to its cancel time (decision came too late)"); self._rec(self.fo, event="late", strategy=strategy, T=T, side=side); return None
            token = token_up if side == "UP" else token_down; size = self._size(price); price = self._floor2(price)   # never bid above the print
            try:
                r, taker = self._place(token, "BUY", price, size)
            except Exception as e:
                log(f"[{strategy}] BUY {side} @{price:.2f} x{size} FAILED: {repr(e)[:160]}"); return None
            oid = r["order_id"]; self.orders[oid] = dict(strategy=strategy, T=T, side=side, token=token, cond=condition_id, price=price, size=size, kind="buy", cancel_at=cancel_at, filled=0.0, tp=tp, placed=time.time(), taker=taker)
            self._rec(self.fo, event="placed", order_id=oid, strategy=strategy, T=T, side=side, price=price, size=size, kind="buy", status=r.get("status"), taker=taker)
            log(f"[{strategy}] BUY {side} @{price:.2f} x{size} order {oid} status {r.get('status')}{' TAKER' if taker else ''}")
            self._save_state(); return oid

    def _post_sell(self, st, T, p, tp):
        """Rest a SELL limit at tp for the shares held that are not yet sold or resting; floors to 2 decimals and respects the
        5-share minimum (a smaller remainder waits for more fills, or settles)."""
        if tp is None or DRY_RUN and False: return
        avail = self._floor2(p["size"] - p["sold"] - p.get("resting", 0.0))
        if avail < MIN_SHARES: return
        try:
            r, taker = self._place(p["token"], "SELL", tp, avail)
            p.setdefault("sell_orders", []).append(r["order_id"]); p["resting"] = p.get("resting", 0.0) + avail
            self.orders[r["order_id"]] = dict(strategy=st, T=T, side=p["side"], token=p["token"], cond=p["cond"], price=tp, size=avail, kind="sell", cancel_at=None, filled=0.0, tp=None, placed=time.time(), taker=taker)
            self._rec(self.fo, event="placed", order_id=r["order_id"], strategy=st, T=T, side=p["side"], price=tp, size=avail, kind="sell", status=r.get("status"), taker=taker)
            log(f"[{st}] SELL limit @{tp:.2f} x{avail:.2f} order {r['order_id']}{' TAKER' if taker else ''}")
            p["sell_fail"] = 0
        except Exception as e:
            p["sell_fail"] = p.get("sell_fail", 0) + 1
            if p["sell_fail"] in (1, 20): log(f"[{st}] SELL limit @{tp:.2f} not placed x{p['sell_fail']} ({repr(e)[:120]}); retried every 3 s until the close, else held to settlement")
        self._save_state()

    def _on_fill(self, oid, matched):
        o = self.orders[oid]; new = matched - o["filled"]
        if new <= 0: return
        o["filled"] = matched
        key = (o["strategy"], o["T"])
        p = self.positions.setdefault(key, dict(side=o["side"], token=o["token"], cond=o["cond"], size=0.0, entry=o["price"], sell_orders=[], resting=0.0, settled=False, redeemed=True, sold=0.0, pnl=0.0, next_check=0.0, tp=o["tp"], next_sell_try=0.0))
        p["size"] += new; fee = taker_fee(o["price"], new) if o.get("taker") else 0.0; p["pnl"] -= fee; self._book(-fee)
        self._rec(self.ft, event="fill", order_id=oid, strategy=o["strategy"], T=o["T"], side=o["side"], price=o["price"], size=new, kind="buy", fee=round(fee, 4), pnl=round(-fee, 4))
        log(f"[{o['strategy']}] FILLED {o['side']} @{o['price']:.2f} x{new:.2f} (T{time.time()-o['T']:+.1f}s)" + (f" taker fee {fee:.2f}" if fee else ""))
        if time.time() < o["T"] + 300: self._post_sell(o["strategy"], o["T"], p, o["tp"])
        self._save_state()

    def _on_sell_fill(self, oid, matched):
        o = self.orders[oid]; new = matched - o["filled"]
        if new <= 0: return
        o["filled"] = matched; key = (o["strategy"], o["T"]); p = self.positions[key]
        p["sold"] += new; p["resting"] = max(p.get("resting", 0.0) - new, 0.0); fee = taker_fee(o["price"], new) if o.get("taker") else 0.0
        pnl = (o["price"] - p["entry"]) * new - fee; p["pnl"] += pnl; self._book(pnl)
        self._rec(self.ft, event="fill", order_id=oid, strategy=o["strategy"], T=o["T"], side=o["side"], price=o["price"], size=new, kind="sell", fee=round(fee, 4), pnl=round(pnl, 4))
        log(f"[{o['strategy']}] SOLD @{o['price']:.2f} x{new:.2f}  pnl {pnl:+.2f}  (day {self.day_pnl:+.2f})")
        if p["sold"] >= p["size"] - 1e-6: p["settled"] = True
        self._save_state()

    def _status(self, oid, o):
        """(status, matched) or None when the lookup failed; an order whose lookup keeps failing is cancelled and dropped with a warning."""
        try:
            st, matched = self.sdk.status(oid); o["err"] = 0; return str(st).lower(), matched
        except Exception as e:
            o["err"] = o.get("err", 0) + 1
            if o["err"] in (1, 10): log(f"status {oid} failed x{o['err']}: {repr(e)[:100]}")
            if o["err"] >= 40:                                      # ~20 s of failures: stop trusting it
                try: self.sdk.cancel(oid)
                except Exception: pass
                log(f"status {oid} failed {o['err']} times: cancelled and dropped from tracking - CHECK THE POLYMARKET UI for this {o['kind']} ({o['strategy']} {o['side']} @{o['price']:.2f} x{o['size']:.2f})")
                self._rec(self.fo, event="dropped", order_id=oid, strategy=o["strategy"], T=o["T"], kind=o["kind"], reason="status lookups failing")
                del self.orders[oid]
            return None

    # ---------------------------------------------------------------- periodic maintenance (called from the trader loop)
    def poll(self, now, outcome_fn):
        if now - getattr(self, "_last_poll", 0) < 0.5: return
        self._last_poll = now
        with self.lock:
            try:
                stop = os.path.exists(STOP_FILE)
                for oid, o in list(self.orders.items()):
                    if DRY_RUN:
                        st, matched = ("cancelled" if o.get("cancel_sent") else "live"), o["filled"]
                    else:
                        r = self._status(oid, o)
                        if r is None: 
                            if o["kind"] == "buy" and (stop or (o["cancel_at"] is not None and now >= o["cancel_at"])) and not o.get("cancel_sent"): self._cancel(oid, o, "status failing")
                            continue
                        st, matched = r
                    if o["kind"] == "buy":
                        if matched > o["filled"]: self._on_fill(oid, matched)
                        want_cancel = o["filled"] < o["size"] and (stop or (o["cancel_at"] is not None and now >= o["cancel_at"]))
                        if want_cancel and not o.get("cancel_sent"):
                            self._cancel(oid, o, "STOP" if stop else "time")
                            self._rec(self.fo, event="cancel_sent", order_id=oid, filled=o["filled"], reason="stop" if stop else "time")
                            log(f"[{o['strategy']}] cancel sent for BUY {o['side']} @{o['price']:.2f} ({o['filled']:.2f}/{o['size']:.2f} filled)" + (" (STOP)" if stop else ""))
                        elif want_cancel and o.get("cancel_err"):
                            self._cancel(oid, o, "retry")
                        if o["filled"] >= o["size"] - 1e-6 or st in ("cancelled", "canceled", "expired", "matched", "filled"):
                            if o["filled"] < o["size"] - 1e-6: self._rec(self.fo, event="cancelled", order_id=oid, filled=o["filled"]); log(f"[{o['strategy']}] BUY {o['side']} @{o['price']:.2f} gone from the book ({o['filled']:.2f}/{o['size']:.2f} filled)")
                            del self.orders[oid]
                    else:
                        if matched > o["filled"]: self._on_sell_fill(oid, matched)
                        if o["filled"] >= o["size"] - 1e-6 or st in ("cancelled", "canceled", "expired", "matched", "filled"):
                            if o["filled"] < o["size"] - 1e-6:
                                p = self.positions.get((o["strategy"], o["T"]))
                                if p: p["resting"] = max(p.get("resting", 0.0) - (o["size"] - o["filled"]), 0.0)
                            del self.orders[oid]
                # a take-profit that could not be posted (rejected, shares not settled yet, fewer than 5 shares until a later fill) or that vanished
                # unfilled is posted again every 3 s while the window is open
                for key, p in self.positions.items():
                    if p["settled"] or p.get("tp") is None or now >= key[1] + 295 or now < p.get("next_sell_try", 0.0): continue
                    if self._floor2(p["size"] - p["sold"] - p.get("resting", 0.0)) >= MIN_SHARES: p["next_sell_try"] = now + 3.0; self._post_sell(key[0], key[1], p, p["tp"])
                # the sell limit lives only inside the window (the backtest takes no exit after the close): cancel resting sells at T+300
                for key, p in self.positions.items():
                    if p["settled"] or now < key[1] + 300: continue
                    for so in list(p.get("sell_orders", [])):
                        o = self.orders.get(so)
                        if o and not o.get("cancel_sent"): self._cancel(so, o, "window closed")
                # settlement: a position still held 60 s after the close resolves to 1 or 0 (looked up every 15 s); book it, queue the redeem
                busy = (240 <= now % 300 < 262) or any(o["kind"] == "buy" and not o.get("cancel_sent") for o in self.orders.values())   # no slow lookups around the decision / while a bid rests
                for key, p in list(self.positions.items()):
                    if busy or p["settled"] or now < key[1] + 300 + 60 or now < p.get("next_check", 0): continue
                    p["next_check"] = now + 15.0
                    if any(so in self.orders for so in p.get("sell_orders", [])): continue      # wait until every sell is confirmed gone (a late fill is booked by the loop above)
                    out = outcome_fn(key[1])
                    if out is None:
                        if now > key[1] + 3600 and int(now) % 900 < 1: log(f"[{key[0]}] window {dt.datetime.fromtimestamp(key[1], dt.UTC):%H:%M} still unresolved after an hour: holding")
                        continue
                    held = self._floor2(p["size"] - p["sold"]); won = (out == 1) == (p["side"] == "UP"); pnl = ((1.0 if won else 0.0) - p["entry"]) * held
                    p["pnl"] += pnl; self._book(pnl); p["settled"] = True
                    self._rec(self.ft, event="settle", strategy=key[0], T=key[1], side=p["side"], size=held, entry=p["entry"], won=won, pnl=round(pnl, 4))
                    log(f"[{key[0]}] SETTLED {p['side']} x{held:.2f} {'WON' if won else 'LOST'}  pnl {pnl:+.2f}  (day {self.day_pnl:+.2f})")
                    if won and held > 0 and p["cond"]:
                        p["redeemed"] = False
                        if p["cond"] not in [q["cond"] for q in self.redeem_q]: self.redeem_q.append(dict(cond=p["cond"], T=key[1], tries=0, next=now + 30.0))
                        self._start_redeemer()
                    self._save_state()
            except Exception:
                log("executor poll error", traceback.format_exc()[-400:])

    # ---------------------------------------------------------------- redemption sweep (background: a relayer call can take long and can fail; retried every minute for a day)
    def _start_redeemer(self):
        if self._redeem_thread is not None and self._redeem_thread.is_alive(): return
        self._redeem_thread = threading.Thread(target=self._redeem_loop, daemon=True); self._redeem_thread.start()

    def _redeem_loop(self):
        while True:
            with self.lock: todo = [q for q in self.redeem_q if time.time() >= q["next"]]
            if not todo and not self.redeem_q: return
            for q in todo:
                try:
                    tx = self.sdk.redeem(q["cond"]); log(f"redeemed {q['cond'][:10]}... (window {dt.datetime.fromtimestamp(q['T'], dt.UTC):%H:%M}) tx {tx}")
                    with self.lock:
                        self.redeem_q = [x for x in self.redeem_q if x["cond"] != q["cond"]]
                        for k, p in self.positions.items():
                            if p["cond"] == q["cond"]: p["redeemed"] = True
                        self._save_state()
                except Exception as e:
                    q["tries"] += 1; q["next"] = time.time() + min(60.0 * q["tries"], 900.0)
                    if q["tries"] in (1, 3, 10, 100): log(f"redeem {q['cond'][:10]}... failed x{q['tries']}: {repr(e)[:120]} (retrying)")
                    if q["tries"] >= 200:
                        log(f"redeem {q['cond'][:10]}... GIVING UP after {q['tries']} tries: redeem it from the Polymarket UI")
                        with self.lock: self.redeem_q = [x for x in self.redeem_q if x["cond"] != q["cond"]]; self._save_state()
            time.sleep(2.0)

    def shutdown(self):
        """On a planned stop: cancel every resting order, keep positions on disk for the next run."""
        with self.lock:
            try:
                if CANCEL_ALL: self.sdk.cancel_all(); log("shutdown: cancelled every resting order of the account")
                else:
                    for oid in list(self.orders): self.sdk.cancel(oid)
                    log(f"shutdown: cancelled this bot's {len(self.orders)} resting order(s)")
            except Exception as e: log(f"shutdown cancel FAILED: {repr(e)[:120]} - check the Polymarket UI")
            self._save_state()

    def summary(self):
        tr = [json.loads(l) for l in open(os.path.join(PAPER, "live_trades.jsonl")) if l.strip()] if os.path.exists(os.path.join(PAPER, "live_trades.jsonl")) else []
        pnl = sum(t.get("pnl", 0.0) for t in tr); n = len([t for t in tr if t["event"] in ("settle", "fill") and t.get("kind") != "buy"])
        fees = sum(t.get("fee", 0.0) for t in tr); nt = len([t for t in tr if t.get("fee", 0.0) > 0])
        return f"live: {len([t for t in tr if t['event']=='fill' and t.get('kind')=='buy'])} buys, {n} closes, realised pnl ${pnl:+.2f}, fees paid ${fees:.2f} on {nt} taker fills"
