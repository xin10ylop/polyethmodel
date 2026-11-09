"""Polymarket 5-minute market trade simulator.

For every window and both sides, precomputes the outcome of: buy `side` at entry price pe at time
t_entry (resting maker order, assumed filled), exit by take-profit limit (maker, fills when any trade
prints at >= tp on our side), stop-loss (taker, when a print <= sl: fill at that print minus slippage,
taker fee), or time stop (taker at the last print before t_stop), never holding to settlement.
Settlement P&L is also recorded for reference.

Fee (crypto_fees_v2, from the market metadata): fee per share = rate * p * (1 - p), taker only.
"""
import numpy as np, pandas as pd, glob, os

TRADES = "/home/user/data_eth/polymarket/trades"
MARKETS = os.environ.get("POLY_MARKETS", "/home/user/data_eth/polymarket/markets.parquet")
FEE_RATE = 0.07


def fee(p, rate=FEE_RATE):
    return rate * p * (1 - p)


def load_windows(days=None):
    """Returns (markets df indexed by window_start, dict window -> (rel seconds, UP-equivalent price,
    trade order) sorted by time)."""
    mk = pd.read_parquet(MARKETS).set_index("window_start")
    mk = mk[mk.outcome_up_price.isin([0.0, 1.0])]
    files = sorted(glob.glob(f"{TRADES}/*.parquet"))
    if days is not None:
        files = [f for f in files if os.path.basename(f)[:10] in days]
    tr = pd.concat([pd.read_parquet(f, columns=["ts", "outcome_index", "price", "window_start"]) for f in files], ignore_index=True)
    tr = tr[tr.window_start.isin(mk.index)]
    tr["rel"] = (tr.ts - tr.window_start).astype(np.int32)
    tr["p_up"] = np.where(tr.outcome_index == 0, tr.price, 1 - tr.price).astype(np.float32)
    tr = tr.sort_values(["window_start", "ts"], kind="stable")
    g = {}
    for w, sub in tr.groupby("window_start", sort=False):
        g[int(w)] = (sub.rel.values, sub.p_up.values)
    return mk, g


def simulate(mk, g, t_entry=-45, pe=0.50, tps=(0.55, 0.58, 0.60, 0.62, 0.65, 0.70, 0.75, 0.80),
             sls=(0.25, 0.30, 0.35, 0.40, 0.42, 0.45, 0.47), tstops=(240, 270, 290, 300), slip=0.01,
             entry_from_trades=False, entry_taker=False, entry_fee=False, entry_next_trade=False,
             entry_mode=None, t_cancel=0, bid=0.50, trail=None):
    """Entry modes:
       'fixed'  (default): filled at `pe` at t_entry (assumed).
       'maker' : resting bid at `bid` placed at t_entry; fills at the first print of our side at <= bid
                 with t_entry <= rel <= t_cancel (otherwise no trade). Exits use prints after the fill.
       'taker' : buy at the first print after t_entry (+slip, +fee).
    Legacy flags entry_from_trades/entry_next_trade map onto 'taker'/'last print'."""
    if entry_mode is None:
        entry_mode = "taker" if (entry_from_trades and entry_next_trade) else ("lastprint" if entry_from_trades else "fixed")
    tps, sls, tstops = np.array(tps), np.array(sls), np.array(tstops)
    W = sorted(g.keys()); n = len(W)
    shape = (n, 2, len(tps), len(sls), len(tstops))
    pnl = np.full(shape, np.nan, np.float32); kind = np.full(shape, -1, np.int8); texit = np.full(shape, np.nan, np.float32)
    settle = np.full((n, 2), np.nan, np.float32); entry_px = np.full((n, 2), np.nan, np.float32)
    pre_last = np.full((n, 2), np.nan, np.float32); fill_rel = np.full((n, 2), np.nan, np.float32)
    for i, w in enumerate(W):
        rel, pup = g[w]
        up = mk.at[w, "outcome_up_price"] == 1.0
        pre = rel < t_entry
        for s in range(2):
            ps = pup if s == 0 else (1 - pup)
            if pre.any():
                pre_last[i, s] = ps[pre][-1]
            win = (up if s == 0 else (not up))
            post_ = (rel >= t_entry) & (rel <= 300)
            if entry_mode == "fixed":
                pe_s = pe; j0 = int(np.searchsorted(rel, t_entry, side="left")); frel = t_entry
            elif entry_mode == "maker":
                cand = post_ & (rel <= t_cancel) & (ps <= bid)
                if not cand.any():
                    continue
                j = int(np.flatnonzero(cand)[0]); pe_s = bid; j0 = j + 1; frel = rel[j]
            elif entry_mode == "maker_last":
                # join the bid at our side's last print; fills at the first print <= that price before t_cancel
                if not pre.any():
                    continue
                b_ = float(ps[pre][-1]); cand = post_ & (rel <= t_cancel) & (ps <= b_)
                if not cand.any():
                    continue
                j = int(np.flatnonzero(cand)[0]); pe_s = b_; j0 = j + 1; frel = rel[j]
            elif entry_mode == "taker":
                if not post_.any():
                    continue
                j = int(np.flatnonzero(post_)[0]); pe_s = float(ps[j]) + slip
                if entry_fee:
                    pe_s += fee(np.clip(pe_s, 0.001, 0.999))
                j0 = j + 1; frel = rel[j]
            else:  # lastprint
                if not pre.any():
                    continue
                pe_s = float(ps[pre][-1]) + (slip if entry_taker else 0.0)
                if entry_fee:
                    pe_s += fee(np.clip(pe_s, 0.001, 0.999))
                j0 = int(np.searchsorted(rel, t_entry, side="left")); frel = t_entry
            entry_px[i, s] = pe_s; fill_rel[i, s] = frel
            settle[i, s] = (1.0 if win else 0.0) - pe_s
            m = np.zeros(len(rel), bool); m[j0:] = True; m &= rel <= 300
            t, p = rel[m], ps[m]
            if len(p) == 0:
                # nothing printed after entry: time stop at entry price (no information) minus costs
                px = pe_s - slip; f = fee(np.clip(px, 0.001, 0.999))
                pnl[i, s] = px - pe_s - f; kind[i, s] = 2; texit[i, s] = tstops[None, None, :]
                continue
            rm = np.maximum.accumulate(p); rn = np.minimum.accumulate(p)
            i_tp = np.searchsorted(rm, tps, side="left"); i_sl = np.searchsorted(-rn, -sls, side="left"); i_ts = np.searchsorted(t, tstops, side="right")
            if trail is not None:
                # trailing stop: once a print >= arm has occurred, the stop becomes `lock`
                arm, lock = trail
                armed = rm >= arm
                hit = np.flatnonzero(armed & (p <= lock))
                i_tr = hit[0] if len(hit) else len(p)
                i_sl = np.minimum(i_sl, i_tr)
            I_tp = i_tp[:, None, None]; I_sl = i_sl[None, :, None]; I_ts = i_ts[None, None, :]
            first = np.minimum(np.minimum(I_tp, I_sl), I_ts)
            tp_px = np.broadcast_to(tps[:, None, None], first.shape)
            sl_px = np.broadcast_to(np.where(I_sl < len(p), p[np.minimum(I_sl, len(p) - 1)], np.nan), first.shape)
            last_before = np.maximum(I_ts - 1, 0)
            ts_px = np.where(I_ts > 0, p[np.minimum(last_before, len(p) - 1)], pe_s)
            ts_px = np.broadcast_to(ts_px, first.shape)
            is_tp = (first == I_tp) & (I_tp < len(p)); is_sl = (first == I_sl) & (I_sl < len(p)) & ~is_tp; is_ts = ~is_tp & ~is_sl
            px = np.where(is_tp, tp_px, np.where(is_sl, sl_px - slip, ts_px - slip))
            f = np.where(is_tp, 0.0, fee(np.clip(px, 0.001, 0.999)))
            pnl[i, s] = px - pe_s - f
            kind[i, s] = np.where(is_tp, 0, np.where(is_sl, 1, 2))
            texit[i, s] = np.where(is_tp, t[np.minimum(I_tp, len(t) - 1)], np.where(is_sl, t[np.minimum(I_sl, len(t) - 1)], tstops[None, None, :]))
    return dict(W=np.array(W), pnl=pnl, kind=kind, texit=texit, settle=settle, entry_px=entry_px, pre_last=pre_last, fill_rel=fill_rel,
                tps=tps, sls=sls, tstops=tstops, t_entry=t_entry, pe=pe, entry_mode=entry_mode)


def path_grid(g, t0=-60, t1=300):
    """UP-equivalent last-trade price per second [t0, t1] per window (NaN before first trade)."""
    W = sorted(g.keys()); n = len(W); m = t1 - t0 + 1
    out = np.full((n, m), np.nan, np.float32)
    for i, w in enumerate(W):
        rel, pup = g[w]
        sel = (rel >= t0) & (rel <= t1)
        r, p = rel[sel], pup[sel]
        if len(r) == 0:
            continue
        last = np.full(m, np.nan, np.float32)
        # last price at each second
        idx = r - t0
        last[idx] = p          # later trades overwrite earlier within the same second (sorted by ts)
        mask = np.isnan(last); j = np.where(~mask, np.arange(m), 0); np.maximum.accumulate(j, out=j)
        out[i] = np.where(np.isnan(last[j]), np.nan, last[j])
    return np.array(W), out


if __name__ == "__main__":
    import time
    t = time.time()
    mk, g = load_windows()
    print("windows", len(g), f"{time.time()-t:.0f}s")
    sim = simulate(mk, g)
    print("sim", f"{time.time()-t:.0f}s")
    tps, sls, tstops = sim["tps"], sim["sls"], sim["tstops"]
    print("== unconditional per-share P&L (both sides averaged; random side) by TP x SL, time stop 290 ==")
    k = list(tstops).index(290)
    tab = pd.DataFrame(np.nanmean(sim["pnl"][:, :, :, :, k], axis=(0, 1)), index=[f"tp{x}" for x in tps], columns=[f"sl{y}" for y in sls])
    print(tab.round(4).to_string())
    print("== P(exit by TP) ==")
    print(pd.DataFrame((sim["kind"][:, :, :, :, k] == 0).mean(axis=(0, 1)), index=[f"tp{x}" for x in tps], columns=[f"sl{y}" for y in sls]).round(3).to_string())
    print("== P(exit by SL) ==")
    print(pd.DataFrame((sim["kind"][:, :, :, :, k] == 1).mean(axis=(0, 1)), index=[f"tp{x}" for x in tps], columns=[f"sl{y}" for y in sls]).round(3).to_string())
    print("settlement pnl mean (random side):", np.nanmean(sim["settle"]).round(4))
    # perfect foresight upper bound: choose the winning side
    up = (mk.loc[sim["W"], "outcome_up_price"].values == 1.0).astype(int)
    side = np.where(up == 1, 0, 1)
    best = sim["pnl"][np.arange(len(side)), side, :, :, k]
    print("== P&L if side were known (upper bound), TP x SL ==")
    print(pd.DataFrame(np.nanmean(best, axis=0), index=[f"tp{x}" for x in tps], columns=[f"sl{y}" for y in sls]).round(4).to_string())
