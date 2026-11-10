"""Audit of the paper trader for one UTC day: did it run, decide on time, follow the rules, and does
the live model agree with the offline pipeline?

  PAPER_DIR=/home/user/paper_eth python3 polyeth/paper_audit.py [--day YYYY-MM-DD] [--sample 4] [--no-replay] [--no-exits]

Sections
  1. coverage   every 5-minute window the trader was up for has an A and a B decision, on time (A at T-44 s, B at T+1.6 s)
  2. rules      a trade exists exactly when the frozen rule says so (A: plausible, |p-0.5| >= 0.05, btc-predict agrees;
                B: |p - market| >= 0.12 -> bid; filled -> trade), from forecasts.jsonl / orders.jsonl / paper_trades.jsonl
  3. parity     the live forecasts are recomputed offline from the archive data for the same seconds (eth3 close
                head and btc-predict z); differences come only from the live feeds (REST/websocket spot, OKX perp proxy)
  5. latency    decision delay, model compute time, feed freshness at each decision, B bid placement time
  4. exits      every trade's exit is replayed on Polymarket's own trade history (REST) and compared with what the
                websocket feed produced; B maker fills are replayed the same way; A pre-open fill plausibility is reported
Writes PAPER_DIR/audit_<day>.json with the numbers.
"""
import os, sys, json, argparse, datetime as dt, time
import numpy as np, pandas as pd, requests

HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import paper_trader as pt  # noqa: E402

DAY = 86400
PAPER = pt.PAPER
pt.LOG = open(os.path.join(PAPER, "audit.log"), "a")      # keep the audit's own lines out of paper.log


def load_jsonl(name):
    p = os.path.join(PAPER, name)
    if not os.path.exists(p): return []
    out = []
    for line in open(p):
        line = line.strip()
        if line:
            try: out.append(json.loads(line))
            except Exception: pass
    return out


def day_bounds(day):
    t0 = int(dt.datetime(day.year, day.month, day.day, tzinfo=dt.UTC).timestamp())
    return t0, t0 + DAY


# ------------------------------------------------------------------------------------ 0. bias / data integrity
def bias(day, fcs):
    """Day-level sanity of the live forecasts themselves (no archive needed): the frozen model is never one-sided for long in the
    archive (over 903 stretches of 30 consecutive windows the share of A forecasts below 0.5 never left 0.13-0.87 and the mean
    close head stayed within 0.465-0.559), and a decision must not have used synthetic perp or forward-filled spot seconds."""
    t0, t1 = day_bounds(day); res = {}
    for st, head in (("A", "close_p"), ("B", "close_p"), ("D", "close_p")):
        R = [r for r in fcs if r["strategy"] == st and t0 <= r["T"] < t1 and np.isfinite(r.get(head, np.nan))]
        if len(R) < 10: continue
        p = np.array([r[head] for r in R]); down = float((p < 0.5).mean()); one = max(down, 1 - down)
        dev = np.abs(p - 0.5); spread = float(dev.mean()); mx = float(dev.max()); conf = float((dev >= 0.05).mean())
        syn = [r.get("synth_last60") for r in R if r.get("synth_last60") is not None]; ff = [r.get("ffill_last60") for r in R if r.get("ffill_last60") is not None]
        res[st] = dict(n=len(R), mean=float(p.mean()), share_below_half=down, one_sided=one, spread=spread, max_dev=mx, confident_share=conf, synth_decisions=int(sum(1 for x in syn if x)), ffill_decisions=int(sum(1 for x in ff if x)), counted=len(syn))
        flag = []
        if st == "A" and len(R) >= 26 and mx < 0.03: flag.append(f"FLAT (max |p-0.5| {mx:.3f} over {len(R)} windows; the archive never has 26 consecutive windows all within 0.025 of 0.5)")
        if st == "A" and len(R) >= 50 and spread < 0.015: flag.append(f"FLAT (mean |p-0.5| {spread:.3f}; archive 0.029-0.038 over a period, 0.012-0.06 over a day)")
        if one > 0.80: flag.append(f"ONE-SIDED ({one:.0%} of forecasts on one side; archive max over 30-window stretches 0.87, over a day 0.62)")
        if st == "A" and not 0.46 <= p.mean() <= 0.56: flag.append(f"MEAN HEAD {p.mean():.3f} outside the archive range 0.465-0.559")
        if res[st]["synth_decisions"]: flag.append(f"{res[st]['synth_decisions']} decisions used synthetic perp seconds")
        if res[st]["ffill_decisions"]: flag.append(f"{res[st]['ffill_decisions']} decisions used forward-filled spot seconds")
        print(f"bias {st}: {len(R)} forecasts, mean {p.mean():.3f}, below 0.5 {down:.0%}, mean |p-0.5| {spread:.3f} (max {mx:.3f}), >= 0.05 in {conf:.0%}" + (" (archive: 16-28% of windows, mean |p-0.5| 0.029-0.038)" if st == "A" else "") + f"; incomplete-bar counters on {len(syn)} records" + (f"  FLAGS: {'; '.join(flag)}" if flag else "  ok"))
        res[st]["flags"] = flag
    if not res: print("bias: fewer than 10 forecasts per strategy, nothing to judge")
    return res


# ------------------------------------------------------------------------------------ 1. coverage
def coverage(day, fcs):
    t0, t1 = day_bounds(day)
    A = {r["T"]: r for r in fcs if r["strategy"] == "A" and t0 <= r["T"] < t1}
    B = {r["T"]: r for r in fcs if r["strategy"] == "B" and t0 <= r["T"] < t1}
    Ts = list(range(t0 + 300, t1 + 1, 300))             # window opens inside the day
    seen = sorted(set(A) | set(B))
    res = {"windows_in_day": len(Ts), "first_seen": None, "last_seen": None, "A_records": len(A), "B_records": len(B), "gaps": [], "late_A": [], "late_B": []}
    if not seen:
        print(f"coverage: no decisions recorded for {day}"); return res
    res["first_seen"] = dt.datetime.fromtimestamp(seen[0], dt.UTC).strftime("%H:%M"); res["last_seen"] = dt.datetime.fromtimestamp(seen[-1], dt.UTC).strftime("%H:%M")
    # gaps: windows between first and last seen with neither record (= downtime or a stuck loop)
    run = []
    for T in Ts:
        if seen[0] <= T <= seen[-1] and T not in A and T not in B: run.append(T)
        elif run:
            res["gaps"].append((run[0], run[-1])); run = []
    if run: res["gaps"].append((run[0], run[-1]))
    for T, r in A.items():
        if "ts" in r and r["ts"] - (T - 44) > 3: res["late_A"].append((T, round(r["ts"] - (T - 44), 1)))
    for T, r in B.items():
        if "ts" in r and r["ts"] - (T + 1.6) > 3: res["late_B"].append((T, round(r["ts"] - (T + 1.6), 1)))
    up = len([T for T in Ts if seen[0] <= T <= seen[-1]])
    print(f"coverage {day}: trader seen {res['first_seen']}-{res['last_seen']} UTC; windows in that span {up}, with A decision {len(A)}, with B decision {len(B)}")
    if res["gaps"]:
        for a, b in res["gaps"]:
            print(f"  GAP {dt.datetime.fromtimestamp(a, dt.UTC):%H:%M} .. {dt.datetime.fromtimestamp(b, dt.UTC):%H:%M} ({(b - a) // 300 + 1} windows without any decision)")
    else:
        print("  no gaps")
    print(f"  late decisions (> 3 s after the scheduled instant): A {len(res['late_A'])}, B {len(res['late_B'])}" + (f"  e.g. {res['late_A'][:3]} {res['late_B'][:3]}" if res["late_A"] or res["late_B"] else ""))
    return res


# ------------------------------------------------------------------------------------ 2. rules
def rules(day, fcs, trades, orders):
    t0, t1 = day_bounds(day)
    tr = {(r["strategy"], r["T"]): r for r in trades if t0 <= r["T"] < t1}
    res = {"A_checked": 0, "A_expected_trades": 0, "A_mismatch": [], "A_legacy": 0, "B_checked": 0, "B_expected_bids": 0, "B_mismatch": [], "B_filled": 0, "B_cancelled": 0}
    for r in fcs:
        if not (t0 <= r["T"] < t1): continue
        p = r.get("close_p", np.nan); T = r["T"]
        if r["strategy"] == "A":
            if "action" not in r: res["A_legacy"] += 1; continue
            res["A_checked"] += 1
            z = r.get("z_s5", np.nan); z = np.nan if z is None else z
            plausible = np.isfinite(r.get("p0", np.nan) or np.nan) and np.isfinite(p) and 0.05 < p < 0.95
            agree = np.isfinite(z) and np.sign(z) == np.sign(p - 0.5)
            expect = plausible and abs(p - 0.5) >= pt.CFG_A["thr"] and agree
            side = "UP" if p > 0.5 else "DOWN"
            want_action = f"BUY {side}" if expect else ("none" if plausible else "skip")
            if r.get("rule", "old") == "new":      # the purpose-built model: trade when |score| >= threshold, side = sign of the score; the old heads are only logged
                ns = r.get("new_score"); ns = np.nan if ns is None else ns; thr = r.get("new_thr") or np.nan
                plausible = bool(np.isfinite(r.get("p0", np.nan) or np.nan) and np.isfinite(ns))
                expect = bool(plausible and abs(ns) >= thr and r.get("market") is not None and np.isfinite(r.get("market") or np.nan)); side = "UP" if ns > 0 else "DOWN"
                want_action = f"BUY {side}" if expect else ("none" if plausible else "skip"); p, z = ns, thr
            if r["action"] == "skip-data":         # the data-quality gate refused a signal: there must be no trade
                res["A_gate_refused"] = res.get("A_gate_refused", 0) + 1
                if ("A", T) in tr: res["A_mismatch"].append(dict(T=T, p=round(float(p), 4), action=r["action"], want="no trade", trade=True))
                continue
            have = ("A", T) in tr
            if expect and not have and T + 300 > time.time(): res.setdefault("A_open", []).append(T); continue   # position still open
            if r["action"] != want_action or have != expect or (have and tr[("A", T)]["side"] != side):
                res["A_mismatch"].append(dict(T=T, p=round(p, 3), z=round(z, 3) if np.isfinite(z) else None, action=r["action"], want=want_action, trade=have))
            res["A_expected_trades"] += int(expect)
        else:
            if "action" not in r: continue
            res["B_checked"] += 1
            mkt = r.get("market", np.nan); mkt = np.nan if mkt is None else mkt
            plausible = np.isfinite(r.get("p0", np.nan) or np.nan) and np.isfinite(p) and 0.05 < p < 0.95
            cfgx = pt.CFGS.get(r["strategy"], pt.CFG_B); st = r["strategy"]
            if st == "D":
                plausible = np.isfinite(r.get("p0", np.nan) or np.nan) and np.isfinite(p) and 0.01 < p < 0.99
                side = "UP" if p >= 0.5 else "DOWN"; psm = p if side == "UP" else 1 - p; psk = mkt if side == "UP" else 1 - mkt
                expect = plausible and np.isfinite(mkt) and psm >= cfgx["q"] and cfgx.get("minpx", 0.0) <= psk <= cfgx["maxpx"]
            else:
                expect = plausible and np.isfinite(mkt) and abs(p - mkt) >= cfgx["delta"]
                side = "UP" if p > mkt else "DOWN"
            want_action = f"BID {side}" if expect else ("none" if plausible else "skip")
            ev = [o for o in orders if o["T"] == T and o.get("strategy", "B") == st]
            placed = any(o["event"] == "placed" for o in ev); filled = any(o["event"] == "filled" for o in ev); cancelled = any(o["event"] == "cancelled" for o in ev)
            have = (st, T) in tr
            if filled and not have and T + cfgx["tstop"] > time.time(): res.setdefault(f"{st}_open", []).append(T); continue   # position still open
            ok = r["action"] == want_action and placed == expect and (not placed or filled != cancelled) and have == filled
            if not ok:
                res["B_mismatch"].append(dict(T=T, p=round(p, 3), market=mkt, action=r["action"], want=want_action, placed=placed, filled=filled, cancelled=cancelled, trade=have))
            res["B_expected_bids"] += int(expect); res["B_filled"] += int(filled); res["B_cancelled"] += int(cancelled)
    print(f"rules A: {res['A_checked']} decisions checked ({res['A_legacy']} legacy records without inputs skipped), {res['A_expected_trades']} should have traded, mismatches {len(res['A_mismatch'])}")
    for m in res["A_mismatch"][:10]: print("   ", m)
    print(f"rules B+C: {res['B_checked']} decisions checked, {res['B_expected_bids']} should have bid, filled {res['B_filled']}, cancelled {res['B_cancelled']}, mismatches {len(res['B_mismatch'])}")
    for m in res["B_mismatch"][:10]: print("   ", m)
    return res


# ------------------------------------------------------------------------------------ 3. model parity
def parity(day, fcs, sample, force_rest=False):
    t0, t1 = day_bounds(day)
    A = {r["T"]: r for r in fcs if r["strategy"] == "A" and t0 <= r["T"] < t1}
    B = {r["T"]: r for r in fcs if r["strategy"] == "B" and t0 <= r["T"] < t1}
    if not A and not B:
        print("parity: nothing to replay"); return {}
    tl0 = (t0 // DAY - 10) * DAY
    tl = pt.LiveTimeline(tl0)
    a_end = pt.load_archive(tl)
    if a_end >= t1:
        tl.finalize_to(t1)
        src = "archive"
    elif force_rest:
        print(f"parity: archive ends {dt.datetime.fromtimestamp(a_end, dt.UTC)}; REST-backfilling the rest (perp synthetic from spot for that part)")
        pt.backfill(tl, a_end); src = "archive+REST"
    else:
        # without the day's Binance archive the perp would be synthesized from spot, which moves the A head by
        # 0.03 on average (0.10 max, 1 direction flip in 4 measured on 2026-11-03): not a parity test. Run after the
        # daily data refresh (archive for day D is published on D+1), or pass --force-rest to see it anyway.
        print(f"parity: archive ends {dt.datetime.fromtimestamp(a_end, dt.UTC)} and does not cover {day}: skipped (re-run after the data refresh, or --force-rest)")
        return {"skipped": "archive does not cover the day"}
    from ep_live import EPLive as BPLive   # eth1 spot variant (forecast(spot, lead, t0))
    bp = BPLive(); model = pt.Model(tl)
    from newa_live import NewA
    newa = NewA()
    rows = []
    Ts = sorted(set(A) | set(B))[::max(1, sample)]
    t_w = time.time()
    for i, T in enumerate(Ts):
        row = dict(T=T)
        if T in A:
            fc = model.forecast(T, -45, 0); row["A_live"] = A[T].get("close_p"); row["A_replay"] = fc.get("close_p")
            if A[T].get("new_score") is not None:
                try:
                    i45 = (T - 45) - tl.t0_epoch; row["N_live"] = A[T]["new_score"]; row["N_replay"] = newa.score(model.last_X, tl.view("s_close", i45 - 15, i45))[0]; row["N_thr"] = A[T].get("new_thr")
                except Exception as ex:
                    row["n_err"] = repr(ex)[:80]
            if "z_s5" in A[T]:
                try:
                    lo = (T - 60) - 60 * 2880 - 900; i0 = lo - tl.t0_epoch; i1 = (T - 60) - tl.t0_epoch
                    v = lambda k: tl.view(k, i0, i1)
                    spot = dict(o=v("s_open"), h=v("s_high"), l=v("s_low"), c=v("s_close"), v=v("s_volume"), qv=v("s_qvol"), n=v("s_count"), bv=v("s_tb_base"))
                    um = dict(o=v("p_open"), h=v("p_high"), l=v("p_low"), c=v("p_close"), v=v("p_buy") + v("p_sell"), qv=v("p_qv"), n=v("p_ntr"), bv=v("p_buy"), nb=v("p_nb"), lbv=v("p_lbv"), lsv=v("p_lsv"), mxq=v("p_maxq"))
                    eth = dict(c=v("e_c"), v=v("e_v"), bv=v("e_bv"))
                    z, _ = bp.forecast(spot, eth, lo); row["z_live"] = A[T].get("z_s5"); row["z_replay"] = z["s5"]
                except Exception as ex:
                    row["z_err"] = repr(ex)[:80]
        if T in B:
            fc = model.forecast(T, -pt.LAG, pt.LAG); row["B_live"] = B[T].get("close_p"); row["B_replay"] = fc.get("close_p"); row["B_market"] = B[T].get("market")
        rows.append(row)
        if (i + 1) % 20 == 0: print(f"  replayed {i + 1}/{len(Ts)} windows ({time.time() - t_w:.0f}s)")
    df = pd.DataFrame(rows); res = {"source": src, "windows": len(df)}
    if "N_live" in df and df.N_live.notna().any():      # the rule that trades: new-model score, live vs recomputed from the archive
        ok = df.N_live.notna() & df.N_replay.notna(); d = (df.N_live - df.N_replay).abs()[ok]; th = df.N_thr[ok].astype(float)
        dec = ((df.N_live[ok].abs() >= th) != (df.N_replay[ok].abs() >= th)).sum(); fl = (np.sign(df.N_live[ok]) != np.sign(df.N_replay[ok])).sum()
        res.update(N_n=int(ok.sum()), N_mean_abs=float(d.mean()), N_max_abs=float(d.max()), N_sign_flips=int(fl), N_decisions_differ=int(dec))
        print(f"parity NEW A score (live vs archive): {int(ok.sum())} windows, mean |diff| {d.mean():.4f}, max {d.max():.4f} (threshold {th.iloc[0]:.4f}); side flips {int(fl)}, trade decisions that differ {int(dec)}")
    for k, thr in (("A", 0.05), ("B", 0.12)):
        if f"{k}_live" not in df: continue
        d = (df[f"{k}_replay"] - df[f"{k}_live"]).abs(); ok = d.notna()
        sg = (df[f"{k}_live"] - df[f"{k}_replay"])[ok]
        if len(sg):
            res[f"{k}_signed_mean"] = float(sg.mean()); res[f"{k}_share_live_above"] = float((sg > 0).mean())
            print(f"parity {k} signed (live - archive): mean {sg.mean():+.4f}, live above the archive in {(sg > 0).mean():.0%} of windows" + ("  FLAG: systematic shift" if abs(sg.mean()) > 0.01 and (abs((sg > 0).mean() - 0.5) > 0.2) else ""))
        res[f"{k}_mean_abs_dp"] = float(d[ok].mean()); res[f"{k}_max_abs_dp"] = float(d[ok].max()); res[f"{k}_n"] = int(ok.sum())
        if k == "A":
            flips = ((np.sign(df.A_live - 0.5) != np.sign(df.A_replay - 0.5)) & ok).sum()
            dec_live = (df.A_live - 0.5).abs() >= thr; dec_rep = (df.A_replay - 0.5).abs() >= thr
        else:
            flips = ((np.sign(df.B_live - df.B_market) != np.sign(df.B_replay - df.B_market)) & ok & df.B_market.notna()).sum()
            dec_live = (df.B_live - df.B_market).abs() >= thr; dec_rep = (df.B_replay - df.B_market).abs() >= thr
        res[f"{k}_sign_flips"] = int(flips); res[f"{k}_threshold_decisions_differ"] = int(((dec_live != dec_rep) & ok).sum())
        print(f"parity {k} ({src}, {int(ok.sum())} windows): mean |dp| {d[ok].mean():.4f}, max {d[ok].max():.4f}, direction flips {flips}, trade/no-trade decisions that differ {res[f'{k}_threshold_decisions_differ']}")
    if "z_live" in df:
        dz = (df.z_replay - df.z_live).abs(); ok = dz.notna()
        flips = ((np.sign(df.z_live) != np.sign(df.z_replay)) & ok).sum()
        res["z_mean_abs_dz"] = float(dz[ok].mean()); res["z_sign_flips"] = int(flips)
        print(f"parity btc-predict z ({int(ok.sum())} windows): mean |dz| {dz[ok].mean():.4f}, max {dz[ok].max():.4f}, sign flips {flips}")
    worst = df.assign(d=(df.get("A_replay", np.nan) - df.get("A_live", np.nan)).abs()).sort_values("d", ascending=False).head(5)
    if len(worst) and worst.d.notna().any():
        print("  largest A differences:"); print(worst[[c for c in ("T", "A_live", "A_replay", "z_live", "z_replay") if c in worst]].round(4).to_string(index=False))
    df.to_csv(os.path.join(PAPER, f"audit_parity_{day}.csv"), index=False)
    return res


# ------------------------------------------------------------------------------------ 4. exits
def pm_prints(cond):
    """UP-equivalent prints (ts seconds, up price, size) from Polymarket's trade history."""
    rows, cursor = [], None
    for _ in range(200):
        p = dict(condition=cond, limit=1000)
        if cursor: p["cursor"] = cursor
        try:
            d = requests.get("https://data-api.polymarket.com/v2/trades", params=p, timeout=30).json()
        except Exception as ex:
            print("  data-api error", repr(ex)[:80]); break
        for r in d.get("data") or []:
            px = float(r["price"]); up = px if int(r["outcome_index"]) == 0 else 1 - px
            rows.append((int(r["timestamp"]), up, float(r["size"])))
        pg = d.get("pagination") or {}; cursor = pg.get("next_cursor")
        if not pg.get("has_more") or not cursor: break
    return sorted(rows)


def replay_exit(rec, prints, cfg, outcome=None):
    cfg = dict(cfg, tp=rec.get("tp", cfg["tp"]), sl=rec.get("sl", cfg.get("sl")), tstop=rec.get("tstop", cfg.get("tstop")))   # the trade's own exits
    side = rec["side"]; T = rec["T"]; fill = T + rec["fill_time"]; tstop = cfg["tstop"] if cfg["tstop"] is not None else 300
    for ts, up, sz in prints:
        if ts < fill - 1: continue
        if ts > T + tstop + 3: break                           # the time-stop fires before later prints (history stamps lag the match by ~2 s)
        ps = up if side == "UP" else 1 - up
        if cfg["tp"] is not None and ps >= cfg["tp"]: return "TP", ts - T, cfg["tp"], 0.0
        if cfg["sl"] is not None and ps <= cfg["sl"]: return "SL", ts - T, ps - cfg["slip"], pt.fee(ps - cfg["slip"], cfg["fee"])
    if cfg["tstop"] is None:                                   # hold to settlement
        if outcome is None: return "NONE", None, None, None
        return "SETTLE", 300, 1.0 if (outcome == 1) == (side == "UP") else 0.0, 0.0
    last = [x for x in prints if x[0] <= T + cfg["tstop"]]
    if last:
        up = last[-1][1]; ps = (up if side == "UP" else 1 - up) - cfg["slip"]
        return "TIME", cfg["tstop"], ps, pt.fee(ps, cfg["fee"])
    return "NONE", None, None, None


def exits(day, trades, orders):
    t0, t1 = day_bounds(day)
    fresh = time.time() - 900                                   # Polymarket's trade history is indexed with a lag of a few minutes
    tr = [r for r in trades if t0 <= r["T"] < t1 and r["T"] + 300 < fresh]
    ords = [o for o in orders if t0 <= o["T"] < t1 and o["event"] == "placed" and o["T"] + 300 < fresh]
    skipped = len([r for r in trades if t0 <= r["T"] < t1 and r["T"] + 300 >= fresh])
    if skipped: print(f"exits: {skipped} trade(s) from the last 15 minutes not replayed yet (history still being indexed); re-run later")
    res = {"trades": len(tr), "exit_mismatch": [], "fill_mismatch": [], "A_fill_plausible": 0, "A_trades": 0, "pnl_live": 0.0, "pnl_replay": 0.0}
    conds = {}
    for T in sorted(set([r["T"] for r in tr] + [o["T"] for o in ords])):
        tk = pt.get_tokens(T); conds[T] = tk[2] if tk else None
    cache = {}
    for r in tr:
        cond = conds.get(r["T"])
        if not cond: continue
        if cond not in cache: cache[cond] = pm_prints(cond)
        pr = cache[cond]; cfg = pt.CFGS.get(r["strategy"], pt.CFG_B)
        kind, et, px, f = replay_exit(r, pr, cfg, outcome=pt.get_outcome(r["T"]) if cfg.get("settle") else None)
        pnl = (px - r["entry"] - f) if px is not None else np.nan
        res["pnl_live"] += r["pnl"]; res["pnl_replay"] += (pnl if np.isfinite(pnl) else 0)
        tol = 0.10 if r["exit_kind"] == "TIME" else 0.02                   # time-stops sell at the live order-book bid, which prints cannot show
        if kind == "SETTLE": et = r["exit_time"]                            # settlement has no exit instant
        same = kind == r["exit_kind"] and (et is None or abs(et - r["exit_time"]) <= 5) and (np.isfinite(pnl) and abs(pnl - r["pnl"]) <= tol)   # REST history has 1 s timestamps
        if not same:
            res["exit_mismatch"].append(dict(strategy=r["strategy"], T=r["T"], live=(r["exit_kind"], round(r["exit_time"], 1), round(r["pnl"], 3)), replay=(kind, et, None if not np.isfinite(pnl) else round(pnl, 3)), prints=len(pr)))
        if r["strategy"] == "A":
            res["A_trades"] += 1
            side = r["side"]
            pre = [1 for ts, up, sz in pr if r["T"] - 60 <= ts <= r["T"] - 40 and (up if side == "UP" else 1 - up) <= r["entry"] + 1e-9]
            res["A_fill_plausible"] += int(bool(pre))
    for o in ords:
        cond = conds.get(o["T"])
        if not cond: continue
        if cond not in cache: cache[cond] = pm_prints(cond)
        side = o["side"]; st = o.get("strategy", "B"); until = o["T"] if st == "C" else o["placed"] + pt.CFGS.get(st, pt.CFG_B)["cancel"]
        hit = [ts for ts, up, sz in cache[cond] if o["placed"] - 1 <= ts <= until + 2 and (up if side == "UP" else 1 - up) <= o["bid"] + 1e-9]
        ev = [x for x in orders if x["T"] == o["T"] and x.get("strategy", "B") == st and x["event"] in ("filled", "cancelled")]
        live_filled = any(x["event"] == "filled" for x in ev)
        if bool(hit) != live_filled:
            res["fill_mismatch"].append(dict(T=o["T"], side=side, bid=o["bid"], live_filled=live_filled, replay_hit=bool(hit)))
    print(f"exits: {len(tr)} trades replayed on Polymarket trade history; exit mismatches {len(res['exit_mismatch'])}; live P&L {res['pnl_live']:+.3f} vs replay {res['pnl_replay']:+.3f} per share")
    for m in res["exit_mismatch"][:10]: print("   ", m)
    print(f"B maker fills: {len(ords)} bids replayed; fill/no-fill mismatches {len(res['fill_mismatch'])}")
    for m in res["fill_mismatch"][:10]: print("   ", m)
    if res["A_trades"]:
        print(f"A pre-open fill plausibility: {res['A_fill_plausible']}/{res['A_trades']} trades had a print at <= 0.50 on our side in [T-60, T-40] (a resting 0.50 bid needs a seller; assumed fills are not checked by the simulator; information only)")
    return res


# ------------------------------------------------------------------------------------ 5. latency
def latency(day, fcs, orders):
    """Timing at every decision: how late after the scheduled instant, model compute time, and how fresh each feed was."""
    t0, t1 = day_bounds(day)
    rec = [r for r in fcs if t0 <= r["T"] < t1 and "decision_delay" in r]
    res = {"decisions_with_timing": len(rec)}
    if not rec:
        print("latency: no decision records with timing fields yet (recorded from the version with latency instrumentation on)"); return res
    df = pd.DataFrame(rec)
    def q(x): x = pd.to_numeric(x, errors="coerce").dropna(); return (f"median {x.median():.2f}s  p95 {x.quantile(.95):.2f}s  max {x.max():.2f}s" if len(x) else "n/a")
    for k, lab in (("decision_delay", "decision after scheduled instant"), ("compute_s", "model compute"), ("spot_lag", "spot feed age (last bar)"), ("perp_lag", "perp feed age (last trade)"), ("pm_age", "Polymarket feed age (last print)")):
        if k in df:
            print(f"latency {lab:34s}: {q(df[k])}"); x = pd.to_numeric(df[k], errors="coerce").dropna(); res[k] = dict(median=float(x.median()), p95=float(x.quantile(.95)), max=float(x.max())) if len(x) else None
    stale = df[pd.to_numeric(df.get("spot_lag"), errors="coerce") > 3] if "spot_lag" in df else df.iloc[0:0]
    res["decisions_on_stale_spot"] = int(len(stale))
    print(f"latency decisions made on a stale spot feed (> 3 s, forward-filled): {len(stale)}" + (f"  e.g. {[dt.datetime.fromtimestamp(t, dt.UTC).strftime('%H:%M') for t in stale['T'].head(5)]}" if len(stale) else ""))
    pl = [o for o in orders if t0 <= o["T"] < t1 and o["event"] == "placed" and "placed_delay" in o]
    if pl:
        x = pd.Series([o["placed_delay"] for o in pl]); res["B_bid_placed_after_open"] = dict(median=float(x.median()), max=float(x.max()), n=len(x))
        print(f"latency B bid placed after the open       : {q(x)}  (n={len(x)})")
    b = df[df.strategy == "B"]
    if "market_preT" in b and len(b):
        d = (pd.to_numeric(b["market"], errors="coerce") - pd.to_numeric(b["market_preT"], errors="coerce")).dropna()
        res["B_market_move_T_to_decision"] = dict(mean_abs=float(d.abs().mean()), max_abs=float(d.abs().max()), n=int(len(d)))
        print(f"latency B market move between the last pre-open print (backtest price) and the price used live: mean |d| {d.abs().mean():.3f}, max {d.abs().max():.3f} (n={len(d)})")
    return res


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--day", default=(dt.datetime.now(dt.UTC).date() - dt.timedelta(days=1)).isoformat())
    ap.add_argument("--sample", type=int, default=4, help="replay every n-th window in the parity check")
    ap.add_argument("--no-replay", action="store_true"); ap.add_argument("--no-exits", action="store_true"); ap.add_argument("--force-rest", action="store_true")
    ap.add_argument("--since", default=None, help="HH:MM UTC: only records from that time of the day on (e.g. after a restart on fixed code)")
    a = ap.parse_args(); day = dt.date.fromisoformat(a.day)
    fcs, trades, orders = load_jsonl("forecasts.jsonl"), load_jsonl("paper_trades.jsonl"), load_jsonl("orders.jsonl")
    if a.since:
        hh, mm = a.since.split(":"); t_since = day_bounds(day)[0] + int(hh) * 3600 + int(mm) * 60
        fcs = [r for r in fcs if r.get("ts", r["T"]) >= t_since]; trades = [r for r in trades if r["T"] >= t_since - 300]; orders = [r for r in orders if r["T"] >= t_since - 300]
    print(f"== paper audit {day}  (PAPER_DIR={PAPER})" + (f"  records from {a.since} UTC on" if a.since else ""))
    out = {"day": a.day, "bias": bias(day, fcs), "coverage": coverage(day, fcs), "rules": rules(day, fcs, trades, orders), "latency": latency(day, fcs, orders)}
    if not a.no_exits: out["exits"] = exits(day, trades, orders)
    if not a.no_replay: out["parity"] = parity(day, fcs, a.sample, a.force_rest)
    json.dump(out, open(os.path.join(PAPER, f"audit_{a.day}.json"), "w"), indent=1, default=str)
    print("saved", os.path.join(PAPER, f"audit_{a.day}.json"))
