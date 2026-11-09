"""Ten trade ideas on top of A / B / C, each a different mechanism, all with resting-limit entries and
sell-limit or settlement exits (no fill assumptions beyond 'a print at or through our price'). Every idea has a
small internal grid; the cell is selected on TUNE (min 150 trades) and read on TEST and Aug 1-19. Results are
ranked on win rate and on P&L. Run with LABEL=july (and MASTER_PATH / POLY_* set) for the July check.
"""
import os, sys, itertools, numpy as np, pandas as pd, warnings
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "polyeth"))
import polysim, strategy_eval as se, experiments as ex  # noqa: E402
warnings.filterwarnings("ignore"); pd.set_option("display.width", 260)
R = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "reports")
LABEL = os.environ.get("LABEL", "main"); master, mk, g = ex.master, ex.mk, ex.g
PERS = {"old": se.OLD, "tune": se.TUNE, "test": se.TEST} if LABEL == "main" else {"july": ("2026-07-01", "2026-07-31")}
SEL = "tune" if LABEL == "main" else "july"
rows = []

def per_stats(pnl, ent, dates):
    out = {}
    for nm, per in PERS.items():
        s = (dates >= per[0]) & (dates <= per[1]); x = np.where(s, pnl, np.nan); mt = se.metrics(x, dates); f = s & np.isfinite(pnl)
        d = pd.Series(pnl[f] / ent[f] * 100).groupby(dates[f]).sum() if f.any() else pd.Series(dtype=float)
        out[nm] = dict(n=mt.get("n", 0), mean=mt.get("mean", np.nan), win=mt.get("win", np.nan), usd=float(d.sum()) if len(d) else 0.0,
                       maxdd=float((d.cumsum().cummax() - d.cumsum()).max()) if len(d) else np.nan, tstat=mt.get("tstat", np.nan), entry=float(np.nanmean(ent[f])) if f.any() else np.nan)
    return out

def add(idea, variant, pnl, ent, dates):
    st = per_stats(pnl, ent, dates); r = {"idea": idea, "variant": variant}
    for nm, d in st.items(): r.update({f"{k}_{nm}": v for k, v in d.items()})
    rows.append(r)

def settle_pnl(sim, idx, side, a, tp):
    """pnl of 'sell limit at tp, else hold to settlement' from a simulate() run with sls=(0.0,), tstops=(300,)."""
    k = sim["kind"][idx, side, a, 0, 0]; ent = sim["entry_px"][idx, side]
    return np.where(k == 0, tp - ent, sim["settle"][idx, side])

def run_sim(t_entry, mode, cancel=None, bid=0.50, tps=(0.85,), sls=(0.0,), tstops=(300,)):
    kw = dict(entry_mode=mode)
    if mode in ("maker", "maker_last"): kw["t_cancel"] = cancel
    if mode == "maker": kw["bid"] = bid
    if mode == "fixed": kw["pe"] = bid
    sim = polysim.simulate(mk, g, t_entry=t_entry, tps=tps, sls=sls, tstops=tstops, slip=0.01, **kw); m, sim = ex.align(master, sim)
    return m, sim

def colz(m, k): return m[k].values if k in m else np.full(len(m), np.nan)

# ------------------------------------------------------------------ 1. late B with settlement exits
for e in (30, 60, 120):
    m, sim = run_sim(e, "maker_last", cancel=e + 10, tps=(0.85, 0.90, 0.95, 1.01)); idx = np.arange(len(m)); dates = m.date.values
    mod = colz(m, f"m2_{e}_close_p"); mkt = sim["pre_last"][:, 0]; dis = mod - mkt; side = np.where(dis > 0, 0, 1)
    for delta, (a, tp) in itertools.product((0.08, 0.12, 0.16, 0.20), enumerate((0.85, 0.90, 0.95, 1.01))):
        sel = np.isfinite(dis) & (np.abs(dis) >= delta); pnl = np.where(sel, settle_pnl(sim, idx, side, a, tp), np.nan)
        add("1 late B + settle", f"T+{e} delta {delta} TP {tp if tp < 1 else 'none'}", pnl, sim["entry_px"][idx, side], dates)
    print("idea 1 done", e, flush=True)

# ------------------------------------------------------------------ 2. post-open overshoot fade
for e in (30, 60):
    m, sim = run_sim(e, "maker_last", cancel=e + 10, tps=(0.50, 0.55, 0.60, 0.70, 1.01)); idx = np.arange(len(m)); dates = m.date.values
    mod = colz(m, f"m2_{e}_close_p"); mkt = sim["pre_last"][:, 0]; cheap = np.where(mkt > 0.5, 1, 0)        # buy the side the market has run away from
    for tilt, band, (a, tp) in itertools.product((0.15, 0.20, 0.25), (0.05, 0.10, "opp"), enumerate((0.50, 0.55, 0.60, 0.70, 1.01))):
        near = (np.abs(mod - 0.5) <= band) if band != "opp" else (np.sign(mod - 0.5) == -np.sign(mkt - 0.5))
        sel = np.isfinite(mod) & np.isfinite(mkt) & (np.abs(mkt - 0.5) >= tilt) & near
        pnl = np.where(sel, settle_pnl(sim, idx, cheap, a, tp), np.nan)
        add("2 overshoot fade", f"T+{e} tilt>={tilt} model {band} TP {tp if tp < 1 else 'none'}", pnl, sim["entry_px"][idx, cheap], dates)
    print("idea 2 done", e, flush=True)

# ------------------------------------------------------------------ 3. A signal, bid 0.50 after the open if the model still agrees at T0
p45 = colz(master, "m2_-45_close_p"); z5 = colz(master, "bp_-60_z_s5")
for cancel in (15, 30, 60):
    m, sim = run_sim(0, "maker", cancel=cancel, bid=0.50, tps=(0.80, 0.85, 1.01)); idx = np.arange(len(m)); dates = m.date.values
    p = colz(m, "m2_-45_close_p"); z = colz(m, "bp_-60_z_s5"); p0 = colz(m, "m2_0_close_p"); side = np.where(p > 0.5, 0, 1)
    base = np.isfinite(p) & (np.abs(p - 0.5) >= 0.05) & np.isfinite(z) & (np.sign(z) == np.sign(p - 0.5))
    for conf, (a, tp) in itertools.product((0.50, 0.55, 0.60), enumerate((0.80, 0.85, 1.01))):
        ok = base & np.isfinite(p0) & (np.where(side == 0, p0, 1 - p0) >= conf)
        pnl = np.where(ok, settle_pnl(sim, idx, side, a, tp), np.nan)
        add("3 A confirmed at open, bid 0.50", f"cancel T+{cancel} model@T0>={conf} TP {tp if tp < 1 else 'none'}", pnl, sim["entry_px"][idx, side], dates)
print("idea 3 done", flush=True)

# ------------------------------------------------------------------ 4. both-sides pre-open maker
for bid in (0.45, 0.46, 0.47, 0.48, 0.49):
    m, sim = run_sim(-45, "maker", cancel=0, bid=bid, tps=(1.01,)); idx = np.arange(len(m)); dates = m.date.values
    fu, fd = np.isfinite(sim["fill_rel"][:, 0]), np.isfinite(sim["fill_rel"][:, 1]); both = fu & fd
    pnl_pair = np.where(both, 1.0 - 2 * bid, np.nan)                                               # both filled: pays 1.00 at settlement
    single = (fu ^ fd); pnl_single = np.where(single, np.where(fu, sim["settle"][:, 0], sim["settle"][:, 1]), np.nan)
    pnl_all = np.where(both, 1.0 - 2 * bid, np.where(single, pnl_single, np.nan))                     # per window, pair counts as one trade of 2*bid stake
    ent_all = np.where(both, 2 * bid, bid)
    add("4 both-sides maker", f"bid {bid} pairs only", pnl_pair, np.full(len(m), 2 * bid), dates)
    add("4 both-sides maker", f"bid {bid} singles only", pnl_single, np.full(len(m), bid), dates)
    add("4 both-sides maker", f"bid {bid} all fills", pnl_all, ent_all, dates)
print("idea 4 done", flush=True)

# ------------------------------------------------------------------ 5. A with pre-open consensus (T-60 & T-45 & bp), TP 0.85 then settle
m, sim = run_sim(-45, "fixed", bid=0.50, tps=(0.85,)); idx = np.arange(len(m)); dates = m.date.values
p = colz(m, "m2_-45_close_p"); p60 = colz(m, "m2_-60_close_p"); z = colz(m, "bp_-60_z_s5"); z1 = colz(m, "bp_-60_z_s1"); bm = colz(m, "bm_-45_close_p"); side = np.where(p > 0.5, 0, 1)
base = np.isfinite(p) & (np.abs(p - 0.5) >= 0.05) & np.isfinite(z) & (np.sign(z) == np.sign(p - 0.5))
pnlA = settle_pnl(sim, idx, side, 0, 0.85); entA = sim["entry_px"][idx, side]
add("5 A consensus", "A as adopted (TP .85 settle)", np.where(base, pnlA, np.nan), entA, dates)
for name, extra in (("+T-60 head agrees", np.sign(p60 - 0.5) == np.sign(p - 0.5)), ("+T-60 agrees & |p60-.5|>=.03", (np.sign(p60 - 0.5) == np.sign(p - 0.5)) & (np.abs(p60 - 0.5) >= 0.03)),
                    ("+bp 1-min agrees", np.sign(z1) == np.sign(p - 0.5)), ("+btc-movement agrees", np.sign(bm - 0.5) == np.sign(p - 0.5)), ("+all three", (np.sign(p60 - 0.5) == np.sign(p - 0.5)) & (np.sign(z1) == np.sign(p - 0.5)) & (np.sign(bm - 0.5) == np.sign(p - 0.5))),
                    ("|p-.5|>=.06", np.abs(p - 0.5) >= 0.06), ("|p-.5|>=.07", np.abs(p - 0.5) >= 0.07), ("|bp z|>=0.1 too", np.abs(z) >= 0.1)):
    add("5 A consensus", name, np.where(base & np.nan_to_num(extra).astype(bool), pnlA, np.nan), entA, dates)
print("idea 5 done", flush=True)

# ------------------------------------------------------------------ 6. high-confidence settlement capture after the open
for e in (60, 90, 120, 180, 240):
    m, sim = run_sim(e, "maker_last", cancel=e + 10, tps=(0.95, 1.01)); idx = np.arange(len(m)); dates = m.date.values
    mod = colz(m, f"m2_{e}_close_p"); mkt = sim["pre_last"][:, 0]; side = np.where(mod > 0.5, 0, 1)
    pm = np.where(side == 0, mod, 1 - mod); pk = np.where(side == 0, mkt, 1 - mkt)
    for q, maxpx, (a, tp) in itertools.product((0.80, 0.85, 0.90, 0.95), (0.70, 0.80, 0.90), enumerate((0.95, 1.01))):
        sel = np.isfinite(pm) & np.isfinite(pk) & (pm >= q) & (pk <= maxpx)
        pnl = np.where(sel, settle_pnl(sim, idx, side, a, tp), np.nan)
        add("6 high-confidence settle", f"T+{e} model>={q} price<={maxpx} TP {tp if tp < 1 else 'none'}", pnl, sim["entry_px"][idx, side], dates)
    print("idea 6 done", e, flush=True)

# ------------------------------------------------------------------ 7 / 8 / 10. B variants: bp agreement, settlement exits, regime filters
m, sim = run_sim(0, "maker_last", cancel=10, tps=(0.80, 0.85, 0.90, 1.01), sls=(0.0, 0.10), tstops=(240, 300)); idx = np.arange(len(m)); dates = m.date.values
mod = colz(m, "m2_0_close_p"); mkt = sim["pre_last"][:, 0]; dis = mod - mkt; side = np.where(dis > 0, 0, 1); baseB = np.isfinite(dis) & (np.abs(dis) >= 0.12)
entB = sim["entry_px"][idx, side]
frozenB = sim["pnl"][idx, side, 0, 1, 0]                       # TP .80 / SL .10 / T+240
add("8 B exits", "B frozen (.80/.10/240)", np.where(baseB, frozenB, np.nan), entB, dates)
for a, tp in enumerate((0.80, 0.85, 0.90, 1.01)):
    add("8 B exits", f"B TP {tp if tp < 1 else 'none'} then settle, no stop", np.where(baseB, settle_pnl(sim, idx, side, a, tp), np.nan), entB, dates)
    k = sim["kind"][idx, side, a, 1, 0]; pnl_s = np.where(k == 0, tp - entB, np.where(k == 1, sim["pnl"][idx, side, a, 1, 0], sim["settle"][idx, side]))
    add("8 B exits", f"B TP {tp if tp < 1 else 'none'} + SL .10 then settle", np.where(baseB, pnl_s, np.nan), entB, dates)
zb = colz(m, "bp_0_z_s5"); zb1 = colz(m, "bp_0_z_s1"); zb10 = colz(m, "bp_0_z_s10s")
for name, f in (("bp 5-min z agrees with B side", np.sign(zb) == np.where(side == 0, 1, -1)), ("bp 1-min z agrees", np.sign(zb1) == np.where(side == 0, 1, -1)), ("bp 10-s z agrees", np.sign(zb10) == np.where(side == 0, 1, -1)),
                ("bp 5-min & 1-min agree", (np.sign(zb) == np.where(side == 0, 1, -1)) & (np.sign(zb1) == np.where(side == 0, 1, -1)))):
    add("7 B + btc-predict agreement", name + " (frozen exits)", np.where(baseB & np.nan_to_num(f).astype(bool), frozenB, np.nan), entB, dates)
    add("7 B + btc-predict agreement", name + " (TP .85 settle)", np.where(baseB & np.nan_to_num(f).astype(bool), settle_pnl(sim, idx, side, 1, 0.85), np.nan), entB, dates)
vol = colz(m, "vol_prev5m_bp"); hr = m.hour_et.values; we = m.weekend.values; vq = np.nanquantile(vol[(dates >= se.TUNE[0]) & (dates <= se.TUNE[1])], [1 / 3, 2 / 3]) if LABEL == "main" else np.nanquantile(vol, [1 / 3, 2 / 3])
for name, f in (("weekday only", we == 0), ("weekend only", we == 1), ("ET 0-8h", hr < 8), ("ET 8-16h", (hr >= 8) & (hr < 16)), ("ET 16-24h", hr >= 16), ("vol low", vol < vq[0]), ("vol mid", (vol >= vq[0]) & (vol < vq[1])), ("vol high", vol >= vq[1]),
                ("not vol high", vol < vq[1]), ("weekday & not vol high", (we == 0) & (vol < vq[1]))):
    add("10 B regime filter", name + " (frozen exits)", np.where(baseB & f, frozenB, np.nan), entB, dates)
print("ideas 7/8/10 done", flush=True)
# C with settlement exits
mC, simC = run_sim(-5, "maker_last", cancel=0, tps=(0.80, 0.85, 0.90, 1.01)); idxC = np.arange(len(mC)); datesC = mC.date.values
modC = colz(mC, "m2_-5_close_p"); mktC = simC["pre_last"][:, 0]; disC = modC - mktC; sideC = np.where(disC > 0, 0, 1); baseC = np.isfinite(disC) & (np.abs(disC) >= 0.08); entC = simC["entry_px"][idxC, sideC]
for a, tp in enumerate((0.80, 0.85, 0.90, 1.01)):
    add("8 C exits", f"C TP {tp if tp < 1 else 'none'} then settle, no stop", np.where(baseC, settle_pnl(simC, idxC, sideC, a, tp), np.nan), entC, datesC)
simC2 = polysim.simulate(mk, g, t_entry=-5, entry_mode="maker_last", t_cancel=0, tps=(0.80,), sls=(0.05,), tstops=(240,), slip=0.01); mC2, simC2 = ex.align(master, simC2)
add("8 C exits", "C frozen (.80/.05/240)", np.where(baseC, simC2["pnl"][idxC, sideC, 0, 0, 0], np.nan), entC, datesC)
print("idea 8C done", flush=True)

# ------------------------------------------------------------------ 9. dip add-on in A windows at T+30
m, sim = run_sim(30, "maker_last", cancel=40, tps=(0.80, 0.85, 1.01)); idx = np.arange(len(m)); dates = m.date.values
p = colz(m, "m2_-45_close_p"); z = colz(m, "bp_-60_z_s5"); p30 = colz(m, "m2_30_close_p"); side = np.where(p > 0.5, 0, 1); mkt = sim["pre_last"][:, 0]
inA = np.isfinite(p) & (np.abs(p - 0.5) >= 0.05) & np.isfinite(z) & (np.sign(z) == np.sign(p - 0.5)); ours = np.where(side == 0, mkt, 1 - mkt); pm = np.where(side == 0, p30, 1 - p30)
for maxpx, conf, (a, tp) in itertools.product((0.35, 0.42, 0.48), (0.55, 0.60, 0.70), enumerate((0.80, 0.85, 1.01))):
    sel = inA & np.isfinite(ours) & (ours <= maxpx) & np.isfinite(pm) & (pm >= conf)
    add("9 dip add-on in A windows", f"T+30 price<={maxpx} model>={conf} TP {tp if tp < 1 else 'none'}", np.where(sel, settle_pnl(sim, idx, side, a, tp), np.nan), sim["entry_px"][idx, side], dates)
print("idea 9 done", flush=True)

# ------------------------------------------------------------------ 11. hedge the other side after A's take-profit print
import glob, pyarrow.parquet as pq
mA = master[np.isfinite(p45) & (np.abs(p45 - 0.5) >= 0.05) & np.isfinite(z5) & (np.sign(z5) == np.sign(p45 - 0.5))]
WA = sorted(set(mA.index.values) & set(mk.index.values)); gg = {}
days = os.environ.get("POLY_DAYS", "").split(",") if os.environ.get("POLY_DAYS") else None
for f in sorted(glob.glob(f"{polysim.TRADES}/*.parquet")):
    if days and os.path.basename(f)[:10] not in days: continue
    t = pq.read_table(f, columns=["ts", "outcome_index", "price", "window_start"], filters=[("window_start", "in", WA)]).to_pandas()
    if not len(t): continue
    t["rel"] = (t.ts - t.window_start).astype(np.int32); t["p_up"] = np.where(t.outcome_index == 0, t.price, 1 - t.price).astype(np.float32); t = t.sort_values(["window_start", "ts"], kind="stable")
    for w, sub in t.groupby("window_start", sort=False): gg[int(w)] = (sub.rel.values, sub.p_up.values)
WA = [w for w in WA if w in gg]; mA = mA.loc[WA]; su = mA["m2_-45_close_p"].values > 0.5; upA = mA.up.values
for tp_lvl, hedge in itertools.product((0.85,), (0.08, 0.10, 0.12, 0.15)):
    pnl = np.full(len(WA), np.nan); ent = np.full(len(WA), hedge)
    for i, w in enumerate(WA):
        rel, pup = gg[w]; ps = pup if su[i] else 1 - pup; mm = (rel >= -45) & (rel <= 300); t_, pr = rel[mm], ps[mm]
        j = np.searchsorted(np.maximum.accumulate(pr), tp_lvl) if len(pr) else len(pr)
        if j >= len(pr): continue
        other = 1 - pr[j + 1:]                                   # other side's price after our TP print
        if len(other) and (other <= hedge + 1e-9).any():          # hedge bid filled
            won_other = (upA[i] == 0) if su[i] else (upA[i] == 1); pnl[i] = (1.0 if won_other else 0.0) - hedge
    add("11 hedge other side after A TP", f"TP {tp_lvl} print, bid other side at {hedge} to close", pnl, ent, mA.date.values)
print("idea 11 done", flush=True)

df = pd.DataFrame(rows); df.to_csv(os.path.join(R, f"ideas10_{LABEL}.csv"), index=False)
ok = df[df[f"n_{SEL}"] >= 150].copy()
cols = ["idea", "variant"] + [f"{k}_{nm}" for nm in PERS for k in ("n", "mean", "win", "usd", "maxdd", "tstat")]
print("\n== per idea: the cell with the best TUNE mean (>= 150 tune trades), read on test / Aug 1-19 ==")
best = ok.sort_values(f"mean_{SEL}", ascending=False).groupby("idea", sort=False).head(1).sort_values(f"mean_{SEL}", ascending=False)
print(best[cols].round(3).to_string(index=False))
print("\n== per idea: the cell with the highest TUNE win rate among cells with tune mean > 0 ==")
pos = ok[ok[f"mean_{SEL}"] > 0]; bw = pos.sort_values(f"win_{SEL}", ascending=False).groupby("idea", sort=False).head(1).sort_values(f"win_{SEL}", ascending=False)
print(bw[cols].round(3).to_string(index=False))
if LABEL == "main":
    print("\n== all cells positive on old, tune AND test with test win rate >= 0.70 (sorted by test mean) ==")
    hi = ok[(ok.mean_old > 0) & (ok.mean_tune > 0) & (ok.mean_test > 0) & (ok.win_test >= 0.70)].sort_values("mean_test", ascending=False)
    print(hi[cols].round(3).head(25).to_string(index=False))
