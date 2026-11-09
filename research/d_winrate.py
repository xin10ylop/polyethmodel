"""Ten ways to raise D's win rate, tested: D = at T+90 the model gives >= 0.90 to a side still trading <= 0.90; bid at the last
print for 10 s; hold to settlement. Each variant is an extra filter or a changed instant; all judged on Aug 1-19 / tune / test
by win rate, edge per share, $/day and whether it stays positive in all three periods.  python research/d_winrate.py"""
import os, sys, numpy as np, pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.join(HERE, "..", "polyeth"))
import polysim, strategy_eval as se, experiments as ex   # noqa: E402
pd.set_option("display.width", 240); master, mk, g = ex.master, ex.mk, ex.g
PERS = {"Aug1-19": se.OLD, "tune": se.TUNE, "test": se.TEST}
SIMS = {}
def sim_at(d):
    if d not in SIMS:
        s = polysim.simulate(mk, g, t_entry=d, entry_mode="maker_last", t_cancel=d + 10, tps=(1.01,), sls=(0.0,), tstops=(300,), slip=0.01)
        m, s = ex.align(master, s); idx = np.arange(len(m))
        for sd in (0, 1): s["pnl"][idx, sd, 0, 0, 0] = s["settle"][idx, sd]
        SIMS[d] = (m, s)
    return SIMS[d]
def evaluate(name, d, q, maxpx, extra=None):
    m, s = sim_at(d); idx = np.arange(len(m)); mod = m[f"m2_{d}_close_p"].values; mkt = s["pre_last"][:, 0]
    side = np.where(mod >= 0.5, 0, 1); pm = np.where(side == 0, mod, 1 - mod); pk = np.where(side == 0, mkt, 1 - mkt)
    sel = np.isfinite(pm) & np.isfinite(pk) & (pm >= q) & (pk <= maxpx)
    if extra is not None: sel &= extra(m, side, pm, pk)
    pnl = np.where(sel, s["pnl"][idx, side, 0, 0, 0], np.nan); entry = s["entry_px"][idx, side]; rows = []
    for per, (a, b) in PERS.items():
        ms = (m.date.values >= a) & (m.date.values <= b); f = ms & np.isfinite(pnl); days = len(set(m.date.values[ms]))
        if f.sum() < 5: rows.append(dict(variant=name, period=per, n=int(f.sum()))); continue
        usd = pd.Series(pnl[f] / entry[f] * 100).groupby(m.date.values[f]).sum()
        rows.append(dict(variant=name, period=per, n=int(f.sum()), per_day=round(f.sum() / days, 1), entry=round(float(entry[f].mean()), 3), win=round(float((pnl[f] > 0).mean()), 3), cents=round(100 * float(np.nanmean(pnl[f])), 2),
                         usd_day=round(float(usd.sum() / days)), worst=round(float(usd.min())), maxdd=round(float((usd.cumsum().cummax() - usd.cumsum()).max()))))
    return rows
def side_val(m, col, side, d):   # value of a model probability column for OUR side
    v = m[f"m2_{d}_{col}"].values; return np.where(side == 0, v, 1 - v)
V = []
V += evaluate("0. D as it runs (T+90, q .90, px <= .90)", 90, 0.90, 0.90)
V += evaluate("1. later: T+150", 150, 0.90, 0.90)
V += evaluate("2. later: T+180", 180, 0.90, 0.90)
V += evaluate("3. stricter model: q .95", 90, 0.95, 0.90)
V += evaluate("4. market agrees too: side trades in [.80, .90]", 90, 0.90, 0.90, lambda m, sd, pm, pk: pk >= 0.80)
V += evaluate("5. short heads agree: 60 s head >= .80 for our side", 90, 0.90, 0.90, lambda m, sd, pm, pk: side_val(m, "h60_p", sd, 90) >= 0.80)
V += evaluate("6. price well past the reference: |move| >= 10 bp our way", 90, 0.90, 0.90, lambda m, sd, pm, pk: np.where(sd == 0, m["m2_90_x_bps"].values, -m["m2_90_x_bps"].values) >= 10)
V += evaluate("7. calm window: 5-min vol below the tune median", 90, 0.90, 0.90, lambda m, sd, pm, pk: m.vol_prev5m_bp.values < float(master.loc[(master.date >= se.TUNE[0]) & (master.date <= se.TUNE[1]), "vol_prev5m_bp"].median()))
V += evaluate("8. no path crossings: touch-back probability <= .15", 90, 0.90, 0.90, lambda m, sd, pm, pk: np.where(sd == 0, m["m2_90_close_touch-5"].values, m["m2_90_close_touch+5"].values) <= 0.15)
V += evaluate("9. model very sure of the direction now: 10 s head >= .90", 90, 0.90, 0.90, lambda m, sd, pm, pk: side_val(m, "h10_p", sd, 90) >= 0.90)
V += evaluate("10. combo: T+150, q .92, side in [.80, .90]", 150, 0.92, 0.90, lambda m, sd, pm, pk: pk >= 0.80)
V += evaluate("11. combo: T+90, q .90, |move| >= 10 bp, 60 s head >= .80", 90, 0.90, 0.90, lambda m, sd, pm, pk: (np.where(sd == 0, m["m2_90_x_bps"].values, -m["m2_90_x_bps"].values) >= 10) & (side_val(m, "h60_p", sd, 90) >= 0.80))
df = pd.DataFrame(V); df.to_csv(os.path.join(HERE, "..", "reports", "D_winrate.csv"), index=False)
piv = df.pivot(index="variant", columns="period", values=["win", "cents", "usd_day", "per_day"])
piv = piv.reindex(sorted(piv.index, key=lambda x: int(x.split(".")[0])))
print(piv.to_string()); print("\nsaved reports/D_winrate.csv")
