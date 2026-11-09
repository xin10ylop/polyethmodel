"""Hypothesis-driven experiments on the TUNE period; final configs re-run on TEST by final_eval.py.
Usage: python3 experiments.py [H1|H2|H3|H4|all]"""
import os
import numpy as np, pandas as pd, sys, warnings
from sklearn.linear_model import LogisticRegression
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "polyeth"))
import polysim, strategy_eval as se
warnings.filterwarnings("ignore")
pd.set_option("display.width", 250)
TUNE, TEST = se.TUNE, se.TEST
master = pd.read_parquet(os.environ.get("MASTER_PATH", "/home/user/data_eth/master.parquet"))
mk, g = polysim.load_windows(days=os.environ["POLY_DAYS"].split(",") if os.environ.get("POLY_DAYS") else None)   # POLY_DAYS limits the trade files loaded


def logit(p): p = np.clip(p, 1e-4, 1 - 1e-4); return np.log(p / (1 - p))


def stacked_signal(master, d, cols=None, label="up_binance_eff"):
    """Logistic stack of pre-reference model outputs at offset d, fitted on TUNE with the Binance label."""
    cols = cols or [c for c in (f"m2_{d}_close_p", f"bm_{d}_close_p", f"m2_{d}_h60_pup_now", f"bm_{d}_h30_pup_now") if c in master.columns]
    X = np.column_stack([master[c].values if c.startswith("bp_") else logit(master[c].values) for c in cols]); ok = np.isfinite(X).all(1)
    tr = ok & (master.date.values >= TUNE[0]) & (master.date.values <= TUNE[1])
    lr = LogisticRegression(C=1.0).fit(X[tr], master[label].values[tr])
    p = np.full(len(master), np.nan); p[ok] = lr.predict_proba(X[ok])[:, 1]
    return p, dict(zip(cols, np.round(lr.coef_[0], 3)))


def align(master, sim):
    keep = np.isin(sim["W"], master.index.values)
    for k in ("pnl", "kind", "texit", "settle", "entry_px", "pre_last", "fill_rel"):
        sim[k] = sim[k][keep]
    sim["W"] = sim["W"][keep]
    return master.loc[sim["W"]], sim


def H1():
    print("\n######## H1: pre-open maker entry at 0.50 (filled by T-45), side from pre-reference forecasts ########")
    sim = polysim.simulate(mk, g, t_entry=-45, pe=0.5, tps=(0.55, 0.58, 0.60, 0.62, 0.65, 0.70, 0.75, 0.80, 0.90),
                           sls=(0.20, 0.25, 0.30, 0.35, 0.40, 0.42, 0.45, 0.47), tstops=(180, 240, 270, 285, 295))
    m, sim = align(master, sim)
    print("windows:", len(m), " tune:", ((m.date >= TUNE[0]) & (m.date <= TUNE[1])).sum(), " test:", ((m.date >= TEST[0]) & (m.date <= TEST[1])).sum())
    res = {}
    for d in (-60, -45, -35):
        for name, sig in ((f"m2_{d}_close_p", m[f"m2_{d}_close_p"].values), (f"bm_{d}_close_p", m[f"bm_{d}_close_p"].values)):
            res[name] = se.select_and_test(m, sim, sig, name, top=8)
        p, coef = stacked_signal(m, d); print("stack coef", coef)
        res[f"stack@{d}"] = se.select_and_test(m, sim, p, f"stack@{d}", top=8)
    if "bp_-60_z_s5" in m.columns:
        z = m["bp_-60_z_s5"].values; sig = 1 / (1 + np.exp(-1.5 * np.nan_to_num(z, nan=0.0))); sig[~np.isfinite(z)] = np.nan
        res["bp_-60_s5"] = se.select_and_test(m, sim, sig, "bp_-60_z_s5 (sigmoid)", top=8)
    rng = np.random.default_rng(0); se.select_and_test(m, sim, rng.uniform(0, 1, len(m)), "random side", top=3)
    # settlement reference for the best stack
    return res


def H2(offsets=(-20, -15, -10, -5, 0, 5, 10, 20, 30, 60, 120), split=None):
    """Post-reference disagreement entries: side = sign(model - market), enter at the NEXT trade print
    (+slippage, taker fee), exits by absolute TP/SL grid or time stop."""
    print("\n######## H2: post-reference 'disagreement' entry at the prevailing market price ########")
    split = split or TUNE
    out = []
    for d in offsets:
        sim = polysim.simulate(mk, g, t_entry=d, pe=0.5, tps=(0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.90, 0.97),
                               sls=(0.05, 0.10, 0.20, 0.30, 0.35, 0.40, 0.45), tstops=(240, 270, 285, 295),
                               entry_from_trades=True, entry_taker=True, entry_fee=True, entry_next_trade=True)
        m, sim = align(master, sim)
        col = f"m2_{d}_close_p"
        if col not in m.columns:
            continue
        mod = m[col].values
        mkt_up = sim["pre_last"][:, 0]                       # last UP print before the decision (what we see)
        insplit = (m.date.values >= split[0]) & (m.date.values <= split[1])
        for delta in (0.03, 0.05, 0.08, 0.12):
            sig = np.where(np.abs(mod - mkt_up) >= delta, 0.5 + (mod - mkt_up), np.nan)
            r = se.grid_eval(m, sim, sig, thresholds=(0.0,), split=split, min_trades=30)
            if len(r) == 0:
                continue
            rs = r[r.tstop == "settle"].iloc[0]; r = r[r.tstop != "settle"].sort_values("mean", ascending=False)
            best = r.iloc[0]
            sel = np.isfinite(sig) & insplit; side = np.where(sig > 0.5, 0, 1)
            out.append(dict(d=d, delta=delta, n=int(best.n), entry_px=float(np.nanmean(sim["entry_px"][sel, side[sel]])),
                            best_mean=best["mean"], tp=best.tp, sl=best.sl, tstop=best.tstop, win=best.win, sharpe=best.sharpe,
                            settle_mean=rs["mean"], settle_win=rs.win,
                            tp60_sl40_t285=float(r[(r.tp == 0.60) & (r.sl == 0.40) & (r.tstop == 285)]["mean"].iloc[0]),
                            tp75_sl30_t285=float(r[(r.tp == 0.75) & (r.sl == 0.30) & (r.tstop == 285)]["mean"].iloc[0]),
                            tp97_sl05_t295=float(r[(r.tp == 0.97) & (r.sl == 0.05) & (r.tstop == 295)]["mean"].iloc[0])))
    df = pd.DataFrame(out); print(df.round(4).to_string(index=False)); return df


if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "all"
    if which in ("H1", "all"): H1()
    if which in ("H2", "all"): H2()
    if which in ("H3", "all"): H3(default_signal)
    if which in ("H4", "all"): H4(default_signal)


# ---------------------------------------------------------------- H3: regimes / filters, H4: dynamic exits
EXITS = {"tp97_sl05_t290": (0.97, 0.05, 290), "tp90_sl10_t290": (0.90, 0.10, 290), "tp80_sl20_t290": (0.80, 0.20, 290),
         "tp70_sl30_t290": (0.70, 0.30, 290), "tp60_sl40_t290": (0.60, 0.40, 290), "tp60_sl45_t290": (0.60, 0.45, 290),
         "tp97_sl05_t270": (0.97, 0.05, 270), "tp97_sl05_t240": (0.97, 0.05, 240)}


def sim_preopen(t_entry=-45, pe=0.5):
    sim = polysim.simulate(mk, g, t_entry=t_entry, pe=pe, tps=(0.60, 0.70, 0.80, 0.90, 0.97), sls=(0.05, 0.10, 0.20, 0.30, 0.40, 0.45),
                           tstops=(240, 270, 290))
    return align(master, sim)


def exit_pnl(sim, name, idx, side):
    tp, sl, ts = EXITS[name]
    a = list(sim["tps"]).index(tp); b = list(sim["sls"]).index(sl); c = list(sim["tstops"]).index(ts)
    return sim["pnl"][idx, side, a, b, c]


def H3(signal_fn, thr=0.04, t_entry=-45):
    print(f"\n######## H3: regime filters for pre-open entry (thr={thr}) ########")
    m, sim = sim_preopen(t_entry)
    p = signal_fn(m); side = np.where(p > 0.5, 0, 1); idx = np.arange(len(m))
    base = np.isfinite(p) & (np.abs(p - 0.5) >= thr)
    vol = m.vol_prev5m_bp.values; vq = np.nanquantile(vol[(m.date.values >= TUNE[0]) & (m.date.values <= TUNE[1])], [1 / 3, 2 / 3])
    hour = m.hour_et.values
    pre = sim["pre_last"][idx, side]          # last pre-open print of our side
    filters = {"all": np.ones(len(m), bool), "vol low": vol < vq[0], "vol mid": (vol >= vq[0]) & (vol < vq[1]), "vol high": vol >= vq[1],
               "weekend": m.weekend.values == 1, "weekday": m.weekend.values == 0, "US session": m.us_session.values == 1,
               "ET 0-8h": (hour < 8), "ET 8-16h": (hour >= 8) & (hour < 16), "ET 16-24h": hour >= 16,
               "pre-open price <= 0.50": pre <= 0.50, "pre-open price >= 0.52": pre >= 0.52,
               "abs(x_pre_m45_m30) small": np.abs(m.x_pre_m45_m30.values) < 1.0}
    rows = []
    for ex in ("tp97_sl05_t290", "tp80_sl20_t290", "tp60_sl40_t290"):
        pnl = exit_pnl(sim, ex, idx, side)
        for fn, f in filters.items():
            for split, nm in ((TUNE, "tune"), (TEST, "test")):
                s = base & f & (m.date.values >= split[0]) & (m.date.values <= split[1])
                r = se.metrics(np.where(s, pnl, np.nan), m.date.values); r.update(exit=ex, filter=fn, split=nm); rows.append(r)
    df = pd.DataFrame(rows)
    piv = df.pivot_table(index=["exit", "filter"], columns="split", values=["n", "mean", "win", "sharpe"])
    print(piv.round(4).to_string())
    return df


def H4(signal_fn, thr=0.04, t_entry=-45):
    print(f"\n######## H4: fixed vs vol-dynamic exits (thr={thr}) ########")
    m, sim = sim_preopen(t_entry)
    p = signal_fn(m); side = np.where(p > 0.5, 0, 1); idx = np.arange(len(m))
    base = np.isfinite(p) & (np.abs(p - 0.5) >= thr)
    vol = m.vol_prev5m_bp.values; vq = np.nanquantile(vol[(m.date.values >= TUNE[0]) & (m.date.values <= TUNE[1])], [1 / 3, 2 / 3])
    tier = np.digitize(vol, vq)
    rows = []
    for ex in EXITS:
        pnl = exit_pnl(sim, ex, idx, side)
        for split, nm in ((TUNE, "tune"), (TEST, "test")):
            for t in (0, 1, 2):
                s = base & (tier == t) & (m.date.values >= split[0]) & (m.date.values <= split[1])
                r = se.metrics(np.where(s, pnl, np.nan), m.date.values); r.update(exit=ex, tier=t, split=nm); rows.append(r)
    df = pd.DataFrame(rows)
    print(df.pivot_table(index=["exit"], columns=["split", "tier"], values="mean").round(4).to_string())
    # dynamic: per tier pick the best exit on TUNE, apply on TEST
    best = {t: df[(df.split == "tune") & (df.tier == t)].sort_values("mean", ascending=False).iloc[0]["exit"] for t in (0, 1, 2)}
    print("best exit per vol tier (tune):", best)
    for split, nm in ((TUNE, "tune"), (TEST, "test")):
        pnl = np.full(len(m), np.nan)
        for t, ex in best.items():
            s = base & (tier == t) & (m.date.values >= split[0]) & (m.date.values <= split[1])
            pnl[s] = exit_pnl(sim, ex, idx, side)[s]
        print(nm, "dynamic:", {k: round(v, 4) for k, v in se.metrics(pnl, m.date.values).items()})
        for ex in ("tp97_sl05_t290", "tp80_sl20_t290"):
            s = base & (m.date.values >= split[0]) & (m.date.values <= split[1])
            print(nm, "fixed", ex, {k: round(v, 4) for k, v in se.metrics(np.where(s, exit_pnl(sim, ex, idx, side), np.nan), m.date.values).items()})
    return df


def default_signal(m):
    p, coef = stacked_signal(m, -45, cols=[c for c in ("m2_-45_close_p", "bm_-45_close_p", "bp_-60_z_s5") if c in m.columns])
    return p
