"""Exhaustive TP x SL x time-stop grid for strategy A (frozen signal: m2 close head @T-45, thr 0.05,
btc-predict agreement; fill at 0.50 before T-45). 1-cent steps. Selection on tune, report on test."""
import os, sys, glob, numpy as np, pandas as pd, pyarrow.parquet as pq, pyarrow.compute as pc, warnings
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "polyeth"))
import polysim  # noqa: E402
warnings.filterwarnings("ignore"); pd.set_option("display.width", 230)
R = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "reports")
m = pd.read_parquet("/home/user/data_eth/master.parquet", columns=["up", "date", "m2_-45_close_p", "bp_-60_z_s5"])
p = m["m2_-45_close_p"].values; z = m["bp_-60_z_s5"].values
sel = np.isfinite(p) & (np.abs(p - 0.5) >= 0.05) & np.isfinite(z) & (np.sign(z) == np.sign(p - 0.5)) & (m.date.values >= "2026-08-20")
m = m[sel]; print("selected windows:", len(m))
mk = pd.read_parquet(polysim.MARKETS).set_index("window_start"); mk = mk[mk.outcome_up_price.isin([0.0, 1.0])]
W = set(m.index.values) & set(mk.index.values)
g = {}
for f in sorted(glob.glob(f"{polysim.TRADES}/*.parquet")):
    t = pq.read_table(f, columns=["ts", "outcome_index", "price", "window_start"], filters=[("window_start", "in", list(W))]).to_pandas()
    if len(t) == 0: continue
    t["rel"] = (t.ts - t.window_start).astype(np.int32); t["p_up"] = np.where(t.outcome_index == 0, t.price, 1 - t.price).astype(np.float32)
    t = t.sort_values(["window_start", "ts"], kind="stable")
    for w, sub in t.groupby("window_start", sort=False): g[int(w)] = (sub.rel.values, sub.p_up.values)
print("windows with trades:", len(g))
TPS = np.round(np.arange(0.52, 1.00, 0.01), 2); SLS = np.round(np.arange(0.01, 0.50, 0.01), 2); TSS = (180, 240, 270, 290)
sim = polysim.simulate(mk, g, t_entry=-45, entry_mode="fixed", pe=0.50, tps=TPS, sls=SLS, tstops=TSS)
mm = m.loc[sim["W"]]; side = np.where(mm["m2_-45_close_p"].values > 0.5, 0, 1); idx = np.arange(len(mm))
P = sim["pnl"][idx, side]                              # [n, tp, sl, ts]
tune = ((mm.date.values >= "2026-08-20") & (mm.date.values <= "2026-09-30")); test = (mm.date.values >= "2026-10-01")
def stats(mask):
    x = P[mask]; d = pd.Series(mask).groupby(mm.date.values[mask] if False else None) if False else None
    mean = np.nanmean(x, 0); win = np.nanmean(x > 0, 0)
    # daily sums -> sharpe & max drawdown
    days = mm.date.values[mask]; ud, inv = np.unique(days, return_inverse=True)
    daily = np.zeros((len(ud),) + x.shape[1:]); np.add.at(daily, inv, np.nan_to_num(x))
    sharpe = daily.mean(0) / (daily.std(0) + 1e-12) * np.sqrt(365)
    cum = np.cumsum(daily, 0); dd = np.max(np.maximum.accumulate(cum, 0) - cum, 0)
    return mean, win, sharpe, dd, int(mask.sum())
mt, wt, st_, dt_, nt = stats(tune); me, we, se_, de, ne = stats(test)
rows = []
for a, tp in enumerate(TPS):
    for b, sl in enumerate(SLS):
        for c, ts in enumerate(TSS):
            rows.append(dict(tp=tp, sl=sl, tstop=ts, mean_tune=mt[a, b, c], win_tune=wt[a, b, c], sharpe_tune=st_[a, b, c], maxdd_tune=dt_[a, b, c],
                             mean_test=me[a, b, c], win_test=we[a, b, c], sharpe_test=se_[a, b, c], maxdd_test=de[a, b, c]))
df = pd.DataFrame(rows); df.to_csv(os.path.join(R, "A_exit_grid_full.csv"), index=False)
print(f"cells: {len(df)}  trades tune {nt} test {ne}")
print("\n== top 25 by TUNE mean (and their test) ==")
print(df.sort_values("mean_tune", ascending=False).head(25).round(4).to_string(index=False))
print("\n== top 15 by TUNE Sharpe ==")
print(df.sort_values("sharpe_tune", ascending=False).head(15).round(4).to_string(index=False))
print("\n== frozen cell and neighbours ==")
print(df[(df.tp.between(0.78, 0.82)) & (df.sl.between(0.18, 0.22)) & (df.tstop == 240)].round(4).to_string(index=False))
print("\n== rank of the frozen cell (0.80/0.20/240) among", len(df), "cells: by tune mean", int((df.mean_tune > df[(df.tp == 0.80) & (df.sl == 0.20) & (df.tstop == 240)].mean_tune.iloc[0]).sum()) + 1,
      "| by test mean", int((df.mean_test > df[(df.tp == 0.80) & (df.sl == 0.20) & (df.tstop == 240)].mean_test.iloc[0]).sum()) + 1)
# test mean of the cell chosen by tune mean / by tune sharpe
for key in ("mean_tune", "sharpe_tune"):
    b = df.sort_values(key, ascending=False).iloc[0]; print(f"chosen by {key}: tp {b.tp} sl {b.sl} ts {b.tstop} -> test mean {b.mean_test:.4f} win {b.win_test:.3f} sharpe {b.sharpe_test:.2f} dd {b.maxdd_test:.2f}")
# smoothed view: average over tstops, coarse heat table
piv = df[df.tstop == 240].pivot(index="tp", columns="sl", values="mean_test")
print("\n== TEST mean (¢) by TP (rows) x SL (cols), time-stop 240, every 3 cents ==")
print((piv.loc[piv.index[::3], piv.columns[::3]] * 100).round(1).to_string())
piv2 = df[df.tstop == 240].pivot(index="tp", columns="sl", values="mean_tune")
print("\n== TUNE mean (¢) ==")
print((piv2.loc[piv2.index[::3], piv2.columns[::3]] * 100).round(1).to_string())
