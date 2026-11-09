"""Exhaustive exits for strategy A: TP 0.52-0.99 x SL 0.01-0.49 (1 cent) x time-stop T+5..T+300 (5 s) =
141,120 combinations, frozen signal (m2 close head @T-45, thr 0.05, btc-predict agreement, fill 0.50).
Batched over windows; per-cell aggregates per period (old / tune / test)."""
import os, sys, glob, time, numpy as np, pandas as pd, pyarrow.parquet as pq, warnings
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "polyeth"))
import polysim  # noqa: E402
warnings.filterwarnings("ignore"); pd.set_option("display.width", 230)
R = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "reports")
m = pd.read_parquet("/home/user/data_eth/master.parquet", columns=["up", "date", "m2_-45_close_p", "bp_-60_z_s5"])
p = m["m2_-45_close_p"].values; z = m["bp_-60_z_s5"].values
sel = np.isfinite(p) & (np.abs(p - 0.5) >= 0.05) & np.isfinite(z) & (np.sign(z) == np.sign(p - 0.5)) & (m.date.values >= "2026-08-01")
m = m[sel]; mk = pd.read_parquet(polysim.MARKETS).set_index("window_start"); mk = mk[mk.outcome_up_price.isin([0.0, 1.0])]
W = sorted(set(m.index.values) & set(mk.index.values)); g = {}
for f in sorted(glob.glob(f"{polysim.TRADES}/*.parquet")):
    t = pq.read_table(f, columns=["ts", "outcome_index", "price", "window_start"], filters=[("window_start", "in", W)]).to_pandas()
    if len(t) == 0: continue
    t["rel"] = (t.ts - t.window_start).astype(np.int32); t["p_up"] = np.where(t.outcome_index == 0, t.price, 1 - t.price).astype(np.float32)
    t = t.sort_values(["window_start", "ts"], kind="stable")
    for w, sub in t.groupby("window_start", sort=False): g[int(w)] = (sub.rel.values, sub.p_up.values)
W = [w for w in W if w in g]; print("windows:", len(W))
TPS = np.round(np.arange(0.52, 1.00, 0.01), 2); SLS = np.round(np.arange(0.01, 0.50, 0.01), 2); TSS = tuple(range(5, 301, 5))
shape = (len(TPS), len(SLS), len(TSS))
per = {"old": lambda d: d < "2026-08-20", "tune": lambda d: (d >= "2026-08-20") & (d <= "2026-09-30"), "test": lambda d: d >= "2026-10-01"}
agg = {k: dict(sum=np.zeros(shape), n=0, wins=np.zeros(shape), daily={}) for k in per}
t0 = time.time()
for i in range(0, len(W), 200):
    ws = W[i:i + 200]; sim = polysim.simulate(mk, {w: g[w] for w in ws}, t_entry=-45, entry_mode="fixed", pe=0.50, tps=TPS, sls=SLS, tstops=TSS)
    mm = m.loc[sim["W"]]; side = np.where(mm["m2_-45_close_p"].values > 0.5, 0, 1)
    P = sim["pnl"][np.arange(len(mm)), side]                     # [n, tp, sl, ts]
    dates = mm.date.values
    for k, fn in per.items():
        mask = fn(dates)
        if not mask.any(): continue
        x = np.nan_to_num(P[mask]); a = agg[k]; a["sum"] += x.sum(0); a["n"] += int(mask.sum()); a["wins"] += (x > 0).sum(0)
        for d in np.unique(dates[mask]):
            a["daily"][d] = a["daily"].get(d, 0) + x[dates[mask] == d].sum(0)
    print(f"  {i + len(ws)}/{len(W)} windows {time.time() - t0:.0f}s", flush=True)
res = {}
for k, a in agg.items():
    D = np.stack([a["daily"][d] for d in sorted(a["daily"])])    # [days, tp, sl, ts]
    cum = np.cumsum(D, 0); dd = np.max(np.maximum.accumulate(cum, 0) - cum, 0)
    res[k] = dict(mean=a["sum"] / a["n"], win=a["wins"] / a["n"], sharpe=D.mean(0) / (D.std(0) + 1e-12) * np.sqrt(365), maxdd=dd, n=a["n"])
np.savez_compressed(os.path.join(R, "A_exit_grid_time.npz"), tps=TPS, sls=SLS, tss=np.array(TSS), **{f"{k}_{s}": v for k, a in res.items() for s, v in a.items() if s != "n"})
ia, ib, ic = np.meshgrid(np.arange(len(TPS)), np.arange(len(SLS)), np.arange(len(TSS)), indexing="ij")
df = pd.DataFrame({"tp": TPS[ia.ravel()], "sl": SLS[ib.ravel()], "tstop": np.array(TSS)[ic.ravel()]})
for k in res:
    for s in ("mean", "win", "sharpe", "maxdd"): df[f"{s}_{k}"] = res[k][s].ravel()
df.to_csv(os.path.join(R, "A_exit_grid_time.csv"), index=False)
print(f"\ncells: {len(df)}; trades old/tune/test: {res['old']['n']}/{res['tune']['n']}/{res['test']['n']}")
cols = ["tp", "sl", "tstop", "mean_old", "mean_tune", "win_tune", "sharpe_tune", "maxdd_tune", "mean_test", "win_test", "sharpe_test", "maxdd_test"]
print("\n== top 25 by TUNE mean ==\n", df.sort_values("mean_tune", ascending=False).head(25)[cols].round(4).to_string(index=False))
print("\n== top 15 by TUNE Sharpe ==\n", df.sort_values("sharpe_tune", ascending=False).head(15)[cols].round(4).to_string(index=False))
print("\n== best cell per time-stop (by tune mean) ==")
best = df.loc[df.groupby("tstop").mean_tune.idxmax()]; print(best[cols].round(4).to_string(index=False))
print("\n== TP 0.80 / SL 0.05: by time-stop ==")
print(df[(df.tp == 0.80) & (df.sl == 0.05)][cols].round(4).to_string(index=False))
print("\n== TP 0.80 / SL 0.20: by time-stop ==")
print(df[(df.tp == 0.80) & (df.sl == 0.20)][cols].round(4).to_string(index=False))
