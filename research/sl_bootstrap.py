"""Paired day-block bootstrap: TP 0.80 / time-stop 240 with SL in {0.01, 0.05, 0.10} vs the frozen SL 0.20 (same trades)."""
import os, sys, glob, numpy as np, pandas as pd, pyarrow.parquet as pq, warnings
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "polyeth"))
import polysim  # noqa: E402
warnings.filterwarnings("ignore")
m = pd.read_parquet("/home/user/data_eth/master.parquet", columns=["up", "date", "m2_-45_close_p", "bp_-60_z_s5"])
p = m["m2_-45_close_p"].values; z = m["bp_-60_z_s5"].values
sel = np.isfinite(p) & (np.abs(p - 0.5) >= 0.05) & np.isfinite(z) & (np.sign(z) == np.sign(p - 0.5)) & (m.date.values >= "2026-08-01")
m = m[sel]; mk = pd.read_parquet(polysim.MARKETS).set_index("window_start"); mk = mk[mk.outcome_up_price.isin([0.0, 1.0])]
W = set(m.index.values) & set(mk.index.values); g = {}
for f in sorted(glob.glob(f"{polysim.TRADES}/*.parquet")):
    t = pq.read_table(f, columns=["ts", "outcome_index", "price", "window_start"], filters=[("window_start", "in", list(W))]).to_pandas()
    if len(t) == 0: continue
    t["rel"] = (t.ts - t.window_start).astype(np.int32); t["p_up"] = np.where(t.outcome_index == 0, t.price, 1 - t.price).astype(np.float32)
    t = t.sort_values(["window_start", "ts"], kind="stable")
    for w, sub in t.groupby("window_start", sort=False): g[int(w)] = (sub.rel.values, sub.p_up.values)
SLS = (0.01, 0.05, 0.10, 0.20)
sim = polysim.simulate(mk, g, t_entry=-45, entry_mode="fixed", pe=0.50, tps=(0.80,), sls=SLS, tstops=(240,))
mm = m.loc[sim["W"]]; side = np.where(mm["m2_-45_close_p"].values > 0.5, 0, 1); idx = np.arange(len(mm))
P = sim["pnl"][idx, side, 0, :, 0]; K = sim["kind"][idx, side, 0, :, 0]
periods = {"old(Aug1-19)": (mm.date.values < "2026-08-20"), "tune": (mm.date.values >= "2026-08-20") & (mm.date.values <= "2026-09-30"), "test": mm.date.values >= "2026-10-01"}
rng = np.random.default_rng(0)
for nm, mask in periods.items():
    days = mm.date.values[mask]; ud, inv = np.unique(days, return_inverse=True)
    D = np.zeros((len(ud), len(SLS))); np.add.at(D, inv, P[mask])
    base = D[:, SLS.index(0.20)]
    print(f"\n== {nm}: {mask.sum()} trades, {len(ud)} days ==")
    for j, sl in enumerate(SLS):
        x = P[mask][:, j]; k = K[mask][:, j]
        diff = D[:, j] - base; boots = np.array([np.mean(diff[rng.integers(0, len(diff), len(diff))]) for _ in range(4000)])
        loss = x[x < 0]
        print(f"SL {sl:.2f}: mean {x.mean()*100:+.2f}c win {np.mean(x>0):.3f} avg loss {loss.mean()*100:+.1f}c exits TP/SL/time {np.mean(k==0):.2f}/{np.mean(k==1):.2f}/{np.mean(k==2):.2f} "
              f"| vs SL 0.20: {diff.mean()*100:+.2f}c/day, 95% CI [{np.quantile(boots,.025)*100:+.2f}, {np.quantile(boots,.975)*100:+.2f}], P(diff<=0)={np.mean(boots<=0):.3f}")
