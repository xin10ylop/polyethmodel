"""Candidate ETH rules (selected on tune in research/eth_A_grid.py): decision T-45 eth3 close head, |p-0.5| >= thr, votes eth1 / eth1+book,
exits sell 0.80 or 0.85 then settle, or settle only.  Per period n / win / edge / $ at $100, AND capacity-constrained $: fill = min(stake,
$ printed at 0.49-0.51 before the open) for stakes 25 / 50 / 100 (ETH windows are thin: research/capacity_eth.py)."""
import glob, sys, warnings, numpy as np, pandas as pd
warnings.filterwarnings("ignore"); pd.set_option("display.width", 260)
sys.path.insert(0, "/home/user/polyethmodel/polyeth"); import polysim, bookdepth  # noqa: E402
W = pd.read_parquet("/home/user/data_eth/master.parquet", columns=["date", "up", "m2_-45_close_p", "bp_-60_z_s5"]); W["date"] = pd.to_datetime(W.date)
T = W.index.values.astype(np.int64); book = bookdepth.archive_imbalance(T); p = W["m2_-45_close_p"].values; z5 = W["bp_-60_z_s5"].values
PER = {"Jun": ("2026-06-01", "2026-06-30"), "Jul": ("2026-07-01", "2026-07-31"), "Aug1-19": ("2026-08-01", "2026-08-19"), "tune": ("2026-08-20", "2026-09-30"), "test": ("2026-10-01", "2026-11-03"), "Nov4-7": ("2026-11-04", "2026-11-07")}
days = [str(d.date()) for d in pd.date_range("2026-06-01", "2026-11-07")]; mk, g = polysim.load_windows(days=days)
cap = {}
for f in sorted(glob.glob(f"{polysim.TRADES}/*.parquet")):
    t = pd.read_parquet(f, columns=["ts", "outcome_index", "price", "size", "window_start"]); pu = np.where(t.outcome_index.values == 0, t.price.values, 1 - t.price.values)
    m = (t.ts.values < t.window_start.values) & (pu >= 0.49) & (pu <= 0.51); s = pd.Series(t.price.values[m] * t["size"].values[m]).groupby(t.window_start.values[m]).sum()
    for k, v in s.items(): cap[int(k)] = cap.get(int(k), 0.0) + float(v)
def sim_for(tp):
    sim = polysim.simulate(mk, g, t_entry=-45, entry_mode="fixed", pe=0.50, tps=(tp if tp else 1.01,), sls=(0.0,), tstops=(300,), slip=0.01); idx = np.arange(len(sim["W"]))
    for sd in (0, 1):
        k = sim["kind"][idx, sd, 0, 0, 0]; sim["pnl"][idx, sd, 0, 0, 0] = np.where((k == 0) & (tp is not None), (tp or 0.0) - 0.5, sim["settle"][idx, sd])
    return pd.DataFrame({"T": sim["W"], "pnl_up": sim["pnl"][idx, 0, 0, 0, 0], "pnl_dn": sim["pnl"][idx, 1, 0, 0, 0]}).set_index("T")
SIMS = {"0.80 then settle": sim_for(0.80), "0.85 then settle": sim_for(0.85), "settle only": sim_for(None)}
rows = []
for ex, sim in SIMS.items():
    j = W.index.get_indexer(sim.index); ok = j >= 0; j = j[ok]; s = sim[ok]; pnl_all = np.where(p[j] > 0.5, s.pnl_up.values, s.pnl_dn.values); dates = W.date.values[j]
    capj = np.array([cap.get(int(t_), 0.0) for t_ in sim.index[ok]])
    for vote in ("eth1", "eth1+book"):
        agree = np.isfinite(z5[j]) & (np.sign(z5[j]) == np.sign(p[j] - 0.5))
        if "book" in vote: agree &= np.isfinite(book[j]) & (np.sign(book[j]) == np.sign(p[j] - 0.5))
        for thr in (0.03, 0.05, 0.06):
            sel = np.isfinite(p[j]) & (np.abs(p[j] - 0.5) >= thr) & agree
            for per, (a, b) in PER.items():
                m = sel & (dates >= np.datetime64(a)) & (dates <= np.datetime64(b)); x = pnl_all[m]; c = capj[m]; nd = (np.datetime64(b) - np.datetime64(a)).astype(int) + 1
                r = dict(exit=ex, vote=vote, thr=thr, period=per, n=len(x), per_day=len(x) / nd, win=(x > 0).mean(), edge_c=x.mean() * 100, usd100=x.sum() * 200)
                for S in (25, 50, 100): r[f"cap${S}/day"] = (x * 2 * np.minimum(S, c)).sum() / nd
                rows.append(r)
E = pd.DataFrame(rows); E.to_csv("/home/user/polyethmodel/reports/eth_A_confirm.csv", index=False)
for ex in SIMS:
    print(f"\n== {ex} ==")
    print(E[E.exit == ex].pivot_table(index=["vote", "thr"], columns="period", values=["win", "edge_c"])[[("win", k) for k in PER] + [("edge_c", k) for k in PER]].round(3).to_string())
print("\n== capacity-constrained $ per day (fill = min(stake, $ printed at 0.49-0.51 pre-open)), sell 0.85 then settle ==")
print(E[E.exit == "0.85 then settle"].pivot_table(index=["vote", "thr"], columns="period", values=["per_day", "cap$25/day", "cap$50/day", "cap$100/day", "usd100"]).round(1).to_string())
