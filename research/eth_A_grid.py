"""ETH strategy A: the rule's own parameters re-selected on ETH (selection on tune Aug 20 - Sep 30, confirmation on test Oct 1 - Nov 3, unseen
June / July / Nov 4-7 reported too).  Signal eth3 close head at T-45; votes: none / eth1 z_s5 at T-60 / book-depth 1 % imbalance / both;
thresholds 0.02..0.10; exits: sell 0.85 then settle (live A), and settle-only.  $100 per trade.  Also the decision time (T-60 / T-45 / T-30 heads)."""
import os, sys, warnings, numpy as np, pandas as pd
warnings.filterwarnings("ignore"); pd.set_option("display.width", 250)
sys.path.insert(0, "/home/user/polyethmodel/polyeth"); import polysim, bookdepth  # noqa: E402
OUT = "/home/user/polyethmodel/reports"
W = pd.read_parquet("/home/user/data_eth/master.parquet", columns=["date", "up", "m2_-60_close_p", "m2_-45_close_p", "m2_-30_close_p", "bp_-60_z_s5", "bp_-60_z_s1", "bp_-120_z_s5"])
W["date"] = pd.to_datetime(W.date); T = W.index.values.astype(np.int64)
book = bookdepth.archive_imbalance(T)
PER = {"Jun": ("2026-06-01", "2026-06-30"), "Jul": ("2026-07-01", "2026-07-31"), "Aug1-19": ("2026-08-01", "2026-08-19"), "tune": ("2026-08-20", "2026-09-30"), "test": ("2026-10-01", "2026-11-03"), "Nov4-7": ("2026-11-04", "2026-11-07")}
days = [str(d.date()) for d in pd.date_range("2026-06-01", "2026-11-07")]
mk, g = polysim.load_windows(days=days)
def sim_for(t_entry, tp):
    sim = polysim.simulate(mk, g, t_entry=t_entry, entry_mode="fixed", pe=0.50, tps=(tp if tp else 1.01,), sls=(0.0,), tstops=(300,), slip=0.01); idx = np.arange(len(sim["W"]))
    for sd in (0, 1):
        k = sim["kind"][idx, sd, 0, 0, 0]; sim["pnl"][idx, sd, 0, 0, 0] = np.where((k == 0) & (tp is not None), (tp or 0.0) - 0.5, sim["settle"][idx, sd])
    return pd.DataFrame({"T": sim["W"], "pnl_up": sim["pnl"][idx, 0, 0, 0, 0], "pnl_dn": sim["pnl"][idx, 1, 0, 0, 0]}).set_index("T")
SIMS = {("-45", 0.85): sim_for(-45, 0.85), ("-45", None): sim_for(-45, None), ("-60", 0.85): sim_for(-60, 0.85), ("-30", 0.85): sim_for(-30, 0.85)}
rows = []
for (dec, tp), sim in SIMS.items():
    j = W.index.get_indexer(sim.index); ok = j >= 0; j = j[ok]; s = sim[ok]; pu, pdn = s.pnl_up.values, s.pnl_dn.values
    p = W[f"m2_{dec}_close_p"].values[j]; z5 = W["bp_-60_z_s5"].values[j]; bk = book[j]; dates = W.date.values[j]
    for vote in ("none", "eth1", "book", "eth1+book"):
        agree = np.ones(len(j), bool)
        if "eth1" in vote: agree &= np.isfinite(z5) & (np.sign(z5) == np.sign(p - 0.5))
        if "book" in vote: agree &= np.isfinite(bk) & (np.sign(bk) == np.sign(p - 0.5))
        for thr in (0.02, 0.03, 0.04, 0.05, 0.06, 0.08, 0.10):
            sel = np.isfinite(p) & (np.abs(p - 0.5) >= thr) & agree; pnl = np.where(p > 0.5, pu, pdn)
            for per, (a, b) in PER.items():
                m = sel & (dates >= np.datetime64(a)) & (dates <= np.datetime64(b)); x = pnl[m]
                if len(x) == 0: continue
                rows.append(dict(decision=dec, exit="0.85 then settle" if tp else "settle only", vote=vote, thr=thr, period=per, n=len(x), win=(x > 0).mean(), edge_c=x.mean() * 100,
                                 usd_at_100=x.sum() * 200, usd_per_day=x.sum() * 200 / ((np.datetime64(b) - np.datetime64(a)).astype(int) + 1), tstat=x.mean() / (x.std() + 1e-9) * np.sqrt(len(x))))
E = pd.DataFrame(rows); E.to_csv(f"{OUT}/eth_A_grid.csv", index=False)
piv = E[(E.decision == "-45") & (E.exit == "0.85 then settle")].pivot_table(index=["vote", "thr"], columns="period", values=["n", "win", "edge_c"])[[("n", p_) for p_ in PER] + [("win", p_) for p_ in PER] + [("edge_c", p_) for p_ in PER]]
print("== decision T-45, sell 0.85 then settle: n / win / edge (cents) by period ==\n" + piv.round(3).to_string())
print("\n== tune-selected cells (edge_c tune >= 2 and n tune >= 300), sorted by tune edge; their test / unseen results ==")
t = E[E.period == "tune"]; good = t[(t.edge_c >= 2) & (t.n >= 300)].sort_values("edge_c", ascending=False).head(15)
for _, r in good.iterrows():
    e = E[(E.decision == r.decision) & (E.exit == r.exit) & (E.vote == r.vote) & (E.thr == r.thr)].set_index("period")
    print(f"{r.decision:>4} {r.exit:17s} {r.vote:10s} thr {r.thr:.2f} | " + " | ".join(f"{p_}: n {int(e.loc[p_, 'n'])} win {e.loc[p_, 'win']:.3f} edge {e.loc[p_, 'edge_c']:+.1f}" if p_ in e.index else f"{p_}: -" for p_ in PER))
