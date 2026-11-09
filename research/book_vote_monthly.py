"""Monthly and daily P&L ($100 per trade, live exits: sell 0.85 else settle) of frozen A vs A + book-depth vote (polybtc/bookdepth.py)."""
import sys, warnings, numpy as np, pandas as pd
warnings.filterwarnings("ignore"); pd.set_option("display.width", 200)
sys.path.insert(0, "/home/user/polyethmodel/polyeth"); import polysim, bookdepth  # noqa: E402
SRC = [("/home/user/data_eth/master.parquet", "/home/user/data_eth/polymarket/markets.parquet", [str(d.date()) for d in pd.date_range("2026-06-01", "2026-11-08")])]
W = pd.concat([pd.read_parquet(mp, columns=["date", "up", "m2_-45_close_p", "bp_-60_z_s5"]) for mp, _, _ in SRC]); W = W[~W.index.duplicated(keep="last")].sort_index()
W["date"] = pd.to_datetime(W.date); p = W["m2_-45_close_p"].values; z5 = W["bp_-60_z_s5"].values
b = bookdepth.archive_imbalance(W.index.values.astype(np.int64))
selA = np.isfinite(p) & (np.abs(p - 0.5) >= 0.05) & np.isfinite(z5) & (np.sign(z5) == np.sign(p - 0.5)); selAB = selA & np.isfinite(b) & (np.sign(b) == np.sign(p - 0.5))
recs = []
for mp, markets, days in SRC:
    polysim.MARKETS = markets; mk, g = polysim.load_windows(days=days)
    sim = polysim.simulate(mk, g, t_entry=-45, entry_mode="fixed", pe=0.50, tps=(0.85,), sls=(0.0,), tstops=(300,), slip=0.01); idx = np.arange(len(sim["W"]))
    j = W.index.get_indexer(sim["W"]); ok = j >= 0; j = j[ok]; idx = idx[ok]
    for sd in (0, 1):
        k = sim["kind"][idx, sd, 0, 0, 0]; sim["pnl"][idx, sd, 0, 0, 0] = np.where(k == 0, 0.35, sim["settle"][idx, sd])
    side = np.where(p[j] > 0.5, 0, 1); pnl = sim["pnl"][idx, side, 0, 0, 0]
    recs.append(pd.DataFrame({"T": sim["W"][ok], "date": W.date.values[j], "pnl": pnl, "A": selA[j], "AB": selAB[j]}))
R = pd.concat(recs).sort_values("T"); R["usd"] = R.pnl * 200; R["month"] = R.date.dt.strftime("%Y-%m")
def tab(groupcol):
    rows = []
    for key, grp in R.groupby(groupcol):
        a, ab = grp[grp.A], grp[grp.AB]
        rows.append({groupcol: key, "A n": len(a), "A win": a.pnl.gt(0).mean(), "A $": a.usd.sum(), "A+book n": len(ab), "A+book win": ab.pnl.gt(0).mean() if len(ab) else np.nan, "A+book $": ab.usd.sum()})
    return pd.DataFrame(rows).set_index(groupcol)
M = tab("month"); M.loc["total"] = M.sum(numeric_only=True); M.loc["total", ["A win", "A+book win"]] = [R[R.A].pnl.gt(0).mean(), R[R.AB].pnl.gt(0).mean()]
print("== monthly P&L, $100 per trade =="); print(M.round(3).to_string())
D = tab("date"); D.index = pd.to_datetime(D.index).strftime("%Y-%m-%d")
print("\n== November 4-7 by day =="); print(D.loc["2026-11-04":"2026-11-07"].round(3).to_string())
print("\n== the 10 worst days for frozen A, and what A + book vote did on them =="); print(D.sort_values("A $").head(10).round(3).to_string())
print("\n== the 10 worst days for A + book vote =="); print(D.sort_values("A+book $").head(10).round(3).to_string())
print(f"\nnegative days: A {int((D['A $'] < 0).sum())}/{len(D)}, A+book {int((D['A+book $'] < 0).sum())}/{len(D)};  worst day: A {D['A $'].min():.0f}, A+book {D['A+book $'].min():.0f}")
M.to_csv("/home/user/polyethmodel/reports/A_book_vote_monthly.csv"); D.to_csv("/home/user/polyethmodel/reports/A_book_vote_daily.csv")
