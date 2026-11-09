"""Produces every number used in the final report: exit grid for strategy B (tune -> test), relative
exits, the pre-open strategy A, combined portfolio, sizing. Writes reports/*.csv|json and trade records."""
import os
import numpy as np, pandas as pd, sys, json, warnings, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "polyeth"))
import polysim, strategy_eval as se, experiments as ex, final_strategy as fs  # noqa: E402
warnings.filterwarnings("ignore"); pd.set_option("display.width", 250)
TUNE, TEST, OLD = se.TUNE, se.TEST, se.OLD
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "reports"); os.makedirs(OUT, exist_ok=True)
master, mk, g = ex.master, ex.mk, ex.g

# ---------------------------------------------------------------- B: exit grid (selection on tune)
print("## B exit grid")
TPS = (0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.97); SLS = (0.05, 0.10, 0.15, 0.20, 0.25, 0.30); TSS = (180, 240, 270, 290)
sim = polysim.simulate(mk, g, t_entry=0, entry_mode="maker_last", t_cancel=10, tps=TPS, sls=SLS, tstops=TSS)
m, sim = ex.align(master, sim)
mod = m["m2_0_close_p"].values; mkt = sim["pre_last"][:, 0]; dis = mod - mkt; side = np.where(dis > 0, 0, 1); idx = np.arange(len(m))
rows = []
for delta in (0.06, 0.08, 0.10, 0.12, 0.15):
    base = np.isfinite(dis) & (np.abs(dis) >= delta)
    for a, tp in enumerate(TPS):
        for b, sl in enumerate(SLS):
            for c, ts in enumerate(TSS):
                pnl = sim["pnl"][idx, side, a, b, c]
                r = {"delta": delta, "tp": tp, "sl": sl, "tstop": ts}
                for split, nm in ((OLD, "old"), (TUNE, "tune"), (TEST, "test")):
                    s = base & (m.date.values >= split[0]) & (m.date.values <= split[1])
                    mm = se.metrics(np.where(s, pnl, np.nan), m.date.values)
                    for k_ in ("n", "mean", "win", "sharpe", "maxdd", "tstat"):
                        r[f"{k_}_{nm}"] = mm.get(k_, np.nan)
                rows.append(r)
    # relative exits: tp = entry + x, sl = entry - y (nearest grid cell per trade)
    ent = sim["entry_px"][idx, side]
    for x in (0.20, 0.25, 0.30, 0.40):
        for yv in (0.30, 0.40, 0.45):
            a_i = np.abs((ent[:, None] + x) - np.array(TPS)[None, :]).argmin(1); b_i = np.abs((ent[:, None] - yv) - np.array(SLS)[None, :]).argmin(1)
            c = TSS.index(240)
            pnl = sim["pnl"][idx, side, a_i, b_i, c]
            r = {"delta": delta, "tp": f"entry+{x}", "sl": f"entry-{yv}", "tstop": 240}
            for split, nm in ((OLD, "old"), (TUNE, "tune"), (TEST, "test")):
                s = base & (m.date.values >= split[0]) & (m.date.values <= split[1])
                mm = se.metrics(np.where(s, pnl, np.nan), m.date.values)
                for k_ in ("n", "mean", "win", "sharpe", "maxdd", "tstat"):
                    r[f"{k_}_{nm}"] = mm.get(k_, np.nan)
            rows.append(r)
grid = pd.DataFrame(rows); grid.to_csv(f"{OUT}/B_exit_grid.csv", index=False)
gg = grid[grid.n_tune >= 300].copy(); gg["score"] = gg.mean_tune * np.sqrt(gg.n_tune)
print(gg.sort_values("score", ascending=False).head(20)[["delta", "tp", "sl", "tstop", "n_tune", "mean_tune", "win_tune", "n_test", "mean_test", "win_test", "sharpe_test", "maxdd_test", "tstat_test"]].round(4).to_string(index=False))
print("\n-- fixed TP/SL 0.80/0.10/t240 across deltas --")
print(grid[(grid.tp == 0.80) & (grid.sl == 0.10) & (grid.tstop == 240)][["delta", "n_old", "mean_old", "n_tune", "mean_tune", "win_tune", "n_test", "mean_test", "win_test", "sharpe_test", "maxdd_test", "tstat_test"]].round(4).to_string(index=False))
print("\n-- user's tight exits 0.60-ish: closest cell tp0.65/sl0.30 --")
print(grid[(grid.tp == 0.65) & (grid.sl == 0.30) & (grid.tstop == 290)][["delta", "n_tune", "mean_tune", "n_test", "mean_test", "win_test"]].round(4).to_string(index=False))

# ---------------------------------------------------------------- locked configs
# frozen live rule (README): m2 close head |p-0.5| >= 0.05, btc-predict 5-min z at T-60 agreeing in sign, TP 0.80 / SL 0.05 / T+240
CFG_A = {"kind": "A", "decision_offset": -45, "fill_by": -45, "entry_price": 0.50, "signal": "m2_-45_close_p", "agree": "bp_-60_z_s5", "thr": 0.05, "tp": 0.85, "sl": None, "tstop": None, "settle": True}
CFG_A0 = {"kind": "A", "decision_offset": -45, "fill_by": -45, "entry_price": 0.50, "signal": "m2_-45_close_p", "agree": "bp_-60_z_s5", "thr": 0.05, "tp": 0.80, "sl": 0.05, "tstop": 240}   # previous frozen exits (shadow)
CFG_AB = dict(CFG_A, book_agree="1.0")   # A + book vote (2026-11-09, research/book_vote_A.py): win rate up in all five periods at ~60% of the trades
CFG_B = dict(fs.CFG_B); CFG_B0 = dict(fs.CFG_B0); CFG_B2 = dict(fs.CFG_B2); CFG_C = dict(fs.CFG_C); CFG_D = dict(fs.CFG_D)


def trades_A2(cfg, master, mk, g):
    settle = cfg.get("settle", False)
    sim = polysim.simulate(mk, g, t_entry=cfg["fill_by"], entry_mode="fixed", pe=cfg["entry_price"], tps=(cfg["tp"],), sls=(0.0 if settle else cfg["sl"],), tstops=(300 if settle else cfg["tstop"],), slip=cfg.get("slip", 0.01))
    m, sim = ex.align(master, sim)
    if settle:   # no stop, no time-stop: whatever did not reach the take-profit is held to settlement (1.00 / 0.00, no fee)
        idx = np.arange(len(m))
        for sd in (0, 1):
            k = sim["kind"][idx, sd, 0, 0, 0]; sim["pnl"][idx, sd, 0, 0, 0] = np.where(k == 0, cfg["tp"] - cfg["entry_price"], sim["settle"][idx, sd]); sim["kind"][idx, sd, 0, 0, 0] = np.where(k == 0, 0, 2)
    p = m[cfg["signal"]].values; side = np.where(p > 0.5, 0, 1)
    sel = np.isfinite(p) & (np.abs(p - 0.5) >= cfg["thr"])
    if cfg.get("agree"):
        z = m[cfg["agree"]].values; sel &= np.isfinite(z) & (np.sign(z) == np.sign(p - 0.5))
    if cfg.get("book_agree"):   # third vote: Binance perp book-depth imbalance within book_agree % of mid, known at T-47 (polybtc/bookdepth.py)
        import bookdepth; b = bookdepth.archive_imbalance(m.index.values.astype(np.int64), level=cfg["book_agree"]); sel &= np.isfinite(b) & (np.sign(b) == np.sign(p - 0.5))
    return m, sim, p, side, sel, {}
fs.trades_A = trades_A2
summary = {}
recs = {}
for nm, cfg in (("A_preopen", CFG_A), ("A0_preopen_frozen_exits", CFG_A0), ("AB_preopen_book_vote", CFG_AB), ("B_open_mispricing", CFG_B), ("B0_open_mispricing_frozen_exits", CFG_B0), ("B2_open_mispricing_delta008", CFG_B2), ("C_preopen_follow", CFG_C), ("D_high_confidence_settle", CFG_D)):
    res, rec = fs.evaluate(cfg, label=nm)
    summary[nm] = {"config": cfg, "results": res.round(5).to_dict()}
    rec.to_csv(f"{OUT}/trades_{nm}.csv", index=False); recs[nm] = rec
    fs.robustness(cfg, nm)
# ---------------------------------------------------------------- combined portfolio & sizing (TEST)
print("\n## combined A + B on TEST, $100 stake per trade")
ra, rb = recs["A_preopen"], recs["B_open_mispricing"]
def daily(rec):
    t = rec[(rec.date >= TEST[0]) & (rec.date <= TEST[1]) & rec.filled]
    return (t.pnl / t.entry_px * 100).groupby(t.date).sum()
da, db = daily(ra), daily(rb)
comb = pd.concat([da.rename("A"), db.rename("B")], axis=1).fillna(0); comb["A+B"] = comb.A + comb.B
print(comb.describe().round(2).to_string()); print("corr(A,B) daily:", round(comb.A.corr(comb.B), 3))
for c in ("A", "B", "A+B"):
    eq = comb[c].cumsum(); print(f"{c}: total {eq.iloc[-1]:.0f}  maxdd {(eq.cummax()-eq).max():.0f}  worst day {comb[c].min():.0f}  sharpe {comb[c].mean()/comb[c].std()*np.sqrt(365):.1f}  days {len(eq)}")
# C ("B five seconds early") and the executable book: C first (decided at T-5), B at T on the windows where C did not fill
rc = recs["C_preopen_follow"]; cf = rc[rc.filled]; bf = rb[rb.filled & ~rb["T"].isin(cf["T"])]; bc = pd.concat([cf, bf]).sort_values("T")
dc, dbc = daily(rc), daily(bc)
comb3 = pd.concat([da.rename("A"), db.rename("B"), dc.rename("C"), dbc.rename("C+B")], axis=1).fillna(0); comb3["A+C+B"] = comb3.A + comb3["C+B"]
print("\n## with C (T-5 disagreement >= 0.08, bid at the last print until the open, 0.80/0.05/240) on TEST, $100 per trade")
for c in ("C", "C+B", "A+C+B"):
    eq = comb3[c].cumsum(); print(f"{c}: total {eq.iloc[-1]:.0f}  maxdd {(eq.cummax()-eq).max():.0f}  worst day {comb3[c].min():.0f}  sharpe {comb3[c].mean()/comb3[c].std()*np.sqrt(365):.1f}  days {len(eq)}  trades/day {len(bc[(bc.date >= TEST[0]) & (bc.date <= TEST[1])]) / len(eq) if c == 'C+B' else len(cf[(cf.date >= TEST[0]) & (cf.date <= TEST[1])]) / len(eq):.1f}")
print("corr daily A vs C+B:", round(comb3.A.corr(comb3["C+B"]), 3))
rd = recs["D_high_confidence_settle"]; dd_ = daily(rd); comb3["D"] = dd_.reindex(comb3.index).fillna(0); comb3["A+C+B+D"] = comb3["A+C+B"] + comb3["D"]
comb3["A+B+D"] = comb3.A + comb3.B + comb3["D"]
for c in ("D", "A+C+B+D", "A+B+D"):
    eq = comb3[c].cumsum(); print(f"{c}: total {eq.iloc[-1]:.0f}  maxdd {(eq.cummax()-eq).max():.0f}  worst day {comb3[c].min():.0f}  sharpe {comb3[c].mean()/comb3[c].std()*np.sqrt(365):.1f}  days {len(eq)}")
summary["combined_test_daily_with_C"] = comb3.round(3).to_dict()
# Kelly-style sizing from per-trade returns (B, test)
t = rb[(rb.date >= TEST[0]) & (rb.date <= TEST[1]) & rb.filled]; r = (t.pnl / t.entry_px).values
kelly = r.mean() / r.var(); print(f"\nB per-trade return: mean {r.mean():.4f} sd {r.std():.4f}  full Kelly fraction of bankroll per trade {kelly:.3f}  (suggest 1/4 Kelly = {kelly/4:.3f})")
# concurrency: trades per window = 1; max simultaneous = 1 window at a time (5-min windows are sequential)
summary["combined_test_daily"] = comb.round(3).to_dict()
summary["kelly_B"] = {"mean_ret": float(r.mean()), "sd_ret": float(r.std()), "full_kelly": float(kelly)}
json.dump(summary, open(f"{OUT}/summary.json", "w"), indent=1, default=float)
print("\nsaved", OUT)
