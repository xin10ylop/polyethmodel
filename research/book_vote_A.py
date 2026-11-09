"""A third agreement vote for rule A: the Binance futures book-depth imbalance (bid minus ask notional within X% of mid,
30-s snapshots, data.binance.vision bookDepth) must have the sign of the traded side.  Found in new_inputs_A2/A3: the
1 % imbalance is the one new feature whose relation to the outcome has the same sign in all six months.
Per period (Jun, Jul, tune, test, Nov 4-7), trade simulator with live exits (sell 0.85 else settle), $100 per trade:
frozen A vs A + book vote, at several depth levels, dead zones, m2 thresholds, and with the snapshot delayed 60 s."""
import os, sys, glob, warnings, numpy as np, pandas as pd
warnings.filterwarnings("ignore"); pd.set_option("display.width", 250)
sys.path.insert(0, "/home/user/polyethmodel/polyeth"); import polysim  # noqa: E402
EXTRA = "/home/user/data_eth/extra"; LAT = 47; OUT = "/home/user/polyethmodel/reports"
HEADS = ["m2_-45_close_p", "bp_-60_z_s5"]
SRC = [("all", "/home/user/data_eth/master.parquet", "/home/user/data_eth/polymarket/markets.parquet")]
parts = [pd.read_parquet(mp, columns=["date", "up"] + HEADS) for _, mp, _ in SRC]
W = pd.concat(parts); W = W[~W.index.duplicated(keep="last")].sort_index(); W["date"] = pd.to_datetime(W.date)
T = W.index.values.astype(np.int64)
bd = pd.concat([pd.read_parquet(f) for f in sorted(glob.glob(f"{EXTRA}/bookdepth/*.parquet"))]).sort_index(); bd = bd[~bd.index.duplicated()]
LV = ("0.2", "1.0", "2.0", "3.0", "5.0")
for lv in LV: bd[f"imb{lv}"] = (bd[f"n-{lv}"] - bd[f"n+{lv}"]) / (bd[f"n-{lv}"] + bd[f"n+{lv}"])
bd["imb_outer"] = ((bd["n-1.0"] - bd["n-0.2"]) - (bd["n+1.0"] - bd["n+0.2"])) / ((bd["n-1.0"] - bd["n-0.2"]) + (bd["n+1.0"] - bd["n+0.2"]))
COLS = [f"imb{lv}" for lv in LV] + ["imb_outer"]
def asof(when, maxage):
    idx = bd.index.values; j = np.searchsorted(idx, when, side="right") - 1; ok = j >= 0
    out = pd.DataFrame(np.where(ok[:, None], bd[COLS].values[np.maximum(j, 0)], np.nan), columns=COLS, index=W.index)
    age = np.where(ok, when - idx[np.maximum(j, 0)], np.nan); out[age > maxage] = np.nan; out["age"] = age; return out
B0 = asof(T - LAT, 120); B60 = asof(T - LAT - 60, 180); B300 = asof(T - LAT - 300, 420)
print("book snapshot age at T-47: median %.0f s, p90 %.0f s, missing %.1f%%" % (np.nanmedian(B0.age), np.nanpercentile(B0.age.dropna(), 90), 100 * B0["imb1.0"].isna().mean()), flush=True)

def period_sim(markets, days):
    polysim.MARKETS = markets; mk, g = polysim.load_windows(days=days)
    sim = polysim.simulate(mk, g, t_entry=-45, entry_mode="fixed", pe=0.50, tps=(0.85,), sls=(0.0,), tstops=(300,), slip=0.01)
    idx = np.arange(len(sim["W"]))
    for sd in (0, 1):
        k = sim["kind"][idx, sd, 0, 0, 0]; sim["pnl"][idx, sd, 0, 0, 0] = np.where(k == 0, 0.35, sim["settle"][idx, sd])
    return pd.DataFrame({"T": sim["W"], "pnl_up": sim["pnl"][idx, 0, 0, 0, 0], "pnl_dn": sim["pnl"][idx, 1, 0, 0, 0]}).set_index("T")
PER = {"Jun": (SRC[0][2], [f"2026-06-{d:02d}" for d in range(1, 31)]), "Jul": (SRC[0][2], [f"2026-07-{d:02d}" for d in range(1, 32)]),
       "tune": (SRC[0][2], [str(d.date()) for d in pd.date_range("2026-08-20", "2026-09-30")]), "test": (SRC[0][2], [str(d.date()) for d in pd.date_range("2026-10-01", "2026-11-03")]),
       "Nov4-7": (SRC[0][2], [f"2026-11-{d:02d}" for d in range(4, 8)])}
SIM = {k: period_sim(*v) for k, v in PER.items()}
p = W["m2_-45_close_p"].values; z5 = W["bp_-60_z_s5"].values; up = W.up.values == 1; sideup = p > 0.5
rows = []
def ev(per, name, sel, j, pnl_up, pnl_dn, extra=None):
    pnl = np.where(sideup[j], pnl_up, pnl_dn)[sel]
    if len(pnl) == 0: return
    r = dict(period=per, rule=name, n=int(len(pnl)), win=float((pnl > 0).mean()), edge_c=float(pnl.mean() * 100), usd_at_100=float(pnl.sum() * 200),
             usd_per_day=float(pnl.sum() * 200 / len(PER[per][1])), dir_acc=float((up[j] == sideup[j])[sel].mean())); rows.append(r)
for per, sim in SIM.items():
    j = W.index.get_indexer(sim.index); ok = j >= 0; simv = sim[ok]; j = j[ok]; pnl_up, pnl_dn = simv.pnl_up.values, simv.pnl_dn.values
    sgn = np.where(sideup[j], 1, -1)
    def A(thr): return np.isfinite(p[j]) & (np.abs(p[j] - 0.5) >= thr) & np.isfinite(z5[j]) & (np.sign(z5[j]) == np.sign(p[j] - 0.5))
    base = A(0.05); ev(per, "FROZEN A", base, j, pnl_up, pnl_dn)
    for B, lab in ((B0, ""), (B60, " [snap 60s older]"), (B300, " [snap 300s older]")):
        for col in COLS:
            b = B[col].values[j]
            for dz in (0.0, 0.05, 0.10):
                agree = np.isfinite(b) & (np.sign(b) == sgn) & (np.abs(b) >= dz)
                ev(per, f"A & book {col} agrees (|imb|>={dz:.2f}){lab}", base & agree, j, pnl_up, pnl_dn)
                if dz == 0.0: ev(per, f"A & book {col} DISagrees{lab}", base & np.isfinite(b) & (np.sign(b) != sgn), j, pnl_up, pnl_dn)
        if lab: continue
        b = B["imb1.0"].values[j]; agree = np.isfinite(b) & (np.sign(b) == sgn)
        for thr in (0.02, 0.03, 0.04):
            ev(per, f"A(m2 thr {thr:.2f}) & book imb1.0 agrees", A(thr) & agree, j, pnl_up, pnl_dn)
        # neutral when the snapshot is missing: keep the trade (what live would do if the feed is down)
        ev(per, "A & (book imb1.0 agrees or missing)", base & (agree | ~np.isfinite(b)), j, pnl_up, pnl_dn)
E = pd.DataFrame(rows); E.to_csv(f"{OUT}/A_book_vote.csv", index=False)
for per in PER:
    print(f"\n--- {per} ---"); e = E[E.period == per].drop(columns="period")
    print(e[~e.rule.str.contains("older")].round(3).to_string(index=False))
print("\n== snapshot-age robustness, imb1.0 no dead zone ==")
print(E[E.rule.str.contains("imb1.0 agrees \\(\\|imb\\|>=0.00\\)") | (E.rule == "FROZEN A")].pivot(index="rule", columns="period", values="win")[list(PER)].round(3).to_string())
print(E[E.rule.str.contains("imb1.0 agrees \\(\\|imb\\|>=0.00\\)") | (E.rule == "FROZEN A")].pivot(index="rule", columns="period", values="n")[list(PER)].to_string())
