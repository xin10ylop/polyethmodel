"""A exits that hold to settlement: no exits at all, TP 0.80 then settle, TP 0.80 + stop then settle (no
time-stop), TP 0.90/0.95 then settle. Settlement pays 1.00 / 0.00 with no fee. Paired day bootstrap vs the
frozen exits (0.80 / 0.05 / T+240) on Aug 1-19, tune, test (LABEL=main) or July (LABEL=july)."""
import os, sys, glob, numpy as np, pandas as pd, pyarrow.parquet as pq, warnings
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "polyeth"))
import polysim  # noqa: E402
warnings.filterwarnings("ignore"); pd.set_option("display.width", 220)
LABEL = os.environ.get("LABEL", "main")
m = pd.read_parquet(os.environ.get("MASTER_PATH", "/home/user/data_eth/master.parquet"), columns=["up", "date", "m2_-45_close_p", "bp_-60_z_s5"])
p = m["m2_-45_close_p"].values; z = m["bp_-60_z_s5"].values
sel = np.isfinite(p) & (np.abs(p - 0.5) >= 0.05) & np.isfinite(z) & (np.sign(z) == np.sign(p - 0.5))
m = m[sel]; mk = pd.read_parquet(polysim.MARKETS).set_index("window_start"); mk = mk[mk.outcome_up_price.isin([0.0, 1.0])]
W = sorted(set(m.index.values) & set(mk.index.values)); g = {}
days = os.environ.get("POLY_DAYS", "").split(",") if os.environ.get("POLY_DAYS") else None
for f in sorted(glob.glob(f"{polysim.TRADES}/*.parquet")):
    if days and os.path.basename(f)[:10] not in days: continue
    t = pq.read_table(f, columns=["ts", "outcome_index", "price", "window_start"], filters=[("window_start", "in", W)]).to_pandas()
    if not len(t): continue
    t["rel"] = (t.ts - t.window_start).astype(np.int32); t["p_up"] = np.where(t.outcome_index == 0, t.price, 1 - t.price).astype(np.float32)
    t = t.sort_values(["window_start", "ts"], kind="stable")
    for w, sub in t.groupby("window_start", sort=False): g[int(w)] = (sub.rel.values, sub.p_up.values)
W = [w for w in W if w in g]; m = m.loc[W]; side_up = m["m2_-45_close_p"].values > 0.5; up = m.up.values; won = np.where(side_up, up == 1, up == 0); dates = m.date.values
def fee(x): return 0.07 * x * (1 - x)
def run(tp=None, sl=None, tstop=None, settle=True):
    pnl = np.full(len(W), np.nan); kind = np.full(len(W), "", dtype=object)
    for i, w in enumerate(W):
        rel, pup = g[w]; ps = pup if side_up[i] else 1 - pup
        mm = (rel >= -45) & (rel <= 300); t, pr = rel[mm], ps[mm]
        if not len(pr): pnl[i] = (1.0 if won[i] else 0.0) - 0.5; kind[i] = "SETTLE"; continue
        i_tp = int(np.searchsorted(np.maximum.accumulate(pr), tp)) if tp else len(pr)
        i_sl = int(np.searchsorted(-np.minimum.accumulate(pr), -sl)) if sl else len(pr)
        i_ts = int(np.searchsorted(t, tstop, side="right")) if tstop else len(pr)
        first = min(i_tp, i_sl, i_ts)
        if first == i_tp and i_tp < len(pr): pnl[i] = tp - 0.5; kind[i] = "TP"
        elif first == i_sl and i_sl < len(pr): e = pr[i_sl] - 0.01; pnl[i] = e - 0.5 - fee(e); kind[i] = "SL"
        elif tstop and first == i_ts and i_ts < len(pr): e = (pr[i_ts - 1] if i_ts > 0 else 0.5) - 0.01; pnl[i] = e - 0.5 - fee(e); kind[i] = "TIME"
        else: pnl[i] = (1.0 if won[i] else 0.0) - 0.5; kind[i] = "SETTLE"
    return pnl, kind
pers = {"Aug1-19": dates < "2026-08-20", "tune": (dates >= "2026-08-20") & (dates <= "2026-09-30"), "test": dates >= "2026-10-01"} if LABEL == "main" else {"July": np.ones(len(dates), bool)}
base, _ = run(tp=0.80, sl=0.05, tstop=240)
rules = [("frozen: TP .80 / SL .05 / T+240", dict(tp=0.80, sl=0.05, tstop=240)), ("hold to settlement, no exits", dict())]
for tp in (0.70, 0.75, 0.80, 0.85, 0.90, 0.95):
    rules.append((f"TP {tp:.2f}, then settle", dict(tp=tp)))
    rules.append((f"TP {tp:.2f} + SL .05, then settle", dict(tp=tp, sl=0.05)))
    rules.append((f"TP {tp:.2f} + SL .10, then settle", dict(tp=tp, sl=0.10)))
rows = []
for lab, kw in rules:
    pnl, kind = run(**kw); r = {"rule": lab}
    for pn, msk in pers.items():
        x = pnl[msk]; d = pd.Series(x / 0.5 * 100).groupby(dates[msk]).sum(); dd = pd.Series(pnl[msk] - base[msk]).groupby(dates[msk]).sum().values
        rng = np.random.default_rng(0); b = [np.mean(dd[rng.integers(0, len(dd), len(dd))]) for _ in range(2000)]
        r.update({f"mean_{pn}": 100 * np.nanmean(x), f"win_{pn}": np.nanmean(x > 0), f"usd_{pn}": d.sum(), f"maxdd_{pn}": (d.cumsum().cummax() - d.cumsum()).max(), f"worst_{pn}": d.min(), f"negdays_{pn}": int((d < 0).sum()), f"p_{pn}": float(np.mean(np.array(b) <= 0)), f"settle_share_{pn}": float(np.mean(kind[msk] == "SETTLE"))})
    rows.append(r)
df = pd.DataFrame(rows); df.to_csv(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "reports", f"A_settle_exit_{LABEL}.csv"), index=False)
print(f"{LABEL}: {len(W)} A trades  (mean in cents/share; usd, maxdd, worst day at $100/trade; p = paired day bootstrap P(no gain vs frozen); settle_share = trades that went to settlement)")
for pn in pers:
    print(f"\n-- {pn} --"); print(df[["rule"] + [f"{k}_{pn}" for k in ("mean", "win", "usd", "maxdd", "worst", "negdays", "p", "settle_share")]].round(3).to_string(index=False))
