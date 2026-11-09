"""Honest walk-forward minute forecasts of the eth1 model (ethpredict), SPOT variant, core feature set.

Mirrors /home/user/polybtcmodel/scripts/train_bp_june.py / train_bp_july.py for ETH:
  * 2-month blocks, each model trained only on data strictly before its block (purge h+10 min):
        Jun-Jul 2026  <- trained on 2026-02-03 .. 2026-05-31
        Aug-Sep 2026  <- trained on 2026-02-03 .. 2026-07-31
        Oct-Nov 2026  <- trained on 2026-02-03 .. 2026-09-30
  * E.make_folds / E.walk_forward / M.lgbm_factory(rounds=400), target y / F.target_scale.
  * Feature set = meta.json spot/core list (spot_s5_core: 136 cols, spot_s1_core: 169 cols),
    restricted to columns present in X (all of them are).
Data: eth1/data/sec/{eth,spot} are symlinks to the btc-predict per-second store; eth1/data/sec/ethum
is the ETH spot file with nb/lbv/lsv/mxq = 0 (exactly what live/sources.build_features does for the
spot variant: F = S.copy() with those four columns zeroed).  No premium-index / Coinbase / alt data
-> those columns are simply absent, and none of them is in the spot core list.
Original eth1 code untouched; everything here only calls ethpredict's public functions.
Output: /home/user/data_eth/forecasts/ep_minute.parquet (one row per minute bar `ts` = bar open; the
forecast is issued at ts + 60 s, same convention as bp_minute_*.parquet).
"""
import os, sys, time, json
import numpy as np, pandas as pd
sys.path.insert(0, "/home/user/xin10ylop/eth1")
os.chdir("/home/user/xin10ylop/eth1")
from ethpredict.features import panel as P, matrix as F
from ethpredict.modeling import gbm as M, evaluate as E
from ethpredict.config import FEAT, MODELS

START, END = "2026-02-01", "2026-11-07"          # last per-second day available locally
MONTHS = ["2026-06", "2026-07", "2026-08", "2026-09", "2026-10", "2026-11"]   # 3 blocks of 2 months
OUT = "/home/user/data_eth/forecasts"; os.makedirs(OUT, exist_ok=True)
META = json.load(open(MODELS / "meta.json"))["spot"]["targets"]

t0 = time.time()
pp = FEAT / "panel_ep.parquet"
if pp.exists():
    df = pd.read_parquet(pp)
else:
    df = P.build(START, END, workers=3, out_name="panel_ep")
print("panel", df.shape, df["ts"].min(), df["ts"].max(), f"{time.time()-t0:.0f}s", flush=True)
df = df[df["ts"] >= "2026-02-03"].reset_index(drop=True)      # first two days = warm-up of 24h rolling stats
X = pd.concat([F.model_matrix(df), F.extra_features(df)], axis=1)
print("X", X.shape, f"{time.time()-t0:.0f}s", flush=True)
res = {"ts": df["ts"].values, "sigma": df["sigma"].values, "s_close": df["s_close"].values}
summary = {}
for tok in ["1", "5"]:
    venue = "s"
    h = int(tok)
    key = f"{venue}{tok}"
    want = META[key]["features"]["core"]
    cols = [c for c in want if c in X.columns]
    missing = [c for c in want if c not in X.columns]
    print(key, "features wanted", len(want), "present", len(cols), "missing", missing, flush=True)
    scale = F.target_scale(df, h)
    yz = df[f"y_{venue}{tok}"] / scale
    folds = E.make_folds(df["ts"], MONTHS, purge_minutes=h + 10, block=2)
    for f in folds:
        print(f"  {key} fold {f.name}: train {df['ts'].iloc[f.train[0]]} .. {df['ts'].iloc[f.train[-1]]} "
              f"({len(f.train)}), test {df['ts'].iloc[f.test[0]]} .. {df['ts'].iloc[f.test[-1]]} ({len(f.test)})", flush=True)
    fp = M.lgbm_factory(rounds=400)
    pred = E.walk_forward(fp, X[cols], yz, folds, verbose=True)
    res[f"z_{key}"] = pred.values
    res[f"y_{key}"] = df[f"y_{venue}{tok}"].values
    y = df[f"y_{venue}{tok}"].values
    summary[key] = {}
    for f in folds:
        m = np.zeros(len(df), bool); m[f.test] = True
        m &= pred.notna().values & np.isfinite(y)
        r = E.metrics(y[m], pred.values[m])
        summary[key][f.name] = r
        print(f"  {key} fold {f.name}: n={r['n']} ic={r['ic']:.4f} rank_ic={r['rank_ic']:.4f} hit={r['hit']:.4f} "
              f"hit_top10={r['hit_top10']:.4f} edge_bps={r['edge_bps']:.3f}", flush=True)
    m = pred.notna().values & np.isfinite(y)
    r = E.metrics(y[m], pred.values[m])
    summary[key]["all"] = r
    print(key, len(cols), "feats ALL", {k: round(v, 4) for k, v in r.items()}, f"{time.time()-t0:.0f}s", flush=True)
    mi = E.monthly_ic(df["ts"][m], y[m], pred.values[m])
    print(key, "monthly IC", mi.round(4).to_dict(), flush=True)
out = pd.DataFrame(res)
out.to_parquet(f"{OUT}/ep_minute.parquet", index=False)
json.dump(summary, open(f"{OUT}/ep_minute_metrics.json", "w"), indent=1)
print("wrote", f"{OUT}/ep_minute.parquet", out.shape, "z_s5 non-null", int(out["z_s5"].notna().sum()),
      "z_s1 non-null", int(out["z_s1"].notna().sum()), f"{time.time()-t0:.0f}s", flush=True)
print("done")
