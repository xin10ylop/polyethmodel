"""Live feature path for the eth1 model (ethpredict, SPOT variant, core feature set): ETH spot + BTC spot per-second bars ->
minute panel -> model matrix -> z forecast.  Uses ethpredict's own feature functions, so the live path equals training.
The spot variant has no perp input: the perp dict is the spot dict with nb/lbv/lsv/mxq = 0, exactly as ethpredict/live/sources.py does.

Per-second arrays (dict of numpy arrays, one entry per second from `t0`):  spot (ETH): o h l c v qv n bv     lead (BTC spot): c v bv
Models: production boosters models/spot_s{5,1}_core.txt.gz (trained through 2026-11-02) for live; the honest walk-forward folds used in the
backtest are scripts/train_ep_folds.py (same features, same learner)."""
import os, sys, json, gzip, datetime as dt
import numpy as np, pandas as pd
ETH1 = "/home/user/xin10ylop/eth1"; sys.path.insert(0, ETH1)
from ethpredict.features import seconds as secfeat, panel as P, matrix as F  # noqa: E402
import lightgbm as lgb  # noqa: E402
MODELS = os.path.join(ETH1, "models"); META = json.load(open(os.path.join(MODELS, "meta.json")))
PRE = 900; HIST_MIN = 2880
SPOT_COLS = ["o", "h", "l", "c", "v", "qv", "n", "bv"]; UM_COLS = SPOT_COLS + ["nb", "lbv", "lsv", "mxq"]; LEAD_COLS = ["c", "v", "bv"]


def _ffill(a):
    a = np.asarray(a, float).copy(); m = np.isnan(a)
    if m.all(): return a
    idx = np.where(~m, np.arange(len(a)), 0); np.maximum.accumulate(idx, out=idx); out = a[idx]
    first = np.flatnonzero(~m)[0]; out[:first] = a[first]; return out


def _prep(d, cols):
    out = {c: np.asarray(d[c], dtype=np.float64) for c in cols}; out["lp"] = np.log(_ffill(out["c"])); return out


class EPLive:
    def __init__(self, keys=("s5", "s1"), variant="spot"):
        self.cols, self.boosters, self.calib = {}, {}, {}
        for k in keys:
            t = META[variant]["targets"][k]; self.cols[k] = t["features"]["core"]; self.calib[k] = t["calib"]
            path = os.path.join(MODELS, t["files"]["core"])
            self.boosters[k] = lgb.Booster(model_file=path) if os.path.exists(path) else lgb.Booster(model_str=gzip.open(path + ".gz", "rt").read())

    def features(self, spot, lead, t0):
        """spot (ETH) / lead (BTC spot): per-second dicts starting at epoch t0 (t0 % 60 == 0), same length N (seconds < now)."""
        N = len(spot["c"]); nmin = (N - PRE) // 60; ends = PRE + 60 * np.arange(nmin) + 59
        ts = (pd.Timestamp(t0 + PRE, unit="s") + pd.to_timedelta(60 * np.arange(nmin), unit="s")).values
        um = {c: spot[c] for c in SPOT_COLS}; um.update({c: np.zeros(N) for c in ("nb", "lbv", "lsv", "mxq")})
        S, Fm, E = _prep(spot, SPOT_COLS), _prep(um, UM_COLS), _prep(lead, LEAD_COLS)
        out = secfeat.features_from_arrays(S, Fm, E, ends, ts, forward=False)
        for c in [c for c in out.columns if c.endswith(("_close", "_open", "_hi", "_lo")) and not c.startswith("e_") and c != "e_close"]:
            out[c] = out[c] + out["ref"]
        out = out.drop(columns=["ref"]); fc = out.select_dtypes("float64").columns
        keep64 = [c for c in fc if c.endswith(("_close", "_open", "_hi", "_lo")) or c.startswith(("px_", "basis"))]
        out[[c for c in fc if c not in keep64]] = out[[c for c in fc if c not in keep64]].astype("float32")
        out["ts"] = pd.to_datetime(out["ts"]); df = P.add_minute_features(out)
        return pd.concat([F.model_matrix(df), F.extra_features(df)], axis=1), df

    def forecast(self, spot, lead, t0):
        X, df = self.features(spot, lead, t0); row = X.iloc[-1:]
        return {k: float(b.predict(row[self.cols[k]].values.astype(np.float64))[0]) for k, b in self.boosters.items()}, df["ts"].iloc[-1]


def archive_arrays(day0, day1, sec="/home/user/btc-predict/data/sec"):
    """Per-second dicts from the archive parquet files (ETH spot = sec/eth, BTC spot = sec/spot) for days day0..day1 inclusive."""
    days = [day0 + dt.timedelta(days=i) for i in range((day1 - day0).days + 1)]; out = {}
    for venue, cols in (("eth", SPOT_COLS), ("spot", LEAD_COLS)):
        df = pd.concat([pd.read_parquet(f"{sec}/{venue}/{d}.parquet") for d in days], ignore_index=True); out[venue] = {c: df[c].to_numpy(float) for c in cols}
    return out["eth"], out["spot"], int(dt.datetime(day0.year, day0.month, day0.day, tzinfo=dt.UTC).timestamp())


if __name__ == "__main__":   # parity: live path on archive seconds vs the training panel (panel_ep.parquet) for the same minute
    import time
    ep = EPLive(); d1 = dt.date(2026, 11, 3); d0 = d1 - dt.timedelta(days=2); spot, lead, t0 = archive_arrays(d0, d1)
    inst = int(dt.datetime(2026, 11, 3, 17, 23, tzinfo=dt.UTC).timestamp()); n = inst - t0; sl = lambda d: {k: v[:n] for k, v in d.items()}
    t = time.time(); z, ts_last = ep.forecast(sl(spot), sl(lead), t0); print("live-path z", z, "last bar", ts_last, f"{time.time()-t:.1f}s")
    panel = pd.read_parquet(f"{ETH1}/data/features/panel_ep.parquet"); panel = panel[(panel.ts >= "2026-10-30") & (panel.ts <= "2026-11-03 17:30")].reset_index(drop=True)
    X = pd.concat([F.model_matrix(panel), F.extra_features(panel)], axis=1); j = np.flatnonzero(panel.ts.values == np.datetime64(ts_last))[0]
    for k, b in ep.boosters.items():
        zt = float(b.predict(X[ep.cols[k]].iloc[[j]].values.astype(np.float64))[0]); print(f"{k}: training-path z {zt:.5f} vs live-path z {z[k]:.5f}")
    Xl, _ = ep.features(sl(spot), sl(lead), t0); rowl = Xl.iloc[-1][ep.cols["s5"]].astype(float); rowt = X.iloc[j][ep.cols["s5"]].astype(float)
    diff = (rowl - rowt).abs() / (rowt.abs() + 1e-6); print("features with rel diff > 1%:", diff[diff > 0.01].round(3).to_dict())
