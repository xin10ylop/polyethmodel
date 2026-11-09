"""Batch out-of-sample forecasts from the FROZEN eth3 model (xin10ylop/eth3, models/wf5_20260430, trained
through 2026-04-30) for every 5-minute Polymarket window and a grid of decision times.

The Polymarket reference is sampled ~L seconds BEFORE the nominal window start T (L ~ 30 s from
mid-Aug 2026; see work/ref_timing3.py). The model is therefore run on a time-shifted view of the
timeline so that its 'window' is the EFFECTIVE window [T-L, T+300-L):
  group A (pre-reference decisions): decision at T+d (d <= -L): pseudo window starting at T+d, e = 0
          -> forecasts the move over the next H seconds from T+d (what a trader must use before the
             reference is fixed).
  group B (post-reference): window start T-L (P0 = Binance open of second T-L), e = d + L.
Mirrors btcmodel.predictor.Predictor.predict exactly (vectorised).  Output: /home/user/data_eth/forecasts/m2_*.parquet
"""
import os, sys, time
import numpy as np, pandas as pd
from scipy.special import logit, expit
os.environ["BTC_DATA_DIR"] = os.environ.get("ETH_DATA_DIR", "/home/user/data_eth"); os.environ["BTC_ASSET"] = "ETH"
sys.path.insert(0, "/home/user/xin10ylop/eth3")
from btcmodel.timeline import load_timeline
from btcmodel.features import Base, FeatureEngine, W
from btcmodel.models import add_derived
from btcmodel.regime import regime_features, REGIME_FEATS
from btcmodel.predictor import Predictor, HEAD_H, DIRECT_FEATS
from btcmodel.events import TOUCH_LEVELS, FP_LEVELS, TOUCH_FEATS, FP_FEATS
from btcmodel.prob import p_touch, p_double_barrier

OUT = os.environ.get("M2_OUT", "/home/user/data_eth/forecasts"); os.makedirs(OUT, exist_ok=True)
L = int(os.environ.get("M2_REF_LAG", "30"))      # reference lag: 30 s from Sept 2026 (60 s TWAP ending at T); 0 for the earlier point-price rule
PRE = [-120, -90, -75, -60, -50, -45, -40, -35, -30]                 # group A: pseudo-start at T+d
POST = [-25, -20, -15, -10, -5, 0, 5, 10, 15, 20, 30, 45, 60, 90, 120, 150, 180, 210, 240, 260]  # group B
KEEP_FEATS = ["pls_1", "pls_2", "pls_5", "pls_10", "pls_30", "pls_60", "bdev_10", "bdev_60", "bdev_300", "bdev_1800",
              "r_1", "r_2", "r_3", "r_5", "r_10", "r_30", "r_60", "r_300", "simb_5", "pimb_5", "simb_10", "pimb_10",
              "simb_30", "pimb_30", "pimb_60", "netflow_10", "netflow_60", "lrv_60", "lrv_300", "lrvsa_300", "lrv_3600",
              "lrv_86400", "seas_now", "jump_300", "lcnt_ratio", "lvol_ratio", "lcnt_60", "maxq_10", "x_v", "x_bps",
              "wmax_v", "wmin_v", "ncross", "prev_ret_v", "prev2_ret_v", "mtf_15m", "mtf_1h", "mtf_4h", "round_1000",
              "round_100", "how", "hod_et", "dow_et", "rskew_300", "rngpos_300"] + REGIME_FEATS


class ShiftedTL:
    """Timeline view starting `s` seconds later (so that 5-minute grids land at T - L etc.)."""
    def __init__(self, tl, s):
        self.tl, self.s = tl, int(s)
        self.n = tl.n - self.s
        self.t0_epoch = tl.t0_epoch + self.s
        self.present_sec = np.ones(self.n, dtype=bool)
    def __getitem__(self, k):
        return self.tl[k][self.s:self.s + self.n]


def touch_batch(d, hm, tm):
    mu, s, H = hm.bm_params(d); x = d["xcur"].values; sig = d["sig"].values; res = {}
    for k in TOUCH_LEVELS:
        Lv = np.log1p(k * 1e-4); dist = (Lv - x) if k > 0 else (x - Lv); sgn = 1.0 if k > 0 else -1.0
        ok = dist > 0; out = np.ones(len(d))
        if ok.any():
            p = np.clip(p_touch(dist[ok], sgn * mu[ok], s[ok], H[ok]), 1e-6, 1 - 1e-6)
            row = pd.DataFrame({"d_std": dist[ok] / sig[ok], "mu_to": sgn * d["mu"].values[ok], "is_down": float(k < 0),
                                "lev_std": abs(Lv) / sig[ok], "p_struct": p, "lp_struct": logit(p)})
            for c in ["lsig_rel", "e", "lH", "lrvsa_300", "jump_300", "som", "ncross", "hod_et"]:
                row[c] = d[c].values[ok]
            row = row.astype({c: np.float32 for c in TOUCH_FEATS}); out[ok] = tm.predict(row)
        res[f"touch{k:+d}"] = out
    return res


def fp_batch(d, hm, fm):
    mu, s, H = hm.bm_params(d); x = d["xcur"].values; sig = d["sig"].values; res = {}
    for k in FP_LEVELS:
        U, D = np.log1p(k * 1e-4), np.log1p(-k * 1e-4); inside = (x > D) & (x < U)
        pu_o = (x >= U).astype(float); pd_o = (x <= D).astype(float); pn_o = np.zeros(len(d))
        if inside.any():
            a, b = U - x[inside], x[inside] - D
            pu, pd_ = p_double_barrier(a, b, mu[inside], s[inside], H[inside])
            pu, pd_ = np.maximum(pu, 1e-6), np.maximum(pd_, 1e-6); pn = np.maximum(1 - pu - pd_, 1e-6); z = pu + pd_ + pn
            row = pd.DataFrame({"pn": pn / z, "pu": pu / z, "pd": pd_ / z, "a_std": a / sig[inside], "b_std": b / sig[inside]})
            row["lpu"] = np.log(row["pu"] / row["pn"]); row["lpd"] = np.log(row["pd"] / row["pn"])
            for c in ["mu", "lsig_rel", "e", "lH", "lrvsa_300", "jump_300", "som", "hod_et"]:
                row[c] = d[c].values[inside]
            row = row.astype({c: np.float32 for c in FP_FEATS}); P = fm.predict(row)
            pn_o[inside], pu_o[inside], pd_o[inside] = P[:, 0], P[:, 1], P[:, 2]
        res[f"fp{k}_up"] = pu_o; res[f"fp{k}_dn"] = pd_o; res[f"fp{k}_none"] = pn_o
    return res


def score_points(tl, pr, Tw_local, d0, es, c0_abs, c1_abs):
    """All windows Tw_local (indices in tl frame, multiples of 300) x elapsed list `es`, with the window
    start at real offset d0 from T. Returns DataFrame with T (epoch), d, e."""
    s = d0 % 300
    tls = ShiftedTL(tl, s)
    a = ((c0_abs - s - 2 * 86400) // 3600) * 3600 - s            # hour-aligned in the shifted frame
    a = max(a, 0)
    while (tls.t0_epoch + a) % 3600:
        a += 1
    b = min(tls.n, c1_abs - s + 900)
    base = Base(tls, a, b); fe = FeatureEngine(base, pr.season)
    T0_s = Tw_local + d0 - s                                      # window start index in shifted frame
    nT, nE = len(Tw_local), len(es)
    T0_abs = np.repeat(T0_s, nE); e = np.tile(np.array(es, dtype=np.int64), nT)
    T0 = (T0_abs - a).astype(np.int64)
    X, aux = fe.compute(T0, e)
    R = regime_features(tls, T0_abs.astype(np.int64), e)
    for k in R.columns:
        X[k] = R[k].values
    X["xcur"] = aux["cur"] - aux["p0"]; X["v"] = aux["v"].astype(np.float32)
    out = pd.DataFrame({"T": tl.t0_epoch + np.repeat(Tw_local, nE), "d": d0 + e, "d0": d0, "e": e,
                        "p0_model": np.exp(aux["p0"]), "price_now": np.exp(aux["cur"])})
    for c in KEEP_FEATS:
        out[c] = X[c].values.astype(np.float32)
    drift_px = {}
    for head, h in HEAD_H.items():
        H = (W - e) if h is None else np.full(len(e), h)
        valid = (e + H) <= W
        d = X[valid].copy().reset_index(drop=True); Hv = H[valid]
        d[f"{head}__H"] = Hv.astype(np.int16)
        d[f"{head}__sint"] = pr.season.integrated(T0[valid] + e[valid], Hv.astype(np.int64), base.how_hour).astype(np.float32)
        hm = pr.models[head]["head"]
        for sh in HEAD_H:
            if f"mu_{sh}_px" in hm.extra:
                d[f"mu_{sh}_px"] = np.float32(drift_px[sh][valid]) if sh in drift_px else np.float32(np.nan)
        d = add_derived(d, head)
        if any(c.startswith("mu_h") for c in hm.extra):
            for c in hm.extra:
                d[c] = (d[c] / d["naive_scale"]).astype(np.float32)
        hm.prepare(d)
        p_m = np.clip(hm.p_up(d), 1e-5, 1 - 1e-5)
        dm = pr.models[head]["direct"]; dfe = list(dm.feature_name())   # eth3: the direct model names its own columns
        p_d = np.clip(dm.predict(d[dfe], num_iteration=dm.best_iteration), 1e-5, 1 - 1e-5)
        p_blend = expit(0.5 * logit(p_m) + 0.5 * logit(p_d))
        dp = np.full(len(e), np.nan); dp[valid] = d["mu"].values * d["sig"].values; drift_px[head] = dp
        pc = hm.dist.sf(-d["mu"].values, Hv); pi0 = d["pi0"].values
        cols = {"p": p_blend, "pm": p_m, "pd": p_d, "mu": d["mu"].values, "sig": d["sig"].values, "pi0": pi0,
                "pup_now": pi0 + (1 - pi0) * pc, "pc_now": pc, "H": Hv}
        if head in ("close", "h60"):
            cols.update(touch_batch(d, hm, pr.models[head]["touch"])); cols.update(fp_batch(d, hm, pr.models[head]["fp"]))
        for k, v in cols.items():
            full = np.full(len(e), np.nan); full[valid] = v; out[f"{head}_{k}"] = full.astype(np.float32)
    return out


def score_chunk(tl, pr, c0_abs, c1_abs, lag=L, pre=PRE, post=POST):
    Tw = np.arange((c0_abs + W - 1) // W * W, c1_abs, W); Tw = Tw[(Tw + W + 60 <= tl.n) & (Tw >= 10 * 86400)]
    parts = [score_points(tl, pr, Tw, -lag, [d + lag for d in post], c0_abs, c1_abs)]
    for d in pre:
        if d == -lag:
            continue
        parts.append(score_points(tl, pr, Tw, d, [0], c0_abs, c1_abs))
    return pd.concat(parts, ignore_index=True).sort_values(["T", "d"]).reset_index(drop=True)


def consistency_test(tl, pr, n=4, seed=1):
    """Shift-free path (d0 = 0) must reproduce Predictor.predict bit-for-bit (up to float32)."""
    from btcmodel.timeline import FIELDS
    rng = np.random.default_rng(seed)
    c0 = tl.index_of(__import__("datetime").date(2026, 10, 10)); c1 = c0 + 86400
    Tw = np.arange(c0, c1, 300)[rng.choice(288, n, replace=False)]
    worst = 0.0
    for e in (0, 37, 200):
        o = score_points(tl, pr, Tw, 0, [e], c0, c1)
        for j, T in enumerate(Tw):
            start = (T - 9 * 86400) // 3600 * 3600; end = T + e
            arrays = {f: np.asarray(tl[f][start:end]) for f in FIELDS}
            pr.ingest(arrays, tl.t0_epoch + start)
            ref = pr.predict(tl.t0_epoch + T + e, p0=float(np.asarray(tl["s_open"][T])))
            for head in ref["heads"]:
                dv = abs(ref["heads"][head]["p_above_ref_blend"] - o[f"{head}_p"].values[j]); worst = max(worst, dv)
                if head == "close":
                    worst = max(worst, abs(ref["heads"][head]["touch"]["+10bps"] - o["close_touch+10"].values[j]))
    print("consistency: max |batch - Predictor| =", worst)
    return worst


if __name__ == "__main__":
    tl = load_timeline()
    pr = Predictor.load_dir(os.environ.get("M3_MODEL", "/home/user/xin10ylop/eth3/models/wf5_20260430"))
    if sys.argv[1] == "test":
        consistency_test(tl, pr); sys.exit()
    from datetime import date, timedelta
    d0, d1 = date.fromisoformat(sys.argv[1]), date.fromisoformat(sys.argv[2])
    step = int(sys.argv[3]) if len(sys.argv) > 3 else 4
    d = d0
    while d <= d1:
        d2 = min(d + timedelta(days=step - 1), d1)
        f = f"{OUT}/m2_{d}_{d2}.parquet"
        if not os.path.exists(f):
            t = time.time()
            o = score_chunk(tl, pr, tl.index_of(d), tl.index_of(d2) + 86400)
            o.to_parquet(f, index=False)
            print(f"{d}..{d2}: {len(o)} rows {time.time()-t:.0f}s", flush=True)
        d = d2 + timedelta(days=1)
    print("done")
