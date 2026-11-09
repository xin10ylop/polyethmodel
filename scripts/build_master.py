"""One row per Polymarket window: outcome, Binance truth, regime descriptors, and every model forecast
(wide: <model>_<offset>_<column>). Output: /home/user/data/master.parquet"""
import numpy as np, pandas as pd, glob, sys, os
TRUTH = os.environ.get("TRUTH_DIR", "/home/user/data_eth/truth"); MARKETS = os.environ.get("POLY_MARKETS", "/home/user/data_eth/polymarket/markets_2026-06-01_2026-11-09.parquet")
BP = os.environ.get("BP_MINUTE", "/home/user/data_eth/forecasts/ep_minute.parquet"); MASTER_OUT = os.environ.get("MASTER_OUT", "/home/user/data_eth/master.parquet")
OFFS_PRE = [-120, -90, -75, -60, -50, -45, -40, -35, -30]
OFFS_POST = [-25, -20, -15, -10, -5, 0, 5, 10, 15, 20, 30, 45, 60, 90, 120, 150, 180, 210, 240, 260]
M2_COLS = ["h10_p", "h10_pup_now", "h10_mu", "h15_pup_now", "h30_p", "h30_pup_now", "h30_mu", "h60_p", "h60_pup_now", "h60_mu",
           "close_p", "close_pm", "close_pd", "close_pup_now", "close_mu", "close_sig", "close_pi0",
           "close_touch+5", "close_touch-5", "close_touch+10", "close_touch-10", "close_fp5_up", "close_fp5_dn",
           "close_fp10_up", "close_fp10_dn", "h60_fp5_up", "h60_fp5_dn", "x_v", "x_bps", "lrv_300", "lrvsa_300", "lrv_3600",
           "seas_now", "jump_300", "pls_5", "pls_10", "bdev_60", "bdev_300", "r_10", "r_30", "pimb_10", "prev_ret_v",
           "reg_ll_beta_1h", "reg_catchup_1h", "reg_vr300_1d", "reg_rv_1d_7d", "reg_cp10_1d", "reg_cp10_7d", "reg_trend_1d", "p0_model", "price_now"]
BM_COLS = ["h10_p", "h10_pup_now", "h30_p", "h30_pup_now", "h60_p", "h60_pup_now", "close_p", "close_praw", "close_pst", "close_pup_now",
           "close_mu", "close_sd", "close_mz", "close_fp5_up", "close_fp5_dn", "close_fp10_up", "close_fp10_dn", "h60_fp5_up", "h60_fp5_dn",
           "close_touch+5", "close_touch-5", "basis_dev_60", "basis_dev_300", "zfret_5", "fofi_5", "lrv_300", "K_bm", "S_bm"]

def wide(prefix, cols, offs):
    files = sorted(glob.glob(f"{os.environ.get('FORECASTS_DIR', '/home/user/data_eth/forecasts')}/{prefix}_*.parquet"))
    if not files:
        return None
    fc = pd.concat([pd.read_parquet(f, columns=["T", "d"] + [c for c in cols]) for f in files], ignore_index=True)
    fc = fc[fc.d.isin(offs)].drop_duplicates(["T", "d"])
    w = fc.pivot(index="T", columns="d", values=cols)
    w.columns = [f"{prefix}_{d}_{c}" for c, d in w.columns]
    return w

tw = pd.read_parquet(f"{TRUTH}/windows.parquet").set_index("T")
paths = np.load(f"{TRUTH}/paths.npy")         # cols T-120..T+300
mk = pd.read_parquet(MARKETS).set_index("window_start")
mk = mk[mk.outcome_up_price.isin([0.0, 1.0])]
m = tw.join(mk[["outcome_up_price", "price_to_beat", "volume"]], how="inner")
m["up"] = (m.outcome_up_price == 1.0).astype(int)
ti = {t: i for i, t in enumerate(tw.index)}
rows = np.array([ti[t] for t in m.index])
lp = np.log(paths[rows])
m["ref30"] = np.exp(lp[:, 120 - 30]); m["end270"] = np.exp(lp[:, 120 + 270])
m["x_eff_bps"] = 1e4 * (lp[:, 120 + 270] - lp[:, 120 - 30])            # effective-window Binance move
m["up_binance_eff"] = (m.x_eff_bps >= 0).astype(int)
m["x_nominal_bps"] = 1e4 * (lp[:, 120 + 300] - lp[:, 120])
m["x_pre_m45_m30"] = 1e4 * (lp[:, 120 - 30] - lp[:, 120 - 45])         # move between decision and reference
m["x_m30_0"] = 1e4 * (lp[:, 120] - lp[:, 120 - 30])
m["x_m30_30"] = 1e4 * (lp[:, 120 + 30] - lp[:, 120 - 30])
m["absmove_eff"] = m.x_eff_bps.abs()
m["vol_prev5m_bp"] = 1e4 * np.sqrt(m.rv_prev5m)
m["vol_win_bp"] = 1e4 * np.sqrt(m.rv_win)
ts = pd.to_datetime(m.index, unit="s", utc=True)
et = ts.tz_convert("America/New_York")
m["hour_et"] = et.hour; m["dow"] = et.dayofweek; m["date"] = ts.strftime("%Y-%m-%d"); m["month"] = ts.strftime("%Y-%m")
m["weekend"] = (m.dow >= 5).astype(int)
m["us_session"] = ((m.hour_et >= 9) & (m.hour_et < 16) & (m.dow < 5)).astype(int)
for pre, cols in (("m2", M2_COLS), ("bm", BM_COLS)):
    w = wide(pre, cols, OFFS_PRE + OFFS_POST)
    if w is not None:
        m = m.join(w, how="left")
        print(pre, "joined", w.shape)
bp = BP
if os.path.exists(bp):
    b = pd.read_parquet(bp); b["T"] = ((pd.DatetimeIndex(b.ts).tz_localize(None) - pd.Timestamp("1970-01-01")) // pd.Timedelta("1s") + 60).astype(np.int64)  # forecast instant
    zc = [c for c in b.columns if c.startswith("z_")]
    for off in (-120, -60, 0):
        bb = b[["T", "sigma"] + zc].copy(); bb["T"] = bb["T"] - off; bb = bb.set_index("T")
        bb.columns = [f"bp_{off}_{c}" for c in bb.columns]
        m = m.join(bb, how="left")
    print("bp joined")
m.to_parquet(MASTER_OUT)
print(m.shape, m.index.min(), m.index.max()); print(m[["up", "up_binance_eff", "x_eff_bps", "vol_prev5m_bp"]].describe().round(3))
print("agreement Polymarket outcome vs Binance effective label:", (m.up == m.up_binance_eff).mean().round(4))
