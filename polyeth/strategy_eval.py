"""Strategy evaluation: signal -> side, confidence filter, TP/SL/time-stop grid; tune / test split.

P&L unit: USDC per share bought at pe (0.50). 'ret' = pnl / pe (return on stake).
"""
import os
import numpy as np, pandas as pd, sys, itertools
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "polyeth"))
import polysim

TUNE = ("2026-08-20", "2026-09-30"); TEST = ("2026-10-01", "2026-11-03"); OLD = ("2026-08-01", "2026-08-19")


def load(t_entry=-45, pe=0.5, tps=(0.55, 0.58, 0.60, 0.62, 0.65, 0.70, 0.75, 0.80, 0.90),
         sls=(0.20, 0.25, 0.30, 0.35, 0.40, 0.42, 0.45, 0.47), tstops=(180, 240, 270, 285, 295), slip=0.01):
    master = pd.read_parquet("/home/user/data_eth/master.parquet")
    mk, g = polysim.load_windows()
    sim = polysim.simulate(mk, g, t_entry=t_entry, pe=pe, tps=tps, sls=sls, tstops=tstops, slip=slip)
    keep = np.isin(sim["W"], master.index.values)
    for k in ("pnl", "kind", "texit", "settle", "entry_px", "pre_last", "fill_rel"):
        sim[k] = sim[k][keep]
    sim["W"] = sim["W"][keep]
    master = master.loc[sim["W"]]
    return master, sim


def metrics(pnl, dates, pe=0.5):
    """pnl: per-trade P&L per share (NaN = no trade)."""
    t = np.isfinite(pnl)
    n = int(t.sum())
    if n == 0:
        return dict(n=0)
    p = pnl[t]; d = pd.Series(p).groupby(np.asarray(dates)[t]).sum()
    cum = np.cumsum(p); dd = np.max(np.maximum.accumulate(cum) - cum) if n else 0
    return dict(n=n, mean=p.mean(), ret=p.mean() / pe, win=np.mean(p > 0), total=p.sum(),
                sharpe=(d.mean() / (d.std() + 1e-12)) * np.sqrt(365) if len(d) > 2 else np.nan, maxdd=dd, days=len(d),
                tstat=p.mean() / (p.std() / np.sqrt(n) + 1e-12))


def grid_eval(master, sim, signal, thresholds=(0.0, 0.02, 0.04, 0.06, 0.08, 0.10, 0.12, 0.15), split=None, filt=None,
              min_trades=200):
    """signal: array of P(up) aligned with master. Returns DataFrame over (thr, tp, sl, ts) with metrics."""
    p = np.asarray(signal, float)
    dates = master["date"].values
    m = np.isfinite(p)
    if split is not None:
        m &= (dates >= split[0]) & (dates <= split[1])
    if filt is not None:
        m &= np.asarray(filt, bool)
    side = np.where(p > 0.5, 0, 1); strength = np.abs(p - 0.5)
    idx = np.arange(len(p))
    rows = []
    for thr in thresholds:
        sel = m & (strength >= thr)
        if sel.sum() < min_trades:
            continue
        P = sim["pnl"][idx[sel], side[sel]]            # [n, tp, sl, ts]
        K = sim["kind"][idx[sel], side[sel]]
        d = dates[sel]
        for a, tp in enumerate(sim["tps"]):
            for b, sl in enumerate(sim["sls"]):
                for c, ts in enumerate(sim["tstops"]):
                    r = metrics(P[:, a, b, c], d); r.update(thr=thr, tp=tp, sl=sl, tstop=ts,
                                                           p_tp=np.mean(K[:, a, b, c] == 0), p_sl=np.mean(K[:, a, b, c] == 1))
                    rows.append(r)
        r = metrics(sim["settle"][idx[sel], side[sel]], d); r.update(thr=thr, tp=np.nan, sl=np.nan, tstop="settle", p_tp=np.nan, p_sl=np.nan)
        rows.append(r)
    return pd.DataFrame(rows)


def select_and_test(master, sim, signal, name, filt=None, top=10, min_trades=300, verbose=True):
    tune = grid_eval(master, sim, signal, split=TUNE, filt=filt, min_trades=min_trades)
    test = grid_eval(master, sim, signal, split=TEST, filt=filt, min_trades=50)
    key = ["thr", "tp", "sl", "tstop"]
    tune = tune[tune.n >= min_trades].sort_values("mean", ascending=False)
    merged = tune.head(top).merge(test, on=key, suffixes=("_tune", "_test"), how="left")
    if verbose:
        pd.set_option("display.width", 250)
        print(f"\n=== {name}: top-{top} configs on TUNE {TUNE} and their TEST {TEST} results ===")
        print(merged[key + ["n_tune", "mean_tune", "win_tune", "sharpe_tune", "p_tp_tune", "n_test", "mean_test", "win_test", "sharpe_test", "maxdd_test", "tstat_test"]].round(4).to_string(index=False))
    return tune, test, merged


if __name__ == "__main__":
    master, sim = load()
    print("windows with trades+forecasts:", len(master), master.index.min(), master.index.max())
    sigs = [c for c in sys.argv[1:]] or [c for c in master.columns if c in ("m2_-45_close_p", "m2_-45_close_pup_now", "m2_-45_h60_pup_now", "bm_-45_close_p", "bm_-45_close_pup_now")]
    for s in sigs:
        if s in master.columns and np.isfinite(master[s].values).sum() > 500:
            select_and_test(master, sim, master[s].values, s)
    # random-side baseline (costs only)
    rng = np.random.default_rng(0)
    select_and_test(master, sim, rng.uniform(0, 1, len(master)), "random side")


# ---------------------------------------------------------------- dynamic exits & sizing
def nearest_idx(grid, values):
    grid = np.asarray(grid, float); values = np.asarray(values, float)
    return np.abs(values[:, None] - grid[None, :]).argmin(1)


def dynamic_eval(master, sim, signal, tp_fn, sl_fn, ts_fn, thresholds=(0.0, 0.02, 0.04, 0.06, 0.08, 0.10), split=None, filt=None,
                 min_trades=50, name="dynamic"):
    """tp_fn/sl_fn/ts_fn: callables master -> per-window level arrays (mapped to the nearest grid cell)."""
    p = np.asarray(signal, float); dates = master["date"].values
    m = np.isfinite(p)
    if split is not None:
        m &= (dates >= split[0]) & (dates <= split[1])
    if filt is not None:
        m &= np.asarray(filt, bool)
    side = np.where(p > 0.5, 0, 1); strength = np.abs(p - 0.5); idx = np.arange(len(p))
    a = nearest_idx(sim["tps"], tp_fn(master)); b = nearest_idx(sim["sls"], sl_fn(master)); c = nearest_idx(sim["tstops"], ts_fn(master))
    rows = []
    for thr in thresholds:
        sel = m & (strength >= thr)
        if sel.sum() < min_trades:
            continue
        P = sim["pnl"][idx[sel], side[sel], a[sel], b[sel], c[sel]]
        r = metrics(P, dates[sel]); r.update(thr=thr, rule=name); rows.append(r)
    return pd.DataFrame(rows)


def equity_curve(pnl, dates, stake=1.0, pe=0.5):
    """Daily P&L in USDC per `stake` dollars per trade."""
    t = np.isfinite(pnl)
    d = pd.Series(pnl[t] / pe * stake).groupby(np.asarray(dates)[t]).sum()
    return d.cumsum()
