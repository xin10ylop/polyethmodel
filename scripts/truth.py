"""Per-window ground truth from Binance spot 1s bars (eth3 day files):
P0 (open of window), close, running path, plus a per-second close path from T-120 .. T+300.
Output: /home/user/data/truth/windows.parquet, /home/user/data/truth/paths.npy (float32 [n, 421]).
"""
import os, sys, glob, datetime as dt
import numpy as np, pandas as pd
DAYS = os.environ.get("DAYS_DIR", "/home/user/data_eth/days"); OUT = os.environ.get("TRUTH_DIR", "/home/user/data_eth/truth"); os.makedirs(OUT, exist_ok=True)
PRE, POST = 120, 300

def main(start, end):
    d0, d1 = dt.date.fromisoformat(start), dt.date.fromisoformat(end)
    days = [d0 + dt.timedelta(days=i) for i in range((d1 - d0).days + 1)]
    # continuous arrays over [d0-1 day, d1+1 day]
    allc, allo, allh, alll, allpc = [], [], [], [], []
    for d in [d0 - dt.timedelta(days=1)] + days + [d1 + dt.timedelta(days=1)]:
        f = f"{DAYS}/{d}.npz"
        if os.path.exists(f):
            z = np.load(f)
            allc.append(z["s_close"]); allo.append(z["s_open"]); allh.append(z["s_high"]); alll.append(z["s_low"])
            allpc.append(z["p_close"] if "p_close" in z.files else np.full(86400, np.nan))
        else:
            for L in (allc, allo, allh, alll, allpc):
                L.append(np.full(86400, np.nan))
    c = np.concatenate(allc); o = np.concatenate(allo); h = np.concatenate(allh); l = np.concatenate(alll)
    pc = np.concatenate(allpc)
    def ffill(a):
        m = np.isnan(a); idx = np.where(~m, np.arange(len(a)), 0); np.maximum.accumulate(idx, out=idx); return a[idx]
    c = ffill(c); pc = ffill(pc)
    t0 = int(dt.datetime(d0.year, d0.month, d0.day, tzinfo=dt.UTC).timestamp()) - 86400
    Tw = np.arange(86400, 86400 + len(days) * 86400, 300)
    rows = []
    paths = np.full((len(Tw), PRE + POST + 1), np.nan, np.float32)
    ppaths = np.full((len(Tw), PRE + POST + 1), np.nan, np.float32)
    for i, T in enumerate(Tw):
        p0 = o[T]
        if not np.isfinite(p0):
            p0 = c[T - 1]
        win_c = c[T:T + 300]; win_h = h[T:T + 300]; win_l = l[T:T + 300]
        rows.append(dict(T=t0 + T, p0_binance=p0, close_binance=c[T + 299], prev_close=c[T - 1],
                         hi=np.nanmax(win_h), lo=np.nanmin(win_l), perp_prev=pc[T - 1], perp_close=pc[T + 299],
                         rv_prev5m=float(np.nansum(np.diff(np.log(c[T - 300:T])) ** 2)),
                         rv_win=float(np.nansum(np.diff(np.log(win_c)) ** 2))))
        paths[i] = c[T - PRE:T + POST + 1]
        ppaths[i] = pc[T - PRE:T + POST + 1]
    df = pd.DataFrame(rows)
    df["up_binance"] = (df["close_binance"] >= df["p0_binance"]).astype(int)
    df.to_parquet(f"{OUT}/windows.parquet", index=False)
    np.save(f"{OUT}/paths.npy", paths); np.save(f"{OUT}/perp_paths.npy", ppaths)
    print(df.shape, df["up_binance"].mean())

if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
