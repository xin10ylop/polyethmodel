"""ETH market depth for our entries: per window, $ printed at 0.49-0.51 (UP-equivalent) before the nominal open, prints before the
open at all, and prints >= 0.85 after the open (A's sell limit).  By month, with the BTC numbers alongside."""
import glob, numpy as np, pandas as pd
pd.set_option("display.width", 200)
def stats(trades_dir, label):
    rows = []
    for f in sorted(glob.glob(f"{trades_dir}/*.parquet")):
        t = pd.read_parquet(f, columns=["ts", "outcome_index", "price", "size", "window_start"])
        if len(t) == 0: continue
        pu = np.where(t.outcome_index.values == 0, t.price.values, 1 - t.price.values); rel = t.ts.values - t.window_start.values; notional = t.price.values * t["size"].values
        pre = rel < 0; mid = pre & (pu >= 0.49) & (pu <= 0.51); hi = (rel >= 0) & (rel < 300) & (np.maximum(pu, 1 - pu) >= 0.85)
        g = pd.DataFrame({"w": t.window_start.values, "pre_n": pre.astype(int), "mid_usd": np.where(mid, notional, 0.0), "hi_usd": np.where(hi, notional, 0.0), "all_usd": notional}).groupby("w").sum()
        rows.append(g)
    g = pd.concat(rows); g["month"] = pd.to_datetime(g.index, unit="s").strftime("%Y-%m")
    out = g.groupby("month").agg(windows=("all_usd", "size"), vol_med=("all_usd", "median"), pre_prints_med=("pre_n", "median"), share_with_preopen_prints=("pre_n", lambda x: (x > 0).mean()),
                                 mid_med=("mid_usd", "median"), share_mid_ge_100=("mid_usd", lambda x: (x >= 100).mean()), share_mid_ge_50=("mid_usd", lambda x: (x >= 50).mean()),
                                 hi_med=("hi_usd", "median"), share_hi_ge_100=("hi_usd", lambda x: (x >= 100).mean()))
    print(f"\n== {label}: per window, by month ==\n" + out.round(2).to_string())
stats("/home/user/data_eth/polymarket/trades", "ETH")
stats("/home/user/data/polymarket/trades", "BTC")
