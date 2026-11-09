"""Polymarket 5-minute ETH markets + trades for an arbitrary date range, without touching the main
markets.parquet: writes <OUT>/markets_<start>_<end>.parquet and the per-day trade files (same layout).
  python scripts/fetch_polymarket_range.py 2026-07-01 2026-08-01 [workers]
"""
import sys, os, time, datetime as dt
import pandas as pd
from concurrent.futures import ThreadPoolExecutor
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import fetch_polymarket as fp

a = int(dt.datetime.fromisoformat(sys.argv[1] + "T00:00:00+00:00").timestamp()); b = int(dt.datetime.fromisoformat(sys.argv[2] + "T00:00:00+00:00").timestamp())
workers = int(sys.argv[3]) if len(sys.argv) > 3 else 16
mpath = f"{fp.OUT}/markets_{sys.argv[1]}_{sys.argv[2]}.parquet"
if os.path.exists(mpath): mk = pd.read_parquet(mpath)
else: mk = fp.list_markets(a, b); mk.to_parquet(mpath, index=False)
print("markets:", len(mk), mk["window_start"].min(), mk["window_start"].max(), flush=True)
mk["day"] = pd.to_datetime(mk["window_start"], unit="s", utc=True).dt.strftime("%Y-%m-%d")
days = [d for d in sorted(mk["day"].unique()) if not os.path.exists(f"{fp.OUT}/trades/{d}.parquet")]
t0 = time.time()
for i, d in enumerate(days):
    md = mk[mk["day"] == d]; conds = md["condition_id"].tolist()
    batches = [conds[j:j + 5] for j in range(0, len(conds), 5)]; rows = []
    with ThreadPoolExecutor(workers) as ex:
        for r in ex.map(fp.fetch_trades, batches): rows += r
    df = pd.DataFrame(rows, columns=["condition_id", "ts", "side", "outcome_index", "price", "size", "tx"]).merge(md[["condition_id", "window_start"]], on="condition_id", how="left")
    out = f"{fp.OUT}/trades/{d}.parquet"; df.to_parquet(out + ".tmp", index=False); os.replace(out + ".tmp", out)
    print(f"{d}: {len(df)} trades, {len(md)} markets [{i+1}/{len(days)} {time.time()-t0:.0f}s]", flush=True)
print("done", flush=True)
