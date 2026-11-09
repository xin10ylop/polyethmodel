"""Extra public feeds for the A-improvement study (all public daily archives, no keys):
  * Binance USDT-M futures bookDepth (30-s snapshots of bid/ask notional within 0.2 / 1 / 2 / 3 / 4 / 5 %)
  * Binance USDT-M futures metrics (5-min open interest, long/short ratios, taker buy/sell ratio)
  * Bybit linear ETHUSDT trades  -> 1-s bars (last price, buy / sell base volume, count)
  * OKX ETH-USDT-SWAP trades     -> 1-s bars (contracts of 0.01 BTC -> base volume)
Raw trade files are deleted after reduction.  Usage: fetch_extra_feeds.py START END [--only bybit,okx,binance]"""
import os, sys, io, zipfile, gzip, subprocess, argparse, numpy as np, pandas as pd
from datetime import date, timedelta
OUT = os.environ.get("EXTRA_DIR", "/home/user/data_eth/extra"); TMP = os.environ.get("TMPDIR", "/tmp/claude-0/-home-user/543a293e-08d9-5eff-abf0-cbf9b740975e/scratchpad/dl")
os.makedirs(TMP, exist_ok=True)

PID = os.getpid()
def epoch_s(x): return (pd.to_datetime(x).astype("datetime64[ns]").astype("int64") // 10**9).values

def get(url, dst):
    r = subprocess.run(["curl", "-sS", "-f", "-o", dst, url], capture_output=True, text=True)
    if r.returncode: raise RuntimeError(f"{url}: {r.stderr.strip()[:200]}")

def bars_1s(ts_s, side_buy, size, day0):
    sec = (np.floor(ts_s).astype(np.int64) - day0); ok = (sec >= 0) & (sec < 86400); sec, side_buy, size = sec[ok], side_buy[ok], size[ok]
    buy = np.bincount(sec, weights=np.where(side_buy, size, 0.0), minlength=86400); sell = np.bincount(sec, weights=np.where(side_buy, 0.0, size), minlength=86400)
    cnt = np.bincount(sec, minlength=86400)
    return buy.astype(np.float32), sell.astype(np.float32), cnt.astype(np.int32), sec

def last_price(sec, price):
    close = np.full(86400, np.nan, np.float64); close[sec] = price   # assignment order = file order -> last trade of the second wins
    return close

def bybit(d):
    f = f"{OUT}/bybit1s/{d}.parquet"
    if os.path.exists(f): return
    raw = f"{TMP}/bybit_{PID}_{d}.csv.gz"; get(f"https://public.bybit.com/trading/ETHUSDT/ETHUSDT{d}.csv.gz", raw)
    t = pd.read_csv(raw, usecols=["timestamp", "side", "size", "price"], dtype={"timestamp": np.float64, "size": np.float64, "price": np.float64})
    os.remove(raw); day0 = int(pd.Timestamp(d).timestamp())
    buy, sell, cnt, sec = bars_1s(t.timestamp.values, t.side.values == "Buy", t["size"].values, day0)
    close = last_price(sec, t.price.values[(np.floor(t.timestamp.values).astype(np.int64) - day0 >= 0) & (np.floor(t.timestamp.values).astype(np.int64) - day0 < 86400)])
    pd.DataFrame({"close": close, "buy": buy, "sell": sell, "cnt": cnt}).to_parquet(f)

def okx_file(d):
    """OKX daily files cover 00:00-24:00 UTC+8; cached per process so consecutive UTC days download each file once."""
    raw = f"{TMP}/okx_{PID}_{d}.zip"; dd = d.replace("-", "")
    if not os.path.exists(raw): get(f"https://www.okx.com/cdn/okex/traderecords/trades/daily/{dd}/ETH-USDT-SWAP-trades-{d}.zip", raw)
    with zipfile.ZipFile(raw) as z: return pd.read_csv(z.open(z.namelist()[0]), usecols=["side", "price", "size", "created_time"])

def okx(d):
    f = f"{OUT}/okx1s/{d}.parquet"
    if os.path.exists(f): return
    nxt = (date.fromisoformat(d) + timedelta(days=1)).isoformat()
    t = pd.concat([okx_file(d), okx_file(nxt)], ignore_index=True)
    prev = f"{TMP}/okx_{PID}_{(date.fromisoformat(d) - timedelta(days=1)).isoformat()}.zip"
    for old in (f"{TMP}/okx_{PID}_{d}.zip", prev):
        if os.path.exists(old): os.remove(old)
    day0 = int(pd.Timestamp(d).timestamp()); ts = t.created_time.values / 1000.0
    buy, sell, cnt, sec = bars_1s(ts, t.side.values == "buy", t["size"].values * 0.01, day0)
    ok = (np.floor(ts).astype(np.int64) - day0 >= 0) & (np.floor(ts).astype(np.int64) - day0 < 86400)
    close = last_price(sec, t.price.values[ok])
    pd.DataFrame({"close": close, "buy": buy, "sell": sell, "cnt": cnt}).to_parquet(f)

def binance(d):
    for kind, sub in (("bookDepth", "bookdepth"), ("metrics", "metrics")):
        f = f"{OUT}/{sub}/{d}.parquet"
        if os.path.exists(f): continue
        raw = f"{TMP}/{kind}_{PID}_{d}.zip"; get(f"https://data.binance.vision/data/futures/um/daily/{kind}/ETHUSDT/ETHUSDT-{kind}-{d}.zip", raw)
        with zipfile.ZipFile(raw) as z: t = pd.read_csv(z.open(z.namelist()[0]))
        os.remove(raw)
        if kind == "bookDepth":
            t["ts"] = epoch_s(t.timestamp)
            t = t.pivot_table(index="ts", columns="percentage", values="notional", aggfunc="last"); t.columns = [f"n{c:+.1f}" for c in t.columns]
        else:
            t["ts"] = epoch_s(t.create_time); t = t.set_index("ts").drop(columns=["symbol", "create_time"])
        t.to_parquet(f)

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("start"); ap.add_argument("end"); ap.add_argument("--only", default="binance,bybit,okx"); a = ap.parse_args()
    d0, d1 = date.fromisoformat(a.start), date.fromisoformat(a.end); d = d0
    while d <= d1:
        s = d.isoformat()
        for name in a.only.split(","):
            try: {"bybit": bybit, "okx": okx, "binance": binance}[name](s); print(s, name, "ok", flush=True)
            except Exception as e: print(s, name, "FAIL", str(e)[:200], flush=True)
        d += timedelta(days=1)
