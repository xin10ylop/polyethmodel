"""Download Polymarket 'ETH Up or Down 5m' (series 10683; BTC was 10684) market metadata and trade prints.

  /home/user/data/polymarket/markets.parquet            one row per 5-minute market
  /home/user/data/polymarket/trades/<YYYY-MM-DD>.parquet   trade prints (taker side) per UTC day of window start
"""
import sys, os, time, json, datetime as dt
from concurrent.futures import ThreadPoolExecutor, as_completed
import numpy as np, pandas as pd, requests

OUT = os.environ.get("POLY_OUT", "/home/user/data_eth/polymarket")
os.makedirs(f"{OUT}/trades", exist_ok=True)
GAMMA = "https://gamma-api.polymarket.com"
DATA = "https://data-api.polymarket.com/v2"
S = requests.Session()

def get_json(url, params=None, retries=8):
    last = None
    for k in range(retries):
        try:
            r = S.get(url, params=params, timeout=60)
            if r.status_code == 429:
                time.sleep(float(r.headers.get("Retry-After", 2)) + 0.5); continue
            r.raise_for_status()
            return r.json()
        except Exception as e:
            last = e; time.sleep(min(2 ** k, 30))
    raise RuntimeError(f"{url} {params}: {last}")

# ---------------------------------------------------------------- markets
def list_markets(t_start, t_end):
    """Enumerate by start_date window (gamma caps 100 per page, offset < 10000)."""
    rows = []
    t = t_start
    while t < t_end:
        t2 = min(t + 6 * 3600, t_end)          # 6 h = 72 markets per window
        a = dt.datetime.fromtimestamp(t, dt.UTC).strftime("%Y-%m-%dT%H:%M:%SZ")
        b = dt.datetime.fromtimestamp(t2 - 1, dt.UTC).strftime("%Y-%m-%dT%H:%M:%SZ")
        got = []
        for off in range(0, 1000, 100):
            d = get_json(f"{GAMMA}/events", dict(series_id=int(os.environ.get("POLY_SERIES", "10683")), limit=100, offset=off,
                                                  start_date_min=a, start_date_max=b))
            if not isinstance(d, list):
                raise RuntimeError(str(d)[:200])
            got += d
            if len(d) < 100:
                break
        for e in got:
            if not e.get("slug", "").startswith(os.environ.get("POLY_PREFIX", "eth-updown-5m-")):
                continue
            m = e["markets"][0]
            rows.append(dict(
                slug=e["slug"], window_start=int(e["slug"].rsplit("-", 1)[1]),
                event_id=int(e["id"]), market_id=int(m["id"]), condition_id=m["conditionId"],
                token_up=json.loads(m["clobTokenIds"])[0], token_down=json.loads(m["clobTokenIds"])[1],
                outcome_up_price=float(json.loads(m["outcomePrices"])[0]) if m.get("outcomePrices") else np.nan,
                price_to_beat=(e.get("eventMetadata") or {}).get("priceToBeat", np.nan),
                volume=float(m.get("volumeNum") or 0), closed=bool(m.get("closed")),
                closed_time=m.get("closedTime"), accepting_from=m.get("acceptingOrdersTimestamp"),
                fee_rate=(m.get("feeSchedule") or {}).get("rate", np.nan),
                fee_exp=(m.get("feeSchedule") or {}).get("exponent", np.nan),
                uma_status=m.get("umaResolutionStatus"),
            ))
        print(f"markets {a}: {len(got)}", flush=True)
        t = t2
    df = pd.DataFrame(rows).drop_duplicates("slug").sort_values("window_start").reset_index(drop=True)
    return df

# ---------------------------------------------------------------- trades
def fetch_trades(cond_ids):
    """All taker-side fills for up to 20 condition ids."""
    rows = []
    params = dict(condition=",".join(cond_ids), limit=1000)
    cursor = None
    while True:
        p = dict(params)
        if cursor:
            p["cursor"] = cursor
        d = get_json(f"{DATA}/trades", p)
        for r in d.get("data") or []:
            rows.append((r["condition_id"], int(r["timestamp"]), r["side"], int(r["outcome_index"]),
                         float(r["price"]), float(r["size"]), r.get("transaction_hash", "")[:18]))
        pg = d.get("pagination") or {}
        cursor = pg.get("next_cursor")
        if not pg.get("has_more") or not cursor:
            break
    return rows

def trades_for_day(day, mk):
    out = f"{OUT}/trades/{day}.parquet"
    if os.path.exists(out):
        return f"{day}: cached"
    conds = mk["condition_id"].tolist()
    rows = []
    for i in range(0, len(conds), 10):
        rows += fetch_trades(conds[i:i + 10])
    df = pd.DataFrame(rows, columns=["condition_id", "ts", "side", "outcome_index", "price", "size", "tx"])
    df = df.merge(mk[["condition_id", "window_start"]], on="condition_id", how="left")
    tmp = out + ".tmp"; df.to_parquet(tmp, index=False); os.replace(tmp, out)
    return f"{day}: {len(df)} trades, {mk.shape[0]} markets"

if __name__ == "__main__":
    a = int(dt.datetime.fromisoformat(sys.argv[1] + "T00:00:00+00:00").timestamp())
    b = int(dt.datetime.fromisoformat(sys.argv[2] + "T00:00:00+00:00").timestamp())
    mpath = f"{OUT}/markets.parquet"
    if os.path.exists(mpath):
        mk = pd.read_parquet(mpath)
    else:
        mk = list_markets(a, b)
        mk.to_parquet(mpath, index=False)
    print("markets:", len(mk), mk["window_start"].min(), mk["window_start"].max(), flush=True)
    mk["day"] = pd.to_datetime(mk["window_start"], unit="s", utc=True).dt.strftime("%Y-%m-%d")
    alld = sorted(mk["day"].unique())
    tune = [d for d in alld if "2026-08-20" <= d <= "2026-09-30"]
    rest = [d for d in alld if d not in tune][::-1]
    days = [d for d in tune + rest if not os.path.exists(f"{OUT}/trades/{d}.parquet")]
    workers = int(sys.argv[3]) if len(sys.argv) > 3 else 16
    t0 = time.time()
    # parallelism across condition batches (each batch paginates sequentially)
    for i, d in enumerate(days):
        md = mk[mk["day"] == d]
        conds = md["condition_id"].tolist()
        batches = [conds[j:j + 5] for j in range(0, len(conds), 5)]
        rows = []
        with ThreadPoolExecutor(workers) as ex:
            for r in ex.map(fetch_trades, batches):
                rows += r
        df = pd.DataFrame(rows, columns=["condition_id", "ts", "side", "outcome_index", "price", "size", "tx"])
        df = df.merge(md[["condition_id", "window_start"]], on="condition_id", how="left")
        out = f"{OUT}/trades/{d}.parquet"; tmp = out + ".tmp"; df.to_parquet(tmp, index=False); os.replace(tmp, out)
        print(f"{d}: {len(df)} trades, {len(md)} markets [{i+1}/{len(days)} {time.time()-t0:.0f}s]", flush=True)
    print("done", flush=True)
