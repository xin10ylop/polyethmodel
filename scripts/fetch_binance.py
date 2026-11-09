"""Binance archive for ETH: one download per (day), ETHUSDT spot 1-s klines + USD-M perp aggTrades.
Writes the eth3 day files (/home/user/data_eth/days/<day>.npz, via eth3's own compact_* code) and keeps the raw zips in
/home/user/data_eth/raw/{spot1s,umagg}/ so the eth1 / eth2 per-second formats can be produced without a second download.
  python scripts/fetch_binance.py 2026-05-15 2026-11-08 [workers]"""
import sys, os, time, datetime as dt, hashlib
from concurrent.futures import ThreadPoolExecutor, as_completed
import numpy as np, requests
sys.path.insert(0, "/home/user/xin10ylop/eth3"); os.environ.setdefault("BTC_DATA_DIR", "/home/user/data_eth"); os.environ.setdefault("BTC_ASSET", "ETH")
from btcmodel import data as m3data
RAW = "/home/user/data_eth/raw"; DAYS = "/home/user/data_eth/days"; BASE = "https://data.binance.vision/data"; SYM = "ETHUSDT"
for d_ in (f"{RAW}/spot1s", f"{RAW}/umagg", DAYS): os.makedirs(d_, exist_ok=True)
S = requests.Session()
def get(url, retries=6):
    last = None
    for k in range(retries):
        try:
            r = S.get(url, timeout=180)
            if r.status_code == 404: return None
            r.raise_for_status(); blob = r.content
            c = S.get(url + ".CHECKSUM", timeout=60)
            if c.status_code == 200 and c.text.strip().split()[0] != hashlib.sha256(blob).hexdigest(): raise IOError("checksum mismatch")
            return blob
        except Exception as e:
            last = e; time.sleep(2 ** k)
    raise RuntimeError(f"{url}: {last}")
def load_raw(kind, url, d):
    p = f"{RAW}/{kind}/{d}.zip"
    if os.path.exists(p): return open(p, "rb").read()
    blob = get(url)
    if blob is None: return None
    open(p + ".part", "wb").write(blob); os.replace(p + ".part", p); return blob
def do_day(d):
    out = f"{DAYS}/{d}.npz"
    if os.path.exists(out) and os.path.exists(f"{RAW}/spot1s/{d}.zip") and os.path.exists(f"{RAW}/umagg/{d}.zip"): return f"{d}: cached"
    spot = load_raw("spot1s", f"{BASE}/spot/daily/klines/{SYM}/1s/{SYM}-1s-{d}.zip", d)
    if spot is None: return f"{d}: spot missing"
    um = load_raw("umagg", f"{BASE}/futures/um/daily/aggTrades/{SYM}/{SYM}-aggTrades-{d}.zip", d)
    if not os.path.exists(out):
        s = m3data.compact_spot_1s(spot, d); p = m3data.compact_perp_agg(um, d) if um is not None else {}
        np.savez_compressed(out + ".tmp.npz", **s, **p); os.replace(out + ".tmp.npz", out)
    return f"{d}: ok" + ("" if um is not None else " (no perp)")
if __name__ == "__main__":
    a, b = dt.date.fromisoformat(sys.argv[1]), dt.date.fromisoformat(sys.argv[2]); workers = int(sys.argv[3]) if len(sys.argv) > 3 else 4
    days = [a + dt.timedelta(days=i) for i in range((b - a).days + 1)][::-1]; t0 = time.time()
    with ThreadPoolExecutor(workers) as ex:
        futs = {ex.submit(do_day, d): d for d in days}
        for i, f in enumerate(as_completed(futs)):
            try: print(f.result(), f"[{i+1}/{len(days)} {time.time()-t0:.0f}s]", flush=True)
            except Exception as e: print(futs[f], "ERROR", repr(e), flush=True)
    print("done", flush=True)
