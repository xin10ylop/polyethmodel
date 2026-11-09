"""Order-book depth imbalance: the third agreement vote for strategy A (research/book_vote_A.py, reports/A_book_vote.csv).

Signal: imb = (bid notional - ask notional) / (bid + ask) within +-LEVEL % of the mid on the Binance USDT-M ETHUSDT perp
book.  A trades only if sign(imb) == sign(p - 0.5) (no dead zone).  Archive source: data.binance.vision
futures/um/daily/bookDepth (30-s snapshots; scripts/fetch_extra_feeds.py -> /home/user/data/extra/bookdepth/<day>.parquet,
columns n-5.0 .. n+5.0 = notional per band, index = snapshot epoch second).  Backtests take the last snapshot at or before
T-47 (decision T-45 minus latency) and ignore snapshots older than MAX_AGE seconds.

Live: BookFeed polls a REST depth endpoint every POLL seconds and keeps the latest imbalance.  Binance's main API is not
reachable from US hosts (HTTP 451); the box must either be outside the US (BOOK_FEED=binance_perp) or fall back to the
spot book mirror (BOOK_FEED=binance_spot, untested against the archive: shadow-log it first) - see README section 9.
"""
import os, glob, time, threading, datetime as dt
import numpy as np, pandas as pd

LEVEL = "1.0"; LAT = 47; MAX_AGE = 120
ARCHIVE = os.environ.get("BOOKDEPTH_DIR", "/home/user/data_eth/extra/bookdepth")
_cache = {}


def load_archive(level=LEVEL):
    """Snapshot series of the imbalance within +-level %: pandas Series indexed by epoch second."""
    if level in _cache: return _cache[level]
    files = sorted(glob.glob(f"{ARCHIVE}/*.parquet"))
    if not files: raise FileNotFoundError(f"no book-depth archive in {ARCHIVE} (scripts/fetch_extra_feeds.py START END --only binance)")
    bd = pd.concat([pd.read_parquet(f, columns=[f"n-{level}", f"n+{level}"]) for f in files]).sort_index()
    bd = bd[~bd.index.duplicated()]
    s = ((bd[f"n-{level}"] - bd[f"n+{level}"]) / (bd[f"n-{level}"] + bd[f"n+{level}"])).astype(float)
    _cache[level] = s; return s


def archive_imbalance(T, level=LEVEL, lat=LAT, max_age=MAX_AGE):
    """Imbalance known at T-lat for an array of nominal window opens T (epoch s); NaN where the last snapshot is older than max_age."""
    s = load_archive(level); idx = s.index.values.astype(np.int64); when = np.asarray(T, np.int64) - lat
    j = np.searchsorted(idx, when, side="right") - 1; ok = j >= 0
    val = np.where(ok, s.values[np.maximum(j, 0)], np.nan); age = np.where(ok, when - idx[np.maximum(j, 0)], np.inf)
    return np.where(age <= max_age, val, np.nan)


def imbalance_from_levels(bids, asks, pct=float(LEVEL)):
    """bids/asks: iterables of (price, qty) strings or floats, best first.  Returns dict(imb, bid_notional, ask_notional, mid,
    span_bid_pct, span_ask_pct): the spans say how far the snapshot reaches; a span below pct means the snapshot does not cover
    the band and the imbalance is NOT comparable with the archive (the caller should log and treat it as missing)."""
    if not bids or not asks: return None
    b = np.array([[float(p), float(q)] for p, q in bids]); a = np.array([[float(p), float(q)] for p, q in asks])
    mid = (b[0, 0] + a[0, 0]) / 2; lo, hi = mid * (1 - pct / 100), mid * (1 + pct / 100)
    nb = float((b[:, 0] * b[:, 1])[b[:, 0] >= lo].sum()); na = float((a[:, 0] * a[:, 1])[a[:, 0] <= hi].sum())
    return dict(imb=(nb - na) / (nb + na) if nb + na > 0 else np.nan, bid_notional=nb, ask_notional=na, mid=mid,
                span_bid_pct=100 * (mid - b[:, 0].min()) / mid, span_ask_pct=100 * (a[:, 0].max() - mid) / mid)


SOURCES = {   # name -> (url, params, parser)  parser(json) -> (bids, asks)
    "binance_perp": ("https://fapi.binance.com/fapi/v1/depth", dict(symbol="ETHUSDT", limit=1000), lambda j: (j["bids"], j["asks"])),
    "binance_spot": ("https://data-api.binance.vision/api/v3/depth", dict(symbol="ETHUSDT", limit=5000), lambda j: (j["bids"], j["asks"])),
    "okx_swap": ("https://www.okx.com/api/v5/market/books-full", dict(instId="BTC-USDT-SWAP", sz=5000), lambda j: ([(x[0], float(x[1]) * 0.01) for x in j["data"][0]["bids"]], [(x[0], float(x[1]) * 0.01) for x in j["data"][0]["asks"]])),
}


class BookFeed:
    """Background poller.  latest(max_age) -> dict(imb, age, span_ok, ts, ...) or None.  Records every snapshot to `record`
    (jsonl path) so the live values can be compared with the archive's 30-s snapshots the next day."""
    def __init__(self, source="binance_perp", poll=5.0, pct=float(LEVEL), record=None, log=print):
        import requests
        self.url, self.params, self.parse = SOURCES[source]; self.source = source; self.poll = poll; self.pct = pct
        self.S = requests.Session(); self.last = None; self.lock = threading.Lock(); self.log = log; self.errors = 0
        self.rec = open(record, "a") if record else None
        threading.Thread(target=self.run, daemon=True).start()

    def run(self):
        import json
        while True:
            t0 = time.time()
            try:
                r = self.S.get(self.url, params=self.params, timeout=8)
                if r.status_code != 200: raise RuntimeError(f"HTTP {r.status_code} {r.text[:80]}")
                bids, asks = self.parse(r.json()); d = imbalance_from_levels(bids, asks, self.pct)
                if d is None: raise RuntimeError("empty book")
                d["ts"] = round(t0, 2); d["span_ok"] = bool(min(d["span_bid_pct"], d["span_ask_pct"]) >= self.pct); d["source"] = self.source
                with self.lock: self.last = d
                if self.rec: self.rec.write(json.dumps({k: (round(v, 6) if isinstance(v, float) else v) for k, v in d.items()}) + "\n"); self.rec.flush()
                if not d["span_ok"] and self.errors % 60 == 0: self.log(f"[book] snapshot covers only {min(d['span_bid_pct'], d['span_ask_pct']):.2f}% of mid (< {self.pct}%): vote treated as missing")
                self.errors = 0 if d["span_ok"] else self.errors + 1
            except Exception as e:
                self.errors += 1
                if self.errors in (1, 10, 100) or self.errors % 1000 == 0: self.log(f"[book] {self.source} error #{self.errors}: {repr(e)[:120]}")
            time.sleep(max(0.5, self.poll - (time.time() - t0)))

    def latest(self, max_age=MAX_AGE):
        with self.lock: d = self.last
        if d is None: return None
        d = dict(d); d["age"] = round(time.time() - d["ts"], 1)
        return d if d["age"] <= max_age and d["span_ok"] else None


def vote(imb, p_up):
    """True when the book agrees with the side implied by p_up; False when it disagrees; None when imb is missing."""
    if imb is None or not np.isfinite(imb) or not np.isfinite(p_up): return None
    return bool(np.sign(imb) == np.sign(p_up - 0.5))


if __name__ == "__main__":   # quick check of the archive path and of one live source
    import sys
    s = load_archive(); print(f"archive: {len(s)} snapshots {dt.datetime.fromtimestamp(int(s.index[0]), dt.UTC):%F} .. {dt.datetime.fromtimestamp(int(s.index[-1]), dt.UTC):%F}, imb mean {s.mean():+.4f} sd {s.std():.4f}")
    src = sys.argv[1] if len(sys.argv) > 1 else None
    if src:
        f = BookFeed(src, poll=2.0); time.sleep(6); print(src, f.latest())
