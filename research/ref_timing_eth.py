"""When is the ETH market's 'price to beat' sampled, relative to the nominal window start T?  Compare Polymarket's priceToBeat
(eventMetadata, Chainlink ETH/USD) with the Binance ETHUSDT spot path (truth paths, T-120 .. T+300): the offset d that minimises
|ptb - spot(T+d)| per window, by month, and the TWAP hypothesis (ptb vs the mean of spot over [T-60, T] etc.).  Same method as the
BTC study (research/ref_timing3.py): for BTC the reference sat ~30 s before T with the TWAP rule, ~0 with the point rule."""
import numpy as np, pandas as pd
pd.set_option("display.width", 200)
tw = pd.read_parquet("/home/user/data_eth/truth/windows.parquet").set_index("T"); paths = np.load("/home/user/data_eth/truth/paths.npy")
mk = pd.read_parquet("/home/user/data_eth/polymarket/markets_2026-06-01_2026-11-09.parquet").set_index("window_start")
m = tw.join(mk[["price_to_beat"]], how="inner"); m = m[np.isfinite(m.price_to_beat)]
rows = np.array([tw.index.get_loc(t) for t in m.index]); P = paths[rows]           # cols 0..420 = T-120 .. T+300
offs = np.arange(-120, 61); cols = offs + 120
err = np.abs(P[:, cols] - m.price_to_beat.values[:, None]) / m.price_to_beat.values[:, None] * 1e4   # bp
best = offs[np.nanargmin(err, axis=1)]
m["best_d"] = best; m["month"] = pd.to_datetime(m.index, unit="s").strftime("%Y-%m"); m["twap"] = (m.index >= pd.Timestamp("2026-08-07", tz="UTC").timestamp()).astype(int)
print("windows with a price to beat:", len(m))
print("\n== best offset d (seconds, ptb ~ spot(T+d)) by month: median / mean / IQR ==")
print(m.groupby("month").best_d.describe()[["count", "mean", "25%", "50%", "75%"]].round(1).to_string())
print("\n== mean |error| in bp for candidate references, by regime ==")
cand = {"spot(T)": P[:, 120], "spot(T-30)": P[:, 90], "spot(T-60)": P[:, 60], "TWAP[T-60,T]": np.nanmean(P[:, 60:121], axis=1), "TWAP[T-90,T-30]": np.nanmean(P[:, 30:91], axis=1),
        "TWAP[T-30,T+30]": np.nanmean(P[:, 90:151], axis=1), "TWAP[T,T+60]": np.nanmean(P[:, 120:181], axis=1)}
out = []
for nm, v in cand.items():
    e = np.abs(v - m.price_to_beat.values) / m.price_to_beat.values * 1e4
    out.append(dict(ref=nm, point_months=np.nanmedian(e[m.twap.values == 0]), twap_months=np.nanmedian(e[m.twap.values == 1])))
print(pd.DataFrame(out).set_index("ref").round(2).to_string())
