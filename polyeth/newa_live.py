"""The purpose-built model for rule A, live (README section 10): LightGBM on the feature engine's row at T-45 (the same row the close head
uses) plus `f_disp15`, trained on the Binance TWAP60(end) vs TWAP60(start) move of every 5-minute window since January 2025
(scripts/dump_feats.py, scripts/train_ourcase_prod.py with BOOK=0).  score > 0 = UP.  The rule trades when |score| is at least the
`tier` threshold stored with the model (top 15 % of |score| in the walk-forward out-of-sample months Aug-Nov 2026).
The book-depth inputs of the research variant are NOT used: the 1-5 % depth bands cannot be rebuilt from a live feed."""
import os, json, numpy as np, lightgbm as lgb
ASSET = "ETH"; MODELS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "models")


def disp15(s_close_last60):
    """f_disp15 = 1e4 * log(close of second T-46 / mean close of seconds T-60 .. T-46): the part of the start TWAP already known at T-45.
    `s_close_last60`: spot closes of seconds T-60 .. T-46 (15 values, oldest first); zeros / NaN are forward-filled as in the archive."""
    c = np.asarray(s_close_last60, np.float64).copy(); bad = ~np.isfinite(c) | (c <= 0)
    if bad.all(): return np.nan
    idx = np.where(~bad, np.arange(len(c)), 0); np.maximum.accumulate(idx, out=idx); c = c[idx]; c[: np.flatnonzero(~bad)[0]] = c[np.flatnonzero(~bad)[0]]
    return float(1e4 * np.log(c[-1] / c.mean()))


class NewA:
    def __init__(self, tier=None, path=None):
        base = path or os.path.join(MODELS, f"ourcase_{ASSET}_d-45_live"); self.meta = json.load(open(base + ".json")); self.booster = lgb.Booster(model_file=base + ".txt")
        self.feats = self.meta["features"]; self.tier = tier or os.environ.get("NEWA_TIER", "top15"); self.thr = float(self.meta["tier_abs_score"][self.tier])

    def score(self, X, s_close_last60):
        """X: the one-row feature frame of Model.forecast(T, -45, 0).  Returns (score, f_disp15)."""
        d = disp15(s_close_last60); row = np.array([[d if f == "f_disp15" else float(X[f].values[0]) for f in self.feats]], dtype=np.float32)
        return float(self.booster.predict(row)[0]), d
