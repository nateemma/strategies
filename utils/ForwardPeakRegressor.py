"""ForwardPeakRegressor — offline-trained RF regressor that predicts the
vol-adjusted forward peak gain over the next ``horizon`` bars.

Pure-logic module (no freqtrade dependencies). Used by:
  * the training script (``Debug/TrainForwardPeakRegressor.py``) to fit and
    persist a per-pair model, and
  * ``BaseNNMTStrategy`` to load a saved model and apply it as a magnitude
    filter on trading predictions.

Validated via ``DebugSignalLearnability --target forward_peak``:
Spearman ρ ≈ +0.14 across pairs (R² ≈ 0). The model can rank bars by
expected forward gain but doesn't calibrate absolute magnitude — so the
strategy should use the score as a relative gate, not an absolute one.
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor


# Feature columns the strategy emits via DataframePopulator. Mirrors
# the set in DebugSignalLearnability so RF baseline numbers transfer.
FEATURE_COLS: List[str] = [
    "dow_sin", "dow_cos", "doy_sin", "doy_cos", "mod_sin", "mod_cos",
    "bb_width", "atr_norm", "rsi_scaled", "mfi_scaled", "gain_norm",
    "log_volume_norm", "macd_norm", "macdhist_norm", "ema_fast_norm",
    "fastk_scaled", "fast_diff", "adx_scaled", "aroonosc_scaled",
    "vwap_ratio", "di_diff_scaled", "sar_ratio", "spread_ma",
    "fisher_ss", "cg_ss",
]

LAGS = (1, 2, 3, 6)

DEFAULT_HORIZON = 8


def generate_target(df: pd.DataFrame, horizon: int = DEFAULT_HORIZON) -> np.ndarray:
    """Vol-adjusted forward peak gain.

    target[t] = max(close[t+1 : t+horizon+1]) / close[t] - 1, divided by
    ATR%[t] so the target is stationary across volatility regimes.

    NaN for the trailing ``horizon`` bars (no forward window).
    """
    close = np.asarray(df["close"].values, dtype=float)
    high = np.asarray(df["high"].values, dtype=float) if "high" in df.columns else close
    low = np.asarray(df["low"].values, dtype=float) if "low" in df.columns else close
    n = len(close)

    prev_close = np.roll(close, 1)
    prev_close[0] = close[0]
    tr = np.maximum(
        high - low,
        np.maximum(np.abs(high - prev_close), np.abs(low - prev_close)),
    )
    atr = pd.Series(tr).ewm(alpha=1.0 / 14, adjust=False).mean().to_numpy()
    # ATR floor at 0.1% (1e-3). Tighter than the previous 1e-6 to prevent
    # extreme bars (near-zero ATR runs) from blowing up the target via
    # division — those caused outliers of ±10k in the predicted score
    # distribution. 0.1% is well below realistic crypto 15m volatility.
    atr_pct = np.maximum(atr / np.maximum(close, 1e-12), 1e-3)

    peak = np.full(n, np.nan)
    for t in range(n - 1):
        end_idx = min(t + horizon + 1, n)
        window = close[t + 1 : end_idx]
        if window.size == 0:
            continue
        peak[t] = (window.max() - close[t]) / close[t]

    return peak / atr_pct


def extract_features(df: pd.DataFrame) -> pd.DataFrame:
    """Build the feature matrix: available base columns + lags."""
    avail = [c for c in FEATURE_COLS if c in df.columns]
    out = {col: df[col] for col in avail}
    for col in avail:
        for lag in LAGS:
            out[f"{col}_lag{lag}"] = df[col].shift(lag)
    return pd.DataFrame(out, index=df.index)


def train(
    df: pd.DataFrame,
    horizon: int = DEFAULT_HORIZON,
    n_estimators: int = 200,
    max_depth: int = 10,
) -> RandomForestRegressor:
    """Fit RF regressor on the full dataframe.

    Stores on the returned model:
      * ``feature_columns_`` — column order for inference alignment
      * ``horizon_`` — forward window used for target generation
      * ``score_percentiles_`` — 101 evenly-spaced quantile cutpoints
        (p0 … p100) of the training-set predictions, used at inference to
        map raw scores to a per-pair percentile rank in [0, 1].
    """
    target = generate_target(df, horizon=horizon)
    features = extract_features(df)

    target_series = pd.Series(target, index=df.index, name="_target")
    aligned = pd.concat([features, target_series], axis=1).dropna()

    if len(aligned) < 200:
        raise ValueError(
            f"Not enough training rows after alignment: {len(aligned)} (need ≥200)"
        )

    X = aligned.drop(columns=["_target"])
    y = aligned["_target"]

    model = RandomForestRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        n_jobs=-1,
        random_state=0,
    )
    model.fit(X, y)
    model.feature_columns_ = list(X.columns)
    model.horizon_ = horizon
    model.n_train_rows_ = len(aligned)

    # Percentile breakpoints from training-set self-predictions. Used by
    # predict_percentile() to normalize raw scores into a per-pair rank,
    # so the strategy's threshold parameter has the same meaning across
    # pairs with very different vol-adjusted score scales.
    train_scores = model.predict(X)
    model.score_percentiles_ = np.percentile(
        train_scores, np.linspace(0.0, 100.0, 101)
    )
    return model


def predict(model: RandomForestRegressor, df: pd.DataFrame) -> np.ndarray:
    """Run inference on a dataframe. Returns one raw score per row.

    Missing feature columns are filled with 0; early bars where lag
    columns are NaN also get 0 (these will not contribute usable signals
    but inference must not crash).
    """
    features = extract_features(df)
    cols = getattr(model, "feature_columns_", list(features.columns))
    for c in cols:
        if c not in features.columns:
            features[c] = 0.0
    features = features[cols]
    X = features.fillna(0.0)
    return model.predict(X)


def predict_percentile(model: RandomForestRegressor, df: pd.DataFrame) -> np.ndarray:
    """Inference returning per-pair percentile ranks in [0, 1].

    Each bar's raw score is mapped to its rank in the training-set
    distribution (stored at fit time in ``model.score_percentiles_``).
    Use this when applying a threshold that should mean the same thing
    across pairs with very different score scales — e.g. threshold=0.5
    rejects roughly half the bars on any pair.
    """
    pct = getattr(model, "score_percentiles_", None)
    if pct is None:
        raise RuntimeError(
            "Model is missing score_percentiles_ — retrain via "
            "TrainForwardPeakRegressor.py to populate it."
        )
    scores = predict(model, df)
    # searchsorted returns insertion index in [0, 101]; divide by 100 and
    # clip to [0, 1] so out-of-range scores collapse to 0.0 or 1.0.
    ranks = np.searchsorted(pct, scores) / 100.0
    return np.clip(ranks, 0.0, 1.0)


def save(model: RandomForestRegressor, path: Path) -> None:
    """Persist via joblib. Creates parent directory if needed."""
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, path)


def load(path: Path) -> Optional[RandomForestRegressor]:
    """Load from disk; returns None if the file doesn't exist."""
    if not path.exists():
        return None
    return joblib.load(path)


def model_path(storage_root: Path, pair: str) -> Path:
    """Canonical save location for a pair's regressor."""
    safe = pair.replace("/", "_")
    return Path(storage_root) / "ForwardPeakRegressor" / f"{safe}.joblib"
