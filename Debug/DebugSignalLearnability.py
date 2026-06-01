"""
DebugSignalLearnability — assess how learnable each training-signal type
and gain/loss threshold is from the feature set.

For each (signal_type, threshold) combo:
  1. Generates buy/sell labels via Framework.TrainingSignals
  2. Trains a lightweight RandomForest classifier on the same feature set
     the GAN/strategy uses (with short lagged history as a flat-classifier
     proxy for the LSTM's temporal context).
  3. Reports cross-validated MCC, signal frequency, expected forward gain
     per signal, and a composite "expected total profit" score.

A high MCC means the features carry enough information to predict that
label distribution. A low MCC means the label is essentially noise from
the feature point of view — no architecture will help.

Run:
    python user_data/strategies/Debug/DebugSignalLearnability.py \
        --pair ICP_USDT \
        --timeframe 15m \
        --methods 1 16 17 18 19 \
        --thresholds 0.003 0.005 0.007 0.010 0.013 \
        --save-csv learnability_results.csv

This is NOT a freqtrade strategy — it's a standalone analysis tool. Run
it directly with Python from the repo root.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd

# Make project imports work
strat_dir = Path(__file__).parent.parent
sys.path.insert(0, str(strat_dir))

from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor  # noqa: E402
from sklearn.metrics import matthews_corrcoef, r2_score  # noqa: E402
from sklearn.model_selection import TimeSeriesSplit  # noqa: E402
from sklearn.neural_network import MLPClassifier  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402
from scipy.stats import spearmanr  # noqa: E402

from utils.DataframePopulator import DataframePopulator, DatasetType  # noqa: E402
from Framework.TrainingSignals import (  # noqa: E402
    LabelMethod,
    available_methods,
    get_train_buy_signals,
    get_train_sell_signals,
)


# Feature columns the GAN/model consumes (mirrors BaseNNStrategy.include_list).
FEATURE_COLS = [
    "dow_sin", "dow_cos", "doy_sin", "doy_cos", "mod_sin", "mod_cos",
    "bb_width", "atr_norm", "rsi_scaled", "mfi_scaled", "gain_norm",
    "log_volume_norm", "macd_norm", "macdhist_norm", "ema_fast_norm",
    "fastk_scaled", "fast_diff", "adx_scaled", "aroonosc_scaled",
    "vwap_ratio", "di_diff_scaled", "sar_ratio", "spread_ma",
    "fisher_ss", "cg_ss",
]


def add_lagged_features(
    df: pd.DataFrame, columns: List[str], lags=(1, 2, 3, 6)
) -> pd.DataFrame:
    """Add lagged versions of each feature. Gives the flat classifier some
    temporal context as a partial proxy for the LSTM's lookback window —
    a 1-bar classifier would under-state learnability for time-series labels.
    """
    out = {col: df[col] for col in columns if col in df.columns}
    for col in columns:
        if col not in df.columns:
            continue
        for lag in lags:
            out[f"{col}_lag{lag}"] = df[col].shift(lag)
    return pd.DataFrame(out, index=df.index)


def measure_learnability(
    X: pd.DataFrame,
    y: pd.Series,
    n_splits: int = 5,
    hidden_units: int = 64,
) -> float:
    """TimeSeriesSplit-CV mean MCC using a small MLP.

    The earlier RandomForest implementation overstated learnability by ~20-40%
    relative to the production Conv1D-LSTM (e.g. H=24/thr=0.015 sweep MCC was
    0.58 but the actual classifier reached ~0.13-0.20 on real data). Reasons:
    RF with max_depth=8 + class_weight=balanced + CV averaging is materially
    more powerful on imbalanced tabular data than the production NN, which has
    no class_weight, sees a single train/val split, and is regularised by
    early stopping.

    This MLP version closes the gap: single hidden layer with capacity close
    to the production final dense block, no class_weight (production has none
    either — instead we oversample the minority class to mimic the signal-
    augmentation effect), StandardScaler for the gradient optimizer, and
    early stopping on a held-out 15% slice of each fold.
    """
    if y.nunique() < 2:
        return 0.0

    tscv = TimeSeriesSplit(n_splits=n_splits)
    mccs: List[float] = []
    rng = np.random.RandomState(0)
    for train_idx, test_idx in tscv.split(X):
        y_tr = y.iloc[train_idx].values
        y_te = y.iloc[test_idx].values
        if len(np.unique(y_tr)) < 2 or len(np.unique(y_te)) < 2:
            continue

        # Oversample minority to balance classes (sklearn MLP has no
        # class_weight). Matches the imbalance-handling production gets
        # from signal augmentation, without the synth-distribution drift
        # risk that a real GAN run would introduce.
        pos_idx = np.flatnonzero(y_tr == 1)
        neg_idx = np.flatnonzero(y_tr == 0)
        if len(pos_idx) == 0 or len(neg_idx) == 0:
            continue
        if len(pos_idx) < len(neg_idx):
            extra = rng.choice(pos_idx, size=len(neg_idx) - len(pos_idx),
                               replace=True)
        else:
            extra = rng.choice(neg_idx, size=len(pos_idx) - len(neg_idx),
                               replace=True)
        bal_idx = np.concatenate([np.arange(len(y_tr)), extra])
        X_tr_bal = X.iloc[train_idx].values[bal_idx]
        y_tr_bal = y_tr[bal_idx]

        scaler = StandardScaler()
        X_tr_s = scaler.fit_transform(X_tr_bal)
        X_te_s = scaler.transform(X.iloc[test_idx].values)

        clf = MLPClassifier(
            hidden_layer_sizes=(hidden_units,),
            max_iter=30,
            early_stopping=True,
            validation_fraction=0.15,
            n_iter_no_change=5,
            random_state=0,
            learning_rate_init=0.001,
            alpha=0.001,
        )
        clf.fit(X_tr_s, y_tr_bal)
        pred = clf.predict(X_te_s)
        if len(np.unique(pred)) < 2 or len(np.unique(y_te)) < 2:
            mccs.append(0.0)
            continue
        mccs.append(matthews_corrcoef(y_te, pred))
    return float(np.mean(mccs)) if mccs else 0.0


def conditional_forward_gain(
    close: np.ndarray, signal: np.ndarray, horizon: int, side: str
) -> float:
    """Average forward-window peak gain (buy) or trough loss (sell) on
    bars where the signal fires.

    A proxy for "expected reward per signal" assuming the model captures
    the signal AND the strategy exits at the forward-window extremum.
    """
    if signal.sum() == 0:
        return 0.0

    n = len(close)
    fwd = np.full(n, np.nan)
    for t in range(n):
        end = min(n, t + horizon + 1)
        if end <= t + 1:
            continue
        window = close[t + 1 : end]
        fwd[t] = np.max(window) if side == "buy" else np.min(window)

    if side == "buy":
        ev = (fwd - close) / np.maximum(close, 1e-12)
    else:
        ev = (close - fwd) / np.maximum(close, 1e-12)

    mask = signal.astype(bool)
    out = ev[mask]
    return float(np.nanmean(out)) if out.size else 0.0


def _build_params(
    method: LabelMethod,
    threshold: float,
    horizon: int,
    side: str,
    bb_width_threshold: Optional[float] = None,
) -> dict:
    """Per-method param dict. Methods accept different param names; we
    whitelist what each function will actually consume so we don't crash."""
    params: dict = {}

    # Almost every label function accepts horizon.
    params["horizon"] = horizon

    if side == "buy":
        # Buy functions use min_gain (and most also accept min_loss).
        params["min_gain"] = threshold
        params["min_loss"] = threshold
    else:
        params["min_loss"] = threshold
        params["min_gain"] = threshold

    # Method-specific extra knobs. labels_gbb / labels_gbb_sell accept a
    # bb_width_threshold (the regime gate). Only attach it when sweeping
    # method=17, otherwise it would either be ignored or crash dispatchers
    # that don't accept it.
    if bb_width_threshold is not None and method == LabelMethod.gbb:
        params["bb_width_threshold"] = bb_width_threshold

    # Indicator-vote families: keep the min_indicators_* knobs at defaults.
    # The dispatch in get_train_sell_signals filters unknown keys for sell
    # variants, but the buy dispatch passes everything through, so we keep
    # the param set minimal.
    return params


def _aug_risk(n_signals: int, n_total: int) -> str:
    """Predict whether DDPM augmentation will collapse at this signal density.

    Per memory (feedback_ddpm_collapses_at_sparse_classes.md): when per-pair
    real Buy/Sell counts drop below ~3K, DDPM produces drifted synth and the
    classifier collapses. In a typical 55K-bar training window that is a
    signal frequency floor of ~5.5%. Sweep windows are usually larger than
    the training window, so the cutoff applies as a frequency, not a raw
    count.
    """
    if n_total <= 0:
        return "n/a"
    freq = n_signals / n_total
    if freq < 0.055:
        return "HIGH"
    if freq < 0.09:
        return "MEDIUM"
    return "LOW"


def evaluate_combo(
    df: pd.DataFrame,
    features: pd.DataFrame,
    method: LabelMethod,
    threshold: float,
    horizon: int,
    side: str,
    bb_width_threshold: Optional[float] = None,
) -> dict:
    """One row of the result table: (method, side, threshold) → metrics."""
    params = _build_params(method, threshold, horizon, side, bb_width_threshold)
    base_row = {
        "method": method.name,
        "method_id": int(method.value),
        "side": side,
        "threshold": threshold,
        "bb_width_threshold": (
            bb_width_threshold if bb_width_threshold is not None else float("nan")
        ),
    }
    try:
        if side == "buy":
            y_raw = get_train_buy_signals(df, method=method, params=params)
        else:
            y_raw = get_train_sell_signals(df, method=method, params=params)
    except Exception as exc:
        return {
            **base_row,
            "n_signals": 0,
            "n_total": 0,
            "mcc": np.nan,
            "ev_per_signal_pct": np.nan,
            "score": 0.0,
            "aug_risk": "n/a",
            "error": f"label-gen: {type(exc).__name__}: {exc}",
        }

    y = (y_raw.astype(float) > 0).astype(int)

    aligned = pd.concat([features, y.rename("y")], axis=1).dropna()
    n_total = int(len(aligned))
    n_signals = int(aligned["y"].sum()) if n_total else 0

    if n_total < 200 or n_signals < 20:
        return {
            **base_row,
            "n_signals": n_signals,
            "n_total": n_total,
            "mcc": np.nan,
            "ev_per_signal_pct": np.nan,
            "score": 0.0,
            "aug_risk": _aug_risk(n_signals, n_total),
            "error": "too few signals or rows",
        }

    X = aligned.drop(columns=["y"])
    yy = aligned["y"]
    mcc = measure_learnability(X, yy)

    close = np.asarray(df["close"].values, dtype=float)
    ev = conditional_forward_gain(close, y.values, horizon, side)

    score = max(mcc, 0.0) * n_signals * ev

    return {
        **base_row,
        "n_signals": n_signals,
        "n_total": n_total,
        "mcc": mcc,
        "ev_per_signal_pct": ev * 100.0,
        "score": score,
        "aug_risk": _aug_risk(n_signals, n_total),
        "error": "",
    }


def generate_profit_labels(
    df: pd.DataFrame,
    threshold: float,
    horizon: int = 24,
    atr_scale: float = 0.5,
) -> np.ndarray:
    """Generate triple-barrier profit labels (3-class: LOSS=0, NEUTRAL=1, PROFIT=2).

    Mirrors BaseNNMTStrategy.get_profit_target. For each bar t, scan the next
    ``horizon`` bars and label by which barrier the close hits FIRST:
      - PROFIT if forward gain reaches +threshold before forward loss reaches -threshold
      - LOSS   if forward loss reaches -threshold first
      - NEUTRAL if neither barrier is hit anywhere in the window
    """
    close = np.asarray(df["close"].values, dtype=float)
    high = np.asarray(df["high"].values, dtype=float) if "high" in df.columns else close
    low = np.asarray(df["low"].values, dtype=float) if "low" in df.columns else close
    n = len(close)

    # Compute ATR-pct for volatility-adjusted thresholds (same as get_profit_target).
    prev_close = np.roll(close, 1)
    prev_close[0] = close[0]
    tr1 = high - low
    tr2 = np.abs(high - prev_close)
    tr3 = np.abs(low - prev_close)
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).ewm(alpha=1.0 / 14, adjust=False).mean().to_numpy()
    atr_pct = atr / np.maximum(close, 1e-12)

    classes = np.ones(n, dtype=int)  # default NEUTRAL=1
    for t in range(n - 1):
        end_idx = min(t + horizon + 1, n)
        window = close[t + 1 : end_idx]
        if window.size == 0:
            continue

        entry = close[t]
        gains = (window - entry) / entry

        vol_adj = atr_pct[t] * atr_scale
        pt_eff = max(threshold, vol_adj)
        sl_eff = max(threshold, vol_adj)

        profit_hits = np.flatnonzero(gains >= pt_eff)
        loss_hits = np.flatnonzero(gains <= -sl_eff)
        first_profit = profit_hits[0] if profit_hits.size else np.iinfo(np.int64).max
        first_loss = loss_hits[0] if loss_hits.size else np.iinfo(np.int64).max

        if first_profit == first_loss:
            classes[t] = 1  # NEUTRAL
        elif first_profit < first_loss:
            classes[t] = 2  # PROFIT
        else:
            classes[t] = 0  # LOSS

    return classes


def measure_learnability_multiclass(
    X: pd.DataFrame, y: pd.Series, n_splits: int = 5,
) -> tuple[float, float]:
    """Train multiclass RF on profit labels (3-class). Returns (overall_mcc, binary_mcc_profit_class).

    ``overall_mcc`` is the 3-class Matthews correlation coefficient.
    ``binary_mcc_profit_class`` is the MCC for (profit==PROFIT) vs (else) — the
    relevant metric for using the profit head as a magnitude filter, since the
    filter only activates on PROFIT predictions.
    """
    if y.nunique() < 2:
        return 0.0, 0.0
    tscv = TimeSeriesSplit(n_splits=n_splits)
    mccs_overall: list[float] = []
    mccs_binary: list[float] = []
    for train_idx, test_idx in tscv.split(X):
        y_tr = y.iloc[train_idx]
        y_te = y.iloc[test_idx]
        if y_tr.nunique() < 2 or y_te.nunique() < 2:
            continue
        clf = RandomForestClassifier(
            n_estimators=100, max_depth=8, n_jobs=-1, random_state=0,
            class_weight="balanced",
        )
        clf.fit(X.iloc[train_idx], y_tr)
        pred = clf.predict(X.iloc[test_idx])
        if len(np.unique(pred)) >= 2 and len(np.unique(y_te)) >= 2:
            mccs_overall.append(matthews_corrcoef(y_te, pred))
            mccs_binary.append(matthews_corrcoef(
                (y_te == 2).astype(int),
                (pred == 2).astype(int),
            ))
    overall = float(np.mean(mccs_overall)) if mccs_overall else 0.0
    binary = float(np.mean(mccs_binary)) if mccs_binary else 0.0
    return overall, binary


def evaluate_profit_combo(
    df: pd.DataFrame, features: pd.DataFrame, threshold: float, horizon: int,
    atr_scale: float,
) -> dict:
    """One row of the profit-sweep result table at a given threshold."""
    try:
        labels = generate_profit_labels(df, threshold, horizon=horizon, atr_scale=atr_scale)
    except Exception as exc:
        return {
            "threshold": threshold,
            "n_total": 0,
            "n_loss": 0, "n_neutral": 0, "n_profit": 0,
            "overall_mcc": np.nan, "binary_mcc_profit": np.nan,
            "error": f"label-gen: {type(exc).__name__}: {exc}",
        }

    y_series = pd.Series(labels, index=df.index, name="y")
    aligned = pd.concat([features, y_series], axis=1).dropna()
    n_total = int(len(aligned))
    n_loss = int((aligned["y"] == 0).sum())
    n_neutral = int((aligned["y"] == 1).sum())
    n_profit = int((aligned["y"] == 2).sum())

    if n_total < 200 or min(n_loss, n_neutral, n_profit) < 20:
        return {
            "threshold": threshold,
            "n_total": n_total, "n_loss": n_loss,
            "n_neutral": n_neutral, "n_profit": n_profit,
            "overall_mcc": np.nan, "binary_mcc_profit": np.nan,
            "error": "too few samples in one of the three classes",
        }

    X = aligned.drop(columns=["y"])
    yy = aligned["y"]
    overall_mcc, binary_mcc = measure_learnability_multiclass(X, yy)

    return {
        "threshold": threshold,
        "n_total": n_total, "n_loss": n_loss,
        "n_neutral": n_neutral, "n_profit": n_profit,
        "overall_mcc": overall_mcc,
        "binary_mcc_profit": binary_mcc,
        "error": "",
    }


def generate_forward_gain(
    df: pd.DataFrame, horizon: int = 24,
) -> tuple[np.ndarray, np.ndarray]:
    """End-of-window forward return (path-independent, directional).

    Returns (gain, gain_vol_adj) where:
      - gain[t] = (close[t + horizon] - close[t]) / close[t]
      - gain_vol_adj[t] = gain[t] / atr_pct[t]

    Smoother than peak gain (no extremum chasing) and symmetric around 0.
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
    atr_pct = np.maximum(atr / np.maximum(close, 1e-12), 1e-6)

    gain = np.full(n, np.nan)
    end_idx = np.minimum(np.arange(n) + horizon, n - 1)
    gain[: n - horizon] = (close[end_idx[: n - horizon]] - close[: n - horizon]) / close[: n - horizon]
    gain_vol_adj = gain / atr_pct
    return gain, gain_vol_adj


def generate_forward_peak_gain(
    df: pd.DataFrame, horizon: int = 24,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute path-independent forward peak gain over the next ``horizon`` bars.

    Returns (peak_gain, peak_gain_vol_adj) where:
      - peak_gain[t] = (max(close[t+1:t+h+1]) - close[t]) / close[t]
      - peak_gain_vol_adj[t] = peak_gain[t] / atr_pct[t]

    Unlike triple-barrier labels, this depends only on the forward window's
    extremum — same peak ⇒ same label regardless of the path taken.
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
    atr_pct = np.maximum(atr / np.maximum(close, 1e-12), 1e-6)

    peak = np.full(n, np.nan)
    for t in range(n - 1):
        end_idx = min(t + horizon + 1, n)
        window = close[t + 1 : end_idx]
        if window.size == 0:
            continue
        peak[t] = (window.max() - close[t]) / close[t]

    peak_vol_adj = peak / atr_pct
    return peak, peak_vol_adj


def measure_learnability_regression(
    X: pd.DataFrame, y: pd.Series, n_splits: int = 5,
) -> tuple[float, float]:
    """TimeSeriesSplit-CV mean R² and Spearman ρ for a regression target.

    Spearman is more relevant for filter use — we care whether the model can
    rank bars by expected gain, not whether it nails the absolute magnitude.
    """
    tscv = TimeSeriesSplit(n_splits=n_splits)
    r2s: list[float] = []
    rhos: list[float] = []
    for train_idx, test_idx in tscv.split(X):
        y_tr = y.iloc[train_idx]
        y_te = y.iloc[test_idx]
        if y_tr.std() == 0 or y_te.std() == 0:
            continue
        reg = RandomForestRegressor(
            n_estimators=100, max_depth=8, n_jobs=-1, random_state=0,
        )
        reg.fit(X.iloc[train_idx], y_tr)
        pred = reg.predict(X.iloc[test_idx])
        r2s.append(r2_score(y_te, pred))
        rho, _ = spearmanr(y_te, pred)
        if not np.isnan(rho):
            rhos.append(rho)
    return (
        float(np.mean(r2s)) if r2s else 0.0,
        float(np.mean(rhos)) if rhos else 0.0,
    )


def evaluate_forward_peak_combo(
    df: pd.DataFrame, features: pd.DataFrame, threshold: float,
    peak: np.ndarray,
) -> dict:
    """One row of the forward-peak-gain sweep at a given threshold.

    Binary label: y = 1 if forward peak gain ≥ threshold, else 0. Trains a
    binary classifier and reports MCC — directly comparable to
    ``binary_mcc_profit`` in the triple-barrier sweep at the same threshold.
    """
    y_arr = (peak >= threshold).astype(int)
    y_arr[np.isnan(peak)] = 0
    y_series = pd.Series(y_arr, index=df.index, name="y")
    aligned = pd.concat([features, y_series], axis=1).dropna()
    n_total = int(len(aligned))
    n_signals = int(aligned["y"].sum())

    if n_total < 200 or n_signals < 20 or (n_total - n_signals) < 20:
        return {
            "threshold": threshold, "n_total": n_total, "n_signals": n_signals,
            "binary_mcc": np.nan, "error": "too few samples in one class",
        }

    X = aligned.drop(columns=["y"])
    yy = aligned["y"]
    mcc = measure_learnability(X, yy)

    return {
        "threshold": threshold, "n_total": n_total, "n_signals": n_signals,
        "binary_mcc": mcc, "error": "",
    }


def load_pair_data(data_dir: Path, pair: str, timeframe: str) -> pd.DataFrame:
    """Load OHLCV from freqtrade's feather format."""
    data_path = data_dir / f"{pair}-{timeframe}.feather"
    if not data_path.exists():
        sys.exit(f"Data file not found: {data_path}")
    df = pd.read_feather(data_path)
    if "date" in df.columns:
        df = df.set_index("date")
    return df


def main() -> None:
    ap = argparse.ArgumentParser(description="Assess training-signal learnability.")
    ap.add_argument("--pair", default="ICP_USDT")
    ap.add_argument(
        "--data-dir",
        default=str(strat_dir.parent / "data" / "binanceus"),
        help="Directory containing <pair>-<timeframe>.feather files",
    )
    ap.add_argument("--timeframe", default="15m")
    ap.add_argument("--horizon", type=int, default=24, help="Forward window in bars")
    ap.add_argument(
        "--methods",
        nargs="+",
        type=int,
        default=[1, 15, 16, 17, 18, 19],
        help="Signal method IDs (see Framework.TrainingSignals.LabelMethod)",
    )
    ap.add_argument(
        "--thresholds",
        nargs="+",
        type=float,
        default=[0.003, 0.005, 0.007, 0.010, 0.013],
        help="min_gain / min_loss thresholds to sweep",
    )
    ap.add_argument(
        "--bb-width-thresholds",
        nargs="+",
        type=float,
        default=None,
        help="bb_width_threshold values to sweep. Only honoured for method 17 "
             "(labels_gbb). When omitted, the function default is used and the "
             "sweep dimension collapses to a single value.",
    )
    ap.add_argument("--side", choices=["buy", "sell", "both"], default="buy")
    ap.add_argument(
        "--max-bars", type=int, default=0,
        help="If >0, use only the most recent N bars (speeds up sweeps).",
    )
    ap.add_argument("--save-csv", default=None, help="Optional path to write results")
    ap.add_argument(
        "--list-methods", action="store_true",
        help="Print available method IDs and exit",
    )
    ap.add_argument(
        "--target",
        choices=["trading", "profit", "forward_peak", "forward_gain"],
        default="trading",
        help="What labels to assess. 'trading' = TrainingSignals method labels "
             "(method × threshold sweep); 'profit' = BaseNNMTStrategy triple-barrier "
             "profit labels (threshold sweep, --methods ignored); 'forward_peak' = "
             "path-independent forward peak gain — binary MCC sweep across thresholds "
             "plus continuous regression R² / Spearman ρ for vol-adjusted target; "
             "'forward_gain' = same as forward_peak but uses end-of-window return "
             "instead of the window peak (smoother, directional).",
    )
    ap.add_argument(
        "--atr-scale", type=float, default=0.5,
        help="ATR-scale for profit-label volatility adjustment (matches "
             "BaseNNMTStrategy.PROFIT_ATR_SCALE). Only used when --target=profit.",
    )
    args = ap.parse_args()

    if args.list_methods:
        print("Available signal methods:")
        for mid, name in available_methods().items():
            print(f"  {mid:2d}  {name}")
        return

    data_dir = Path(args.data_dir)

    print(f"Loading {args.pair} {args.timeframe} from {data_dir}…")
    df = load_pair_data(data_dir, args.pair, args.timeframe)
    if args.max_bars > 0 and len(df) > args.max_bars:
        df = df.iloc[-args.max_bars:].copy()
    print(f"  rows: {len(df)}   {df.index[0]} → {df.index[-1]}")

    print("Computing indicators (MINIMAL set)…")
    pop = DataframePopulator()
    df = pop.add_indicators(df, dataset_type=DatasetType.MINIMAL)

    avail = [c for c in FEATURE_COLS if c in df.columns]
    missing = [c for c in FEATURE_COLS if c not in df.columns]
    if missing:
        print(f"  WARNING: missing columns (skipping): {missing}")
    features = add_lagged_features(df, avail)
    print(f"  features: {len(features.columns)} cols  ({len(avail)} base + lags)")

    if args.target in ("forward_peak", "forward_gain"):
        # Path-independent forward target. ``forward_peak`` uses the window
        # maximum (captures opportunity); ``forward_gain`` uses the window
        # endpoint (captures direction). Both run the same binary-MCC sweep +
        # continuous regression on the vol-adjusted variant.
        if args.target == "forward_peak":
            print(f"\nComputing forward peak gain (horizon={args.horizon})…")
            peak, peak_vol_adj = generate_forward_peak_gain(df, args.horizon)
            label_name = "forward peak gain"
        else:
            print(f"\nComputing end-of-window forward gain (horizon={args.horizon})…")
            peak, peak_vol_adj = generate_forward_gain(df, args.horizon)
            label_name = "forward gain (end-of-window)"

        # Regression on continuous vol-adjusted target — one row, all data.
        peak_series = pd.Series(peak_vol_adj, index=df.index, name="y")
        aligned_reg = pd.concat([features, peak_series], axis=1).dropna()
        r2 = rho = float("nan")
        if len(aligned_reg) >= 200:
            X_reg = aligned_reg.drop(columns=["y"])
            y_reg = aligned_reg["y"]
            r2, rho = measure_learnability_regression(X_reg, y_reg)
        print(f"  Continuous regression on vol-adjusted {label_name} "
              f"(N={len(aligned_reg)}):  R²={r2:+.4f}   Spearman ρ={rho:+.4f}")

        print(f"\nSweeping {label_name.upper()} thresholds "
              f"({len(args.thresholds)} thresholds, horizon={args.horizon})…\n")
        print(f"  {'thresh':>7s}  {'N_pos':>8s}  {'N_total':>8s}  {'pos_pct':>7s}  "
              f"{'binary_MCC':>10s}")
        print(f"  {'-'*7}  {'-'*8}  {'-'*8}  {'-'*7}  {'-'*10}")

        rows = []
        for thresh in args.thresholds:
            row = evaluate_forward_peak_combo(df, features, thresh, peak)
            rows.append(row)
            bm = (f"{row['binary_mcc']:+.3f}"
                  if not np.isnan(row.get("binary_mcc", np.nan)) else "   n/a")
            pos_pct = (
                100.0 * row["n_signals"] / row["n_total"]
                if row["n_total"] else 0.0
            )
            print(
                f"  {row['threshold']:7.4f}  {row['n_signals']:8d}  "
                f"{row['n_total']:8d}  {pos_pct:6.1f}%  {bm:>10s}"
            )
    elif args.target == "profit":
        # Profit-label sweep: BaseNNMTStrategy.get_profit_target labels (3-class)
        # across thresholds. ``--methods`` is ignored; we generate profit labels
        # directly via triple-barrier logic.
        print(f"\nSweeping PROFIT-label thresholds "
              f"({len(args.thresholds)} thresholds, horizon={args.horizon}, "
              f"atr_scale={args.atr_scale})…\n")
        print(f"  {'thresh':>7s}  {'N_loss':>8s}  {'N_neut':>8s}  {'N_prof':>8s}  "
              f"{'overall_MCC':>11s}  {'binary_MCC(PROFIT)':>18s}")
        print(f"  {'-'*7}  {'-'*8}  {'-'*8}  {'-'*8}  {'-'*11}  {'-'*18}")

        rows = []
        for thresh in args.thresholds:
            row = evaluate_profit_combo(
                df, features, thresh, args.horizon, args.atr_scale
            )
            rows.append(row)
            oma = (f"{row['overall_mcc']:+.3f}"
                   if not np.isnan(row.get("overall_mcc", np.nan)) else "   n/a")
            bma = (f"{row['binary_mcc_profit']:+.3f}"
                   if not np.isnan(row.get("binary_mcc_profit", np.nan)) else "   n/a")
            print(
                f"  {row['threshold']:7.4f}  {row['n_loss']:8d}  "
                f"{row['n_neutral']:8d}  {row['n_profit']:8d}  "
                f"{oma:>11s}  {bma:>18s}"
            )
    else:
        sides = ["buy", "sell"] if args.side == "both" else [args.side]
        bb_widths = args.bb_width_thresholds if args.bb_width_thresholds else [None]

        n_combos = (
            len(args.methods) * len(args.thresholds) * len(sides) * len(bb_widths)
        )
        if bb_widths != [None]:
            print(f"\nSweeping methods × thresholds × bb_widths × sides "
                  f"({len(args.methods)} × {len(args.thresholds)} × "
                  f"{len(bb_widths)} × {len(sides)} = {n_combos} combos)…\n")
            header = (
                f"  {'method':18s}  {'side':4s}  {'thresh':>7s}  {'bb_w':>6s}  "
                f"{'N_sig':>6s}  {'aug_risk':>8s}  {'MCC':>7s}  "
                f"{'EV/sig%':>8s}  {'score':>10s}"
            )
            sep = (
                f"  {'-'*18}  {'-'*4}  {'-'*7}  {'-'*6}  {'-'*6}  {'-'*8}  "
                f"{'-'*7}  {'-'*8}  {'-'*10}"
            )
        else:
            print(f"\nSweeping methods × thresholds × sides "
                  f"({len(args.methods)} × {len(args.thresholds)} × "
                  f"{len(sides)} = {n_combos} combos)…\n")
            header = (
                f"  {'method':18s}  {'side':4s}  {'thresh':>7s}  "
                f"{'N_sig':>6s}  {'aug_risk':>8s}  {'MCC':>7s}  "
                f"{'EV/sig%':>8s}  {'score':>10s}"
            )
            sep = (
                f"  {'-'*18}  {'-'*4}  {'-'*7}  {'-'*6}  {'-'*8}  "
                f"{'-'*7}  {'-'*8}  {'-'*10}"
            )
        print(header)
        print(sep)

        rows = []
        for mid in args.methods:
            try:
                method = LabelMethod(mid)
            except ValueError:
                print(f"  skip: unknown method id {mid}")
                continue
            for side in sides:
                for thresh in args.thresholds:
                    for bb_w in bb_widths:
                        row = evaluate_combo(
                            df, features, method, thresh, args.horizon, side, bb_w,
                        )
                        rows.append(row)
                        mcc_str = (
                            f"{row['mcc']:+.3f}" if not np.isnan(row["mcc"]) else "   n/a"
                        )
                        ev_str = (
                            f"{row['ev_per_signal_pct']:+6.2f}"
                            if not np.isnan(row["ev_per_signal_pct"]) else "   n/a"
                        )
                        if bb_widths != [None]:
                            bbw_str = (
                                f"{bb_w:6.4f}" if bb_w is not None else "  def "
                            )
                            print(
                                f"  {row['method']:18s}  {row['side']:4s}  "
                                f"{row['threshold']:7.4f}  {bbw_str}  "
                                f"{row['n_signals']:6d}  "
                                f"{row['aug_risk']:>8s}  "
                                f"{mcc_str:>7s}  {ev_str:>8s}  "
                                f"{row['score']:10.2f}"
                            )
                        else:
                            print(
                                f"  {row['method']:18s}  {row['side']:4s}  "
                                f"{row['threshold']:7.4f}  "
                                f"{row['n_signals']:6d}  "
                                f"{row['aug_risk']:>8s}  "
                                f"{mcc_str:>7s}  {ev_str:>8s}  "
                                f"{row['score']:10.2f}"
                            )

    table = pd.DataFrame(rows)
    if args.save_csv:
        out = Path(args.save_csv)
        table.to_csv(out, index=False)
        print(f"\nSaved CSV: {out}")

    if args.target in ("profit", "forward_peak"):
        # No "best by score" — just print the table as-is. User picks the
        # threshold that keeps the relevant MCC above their bar.
        return

    # Summary per (method, side): pick the threshold with the best score.
    print("\n=== Best threshold per (method, side) by score ===")
    valid = table[table["mcc"].notna()].copy()
    if not valid.empty:
        best = (
            valid.sort_values("score", ascending=False)
            .groupby(["method", "side"])
            .head(1)
            .sort_values("score", ascending=False)
        )
        cols = ["method", "side", "threshold", "n_signals", "mcc",
                "ev_per_signal_pct", "score"]
        if "bb_width_threshold" in valid.columns and valid["bb_width_threshold"].notna().any():
            cols.insert(3, "bb_width_threshold")
        print(best[cols].to_string(index=False))
    else:
        print("  (no rows with valid MCC — likely too few signals)")


if __name__ == "__main__":
    main()
