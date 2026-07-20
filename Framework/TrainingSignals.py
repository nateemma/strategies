"""
TrainingSignals - Future-aware labeling utilities for generating buy/sell training signals.

All methods assume access to full future data (OK for training label creation).
Returns pandas Series of 0/1 labels aligned to the input dataframe length.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Dict, Any, Tuple, List, Optional
from enum import IntEnum
from scipy.signal import find_peaks

try:
    import pywt

    HAS_PYWT = True
except ImportError:
    HAS_PYWT = False

DEFAULT_HORIZON = 36
DEFAULT_MIN_GAIN = 0.007
DEFAULT_MIN_LOSS = 0.009
DEFAULT_MAX_DRAWDOWN = 0.02
# ------------------------------
# Helpers
# ------------------------------


def _safe_close(df: pd.DataFrame) -> np.ndarray:
    close = np.asarray(df["close"], dtype=float)
    return np.maximum(close, 1e-12)


def _atr(df: pd.DataFrame, period: int = 14) -> np.ndarray:
    # Lightweight ATR approximation to avoid ta-lib dependency here
    high = np.asarray(df["high"], dtype=float)
    low = np.asarray(df["low"], dtype=float)
    close = _safe_close(df)
    prev_close = np.roll(close, 1)
    prev_close[0] = close[0]
    tr1 = high - low
    tr2 = np.abs(high - prev_close)
    tr3 = np.abs(low - prev_close)
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    # Wilder's smoothing approximation
    atr = pd.Series(tr).ewm(alpha=1.0 / period, adjust=False).mean().to_numpy()
    return np.maximum(atr, 1e-12)


def _rolling_max_forward(arr: np.ndarray, horizon: int) -> np.ndarray:
    n = len(arr)
    out = np.full(n, np.nan, dtype=float)
    # naive but simple; horizon usually modest for labeling
    for t in range(n):
        end = min(n, t + horizon + 1)
        if t + 1 < end:
            out[t] = np.nanmax(arr[t + 1 : end])
    return out


def _rolling_min_forward(arr: np.ndarray, horizon: int) -> np.ndarray:
    n = len(arr)
    out = np.full(n, np.nan, dtype=float)
    for t in range(n):
        end = min(n, t + horizon + 1)
        if t + 1 < end:
            out[t] = np.nanmin(arr[t + 1 : end])
    return out


def forward_excursion(df: pd.DataFrame, horizon: int):
    """Per-row max favorable excursion over ``horizon``, for buy (up) and sell
    (down) directions, as fractional returns aligned with df rows.

    Uses the same primitives as ``labels_forward_return_mae_cap`` so the returned
    magnitude matches the quantity the gbb labeler thresholds on (buy → mfe,
    sell → downside mfe). Intended as a per-sample P&L-magnitude weight for
    training; it is future-looking and MUST NOT be used as a model feature.
    """
    close = _safe_close(df)
    buy_mfe = (_rolling_max_forward(close, horizon) - close) / close
    sell_mfe = (close - _rolling_min_forward(close, horizon)) / close
    return buy_mfe, sell_mfe


def _max_adverse_excursion(close: np.ndarray, horizon: int) -> np.ndarray:
    # MAE as worst adverse move from entry within future window
    n = len(close)
    out = np.full(n, np.nan, dtype=float)
    for t in range(n):
        end = min(n, t + horizon + 1)
        if t + 1 < end:
            future = close[t + 1 : end]
            out[t] = np.nanmax((close[t] - future) / close[t])
    return out


# ------------------------------
# Methods
# ------------------------------


def labels_forward_return_mae_cap(
    df: pd.DataFrame,
    horizon: int = DEFAULT_HORIZON,
    min_gain: float = DEFAULT_MIN_GAIN,
    max_drawdown: float = DEFAULT_MAX_DRAWDOWN,
    atr_scale: Optional[float] = None,
    min_loss: Optional[float] = DEFAULT_MIN_LOSS,  # Used for conflict checking
) -> pd.Series:
    """
    Label 1 if max future gain >= min_gain and MAE <= max_drawdown.
    If atr_scale provided, thresholds are scaled by ATR/close * atr_scale.
    IMPORTANT: Don't signal buy if sell would also be optimal (prioritize sell).
    """
    close = _safe_close(df)
    max_future = _rolling_max_forward(close, horizon)
    mfe = (max_future - close) / close
    mae = _max_adverse_excursion(close, horizon)

    # Calculate ATR once if needed
    if atr_scale is not None:
        atr = _atr(df)
        scale = (atr / close) * atr_scale
        min_gain_arr = np.maximum(min_gain, scale)
        max_dd_arr = np.maximum(max_drawdown, scale)
    else:
        min_gain_arr = np.full_like(mfe, min_gain)
        max_dd_arr = np.full_like(mae, max_drawdown)

    buy = (mfe >= min_gain_arr) & (mae <= max_dd_arr)

    # Check if sell would also be optimal - if so, don't signal buy
    # Use min_loss if provided, otherwise use min_gain as default for sell check
    # Use same horizon as sell function would use (default 64 to match get_train_sell_signals)
    sell_min_loss = min_loss if min_loss is not None else min_gain
    sell_horizon = horizon  # Use same horizon as buy function (should match if passed consistently)
    min_future = _rolling_min_forward(close, sell_horizon)
    sell_mfe = (close - min_future) / close  # favorable move for sell (down)
    sell_mae = _max_adverse_excursion(close[::-1], sell_horizon)[
        ::-1
    ]  # adverse move upward

    if atr_scale is not None:
        # Reuse atr calculated above
        scale = (atr / close) * atr_scale
        sell_min_loss_arr = np.maximum(sell_min_loss, scale)
        sell_max_dd_arr = np.maximum(max_drawdown, scale)
    else:
        sell_min_loss_arr = np.full_like(sell_mfe, sell_min_loss)
        sell_max_dd_arr = np.full_like(sell_mae, max_drawdown)

    sell = (sell_mfe >= sell_min_loss_arr) & (sell_mae <= sell_max_dd_arr)
    sell = np.where(np.isnan(sell_mfe) | np.isnan(sell_mae), False, sell)

    # Only signal buy if buy is optimal AND sell is not optimal
    buy = buy & ~sell

    buy = np.where(np.isnan(mfe) | np.isnan(mae), 0, buy.astype(int))
    return pd.Series(buy, index=df.index, dtype=float)


def labels_triple_barrier(
    df: pd.DataFrame,
    horizon: int = DEFAULT_HORIZON,
    pt: float = 0.03,
    sl: float = 0.02,
    atr_scale: Optional[float] = None,
    min_gain: Optional[float] = DEFAULT_MIN_GAIN,
    min_loss: Optional[float] = None,  # Ignored for buy functions
) -> pd.Series:
    """
    Triple-barrier (profit-take, stop-loss, time) with buy=1 if PT hits first.
    """
    close = _safe_close(df)
    # Allow overriding minimum profit-take via min_gain
    if min_gain is not None:
        pt = max(pt, float(min_gain))
    if atr_scale is not None:
        atr = _atr(df)
        pt_arr = np.maximum(pt, (atr / close) * atr_scale)
        sl_arr = np.maximum(sl, (atr / close) * atr_scale)
    else:
        pt_arr = np.full_like(close, pt)
        sl_arr = np.full_like(close, sl)

    n = len(close)
    out = np.zeros(n, dtype=int)
    for t in range(n):
        end = min(n, t + horizon + 1)
        if t + 1 >= end:
            continue
        entry = close[t]
        upper = entry * (1.0 + pt_arr[t])
        lower = entry * (1.0 - sl_arr[t])
        future = close[t + 1 : end]
        hit_upper = np.where(future >= upper)[0]
        hit_lower = np.where(future <= lower)[0]
        if hit_upper.size == 0 and hit_lower.size == 0:
            # time barrier: optionally require positive return
            out[t] = 1 if (future[-1] - entry) / entry > 0 else 0
        else:
            first_upper = hit_upper[0] if hit_upper.size else np.inf
            first_lower = hit_lower[0] if hit_lower.size else np.inf
            out[t] = 1 if first_upper < first_lower else 0
    return pd.Series(out.astype(float), index=df.index)


def labels_triple_barrier_sell(
    df: pd.DataFrame,
    horizon: int = DEFAULT_HORIZON,
    pt: float = 0.03,
    sl: float = 0.02,
    atr_scale: Optional[float] = None,
    min_loss: Optional[float] = None,
    min_gain: Optional[float] = None,  # Ignored for sell functions
) -> pd.Series:
    """
    Triple-barrier variant for sell labeling: label 1 if SL (down move) hits first.
    """
    close = _safe_close(df)
    # Allow overriding minimum stop-loss (down move) via min_loss
    if min_loss is not None:
        sl = max(sl, float(min_loss))
    if atr_scale is not None:
        atr = _atr(df)
        pt_arr = np.maximum(pt, (atr / close) * atr_scale)
        sl_arr = np.maximum(sl, (atr / close) * atr_scale)
    else:
        pt_arr = np.full_like(close, pt)
        sl_arr = np.full_like(close, sl)

    n = len(close)
    out = np.zeros(n, dtype=int)
    for t in range(n):
        end = min(n, t + horizon + 1)
        if t + 1 >= end:
            continue
        entry = close[t]
        upper = entry * (1.0 + pt_arr[t])
        lower = entry * (1.0 - sl_arr[t])
        future = close[t + 1 : end]
        hit_upper = np.where(future >= upper)[0]
        hit_lower = np.where(future <= lower)[0]
        if hit_upper.size == 0 and hit_lower.size == 0:
            # time barrier: label as sell if return negative
            out[t] = 1 if (future[-1] - entry) / entry < 0 else 0
        else:
            first_upper = hit_upper[0] if hit_upper.size else np.inf
            first_lower = hit_lower[0] if hit_lower.size else np.inf
            out[t] = 1 if first_lower < first_upper else 0
    return pd.Series(out.astype(float), index=df.index)


def labels_quantile_future_return(
    df: pd.DataFrame,
    horizon: int = DEFAULT_HORIZON,
    top_quantile: float = 0.8,
    max_drawdown: Optional[float] = None,
    atr_scale: Optional[float] = None,
    min_gain: Optional[float] = None,
    min_loss: Optional[float] = None,  # Ignored for buy functions
) -> pd.Series:
    """
    Label top quantile of future returns as 1, optionally filter by MAE cap.
    IMPORTANT: Don't signal buy if sell would also be optimal (prioritize sell).
    """
    close = _safe_close(df)
    max_future = _rolling_max_forward(close, horizon)
    fut_ret = (max_future - close) / close
    thr = (
        np.nanquantile(fut_ret[~np.isnan(fut_ret)], top_quantile)
        if np.any(~np.isnan(fut_ret))
        else np.inf
    )
    # If explicit minimum gain provided, enforce it as well
    if min_gain is not None:
        thr = max(thr, float(min_gain))
    buy = fut_ret >= thr

    if max_drawdown is not None:
        mae = _max_adverse_excursion(close, horizon)
        if atr_scale is not None:
            atr = _atr(df)
            max_dd_arr = np.maximum(max_drawdown, (atr / close) * atr_scale)
        else:
            max_dd_arr = np.full_like(mae, max_drawdown)
        buy = buy & (mae <= max_dd_arr)

    # Check if sell would also be optimal - if so, don't signal buy
    # Calculate sell signals using bottom quantile (default 0.2 = bottom 20%)
    min_future = _rolling_min_forward(close, horizon)
    fut_ret_sell = (min_future - close) / close  # Negative for drops
    bottom_quantile = 1.0 - top_quantile  # Default: 0.2 for top_quantile=0.8
    valid_sell = fut_ret_sell[~np.isnan(fut_ret_sell)]
    if np.any(~np.isnan(fut_ret_sell)):
        sell_thr = np.nanquantile(valid_sell, bottom_quantile)
    else:
        sell_thr = -np.inf
    # If min_gain is provided, use it as min_loss for sell check
    if min_gain is not None:
        sell_thr = min(sell_thr, -float(min_gain))
    sell = fut_ret_sell <= sell_thr

    if max_drawdown is not None:
        mae_sell = _max_adverse_excursion(close[::-1], horizon)[::-1]
        if atr_scale is not None:
            atr = _atr(df)
            max_dd_arr_sell = np.maximum(max_drawdown, (atr / close) * atr_scale)
        else:
            max_dd_arr_sell = np.full_like(mae_sell, max_drawdown)
        sell = sell & (mae_sell <= max_dd_arr_sell)

    # Only signal buy if buy is optimal AND sell is not optimal
    buy = buy & ~sell

    buy = np.where(np.isnan(fut_ret), 0, buy.astype(int))
    return pd.Series(buy, index=df.index, dtype=float)


def labels_quantile_future_return_sell(
    df: pd.DataFrame,
    horizon: int = DEFAULT_HORIZON,
    bottom_quantile: float = 0.2,
    max_drawdown: Optional[float] = None,
    atr_scale: Optional[float] = None,
    min_loss: Optional[float] = None,
    min_gain: Optional[float] = None,  # Ignored for sell functions
) -> pd.Series:
    """
    Label bottom quantile of future returns (worst returns) as 1 for sell signals.
    Identifies positions where future price drops significantly.
    """
    close = _safe_close(df)
    min_future = _rolling_min_forward(close, horizon)
    fut_ret = (min_future - close) / close  # Negative for drops
    # Bottom quantile means worst returns (most negative)
    thr = (
        np.nanquantile(fut_ret[~np.isnan(fut_ret)], bottom_quantile)
        if np.any(~np.isnan(fut_ret))
        else -np.inf
    )
    # If explicit minimum loss provided, enforce it as well (loss is positive magnitude)
    if min_loss is not None:
        thr = min(thr, -float(min_loss))  # Negative threshold for loss
    sell = fut_ret <= thr

    if max_drawdown is not None:
        # For sell, MAE is the maximum adverse excursion upward (against the sell)
        mae = _max_adverse_excursion(close[::-1], horizon)[
            ::-1
        ]  # Reverse for sell perspective
        if atr_scale is not None:
            atr = _atr(df)
            max_dd_arr = np.maximum(max_drawdown, (atr / close) * atr_scale)
        else:
            max_dd_arr = np.full_like(mae, max_drawdown)
        sell = sell & (mae <= max_dd_arr)

    sell = np.where(np.isnan(fut_ret), 0, sell.astype(int))
    return pd.Series(sell, index=df.index, dtype=float)


def labels_mfe_mae_ratio(
    df: pd.DataFrame,
    horizon: int = DEFAULT_HORIZON,
    min_gain: float = 0.02,
    min_ratio: float = 2.0,
    min_loss: Optional[float] = None,  # Ignored for buy functions
) -> pd.Series:
    """
    Label 1 if MFE >= min_gain and MFE/MAE >= min_ratio.
    """
    close = _safe_close(df)
    max_future = _rolling_max_forward(close, horizon)
    mfe = (max_future - close) / close
    mae = _max_adverse_excursion(close, horizon)
    ratio = mfe / (mae + 1e-12)
    buy = (mfe >= min_gain) & (ratio >= min_ratio)
    buy = np.where(np.isnan(mfe) | np.isnan(mae), 0, buy.astype(int))
    return pd.Series(buy, index=df.index, dtype=float)


def labels_local_min_followthrough(
    df: pd.DataFrame,
    horizon: int = DEFAULT_HORIZON,
    window_k: int = 3,
    min_gain: float = 0.02,
    max_drawdown: float = 0.1,
    min_loss: Optional[float] = None,  # Used for conflict checking
) -> pd.Series:
    """
    Local minima with profitable follow-through within horizon.
    IMPORTANT: Don't signal buy if sell would also be optimal (prioritize sell).
    """
    close = _safe_close(df)
    n = len(close)
    is_local_min = np.zeros(n, dtype=bool)
    for t in range(n):
        left_idx = max(0, t - window_k)
        r = min(n, t + window_k + 1)
        if np.nanmin(close[left_idx:r]) == close[t]:
            is_local_min[t] = True
    max_future = _rolling_max_forward(close, horizon)
    mfe = (max_future - close) / close
    mae = _max_adverse_excursion(close, horizon)
    buy = is_local_min & (mfe >= min_gain) & (mae <= max_drawdown)

    # Check if sell would also be optimal - if so, don't signal buy
    # Use min_loss if provided, otherwise use min_gain as default for sell check
    sell_min_loss = min_loss if min_loss is not None else min_gain
    is_local_max = np.zeros(n, dtype=bool)
    for t in range(n):
        left_idx = max(0, t - window_k)
        r = min(n, t + window_k + 1)
        if np.nanmax(close[left_idx:r]) == close[t]:
            is_local_max[t] = True
    min_future = _rolling_min_forward(close, horizon)
    sell_mfe = (close - min_future) / close  # favorable move for sell (down)
    sell_mae = _max_adverse_excursion(close[::-1], horizon)[::-1]  # adverse move upward
    sell = is_local_max & (sell_mfe >= sell_min_loss) & (sell_mae <= max_drawdown)
    sell = np.where(np.isnan(sell_mfe) | np.isnan(sell_mae), False, sell)

    # Only signal buy if buy is optimal AND sell is not optimal
    buy = buy & ~sell

    buy = np.where(np.isnan(mfe) | np.isnan(mae), 0, buy.astype(int))
    return pd.Series(buy, index=df.index, dtype=float)


def labels_local_min_followthrough_sell(
    df: pd.DataFrame,
    horizon: int = DEFAULT_HORIZON,
    window_k: int = 3,
    min_loss: float = 0.02,
    max_drawdown: float = 0.1,
    min_gain: Optional[float] = None,  # Ignored for sell functions
) -> pd.Series:
    """
    Local maxima with profitable downward follow-through within horizon.
    This is the sell variant of labels_local_min_followthrough.
    """
    close = _safe_close(df)
    n = len(close)
    is_local_max = np.zeros(n, dtype=bool)
    for t in range(n):
        left_idx = max(0, t - window_k)
        r = min(n, t + window_k + 1)
        if np.nanmax(close[left_idx:r]) == close[t]:
            is_local_max[t] = True
    min_future = _rolling_min_forward(close, horizon)
    mfe = (close - min_future) / close  # favorable move for sell (down)
    mae = _max_adverse_excursion(close[::-1], horizon)[::-1]  # adverse move upward
    sell = is_local_max & (mfe >= min_loss) & (mae <= max_drawdown)
    sell = np.where(np.isnan(mfe) | np.isnan(mae), 0, sell.astype(int))
    return pd.Series(sell, index=df.index, dtype=float)


def labels_risk_adjusted_future_return(
    df: pd.DataFrame,
    horizon: int = DEFAULT_HORIZON,
    window_stats: int = 16,
    min_sharpe: float = 0.5,
    max_drawdown: float = 0.1,
    min_gain: Optional[float] = None,
    min_loss: Optional[float] = None,  # Ignored for buy functions
) -> pd.Series:
    """
    Risk-adjusted selection via future-window Sharpe proxy + MAE cap.
    """
    close = _safe_close(df)
    n = len(close)
    sharpe = np.full(n, np.nan, dtype=float)
    for t in range(n):
        end = min(n, t + horizon + 1)
        if t + 1 < end:
            window = pd.Series(close[t + 1 : end]).pct_change().dropna()
            if len(window) >= max(3, window_stats // 2):
                if len(window) >= window_stats:
                    mu = window.rolling(window_stats).mean().iloc[-1]
                    sd = window.rolling(window_stats).std(ddof=0).iloc[-1]
                else:
                    mu = window.mean()
                    sd = window.std(ddof=0)
                if sd and not np.isnan(sd):
                    sharpe[t] = float(mu / (sd + 1e-12))
    mae = _max_adverse_excursion(close, horizon)
    buy = (sharpe >= min_sharpe) & (mae <= max_drawdown)
    # Optional: also require minimum future gain
    if min_gain is not None:
        max_future = _rolling_max_forward(close, horizon)
        mfe = (max_future - close) / close
        buy = buy & (mfe >= float(min_gain))
    buy = np.where(np.isnan(sharpe) | np.isnan(mae), 0, buy.astype(int))
    return pd.Series(buy, index=df.index, dtype=float)


def labels_multi_horizon_vote(
    df: pd.DataFrame,
    horizons: Tuple[int, int, int] = (16, 32, 64),
    horizon: int = DEFAULT_HORIZON,
    min_gain: float = 0.02,
    max_drawdown: float = 0.02,
    votes_needed: int = 2,
    min_loss: Optional[float] = None,  # Ignored for buy functions
) -> pd.Series:
    """
    Vote across multiple horizons using forward_return_mae_cap criteria.
    If a single horizon is passed (as in other label methods), ensure it's included.
    """
    if horizon is not None and horizon not in horizons:
        horizons = tuple(sorted(set(horizons + (int(horizon),))))
    votes = []
    for h in horizons:
        v = labels_forward_return_mae_cap(
            df, horizon=h, min_gain=min_gain, max_drawdown=max_drawdown
        ).to_numpy(dtype=int)
        votes.append(v)
    votes = np.stack(votes, axis=1)  # [N, H]
    buy = (votes.sum(axis=1) >= votes_needed).astype(float)
    return pd.Series(buy, index=df.index)


def labels_technical_indicators(
    df: pd.DataFrame,
    min_gain: float = 0.01,
    horizon: int = DEFAULT_HORIZON,
    min_indicators_buy: int = 4,
    min_loss: Optional[float] = None,  # Ignored for buy functions
) -> pd.Series:
    """
    Generate buy signals based on technical indicator consensus.
    Uses indicators already present in the dataframe (RSI, CCI, guard_metric, bb_position, etc.).
    This should be highly learnable since the model sees these same features.

    Buy signals: When multiple indicators suggest oversold conditions AND future gain >= min_gain.

    Args:
        df: DataFrame with technical indicators
        min_gain: Minimum future gain required for buy signal
        horizon: Lookahead window for future gain validation
        min_indicators_buy: Minimum number of indicators that must agree for buy signal
    """
    close = _safe_close(df)
    n = len(close)
    labels = np.zeros(n, dtype=float)

    # Get indicator values (handle missing columns gracefully)
    adx = df.get("adx_scaled", pd.Series(np.zeros(n)))
    aroonosc = df.get("aroonosc_scaled", pd.Series(np.zeros(n)))
    guard = df.get("guard_metric", pd.Series(np.zeros(n)))
    sar_ratio = df.get("sar_ratio", pd.Series(np.zeros(n)))
    bb_width = df.get("bb_width", pd.Series(np.zeros(n)))
    vwap_ratio = df.get("vwap_ratio", pd.Series(np.zeros(n)))

    # Convert to numpy arrays, handling NaN
    adx = np.asarray(adx, dtype=float)
    aroonosc = np.asarray(aroonosc, dtype=float)
    guard = np.asarray(guard, dtype=float)
    sar_ratio = np.asarray(sar_ratio, dtype=float)
    bb_width = np.asarray(bb_width, dtype=float)
    vwap_ratio = np.asarray(vwap_ratio, dtype=float)

    # Replace NaN with neutral values (0)
    adx = np.nan_to_num(adx, nan=0.0)
    aroonosc = np.nan_to_num(aroonosc, nan=0.0)
    guard = np.nan_to_num(guard, nan=0.0)
    sar_ratio = np.nan_to_num(sar_ratio, nan=0.0)
    bb_width = np.nan_to_num(bb_width, nan=0.0)
    vwap_ratio = np.nan_to_num(vwap_ratio, nan=0.0)

    # Calculate future gain for validation
    max_future = _rolling_max_forward(close, horizon)
    future_gain = (max_future - close) / close

    # Buy signals: Oversold indicators + future gain validation
    buy_votes = np.zeros(n, dtype=int)
    buy_votes += (adx >= 0.1).astype(int)  # ADX oversold
    buy_votes += (aroonosc <= -0.2).astype(int)  # Aroonosc oversold
    buy_votes += (guard <= -0.2).astype(int)  # Guard metric oversold
    buy_votes += (sar_ratio >= 0.2).astype(int)  # SAR ratio oversold
    buy_votes += (bb_width >= 0.017).astype(int)  # BB width >= 1.7% raw (was 0.2 normalized)
    buy_votes += (vwap_ratio >= 0.2).astype(int)  # VWAP ratio oversold

    # Buy: enough indicators agree AND future gain meets threshold
    buy_mask = (buy_votes >= min_indicators_buy) & (future_gain >= min_gain)
    buy_mask = np.where(np.isnan(future_gain), False, buy_mask)
    labels[buy_mask] = 1.0

    return pd.Series(labels, index=df.index)


def labels_technical_indicators_sell(
    df: pd.DataFrame,
    min_loss: float = 0.01,
    horizon: int = DEFAULT_HORIZON,
    min_indicators_sell: int = 4,
    min_gain: Optional[float] = None,  # Ignored for sell functions
) -> pd.Series:
    """
    Generate sell signals based on technical indicator consensus.
    Similar to labels_technical_indicators but for sell signals.
    """
    close = _safe_close(df)
    n = len(close)
    labels = np.zeros(n, dtype=float)

    # Get indicator values (handle missing columns gracefully)
    adx = df.get("adx_scaled", pd.Series(np.zeros(n)))
    aroonosc = df.get("aroonosc_scaled", pd.Series(np.zeros(n)))
    guard = df.get("guard_metric", pd.Series(np.zeros(n)))
    sar_ratio = df.get("sar_ratio", pd.Series(np.zeros(n)))
    bb_width = df.get("bb_width", pd.Series(np.zeros(n)))
    vwap_ratio = df.get("vwap_ratio", pd.Series(np.zeros(n)))

    # Convert to numpy arrays, handling NaN
    adx = np.asarray(adx, dtype=float)
    aroonosc = np.asarray(aroonosc, dtype=float)
    guard = np.asarray(guard, dtype=float)
    sar_ratio = np.asarray(sar_ratio, dtype=float)
    bb_width = np.asarray(bb_width, dtype=float)
    vwap_ratio = np.asarray(vwap_ratio, dtype=float)

    # Replace NaN with neutral values (0)
    adx = np.nan_to_num(adx, nan=0.0)
    aroonosc = np.nan_to_num(aroonosc, nan=0.0)
    guard = np.nan_to_num(guard, nan=0.0)
    sar_ratio = np.nan_to_num(sar_ratio, nan=0.0)
    bb_width = np.nan_to_num(bb_width, nan=0.0)
    vwap_ratio = np.nan_to_num(vwap_ratio, nan=0.0)

    # Calculate future loss for validation (use min_future for sell signals)
    min_future = _rolling_min_forward(close, horizon)
    future_loss = (close - min_future) / close

    # Sell signals: Overbought indicators + future loss validation
    # Conditions are opposite of buy signals (overbought vs oversold)
    sell_votes = np.zeros(n, dtype=int)
    sell_votes += (adx <= -0.1).astype(int)  # ADX overbought (low trend strength)
    sell_votes += (aroonosc >= 0.2).astype(int)  # Aroonosc overbought (uptrend)
    sell_votes += (guard >= 0.2).astype(int)  # Guard metric overbought
    sell_votes += (sar_ratio <= -0.2).astype(
        int
    )  # SAR ratio overbought (price below SAR, bearish)
    sell_votes += (bb_width <= 0.011).astype(int)  # BB width <= 1.1% raw, low vol (was -0.2 normalized)
    sell_votes += (vwap_ratio <= -0.2).astype(
        int
    )  # VWAP ratio overbought (price below VWAP)

    # Sell: enough indicators agree AND future loss meets threshold
    sell_mask = (sell_votes >= min_indicators_sell) & (future_loss >= min_loss)
    sell_mask = np.where(np.isnan(future_loss), False, sell_mask)
    labels[sell_mask] = 1.0

    return pd.Series(labels, index=df.index)


def labels_indicators2(
    df: pd.DataFrame,
    min_gain: float = 0.01,
    horizon: int = DEFAULT_HORIZON,
    min_indicators_buy: int = 4,
    min_loss: Optional[float] = None,  # Ignored for buy functions
) -> pd.Series:
    """
    Generate buy signals based on technical indicator consensus.
    Uses indicators already present in the dataframe (RSI, CCI, guard_metric, bb_position, etc.).
    This should be highly learnable since the model sees these same features.

    Buy signals: When multiple indicators suggest oversold conditions AND future gain >= min_gain.

    Args:
        df: DataFrame with technical indicators
        min_gain: Minimum future gain required for buy signal
        horizon: Lookahead window for future gain validation
        min_indicators_buy: Minimum number of indicators that must agree for buy signal
    """
    close = _safe_close(df)
    n = len(close)
    labels = np.zeros(n, dtype=float)

    # Get indicator values (handle missing columns gracefully)
    atr = df.get("atr_norm", pd.Series(np.zeros(n)))
    gain = df.get("gain_norm", pd.Series(np.zeros(n)))
    rsi = df.get("rsi_scaled", pd.Series(np.zeros(n)))
    mfi = df.get("mfi_scaled", pd.Series(np.zeros(n)))
    bb_width = df.get("bb_width", pd.Series(np.zeros(n)))
    log_volume = df.get("log_volume_norm", pd.Series(np.zeros(n)))
    close_norm = df.get("close_norm", pd.Series(np.zeros(n)))

    # Convert to numpy arrays, handling NaN
    atr = np.asarray(atr, dtype=float)
    gain = np.asarray(gain, dtype=float)
    rsi = np.asarray(rsi, dtype=float)
    mfi = np.asarray(mfi, dtype=float)
    bb_width = np.asarray(bb_width, dtype=float)
    log_volume = np.asarray(log_volume, dtype=float)
    close_norm = np.asarray(close_norm, dtype=float)

    # Replace NaN with neutral values (0)
    atr = np.nan_to_num(atr, nan=0.0)
    gain = np.nan_to_num(gain, nan=0.0)
    rsi = np.nan_to_num(rsi, nan=0.0)
    mfi = np.nan_to_num(mfi, nan=0.0)
    bb_width = np.nan_to_num(bb_width, nan=0.0)
    log_volume = np.nan_to_num(log_volume, nan=0.0)
    close_norm = np.nan_to_num(close_norm, nan=0.0)

    # Calculate future gain for validation
    max_future = _rolling_max_forward(close, horizon)
    future_gain = (max_future - close) / close

    # Buy signals: Oversold indicators + future gain validation
    buy_votes = np.zeros(n, dtype=int)
    buy_votes += (atr >= 0.5).astype(int)  # ATR oversold
    buy_votes += (gain <= -0.5).astype(int)  # Gain oversold
    buy_votes += (rsi <= -0.0).astype(int)  # RSI oversold
    buy_votes += (mfi <= -0.0).astype(int)  # MFI oversold
    buy_votes += (bb_width <= 0.011).astype(int)  # BB width <= 1.1% raw (was -0.2 normalized)
    buy_votes += (log_volume >= 0.2).astype(int)  # Log volume negative
    buy_votes += (close_norm <= -0.2).astype(int)  # Close norm negative

    # Buy: enough indicators agree AND future gain meets threshold
    buy_mask = (buy_votes >= min_indicators_buy) & (future_gain >= min_gain)
    buy_mask = np.where(np.isnan(future_gain), False, buy_mask)
    labels[buy_mask] = 1.0

    return pd.Series(labels, index=df.index)


def labels_indicators2_sell(
    df: pd.DataFrame,
    min_loss: float = 0.01,
    horizon: int = DEFAULT_HORIZON,
    min_indicators_sell: int = 4,
    min_gain: Optional[float] = None,  # Ignored for sell functions
) -> pd.Series:
    """
    Generate sell signals based on technical indicator consensus.
    Similar to labels_indicators2 but for sell signals.
    """
    close = _safe_close(df)
    n = len(close)
    labels = np.zeros(n, dtype=float)

    # Get indicator values (handle missing columns gracefully)
    atr = df.get("atr_norm", pd.Series(np.zeros(n)))
    gain = df.get("gain_norm", pd.Series(np.zeros(n)))
    rsi = df.get("rsi_scaled", pd.Series(np.zeros(n)))
    mfi = df.get("mfi_scaled", pd.Series(np.zeros(n)))
    bb_width = df.get("bb_width", pd.Series(np.zeros(n)))
    log_volume = df.get("log_volume_norm", pd.Series(np.zeros(n)))
    close_norm = df.get("close_norm", pd.Series(np.zeros(n)))

    # Convert to numpy arrays, handling NaN
    atr = np.asarray(atr, dtype=float)
    gain = np.asarray(gain, dtype=float)
    rsi = np.asarray(rsi, dtype=float)
    mfi = np.asarray(mfi, dtype=float)
    bb_width = np.asarray(bb_width, dtype=float)
    log_volume = np.asarray(log_volume, dtype=float)
    close_norm = np.asarray(close_norm, dtype=float)

    # Replace NaN with neutral values (0)
    atr = np.nan_to_num(atr, nan=0.0)
    gain = np.nan_to_num(gain, nan=0.0)
    rsi = np.nan_to_num(rsi, nan=0.0)
    mfi = np.nan_to_num(mfi, nan=0.0)
    bb_width = np.nan_to_num(bb_width, nan=0.0)
    log_volume = np.nan_to_num(log_volume, nan=0.0)
    close_norm = np.nan_to_num(close_norm, nan=0.0)

    # Calculate future loss for validation (use min_future for sell signals)
    min_future = _rolling_min_forward(close, horizon)
    future_loss = (close - min_future) / close

    # Sell signals: Overbought indicators + future loss validation
    # Conditions are opposite of buy signals (overbought vs oversold)
    sell_votes = np.zeros(n, dtype=int)
    sell_votes += (atr <= -0.5).astype(int)  # ATR oversold
    sell_votes += (gain >= 0.5).astype(int)  # Gain oversold
    sell_votes += (rsi >= 0.0).astype(int)  # RSI oversold
    sell_votes += (mfi >= 0.0).astype(int)  # MFI oversold
    sell_votes += (bb_width >= 0.017).astype(int)  # BB width >= 1.7% raw (was 0.2 normalized)
    sell_votes += (log_volume <= 0.2).astype(int)  # Log volume negative
    sell_votes += (close_norm >= 0.2).astype(int)  # Close norm negative

    # Sell: enough indicators agree AND future loss meets threshold
    sell_mask = (sell_votes >= min_indicators_sell) & (future_loss >= min_loss)
    sell_mask = np.where(np.isnan(future_loss), False, sell_mask)
    labels[sell_mask] = 1.0

    return pd.Series(labels, index=df.index)


def labels_indicators3(
    df: pd.DataFrame,
    min_gain: float = 0.01,
    horizon: int = DEFAULT_HORIZON,
    min_indicators_buy: int = 3,
    min_loss: Optional[float] = None,  # Ignored for buy functions
) -> pd.Series:
    """
    Generate buy signals based on technical indicator consensus.
    Uses indicators already present in the dataframe (RSI, CCI, guard_metric, bb_position, etc.).
    This should be highly learnable since the model sees these same features.

    Buy signals: When multiple indicators suggest oversold conditions AND future gain >= min_gain.

    Args:
        df: DataFrame with technical indicators
        min_gain: Minimum future gain required for buy signal
        horizon: Lookahead window for future gain validation
        min_indicators_buy: Minimum number of indicators that must agree for buy signal
    """
    close = _safe_close(df)
    n = len(close)
    labels = np.zeros(n, dtype=float)

    # Get indicator values (handle missing columns gracefully)
    close_norm = df.get("close_norm", pd.Series(np.zeros(n)))
    adx_scaled = df.get("adx_scaled", pd.Series(np.zeros(n)))
    trend_mode = df.get("trend_mode", pd.Series(np.zeros(n)))
    guard_metric = df.get("guard_metric", pd.Series(np.zeros(n)))

    # Convert to numpy arrays, handling NaN
    close_norm = np.asarray(close_norm, dtype=float)
    adx_scaled = np.asarray(adx_scaled, dtype=float)
    trend_mode = np.asarray(trend_mode, dtype=float)
    guard_metric = np.asarray(guard_metric, dtype=float)

    # Replace NaN with neutral values (0)
    close_norm = np.nan_to_num(close_norm, nan=0.0)
    adx_scaled = np.nan_to_num(adx_scaled, nan=0.0)
    trend_mode = np.nan_to_num(trend_mode, nan=0.0)
    guard_metric = np.nan_to_num(guard_metric, nan=0.0)

    # Calculate future gain for validation
    max_future = _rolling_max_forward(close, horizon)
    future_gain = (max_future - close) / close

    # Buy signals: Oversold indicators + future gain validation

    buy_votes = np.zeros(n, dtype=int)
    buy_votes += (close_norm < -0.0).astype(int)  # Close norm negative
    buy_votes += (adx_scaled > -0.2).astype(int)  # ADX scaled positive
    buy_votes += (trend_mode > 0).astype(int)  # in trend
    buy_votes += (guard_metric < -0.0).astype(int)  # Guard metric negative

    # Buy: enough indicators agree AND future gain meets threshold
    buy_mask = (buy_votes >= min_indicators_buy) & (future_gain >= min_gain)
    buy_mask = np.where(np.isnan(future_gain), False, buy_mask)
    labels[buy_mask] = 1.0

    return pd.Series(labels, index=df.index)


def labels_indicators3_sell(
    df: pd.DataFrame,
    min_loss: float = 0.01,
    horizon: int = DEFAULT_HORIZON,
    min_indicators_sell: int = 3,
    min_gain: Optional[float] = None,  # Ignored for sell functions
) -> pd.Series:
    """
    Generate sell signals based on technical indicator consensus.
    Similar to labels_indicators3 but for sell signals.
    """
    close = _safe_close(df)
    n = len(close)
    labels = np.zeros(n, dtype=float)

    # Get indicator values (handle missing columns gracefully)
    close_norm = df.get("close_norm", pd.Series(np.zeros(n)))
    adx_scaled = df.get("adx_scaled", pd.Series(np.zeros(n)))
    trend_mode = df.get("trend_mode", pd.Series(np.zeros(n)))
    guard_metric = df.get("guard_metric", pd.Series(np.zeros(n)))

    # Convert to numpy arrays, handling NaN
    close_norm = np.asarray(close_norm, dtype=float)
    adx_scaled = np.asarray(adx_scaled, dtype=float)
    trend_mode = np.asarray(trend_mode, dtype=float)
    guard_metric = np.asarray(guard_metric, dtype=float)

    # Replace NaN with neutral values (0)
    close_norm = np.nan_to_num(close_norm, nan=0.0)
    adx_scaled = np.nan_to_num(adx_scaled, nan=0.0)
    trend_mode = np.nan_to_num(trend_mode, nan=0.0)
    guard_metric = np.nan_to_num(guard_metric, nan=0.0)

    # Calculate future loss for validation (use min_future for sell signals)
    min_future = _rolling_min_forward(close, horizon)
    future_loss = (close - min_future) / close

    # Sell signals: Overbought indicators + future loss validation
    # Conditions are opposite of buy signals (overbought vs oversold)
    sell_votes = np.zeros(n, dtype=int)
    sell_votes += (close_norm > 0.0).astype(int)  # Close norm negative
    sell_votes += (adx_scaled > -0.2).astype(int)  # ADX scaled negative
    sell_votes += (trend_mode > 0).astype(int)  # in trend
    sell_votes += (guard_metric > 0.0).astype(int)  # Guard metric positive

    # Sell: enough indicators agree AND future loss meets threshold
    sell_mask = (sell_votes >= min_indicators_sell) & (future_loss >= min_loss)
    sell_mask = np.where(np.isnan(future_loss), False, sell_mask)
    labels[sell_mask] = 1.0

    return pd.Series(labels, index=df.index)


def labels_indicators4(
    df: pd.DataFrame,
    min_gain: float = 0.01,
    horizon: int = DEFAULT_HORIZON,
    min_indicators_buy: int = 3,
    min_loss: Optional[float] = None,  # Ignored for buy functions
) -> pd.Series:
    """
    Generate buy signals based on indicators with high correlation to buy/sell:

    - rsi_scaled
    - mfi_scaled
    - ema_fast_norm
    - fastk_scaled
    - di_diff_scaled
    """
    close = _safe_close(df)
    n = len(close)
    labels = np.zeros(n, dtype=float)

    rsi = df.get("rsi_scaled", pd.Series(np.zeros(n)))
    mfi = df.get("mfi_scaled", pd.Series(np.zeros(n)))
    ema_fast = df.get("ema_fast_norm", pd.Series(np.zeros(n)))
    fastk = df.get("fastk_scaled", pd.Series(np.zeros(n)))
    di_diff = df.get("di_diff_scaled", pd.Series(np.zeros(n)))

    rsi = np.nan_to_num(np.asarray(rsi, dtype=float), nan=0.0)
    mfi = np.nan_to_num(np.asarray(mfi, dtype=float), nan=0.0)
    ema_fast = np.nan_to_num(np.asarray(ema_fast, dtype=float), nan=0.0)
    fastk = np.nan_to_num(np.asarray(fastk, dtype=float), nan=0.0)
    di_diff = np.nan_to_num(np.asarray(di_diff, dtype=float), nan=0.0)

    max_future = _rolling_max_forward(close, horizon)
    future_gain = (max_future - close) / close

    # Consensus: oversold / below norm favours buy; positive trend (di_diff) favours buy
    buy_votes = np.zeros(n, dtype=int)
    buy_votes += (rsi < 0.0).astype(int)
    buy_votes += (mfi < 0.0).astype(int)
    buy_votes += (ema_fast < 0.0).astype(int)
    buy_votes += (fastk < 0.0).astype(int)
    buy_votes += (di_diff > 0.0).astype(int)

    buy_mask = (buy_votes >= min_indicators_buy) & (future_gain >= min_gain)
    buy_mask = np.where(np.isnan(future_gain), False, buy_mask)
    labels[buy_mask] = 1.0

    return pd.Series(labels, index=df.index)


def labels_indicators4_sell(
    df: pd.DataFrame,
    min_loss: float = 0.01,
    horizon: int = DEFAULT_HORIZON,
    min_indicators_sell: int = 3,
    min_gain: Optional[float] = None,  # Ignored for sell functions
) -> pd.Series:
    """
    Sell counterpart to labels_indicators4 using the same buy/sell-correlated indicators.
    """
    close = _safe_close(df)
    n = len(close)
    labels = np.zeros(n, dtype=float)

    rsi = df.get("rsi_scaled", pd.Series(np.zeros(n)))
    mfi = df.get("mfi_scaled", pd.Series(np.zeros(n)))
    ema_fast = df.get("ema_fast_norm", pd.Series(np.zeros(n)))
    fastk = df.get("fastk_scaled", pd.Series(np.zeros(n)))
    di_diff = df.get("di_diff_scaled", pd.Series(np.zeros(n)))

    rsi = np.nan_to_num(np.asarray(rsi, dtype=float), nan=0.0)
    mfi = np.nan_to_num(np.asarray(mfi, dtype=float), nan=0.0)
    ema_fast = np.nan_to_num(np.asarray(ema_fast, dtype=float), nan=0.0)
    fastk = np.nan_to_num(np.asarray(fastk, dtype=float), nan=0.0)
    di_diff = np.nan_to_num(np.asarray(di_diff, dtype=float), nan=0.0)

    min_future = _rolling_min_forward(close, horizon)
    future_loss = (close - min_future) / close

    # Opposite of buy: overbought / above norm; negative trend favours sell
    sell_votes = np.zeros(n, dtype=int)
    sell_votes += (rsi > 0.0).astype(int)
    sell_votes += (mfi > 0.0).astype(int)
    sell_votes += (ema_fast > 0.0).astype(int)
    sell_votes += (fastk > 0.0).astype(int)
    sell_votes += (di_diff < 0.0).astype(int)

    sell_mask = (sell_votes >= min_indicators_sell) & (future_loss >= min_loss)
    sell_mask = np.where(np.isnan(future_loss), False, sell_mask)
    labels[sell_mask] = 1.0

    return pd.Series(labels, index=df.index)


def labels_gbb(
    df: pd.DataFrame,
    guard_threshold: float = -0.2,
    bb_width_threshold: float = 0.035,
    min_gain: Optional[float] = 0.01,
    min_loss: Optional[float] = None,  # Ignored for buy
    horizon: Optional[int] = DEFAULT_HORIZON,
) -> pd.Series:
    """
    Generate buy signals using guard_metric and bb_width only.
    Buy when guard_metric is below threshold and bb_width is above threshold.
    """
    close = _safe_close(df)
    n = len(close)
    labels = np.zeros(n, dtype=float)

    guard_metric = df.get("guard_metric", pd.Series(np.zeros(n)))
    bb_width = df.get("bb_width", pd.Series(np.zeros(n)))

    guard_metric = np.nan_to_num(np.asarray(guard_metric, dtype=float), nan=0.0)
    bb_width = np.nan_to_num(np.asarray(bb_width, dtype=float), nan=0.0)

    max_future = _rolling_max_forward(close, horizon)
    future_gain = (max_future - close) / close
    min_gain_val = min_gain if min_gain is not None else 0.0

    buy_mask = (
        (guard_metric < guard_threshold)
        & (bb_width > bb_width_threshold)
        & (future_gain >= min_gain_val)
    )
    buy_mask = np.where(np.isnan(future_gain), False, buy_mask)
    labels[buy_mask] = 1.0

    return pd.Series(labels, index=df.index)


def labels_gbb_sell(
    df: pd.DataFrame,
    guard_threshold: float = 0.2,
    bb_width_threshold: float = 0.035,
    min_gain: Optional[float] = None,  # Ignored for sell
    min_loss: Optional[float] = 0.01,
    horizon: Optional[int] = DEFAULT_HORIZON,
) -> pd.Series:
    """
    Generate sell signals using guard_metric and bb_width only.
    Sell when guard_metric is above threshold and bb_width is above threshold.
    """
    close = _safe_close(df)
    n = len(close)
    labels = np.zeros(n, dtype=float)

    guard_metric = df.get("guard_metric", pd.Series(np.zeros(n)))
    bb_width = df.get("bb_width", pd.Series(np.zeros(n)))

    guard_metric = np.nan_to_num(np.asarray(guard_metric, dtype=float), nan=0.0)
    bb_width = np.nan_to_num(np.asarray(bb_width, dtype=float), nan=0.0)

    min_future = _rolling_min_forward(close, horizon)
    future_loss = (close - min_future) / close
    min_loss_val = min_loss if min_loss is not None else 0.0

    sell_mask = (
        (guard_metric > guard_threshold)
        & (bb_width > bb_width_threshold)
        & (future_loss >= min_loss_val)
    )
    sell_mask = np.where(np.isnan(future_loss), False, sell_mask)
    labels[sell_mask] = 1.0

    return pd.Series(labels, index=df.index)


def labels_bands(
    df: pd.DataFrame,
    min_gain: float = 0.01,
    horizon: int = DEFAULT_HORIZON,
    min_loss: Optional[float] = None,  # Ignored for buy
) -> pd.Series:
    """
    Generate buy signals where close crosses above lower or upper bands
    (Bollinger, Donchian, Keltner, or Larry Williams).
    """
    close = _safe_close(df)
    n = len(close)
    labels = np.zeros(n, dtype=float)
    high = np.asarray(df["high"], dtype=float)
    low = np.asarray(df["low"], dtype=float)
    open_p = np.asarray(df["open"], dtype=float)

    # 1. Bollinger Bands (from DF if available, else calculate)
    bb_lower = df.get("bb_lowerband")
    bb_upper = df.get("bb_upperband")
    if bb_lower is None or bb_upper is None:
        # Simple fallback
        sma = pd.Series(close).rolling(20).mean()
        std = pd.Series(close).rolling(20).std()
        bb_lower = sma - (2.0 * std)
        bb_upper = sma + (2.0 * std)
    bb_lower = np.nan_to_num(np.asarray(bb_lower, dtype=float), nan=0.0)
    bb_upper = np.nan_to_num(np.asarray(bb_upper, dtype=float), nan=0.0)

    # 2. Donchian Channels
    dc_lower = pd.Series(low).rolling(20).min().to_numpy()
    dc_upper = pd.Series(high).rolling(20).max().to_numpy()
    dc_lower = np.nan_to_num(dc_lower, nan=0.0)
    dc_upper = np.nan_to_num(dc_upper, nan=0.0)

    # 3. Keltner Channels
    atr = _atr(df, period=10)
    ema20 = pd.Series(close).ewm(span=20, adjust=False).mean().to_numpy()
    kc_lower = ema20 - (2.0 * atr)
    kc_upper = ema20 + (2.0 * atr)
    kc_lower = np.nan_to_num(kc_lower, nan=0.0)
    kc_upper = np.nan_to_num(kc_upper, nan=0.0)

    # 4. Larry Williams (Volatility) Bands
    lw_atr = _atr(df, period=20)
    lw_lower = open_p - (1.0 * lw_atr)
    lw_upper = open_p + (0.5 * lw_atr)
    lw_lower = np.nan_to_num(lw_lower, nan=0.0)
    lw_upper = np.nan_to_num(lw_upper, nan=0.0)

    prev_close = np.roll(close, 1)
    prev_close[0] = close[0]

    # Buy signals: Cross above any Lower Band OR Cross above any Upper Band
    def cross_above(curr, band):
        prev_band = np.roll(band, 1)
        res = (curr > band) & (prev_close <= prev_band)
        res[0] = False
        return res

    cross_lower = (
        cross_above(close, bb_lower)
        | cross_above(close, dc_lower)
        | cross_above(close, kc_lower)
        | cross_above(close, lw_lower)
    )
    cross_upper = (
        cross_above(close, bb_upper)
        | cross_above(close, dc_upper)
        | cross_above(close, kc_upper)
        | cross_above(close, lw_upper)
    )

    # Validation
    max_future = _rolling_max_forward(close, horizon)
    future_gain = (max_future - close) / close

    buy_mask = (cross_lower | cross_upper) & (future_gain >= min_gain)
    buy_mask = np.where(np.isnan(future_gain), False, buy_mask)
    labels[buy_mask] = 1.0

    return pd.Series(labels, index=df.index)


def labels_bands_sell(
    df: pd.DataFrame,
    min_loss: float = 0.01,
    horizon: int = DEFAULT_HORIZON,
    min_gain: Optional[float] = None,  # Ignored for sell
) -> pd.Series:
    """
    Generate sell signals where close crosses below upper bands.
    """
    close = _safe_close(df)
    n = len(close)
    labels = np.zeros(n, dtype=float)
    high = np.asarray(df["high"], dtype=float)
    open_p = np.asarray(df["open"], dtype=float)

    # 1. Bollinger Bands
    bb_upper = df.get("bb_upperband")
    if bb_upper is None:
        sma = pd.Series(close).rolling(20).mean()
        std = pd.Series(close).rolling(20).std()
        bb_upper = sma + (2.0 * std)
    bb_upper = np.nan_to_num(np.asarray(bb_upper, dtype=float), nan=0.0)

    # 2. Donchian Channels
    dc_upper = pd.Series(high).rolling(20).max().to_numpy()
    dc_upper = np.nan_to_num(dc_upper, nan=0.0)

    # 3. Keltner Channels
    atr = _atr(df, period=10)
    ema20 = pd.Series(close).ewm(span=20, adjust=False).mean().to_numpy()
    kc_upper = ema20 + (2.0 * atr)
    kc_upper = np.nan_to_num(kc_upper, nan=0.0)

    # 4. Larry Williams Bands
    lw_atr = _atr(df, period=20)
    lw_upper = open_p + (1.0 * lw_atr)  # Cycle sell mult
    lw_upper = np.nan_to_num(lw_upper, nan=0.0)

    prev_close = np.roll(close, 1)
    prev_close[0] = close[0]

    # Sell signals: Cross below any Upper Band
    def cross_below(curr, band):
        prev_band = np.roll(band, 1)
        res = (curr < band) & (prev_close >= prev_band)
        res[0] = False
        return res

    cross_upper = (
        cross_below(close, bb_upper)
        | cross_below(close, dc_upper)
        | cross_below(close, kc_upper)
        | cross_below(close, lw_upper)
    )

    # Validation
    min_future = _rolling_min_forward(close, horizon)
    future_loss = (close - min_future) / close

    sell_mask = (cross_upper) & (future_loss >= min_loss)
    sell_mask = np.where(np.isnan(future_loss), False, sell_mask)
    labels[sell_mask] = 1.0

    return pd.Series(labels, index=df.index)


def labels_trends(
    df: pd.DataFrame,
    min_gain: float = 0.01,
    horizon: int = DEFAULT_HORIZON,
    min_loss: Optional[float] = None,  # Used for conflict checking
) -> pd.Series:
    """
    Generate buy signals based on various trends
    This should be highly learnable since the model sees these same features.

    Buy signals: When indicators suggest oversold conditions AND future gain >= min_gain.
    IMPORTANT: Don't signal buy if sell would also be optimal (prioritize sell).

    Args:
        df: DataFrame with technical indicators
        min_gain: Minimum future gain required for buy signal
    """
    close = _safe_close(df)
    n = len(close)
    labels = np.zeros(n, dtype=float)

    # Get indicator values (handle missing columns gracefully)
    flow = df.get("flow", pd.Series(np.zeros(n)))
    regime = df.get("regime", pd.Series(np.zeros(n)))
    risk = df.get("risk", pd.Series(np.zeros(n)))
    momentum = df.get("momentum", pd.Series(np.zeros(n)))

    # Convert to numpy arrays, handling NaN
    flow = np.asarray(flow, dtype=float)
    regime = np.asarray(regime, dtype=float)
    risk = np.asarray(risk, dtype=float)
    momentum = np.asarray(momentum, dtype=float)

    # Replace NaN with neutral values (0)
    flow = np.nan_to_num(flow, nan=0.0)
    regime = np.nan_to_num(regime, nan=0.0)
    risk = np.nan_to_num(risk, nan=0.0)
    momentum = np.nan_to_num(momentum, nan=0.0)

    # Calculate future gain for validation
    max_future = _rolling_max_forward(close, horizon)
    future_gain = (max_future - close) / close

    # Buy signals: Oversold indicators + future gain validation
    # Use strict equality to avoid overlap (all conditions mutually exclusive):
    # - flow == 0 (DECREASE) for buy, flow == 2 (INCREASE) for sell
    # - regime == 0 (BEAR) for buy, regime == 2 (BULL) for sell
    # - risk <= 1 (LOW or NORMAL) for buy, risk == 2 (HIGH) for sell (mutually exclusive)
    # - momentum == 0 (NEGATIVE) for buy, momentum == 2 (POSITIVE) for sell
    num_votes = np.zeros(n, dtype=int)
    num_votes += (flow == 0).astype(int)  # DECREASE
    num_votes += (regime == 0).astype(int)  # BEAR
    num_votes += (risk <= 1).astype(
        int
    )  # LOW or NORMAL (oversold conditions often have lower risk)
    num_votes += (momentum == 0).astype(int)  # NEGATIVE
    buy_trend_mask = num_votes >= 2  # Require 2 out of 4 conditions

    # Buy: enough indicators agree AND future gain meets threshold
    buy_mask = (buy_trend_mask) & (future_gain >= min_gain)
    buy_mask = np.where(np.isnan(future_gain), False, buy_mask)

    # Check if sell would also be optimal - if so, don't signal buy
    # Use min_loss if provided, otherwise use min_gain as default for sell check
    sell_min_loss = min_loss if min_loss is not None else min_gain
    min_future = _rolling_min_forward(close, horizon)
    future_loss = (close - min_future) / close

    # Sell signals: Overbought indicators + future loss validation
    # Use strict equality to avoid overlap (mutually exclusive with buy conditions)
    sell_num_votes = np.zeros(n, dtype=int)
    sell_num_votes += (flow == 2).astype(int)  # INCREASE
    sell_num_votes += (regime == 2).astype(int)  # BULL
    sell_num_votes += (risk == 2).astype(
        int
    )  # HIGH (overbought conditions often have higher risk)
    sell_num_votes += (momentum == 2).astype(int)  # POSITIVE
    sell_trend_mask = sell_num_votes >= 2  # Require 2 out of 4 conditions

    # Sell: enough indicators agree AND future loss meets threshold
    sell_mask = (sell_trend_mask) & (future_loss >= sell_min_loss)
    sell_mask = np.where(np.isnan(future_loss), False, sell_mask)

    # Only signal buy if buy is optimal AND sell is not optimal
    buy_mask = buy_mask & ~sell_mask

    labels[buy_mask] = 1.0

    return pd.Series(labels, index=df.index)


def labels_trends_sell(
    df: pd.DataFrame,
    min_loss: float = 0.01,
    horizon: int = DEFAULT_HORIZON,
    min_gain: Optional[float] = None,  # Ignored for sell functions
) -> pd.Series:
    """
    Generate sell signals based on various trends

    NOTE: this is very specific to this architecture, these are not 'normal' indicators
    """
    close = _safe_close(df)
    n = len(close)
    labels = np.zeros(n, dtype=float)

    # Get indicator values (handle missing columns gracefully)
    flow = df.get("flow", pd.Series(np.zeros(n)))
    regime = df.get("regime", pd.Series(np.zeros(n)))
    risk = df.get("risk", pd.Series(np.zeros(n)))
    momentum = df.get("momentum", pd.Series(np.zeros(n)))

    # Convert to numpy arrays, handling NaN
    flow = np.asarray(flow, dtype=float)
    regime = np.asarray(regime, dtype=float)
    risk = np.asarray(risk, dtype=float)
    momentum = np.asarray(momentum, dtype=float)

    # Replace NaN with neutral values (0)
    flow = np.nan_to_num(flow, nan=0.0)
    regime = np.nan_to_num(regime, nan=0.0)
    risk = np.nan_to_num(risk, nan=0.0)
    momentum = np.nan_to_num(momentum, nan=0.0)

    # Calculate future loss for validation
    min_future = _rolling_min_forward(close, horizon)
    future_loss = (close - min_future) / close

    # Sell signals: Overbought indicators + future loss validation
    # Use strict equality to avoid overlap (mutually exclusive with buy conditions)
    num_votes = np.zeros(n, dtype=int)
    num_votes += (flow == 2).astype(int)  # INCREASE
    num_votes += (regime == 2).astype(int)  # BULL
    num_votes += (risk == 2).astype(
        int
    )  # HIGH (overbought conditions often have higher risk)
    num_votes += (momentum == 2).astype(int)  # POSITIVE
    sell_trend_mask = num_votes >= 2  # Require 2 out of 4 conditions

    # Sell: enough indicators agree AND future loss meets threshold
    sell_mask = (sell_trend_mask) & (future_loss >= min_loss)
    sell_mask = np.where(np.isnan(future_loss), False, sell_mask)
    labels[sell_mask] = 1.0

    return pd.Series(labels, index=df.index)


def labels_optimal_signals(
    df: pd.DataFrame,
    horizon: int = DEFAULT_HORIZON,
    min_threshold: float = 0.5,
    min_gain: Optional[float] = None,
    min_loss: Optional[float] = None,  # Ignored for buy functions
) -> pd.Series:
    """
    Generate optimal buy signals based on risk-adjusted return.

    For each timestep, looks ahead N candles/periods and calculates:
    - Risk-adjusted return (Sharpe-like: return / volatility)
    - Maximum favorable excursion (MFE) for buying
    - Determines if buy is optimal based on risk-adjusted return

    Args:
        df: DataFrame with price data (lookahead OK for training)
        horizon: Number of candles/periods to look ahead (default: 64)
        min_threshold: Minimum risk-adjusted return ratio to signal (default: 0.5)
        min_gain: Optional minimum absolute gain required (overrides risk-adjusted check)

    Returns:
        Series of 0/1 buy signals
    """
    close = _safe_close(df)
    high = np.asarray(df["high"], dtype=float)
    low = np.asarray(df["low"], dtype=float)
    n = len(close)
    labels = np.zeros(n, dtype=float)

    epsilon = 1e-8

    for t in range(n):
        end_idx = min(n, t + horizon + 1)
        if t + 1 >= end_idx:
            continue

        # Get future price data
        future_close = close[t + 1 : end_idx]
        future_high = high[t + 1 : end_idx]
        future_low = low[t + 1 : end_idx]
        current_price = close[t]

        # Calculate price changes (returns)
        future_returns = (future_close - current_price) / current_price
        future_high_returns = (future_high - current_price) / current_price
        future_low_returns = (current_price - future_low) / current_price

        # Calculate volatility as std of returns
        if len(future_returns) > 1:
            volatility = np.std(future_returns) + epsilon
        else:
            volatility = epsilon

        # Buy scenario: maximum favorable excursion (MFE) for buying
        buy_max_return = np.max(future_high_returns)
        buy_risk_adjusted = buy_max_return / volatility if volatility > epsilon else 0.0

        # Sell scenario: maximum favorable excursion (MFE) for selling
        sell_max_return = np.max(future_low_returns)
        sell_risk_adjusted = (
            sell_max_return / volatility if volatility > epsilon else 0.0
        )

        # Determine if buy is optimal
        # Buy if: buy risk-adjusted > sell risk-adjusted AND buy risk-adjusted > threshold
        # Also check min_gain if specified
        # IMPORTANT: Don't signal buy if sell would also be optimal (prioritize sell)
        buy_optimal = (
            buy_risk_adjusted > sell_risk_adjusted and buy_risk_adjusted > min_threshold
        )

        # Check if sell would also be optimal - if so, don't signal buy
        sell_optimal = (
            sell_risk_adjusted > buy_risk_adjusted
            and sell_risk_adjusted > min_threshold
        )

        if min_gain is not None:
            buy_optimal = buy_optimal and (buy_max_return >= min_gain)

        # Only signal buy if buy is optimal AND sell is not optimal
        if buy_optimal and not sell_optimal:
            labels[t] = 1.0

    return pd.Series(labels, index=df.index)


def labels_optimal_signals_sell(
    df: pd.DataFrame,
    horizon: int = DEFAULT_HORIZON,
    min_threshold: float = 0.5,
    min_loss: Optional[float] = None,
    min_gain: Optional[float] = None,  # Ignored for sell functions
) -> pd.Series:
    """
    Generate optimal sell signals based on risk-adjusted return.

    For each timestep, looks ahead N candles/periods and calculates:
    - Risk-adjusted return (Sharpe-like: return / volatility)
    - Maximum favorable excursion (MFE) for selling
    - Determines if sell is optimal based on risk-adjusted return

    Args:
        df: DataFrame with price data (lookahead OK for training)
        horizon: Number of candles/periods to look ahead (default: 64)
        min_threshold: Minimum risk-adjusted return ratio to signal (default: 0.5)
        min_loss: Optional minimum absolute loss required (overrides risk-adjusted check)

    Returns:
        Series of 0/1 sell signals
    """
    close = _safe_close(df)
    high = np.asarray(df["high"], dtype=float)
    low = np.asarray(df["low"], dtype=float)
    n = len(close)
    labels = np.zeros(n, dtype=float)

    epsilon = 1e-8

    for t in range(n):
        end_idx = min(n, t + horizon + 1)
        if t + 1 >= end_idx:
            continue

        # Get future price data
        future_close = close[t + 1 : end_idx]
        future_high = high[t + 1 : end_idx]
        future_low = low[t + 1 : end_idx]
        current_price = close[t]

        # Calculate price changes (returns)
        future_returns = (future_close - current_price) / current_price
        future_high_returns = (future_high - current_price) / current_price
        future_low_returns = (current_price - future_low) / current_price

        # Calculate volatility as std of returns
        if len(future_returns) > 1:
            volatility = np.std(future_returns) + epsilon
        else:
            volatility = epsilon

        # Buy scenario: maximum favorable excursion (MFE) for buying
        buy_max_return = np.max(future_high_returns)
        buy_risk_adjusted = buy_max_return / volatility if volatility > epsilon else 0.0

        # Sell scenario: maximum favorable excursion (MFE) for selling
        sell_max_return = np.max(future_low_returns)
        sell_risk_adjusted = (
            sell_max_return / volatility if volatility > epsilon else 0.0
        )

        # Determine if sell is optimal
        # Sell if: sell risk-adjusted > buy risk-adjusted AND sell risk-adjusted > threshold
        # Also check min_loss if specified
        sell_optimal = (
            sell_risk_adjusted > buy_risk_adjusted
            and sell_risk_adjusted > min_threshold
        )

        if min_loss is not None:
            sell_optimal = sell_optimal and (sell_max_return >= min_loss)

        if sell_optimal:
            labels[t] = 1.0

    return pd.Series(labels, index=df.index)


def labels_future_sharpe(
    df: pd.DataFrame,
    horizon: int = DEFAULT_HORIZON,
    min_sharpe: float = 0.5,
    min_gain: Optional[float] = None,
    min_loss: Optional[float] = None,  # Ignored for buy functions
) -> pd.Series:
    """
    Generate buy signals based on future Sharpe ratio.

    For each timestep, looks ahead N candles/periods and calculates:
    - Mean future return
    - Standard deviation of future returns
    - Sharpe ratio = mean_return / std_dev
    - Compares buy vs sell Sharpe ratios and signals the better option

    Args:
        df: DataFrame with price data (lookahead OK for training)
        horizon: Number of candles/periods to look ahead (default: 64)
        min_sharpe: Minimum Sharpe ratio to signal (default: 0.5)
        min_gain: Optional minimum absolute gain required

    Returns:
        Series of 0/1 buy signals
    """
    close = _safe_close(df)
    high = np.asarray(df["high"], dtype=float)
    low = np.asarray(df["low"], dtype=float)
    n = len(close)
    labels = np.zeros(n, dtype=float)

    epsilon = 1e-8

    for t in range(n):
        end_idx = min(n, t + horizon + 1)
        if t + 1 >= end_idx:
            continue

        current_price = close[t]
        future_high = high[t + 1 : end_idx]
        future_low = low[t + 1 : end_idx]

        # Calculate returns for buy scenario (using high prices)
        buy_returns = (future_high - current_price) / current_price

        # Calculate returns for sell scenario (using low prices)
        sell_returns = (current_price - future_low) / current_price

        # Buy Sharpe: mean return / std dev
        if len(buy_returns) > 1:
            buy_mean = np.mean(buy_returns)
            buy_std = np.std(buy_returns) + epsilon
            buy_sharpe = buy_mean / buy_std if buy_std > epsilon else 0.0
        else:
            buy_sharpe = 0.0

        # Sell Sharpe: mean return / std dev
        if len(sell_returns) > 1:
            sell_mean = np.mean(sell_returns)
            sell_std = np.std(sell_returns) + epsilon
            sell_sharpe = sell_mean / sell_std if sell_std > epsilon else 0.0
        else:
            sell_sharpe = 0.0

        # Signal buy if buy Sharpe > sell Sharpe and exceeds threshold
        # IMPORTANT: Don't signal buy if sell would also be optimal (prioritize sell)
        buy_optimal = buy_sharpe > sell_sharpe and buy_sharpe >= min_sharpe

        # Check if sell would also be optimal - if so, don't signal buy
        sell_optimal = sell_sharpe > buy_sharpe and sell_sharpe >= min_sharpe

        if min_gain is not None:
            buy_max_return = np.max(buy_returns)
            buy_optimal = buy_optimal and (buy_max_return >= min_gain)

        # Only signal buy if buy is optimal AND sell is not optimal
        if buy_optimal and not sell_optimal:
            labels[t] = 1.0

        # Use ADX to filter out sideays markets
        adx_scaled = df.get("adx_scaled", pd.Series(np.zeros(n)))
        labels = np.where(adx_scaled > -0.2, labels, 0)

    return pd.Series(labels, index=df.index)


def labels_future_sharpe_sell(
    df: pd.DataFrame,
    horizon: int = DEFAULT_HORIZON,
    min_sharpe: float = 0.5,
    min_loss: Optional[float] = None,
    min_gain: Optional[float] = None,  # Ignored for sell functions
) -> pd.Series:
    """
    Generate sell signals based on future Sharpe ratio.

    For each timestep, looks ahead N candles/periods and calculates:
    - Mean future return
    - Standard deviation of future returns
    - Sharpe ratio = mean_return / std_dev
    - Compares buy vs sell Sharpe ratios and signals the better option

    Args:
        df: DataFrame with price data (lookahead OK for training)
        horizon: Number of candles/periods to look ahead (default: 64)
        min_sharpe: Minimum Sharpe ratio to signal (default: 0.5)
        min_loss: Optional minimum absolute loss required

    Returns:
        Series of 0/1 sell signals
    """
    close = _safe_close(df)
    high = np.asarray(df["high"], dtype=float)
    low = np.asarray(df["low"], dtype=float)
    n = len(close)
    labels = np.zeros(n, dtype=float)

    epsilon = 1e-8

    for t in range(n):
        end_idx = min(n, t + horizon + 1)
        if t + 1 >= end_idx:
            continue

        current_price = close[t]
        future_high = high[t + 1 : end_idx]
        future_low = low[t + 1 : end_idx]

        # Calculate returns for buy scenario (using high prices)
        buy_returns = (future_high - current_price) / current_price

        # Calculate returns for sell scenario (using low prices)
        sell_returns = (current_price - future_low) / current_price

        # Buy Sharpe: mean return / std dev
        if len(buy_returns) > 1:
            buy_mean = np.mean(buy_returns)
            buy_std = np.std(buy_returns) + epsilon
            buy_sharpe = buy_mean / buy_std if buy_std > epsilon else 0.0
        else:
            buy_sharpe = 0.0

        # Sell Sharpe: mean return / std dev
        if len(sell_returns) > 1:
            sell_mean = np.mean(sell_returns)
            sell_std = np.std(sell_returns) + epsilon
            sell_sharpe = sell_mean / sell_std if sell_std > epsilon else 0.0
        else:
            sell_sharpe = 0.0

        # Signal sell if sell Sharpe > buy Sharpe and exceeds threshold
        sell_optimal = sell_sharpe > buy_sharpe and sell_sharpe >= min_sharpe

        if min_loss is not None:
            sell_max_return = np.max(sell_returns)
            sell_optimal = sell_optimal and (sell_max_return >= min_loss)

        if sell_optimal:
            labels[t] = 1.0

        # Use ADX to filter out sideays markets
        adx_scaled = df.get("adx_scaled", pd.Series(np.zeros(n)))
        labels = np.where(adx_scaled > -0.2, labels, 0)

    return pd.Series(labels, index=df.index)


def labels_future_sortino(
    df: pd.DataFrame,
    horizon: int = DEFAULT_HORIZON,
    min_sortino: float = 0.5,
    min_gain: Optional[float] = None,
    min_loss: Optional[float] = None,  # Ignored for buy functions
) -> pd.Series:
    """
    Generate buy signals based on future Sortino ratio.

    Sortino ratio is similar to Sharpe but only penalizes downside volatility.
    Sortino = mean_return / downside_deviation

    For each timestep, looks ahead N candles/periods and calculates:
    - Mean future return
    - Downside deviation (std dev of negative returns only)
    - Sortino ratio = mean_return / downside_deviation
    - Compares buy vs sell Sortino ratios and signals the better option

    Args:
        df: DataFrame with price data (lookahead OK for training)
        horizon: Number of candles/periods to look ahead (default: 64)
        min_sortino: Minimum Sortino ratio to signal (default: 0.5)
        min_gain: Optional minimum absolute gain required

    Returns:
        Series of 0/1 buy signals
    """
    close = _safe_close(df)
    high = np.asarray(df["high"], dtype=float)
    low = np.asarray(df["low"], dtype=float)
    n = len(close)
    labels = np.zeros(n, dtype=float)

    epsilon = 1e-8

    for t in range(n):
        end_idx = min(n, t + horizon + 1)
        if t + 1 >= end_idx:
            continue

        current_price = close[t]
        future_high = high[t + 1 : end_idx]
        future_low = low[t + 1 : end_idx]

        # Calculate returns for buy scenario (using high prices)
        buy_returns = (future_high - current_price) / current_price

        # Calculate returns for sell scenario (using low prices)
        sell_returns = (current_price - future_low) / current_price

        # Buy Sortino: mean return / downside deviation
        if len(buy_returns) > 1:
            buy_mean = np.mean(buy_returns)
            # Downside deviation: std dev of negative returns only
            buy_downside = buy_returns[buy_returns < 0]
            if len(buy_downside) > 0:
                buy_downside_dev = np.std(buy_downside) + epsilon
            else:
                buy_downside_dev = epsilon  # No downside = perfect
            buy_sortino = (
                buy_mean / buy_downside_dev if buy_downside_dev > epsilon else 0.0
            )
        else:
            buy_sortino = 0.0

        # Sell Sortino: mean return / downside deviation
        if len(sell_returns) > 1:
            sell_mean = np.mean(sell_returns)
            # Downside deviation: std dev of negative returns only
            sell_downside = sell_returns[sell_returns < 0]
            if len(sell_downside) > 0:
                sell_downside_dev = np.std(sell_downside) + epsilon
            else:
                sell_downside_dev = epsilon  # No downside = perfect
            sell_sortino = (
                sell_mean / sell_downside_dev if sell_downside_dev > epsilon else 0.0
            )
        else:
            sell_sortino = 0.0

        # Signal buy if buy Sortino > sell Sortino and exceeds threshold
        # IMPORTANT: Don't signal buy if sell would also be optimal (prioritize sell)
        buy_optimal = buy_sortino > sell_sortino and buy_sortino >= min_sortino

        # Check if sell would also be optimal - if so, don't signal buy
        sell_optimal = sell_sortino > buy_sortino and sell_sortino >= min_sortino

        if min_gain is not None:
            buy_max_return = np.max(buy_returns)
            buy_optimal = buy_optimal and (buy_max_return >= min_gain)

        # Only signal buy if buy is optimal AND sell is not optimal
        if buy_optimal and not sell_optimal:
            labels[t] = 1.0

        # Use ADX to filter out sideays markets
        adx_scaled = df.get("adx_scaled", pd.Series(np.zeros(n)))
        labels = np.where(adx_scaled > -0.2, labels, 0)

    return pd.Series(labels, index=df.index)


def labels_future_sortino_sell(
    df: pd.DataFrame,
    horizon: int = DEFAULT_HORIZON,
    min_sortino: float = 0.5,
    min_loss: Optional[float] = None,
    min_gain: Optional[float] = None,  # Ignored for sell functions
) -> pd.Series:
    """
    Generate sell signals based on future Sortino ratio.

    Sortino ratio is similar to Sharpe but only penalizes downside volatility.
    Sortino = mean_return / downside_deviation

    Args:
        df: DataFrame with price data (lookahead OK for training)
        horizon: Number of candles/periods to look ahead (default: 64)
        min_sortino: Minimum Sortino ratio to signal (default: 0.5)
        min_loss: Optional minimum absolute loss required

    Returns:
        Series of 0/1 sell signals
    """
    close = _safe_close(df)
    high = np.asarray(df["high"], dtype=float)
    low = np.asarray(df["low"], dtype=float)
    n = len(close)
    labels = np.zeros(n, dtype=float)

    epsilon = 1e-8

    for t in range(n):
        end_idx = min(n, t + horizon + 1)
        if t + 1 >= end_idx:
            continue

        current_price = close[t]
        future_high = high[t + 1 : end_idx]
        future_low = low[t + 1 : end_idx]

        # Calculate returns for buy scenario (using high prices)
        buy_returns = (future_high - current_price) / current_price

        # Calculate returns for sell scenario (using low prices)
        sell_returns = (current_price - future_low) / current_price

        # Buy Sortino: mean return / downside deviation
        if len(buy_returns) > 1:
            buy_mean = np.mean(buy_returns)
            buy_downside = buy_returns[buy_returns < 0]
            if len(buy_downside) > 0:
                buy_downside_dev = np.std(buy_downside) + epsilon
            else:
                buy_downside_dev = epsilon
            buy_sortino = (
                buy_mean / buy_downside_dev if buy_downside_dev > epsilon else 0.0
            )
        else:
            buy_sortino = 0.0

        # Sell Sortino: mean return / downside deviation
        if len(sell_returns) > 1:
            sell_mean = np.mean(sell_returns)
            sell_downside = sell_returns[sell_returns < 0]
            if len(sell_downside) > 0:
                sell_downside_dev = np.std(sell_downside) + epsilon
            else:
                sell_downside_dev = epsilon
            sell_sortino = (
                sell_mean / sell_downside_dev if sell_downside_dev > epsilon else 0.0
            )
        else:
            sell_sortino = 0.0

        # Signal sell if sell Sortino > buy Sortino and exceeds threshold
        sell_optimal = sell_sortino > buy_sortino and sell_sortino >= min_sortino

        if min_loss is not None:
            sell_max_return = np.max(sell_returns)
            sell_optimal = sell_optimal and (sell_max_return >= min_loss)

        if sell_optimal:
            labels[t] = 1.0

        # Use ADX to filter out sideays markets
        adx_scaled = df.get("adx_scaled", pd.Series(np.zeros(n)))
        labels = np.where(adx_scaled > -0.2, labels, 0)

    return pd.Series(labels, index=df.index)


def labels_future_expectancy(
    df: pd.DataFrame,
    horizon: int = DEFAULT_HORIZON,
    min_expectancy: float = 0.01,
    min_gain: Optional[float] = None,
    min_loss: Optional[float] = None,  # Ignored for buy functions
) -> pd.Series:
    """
    Generate buy signals based on future expectancy.

    Expectancy = (Win rate × Average win) - (Loss rate × Average loss)
    Measures expected value per trade.

    For each timestep, looks ahead N candles/periods and calculates:
    - Win rate and average win for buy scenario
    - Loss rate and average loss for buy scenario
    - Buy expectancy = (win_rate × avg_win) - (loss_rate × avg_loss)
    - Same for sell scenario
    - Signals the option with higher expectancy

    Args:
        df: DataFrame with price data (lookahead OK for training)
        horizon: Number of candles/periods to look ahead (default: 64)
        min_expectancy: Minimum expectancy to signal (default: 0.01 = 1%)
        min_gain: Optional minimum absolute gain required

    Returns:
        Series of 0/1 buy signals
    """
    close = _safe_close(df)
    high = np.asarray(df["high"], dtype=float)
    low = np.asarray(df["low"], dtype=float)
    n = len(close)
    labels = np.zeros(n, dtype=float)

    for t in range(n):
        end_idx = min(n, t + horizon + 1)
        if t + 1 >= end_idx:
            continue

        current_price = close[t]
        future_high = high[t + 1 : end_idx]
        future_low = low[t + 1 : end_idx]

        # Calculate returns for buy scenario (using high prices)
        buy_returns = (future_high - current_price) / current_price

        # Calculate returns for sell scenario (using low prices)
        sell_returns = (current_price - future_low) / current_price

        # Buy expectancy
        if len(buy_returns) > 0:
            buy_wins = buy_returns[buy_returns > 0]
            buy_losses = buy_returns[buy_returns <= 0]
            buy_win_rate = (
                len(buy_wins) / len(buy_returns) if len(buy_returns) > 0 else 0.0
            )
            buy_loss_rate = (
                len(buy_losses) / len(buy_returns) if len(buy_returns) > 0 else 0.0
            )
            buy_avg_win = np.mean(buy_wins) if len(buy_wins) > 0 else 0.0
            buy_avg_loss = np.abs(np.mean(buy_losses)) if len(buy_losses) > 0 else 0.0
            buy_expectancy = (buy_win_rate * buy_avg_win) - (
                buy_loss_rate * buy_avg_loss
            )
        else:
            buy_expectancy = 0.0

        # Sell expectancy
        if len(sell_returns) > 0:
            sell_wins = sell_returns[sell_returns > 0]
            sell_losses = sell_returns[sell_returns <= 0]
            sell_win_rate = (
                len(sell_wins) / len(sell_returns) if len(sell_returns) > 0 else 0.0
            )
            sell_loss_rate = (
                len(sell_losses) / len(sell_returns) if len(sell_returns) > 0 else 0.0
            )
            sell_avg_win = np.mean(sell_wins) if len(sell_wins) > 0 else 0.0
            sell_avg_loss = (
                np.abs(np.mean(sell_losses)) if len(sell_losses) > 0 else 0.0
            )
            sell_expectancy = (sell_win_rate * sell_avg_win) - (
                sell_loss_rate * sell_avg_loss
            )
        else:
            sell_expectancy = 0.0

        # Signal buy if buy expectancy > sell expectancy and exceeds threshold
        # IMPORTANT: Don't signal buy if sell would also be optimal (prioritize sell)
        buy_optimal = (
            buy_expectancy > sell_expectancy and buy_expectancy >= min_expectancy
        )

        # Check if sell would also be optimal - if so, don't signal buy
        sell_optimal = (
            sell_expectancy > buy_expectancy and sell_expectancy >= min_expectancy
        )

        if min_gain is not None:
            buy_max_return = np.max(buy_returns)
            buy_optimal = buy_optimal and (buy_max_return >= min_gain)

        # Only signal buy if buy is optimal AND sell is not optimal
        if buy_optimal and not sell_optimal:
            labels[t] = 1.0

        adx_scaled = df.get("adx_scaled", pd.Series(np.zeros(n)))
        labels = np.where(adx_scaled > -0.2, labels, 0)

    return pd.Series(labels, index=df.index)


def labels_future_expectancy_sell(
    df: pd.DataFrame,
    horizon: int = DEFAULT_HORIZON,
    min_expectancy: float = 0.01,
    min_loss: Optional[float] = None,
    min_gain: Optional[float] = None,  # Ignored for sell functions
) -> pd.Series:
    """
    Generate sell signals based on future expectancy.

    Expectancy = (Win rate × Average win) - (Loss rate × Average loss)
    Measures expected value per trade.

    Args:
        df: DataFrame with price data (lookahead OK for training)
        horizon: Number of candles/periods to look ahead (default: 64)
        min_expectancy: Minimum expectancy to signal (default: 0.01 = 1%)
        min_loss: Optional minimum absolute loss required

    Returns:
        Series of 0/1 sell signals
    """
    close = _safe_close(df)
    high = np.asarray(df["high"], dtype=float)
    low = np.asarray(df["low"], dtype=float)
    n = len(close)
    labels = np.zeros(n, dtype=float)

    for t in range(n):
        end_idx = min(n, t + horizon + 1)
        if t + 1 >= end_idx:
            continue

        current_price = close[t]
        future_high = high[t + 1 : end_idx]
        future_low = low[t + 1 : end_idx]

        # Calculate returns for buy scenario (using high prices)
        buy_returns = (future_high - current_price) / current_price

        # Calculate returns for sell scenario (using low prices)
        sell_returns = (current_price - future_low) / current_price

        # Buy expectancy
        if len(buy_returns) > 0:
            buy_wins = buy_returns[buy_returns > 0]
            buy_losses = buy_returns[buy_returns <= 0]
            buy_win_rate = (
                len(buy_wins) / len(buy_returns) if len(buy_returns) > 0 else 0.0
            )
            buy_loss_rate = (
                len(buy_losses) / len(buy_returns) if len(buy_returns) > 0 else 0.0
            )
            buy_avg_win = np.mean(buy_wins) if len(buy_wins) > 0 else 0.0
            buy_avg_loss = np.abs(np.mean(buy_losses)) if len(buy_losses) > 0 else 0.0
            buy_expectancy = (buy_win_rate * buy_avg_win) - (
                buy_loss_rate * buy_avg_loss
            )
        else:
            buy_expectancy = 0.0

        # Sell expectancy
        if len(sell_returns) > 0:
            sell_wins = sell_returns[sell_returns > 0]
            sell_losses = sell_returns[sell_returns <= 0]
            sell_win_rate = (
                len(sell_wins) / len(sell_returns) if len(sell_returns) > 0 else 0.0
            )
            sell_loss_rate = (
                len(sell_losses) / len(sell_returns) if len(sell_returns) > 0 else 0.0
            )
            sell_avg_win = np.mean(sell_wins) if len(sell_wins) > 0 else 0.0
            sell_avg_loss = (
                np.abs(np.mean(sell_losses)) if len(sell_losses) > 0 else 0.0
            )
            sell_expectancy = (sell_win_rate * sell_avg_win) - (
                sell_loss_rate * sell_avg_loss
            )
        else:
            sell_expectancy = 0.0

        # Signal sell if sell expectancy > buy expectancy and exceeds threshold
        sell_optimal = (
            sell_expectancy > buy_expectancy and sell_expectancy >= min_expectancy
        )

        if min_loss is not None:
            sell_max_return = np.max(sell_returns)
            sell_optimal = sell_optimal and (sell_max_return >= min_loss)

        if sell_optimal:
            labels[t] = 1.0

        # Use ADX to filter out sideays markets
        adx_scaled = df.get("adx_scaled", pd.Series(np.zeros(n)))
        labels = np.where(adx_scaled > -0.2, labels, 0)

    return pd.Series(labels, index=df.index)


def labels_local_extrema(
    df: pd.DataFrame,
    horizon: int = DEFAULT_HORIZON,
    min_gain: Optional[float] = None,
    min_loss: Optional[float] = None,  # Ignored for buy functions
) -> pd.Series:
    """
    Generate buy signals based on local minima in forward-looking window.

    For each timestep, looks ahead N candles/periods and checks if the current
    closing price is the lowest in that window. If so, signals a buy.

    This identifies local minima (good buy points) by looking ahead.

    Args:
        df: DataFrame with price data (lookahead OK for training)
        horizon: Number of candles/periods to look ahead (default: 72)
        min_gain: Optional minimum gain required from current price to max in window

    Returns:
        Series of 0/1 buy signals
    """
    close = _safe_close(df)
    n = len(close)
    labels = np.zeros(n, dtype=float)

    for t in range(n):
        end_idx = min(n, t + horizon + 1)
        if t + 1 >= end_idx:
            continue

        # Get forward window
        forward_window = close[t:end_idx]

        # Check if current price is the minimum in the forward window
        is_min = close[t] == np.min(forward_window)
        # Check if current price is also the maximum (would conflict with sell)
        is_max = close[t] == np.max(forward_window)
        # Require that the first minimum occurs before the first maximum in the window
        min_idx = int(np.argmin(forward_window))
        max_idx = int(np.argmax(forward_window))

        # Only signal buy if it's a minimum AND not a maximum (prioritize sell)
        if is_min and not is_max and (min_idx < max_idx):
            # Optional: also check if there's sufficient gain potential
            if min_gain is not None:
                max_in_window = np.max(forward_window)
                potential_gain = (max_in_window - close[t]) / close[t]
                if potential_gain >= min_gain:
                    labels[t] = 1.0
            else:
                labels[t] = 1.0

    return pd.Series(labels, index=df.index)


def labels_local_extrema_sell(
    df: pd.DataFrame,
    horizon: int = DEFAULT_HORIZON,
    min_loss: Optional[float] = None,
    min_gain: Optional[float] = None,  # Ignored for sell functions
) -> pd.Series:
    """
    Generate sell signals based on local maxima in forward-looking window.

    For each timestep, looks ahead N candles/periods and checks if the current
    closing price is the highest in that window. If so, signals a sell.

    This identifies local maxima (good sell points) by looking ahead.

    Args:
        df: DataFrame with price data (lookahead OK for training)
        horizon: Number of candles/periods to look ahead (default: 72)
        min_loss: Optional minimum loss potential from current price to min in window

    Returns:
        Series of 0/1 sell signals
    """
    close = _safe_close(df)
    n = len(close)
    labels = np.zeros(n, dtype=float)

    for t in range(n):
        end_idx = min(n, t + horizon + 1)
        if t + 1 >= end_idx:
            continue

        # Get forward window
        forward_window = close[t:end_idx]

        # Check if current price is the maximum in the forward window
        is_max = close[t] == np.max(forward_window)
        is_min = close[t] == np.min(forward_window)
        max_idx = int(np.argmax(forward_window))
        min_idx = int(np.argmin(forward_window))
        if is_max and not is_min and (max_idx < min_idx):
            # Optional: also check if there's sufficient loss potential
            if min_loss is not None:
                min_in_window = np.min(forward_window)
                potential_loss = (close[t] - min_in_window) / close[t]
                if potential_loss >= min_loss:
                    labels[t] = 1.0
            else:
                labels[t] = 1.0

    return pd.Series(labels, index=df.index)


def _dwt_smooth(data: np.ndarray) -> np.ndarray:
    """Apply DWT smoothing to the data (zero-lag approximation)."""
    if not HAS_PYWT:
        # Fallback to simple moving average if pywt not available
        return (
            pd.Series(data)
            .rolling(window=5, center=True)
            .mean()
            .fillna(method="bfill")
            .fillna(method="ffill")
            .to_numpy()
        )

    wavelet = "db4"
    level = 1  # Reduced from 2 to 1 for less smoothing
    threshold = 0.2  # Reduced from 0.4 to 0.2 for less smoothing

    # Perform the multilevel DWT
    coeffs = pywt.wavedec(data, wavelet, level=level, mode="per")

    # Apply soft thresholding to the coefficients
    thresh = threshold * np.nanmax(data)
    coeffs[1:] = [pywt.threshold(c, thresh, "soft") for c in coeffs[1:]]

    # Perform the multilevel IDWT on the modified coefficients
    poly = pywt.waverec(coeffs, wavelet, mode="per")

    if len(data) != len(poly):
        dlen = min(len(data), len(poly))
        smoothed = data.copy()
        smoothed[-dlen:] = poly[-dlen:]
    else:
        smoothed = poly
    return smoothed


def labels_geometry(
    df: pd.DataFrame,
    min_gain: Optional[float] = None,
    min_loss: Optional[float] = None,  # Ignored for buy functions
    prominence: Optional[float] = None,
    horizon: Optional[int] = None,
) -> pd.Series:
    """
    Generate buy signals based on DWT-smoothed price geometry (valleys/minima).

    Uses DWT smoothing to create a zero-lag approximation of close prices, then finds
    local minima (valleys) as buy signals. Filters based on guard_metric and adx criteria
    similar to populate_entry_trend.

    Args:
        df: DataFrame with price data and indicators (guard_metric, adx must be present)
        min_gain: Optional minimum gain threshold (not used for geometry method)
        min_loss: Ignored for buy functions
        entry_guard_threshold: Threshold for guard_metric filter
            (default: -0.5, similar to entry)
        entry_adx_threshold: Threshold for adx filter (default: 50.0, similar to entry)
        prominence: Minimum prominence for peak detection (default: None = auto)
        horizon: Minimum distance between peaks (default: None = auto)

    Returns:
        Series of 0/1 buy signals
    """
    close = _safe_close(df)
    n = len(close)
    labels = np.zeros(n, dtype=float)

    # Apply DWT smoothing to create zero-lag approximation
    # smoothed = _dwt_smooth(close)
    smoothed = close

    # Find local minima (valleys) - these are buy signals
    # Invert the signal to find minima (find_peaks finds maxima)
    inverted = -smoothed

    # Set default parameters for peak detection
    if prominence is None:
        # Auto-calculate prominence as a fraction of price range
        # price_range = np.nanmax(smoothed) - np.nanmin(smoothed)
        # price_max = np.max(smoothed)
        price_mean = np.mean(smoothed)
        price_std = np.std(smoothed)
        # prominence = price_range * min_gain * 0.5
        # prominence = (price_max - price_mean) * 0.001  # 1% of price range
        prominence = min(price_std, price_mean * min_gain) * 0.5

    if horizon is None:
        # horizon = max(1, n // 100)  # At least 1% of data length apart
        horizon = 9

    # Find peaks in inverted signal (which are valleys in original)
    peaks, properties = find_peaks(
        inverted,
        prominence=prominence,
        distance=horizon // 2,
    )

    for peak_idx in peaks:
        labels[peak_idx] = 1.0
        # Also set previous 2 labels to 1.0 (handle edge cases)
        if peak_idx > 0:
            labels[peak_idx - 1] = 1.0
        if peak_idx > 1:
            labels[peak_idx - 2] = 1.0
        if peak_idx > 2:
            labels[peak_idx - 3] = 1.0

    return pd.Series(labels, index=df.index)


def labels_geometry_sell(
    df: pd.DataFrame,
    min_loss: Optional[float] = None,
    min_gain: Optional[float] = None,  # Ignored for sell functions
    prominence: Optional[float] = None,
    horizon: Optional[int] = None,
) -> pd.Series:
    """
    Generate sell signals based on DWT-smoothed price geometry (peaks/maxima).

    Uses DWT smoothing to create a zero-lag approximation of close prices, then finds
    local maxima (peaks) as sell signals. Filters based on guard_metric criteria
    similar to populate_exit_trend.

    Args:
        df: DataFrame with price data and indicators (guard_metric must be present)
        min_loss: Optional minimum loss threshold (not used for geometry method)
        min_gain: Ignored for sell functions
        exit_guard_threshold: Threshold for guard_metric filter (default: 0.5, similar to exit)
        prominence: Minimum prominence for peak detection (default: None = auto)
        horizon: Minimum distance between peaks (default: None = auto)

    Returns:
        Series of 0/1 sell signals
    """
    close = _safe_close(df)
    n = len(close)
    labels = np.zeros(n, dtype=float)

    # Apply DWT smoothing to create zero-lag approximation
    smoothed = _dwt_smooth(close)
    # smoothed = close

    # Find local maxima (peaks) - these are sell signals
    # Set default parameters for peak detection
    if prominence is None:
        # Auto-calculate prominence as a fraction of price range
        # price_range = np.nanmax(smoothed) - np.nanmin(smoothed)
        # price_max = np.max(smoothed)
        price_mean = np.mean(smoothed)
        price_std = np.std(smoothed)
        # prominence = price_range * min_gain * 0.5
        # prominence = (price_max - price_mean) * 0.001  # 1% of price range
        prominence = min(price_std * 0.5, price_mean * min_loss)

    if horizon is None:
        # horizon = max(1, n // 100)  # At least 1% of data length apart
        horizon = 9

    # Find peaks (maxima) in smoothed signal
    peaks, properties = find_peaks(
        smoothed,
        prominence=prominence,
        distance=horizon // 2,
    )

    for peak_idx in peaks:
        labels[peak_idx] = 1.0
        # Also set previous 3 labels to 1.0 (handle edge cases)
        if peak_idx > 0:
            labels[peak_idx - 1] = 1.0
        if peak_idx > 1:
            labels[peak_idx - 2] = 1.0
        if peak_idx > 2:
            labels[peak_idx - 3] = 1.0

    return pd.Series(labels, index=df.index)


def labels_breakout(
    df: pd.DataFrame,
    min_gain: float = 0.01,
    horizon: Optional[int] = DEFAULT_HORIZON,
    lookback: int = 20,
    min_loss: Optional[float] = None,  # ignored for buy (call-site compat)
) -> pd.Series:
    """
    Breakout (momentum) BUY labels — the opposite entry condition to the
    mean-reversion labelers (gbb/trends buy dips). Label a bar as Buy when the
    close breaks OUT above the prior `lookback`-bar high (a Donchian upside
    breakout) AND the move follows through (max future gain over `horizon`
    >= min_gain). A model trained on this buys STRENGTH, not weakness — so a
    BTC-uptrend / trend filter is ALIGNED with it rather than fighting it.
    """
    close = _safe_close(df)
    n = len(close)
    labels = np.zeros(n, dtype=float)
    high = np.nan_to_num(np.asarray(df.get("high", pd.Series(close)), dtype=float))
    prior_high = pd.Series(high).rolling(lookback).max().shift(1).to_numpy()
    breakout = close > prior_high                       # new lookback-bar high
    max_future = _rolling_max_forward(close, horizon)
    future_gain = (max_future - close) / close
    min_gain_val = min_gain if min_gain is not None else 0.0
    buy_mask = breakout & (future_gain >= min_gain_val)
    buy_mask = np.where(np.isnan(future_gain) | np.isnan(prior_high), False, buy_mask)
    labels[buy_mask] = 1.0
    return pd.Series(labels, index=df.index)


def labels_breakout_sell(
    df: pd.DataFrame,
    min_loss: float = 0.01,
    horizon: Optional[int] = DEFAULT_HORIZON,
    lookback: int = 20,
    min_gain: Optional[float] = None,  # ignored for sell (call-site compat)
) -> pd.Series:
    """
    Breakdown (momentum-down) SELL labels: close breaks BELOW the prior
    `lookback`-bar low AND the drop follows through (max future loss over
    `horizon` >= min_loss). Mirror of labels_breakout for the sell side.
    """
    close = _safe_close(df)
    n = len(close)
    labels = np.zeros(n, dtype=float)
    low = np.nan_to_num(np.asarray(df.get("low", pd.Series(close)), dtype=float))
    prior_low = pd.Series(low).rolling(lookback).min().shift(1).to_numpy()
    breakdown = close < prior_low                       # new lookback-bar low
    min_future = _rolling_min_forward(close, horizon)
    future_loss = (close - min_future) / close
    min_loss_val = min_loss if min_loss is not None else 0.0
    sell_mask = breakdown & (future_loss >= min_loss_val)
    sell_mask = np.where(np.isnan(future_loss) | np.isnan(prior_low), False, sell_mask)
    labels[sell_mask] = 1.0
    return pd.Series(labels, index=df.index)


def labels_breakout_tb(
    df: pd.DataFrame,
    min_gain: float = 0.01,
    horizon: Optional[int] = DEFAULT_HORIZON,
    lookback: int = 20,
    min_loss: Optional[float] = None,
) -> pd.Series:
    """Breakout with CLEAN (path-aware) follow-through — the fix for the crude
    labels_breakout target. Label Buy only when the breakout price is still up at
    the horizon END (not a transient spike) AND the drawdown during the window
    stayed shallow (<= min_gain). Removes the fakeouts that pollute the max-
    favorable-excursion target and cap its learnability/EV."""
    close = _safe_close(df)
    n = len(close)
    labels = np.zeros(n, dtype=float)
    high = np.nan_to_num(np.asarray(df.get("high", pd.Series(close)), dtype=float))
    prior_high = pd.Series(high).rolling(lookback).max().shift(1).to_numpy()
    breakout = close > prior_high
    end = pd.Series(close).shift(-horizon).to_numpy()
    fwd_end = (end - close) / close                         # return held to t+H
    min_future = _rolling_min_forward(close, horizon)
    mae = (close - min_future) / close                      # max drawdown in window
    mg = min_gain if min_gain is not None else 0.0
    buy = breakout & (fwd_end >= mg) & (mae <= mg)
    buy = np.where(np.isnan(fwd_end) | np.isnan(prior_high), False, buy)
    labels[buy] = 1.0
    return pd.Series(labels, index=df.index)


def labels_breakout_vol(
    df: pd.DataFrame,
    min_gain: float = 0.01,
    horizon: Optional[int] = DEFAULT_HORIZON,
    lookback: int = 20,
    rvol_min: float = 1.5,
    min_loss: Optional[float] = None,
) -> pd.Series:
    """Volume-confirmed breakout: new lookback-high AND above-average relative
    volume (real breakouts trade on volume; low-volume breaks tend to fail)."""
    close = _safe_close(df)
    n = len(close)
    labels = np.zeros(n, dtype=float)
    high = np.nan_to_num(np.asarray(df.get("high", pd.Series(close)), dtype=float))
    prior_high = pd.Series(high).rolling(lookback).max().shift(1).to_numpy()
    breakout = close > prior_high
    rvol = np.nan_to_num(np.asarray(df.get("rvol", pd.Series(np.ones(n))), dtype=float), nan=1.0)
    max_future = _rolling_max_forward(close, horizon)
    fg = (max_future - close) / close
    mg = min_gain if min_gain is not None else 0.0
    buy = breakout & (rvol > rvol_min) & (fg >= mg)
    buy = np.where(np.isnan(fg) | np.isnan(prior_high), False, buy)
    labels[buy] = 1.0
    return pd.Series(labels, index=df.index)


def labels_breakout_squeeze(
    df: pd.DataFrame,
    min_gain: float = 0.01,
    horizon: Optional[int] = DEFAULT_HORIZON,
    lookback: int = 20,
    squeeze_pct: float = 0.3,
    min_loss: Optional[float] = None,
) -> pd.Series:
    """Squeeze breakout: breakout out of a low-volatility consolidation (prior
    bb_width in the bottom `squeeze_pct` of its recent 100-bar range) — the
    energy-release setups that tend to follow through hardest."""
    close = _safe_close(df)
    n = len(close)
    labels = np.zeros(n, dtype=float)
    high = np.nan_to_num(np.asarray(df.get("high", pd.Series(close)), dtype=float))
    prior_high = pd.Series(high).rolling(lookback).max().shift(1).to_numpy()
    breakout = close > prior_high
    bb_width = pd.Series(np.nan_to_num(np.asarray(df.get("bb_width", pd.Series(np.zeros(n))), dtype=float)))
    bw_prev = bb_width.shift(1)
    bw_thresh = bb_width.rolling(100).quantile(squeeze_pct)
    squeeze = (bw_prev <= bw_thresh).to_numpy()
    max_future = _rolling_max_forward(close, horizon)
    fg = (max_future - close) / close
    mg = min_gain if min_gain is not None else 0.0
    buy = breakout & squeeze & (fg >= mg)
    buy = np.where(np.isnan(fg) | np.isnan(prior_high), False, buy)
    labels[buy] = 1.0
    return pd.Series(labels, index=df.index)


def labels_breakout_gbb(
    df: pd.DataFrame,
    guard_threshold: float = 0.2,
    bb_width_threshold: float = 0.035,
    min_gain: float = 0.01,
    horizon: Optional[int] = DEFAULT_HORIZON,
    min_loss: Optional[float] = None,
) -> pd.Series:
    """Inverse-gbb BREAKOUT label — the EXACT gbb structure and indicators
    (guard_metric, bb_width), but the guard condition FLIPPED: buy when
    guard_metric is HIGH (strong / overbought momentum) instead of low (a dip),
    still volatile (bb_width) and with future follow-through. gbb is highly
    learnable partly because it is defined on guard_metric — a feature the model
    sees directly; this tests whether a FEATURE-based breakout is more learnable
    than the price-based labels_breakout (i.e. is the breakout problem the label
    FORM, or the momentum PHENOMENON?).
    """
    close = _safe_close(df)
    n = len(close)
    labels = np.zeros(n, dtype=float)
    guard_metric = np.nan_to_num(np.asarray(df.get("guard_metric", pd.Series(np.zeros(n))), dtype=float), nan=0.0)
    bb_width = np.nan_to_num(np.asarray(df.get("bb_width", pd.Series(np.zeros(n))), dtype=float), nan=0.0)
    max_future = _rolling_max_forward(close, horizon)
    future_gain = (max_future - close) / close
    mg = min_gain if min_gain is not None else 0.0
    buy_mask = (guard_metric > guard_threshold) & (bb_width > bb_width_threshold) & (future_gain >= mg)
    buy_mask = np.where(np.isnan(future_gain), False, buy_mask)
    labels[buy_mask] = 1.0
    return pd.Series(labels, index=df.index)


def labels_breakout_gbb_sell(
    df: pd.DataFrame,
    guard_threshold: float = 0.2,
    bb_width_threshold: float = 0.035,
    min_loss: float = 0.01,
    horizon: Optional[int] = DEFAULT_HORIZON,
    min_gain: Optional[float] = None,
) -> pd.Series:
    """Inverse-gbb breakDOWN sell label: sell when guard_metric is LOW (weak /
    breaking down) + volatile + future follow-through DOWN. Mirror of
    labels_breakout_gbb for the sell side."""
    close = _safe_close(df)
    n = len(close)
    labels = np.zeros(n, dtype=float)
    guard_metric = np.nan_to_num(np.asarray(df.get("guard_metric", pd.Series(np.zeros(n))), dtype=float), nan=0.0)
    bb_width = np.nan_to_num(np.asarray(df.get("bb_width", pd.Series(np.zeros(n))), dtype=float), nan=0.0)
    min_future = _rolling_min_forward(close, horizon)
    future_loss = (close - min_future) / close
    ml = min_loss if min_loss is not None else 0.0
    sell_mask = (guard_metric < -guard_threshold) & (bb_width > bb_width_threshold) & (future_loss >= ml)
    sell_mask = np.where(np.isnan(future_loss), False, sell_mask)
    labels[sell_mask] = 1.0
    return pd.Series(labels, index=df.index)


def labels_breakout_consensus(
    df: pd.DataFrame,
    min_gain: float = 0.01,
    horizon: int = DEFAULT_HORIZON,
    min_indicators_buy: int = 4,
    min_loss: Optional[float] = None,
) -> pd.Series:
    """Inverse of labels_technical_indicators — a MOMENTUM/breakout indicator
    consensus. Same indicators, but the directional oscillators flipped to
    OVERBOUGHT/strong (guard_metric>=+0.2, aroonosc>=+0.2); keeps the trend/
    volatility/bullish filters (adx trending, bb_width volatile, sar/vwap
    bullish). A HIGH-THROUGHPUT feature-based breakout candidate (cf. inverse-gbb),
    aimed at fixing the sparsity that made NNNC_Breakout untestable."""
    close = _safe_close(df)
    n = len(close)
    labels = np.zeros(n, dtype=float)

    def _g(c):
        return np.nan_to_num(np.asarray(df.get(c, pd.Series(np.zeros(n))), dtype=float), nan=0.0)

    adx = _g("adx_scaled"); aroonosc = _g("aroonosc_scaled"); guard = _g("guard_metric")
    sar_ratio = _g("sar_ratio"); bb_width = _g("bb_width"); vwap_ratio = _g("vwap_ratio")
    max_future = _rolling_max_forward(close, horizon)
    future_gain = (max_future - close) / close

    votes = np.zeros(n, dtype=int)
    votes += (adx >= 0.1).astype(int)            # trending (keep)
    votes += (aroonosc >= 0.2).astype(int)       # aroon UP (flipped)
    votes += (guard >= 0.2).astype(int)          # strong/overbought (flipped)
    votes += (sar_ratio >= 0.2).astype(int)      # SAR bullish (keep)
    votes += (bb_width >= 0.017).astype(int)     # volatile (keep)
    votes += (vwap_ratio >= 0.2).astype(int)     # above VWAP / momentum (keep)
    mg = min_gain if min_gain is not None else 0.0
    buy = (votes >= min_indicators_buy) & (future_gain >= mg)
    buy = np.where(np.isnan(future_gain), False, buy)
    labels[buy] = 1.0
    return pd.Series(labels, index=df.index)


def labels_breakout_consensus_sell(
    df: pd.DataFrame,
    min_loss: float = 0.01,
    horizon: int = DEFAULT_HORIZON,
    min_indicators_sell: int = 4,
    min_gain: Optional[float] = None,
) -> pd.Series:
    """Breakdown (momentum-down) consensus — mirror of labels_breakout_consensus:
    directional oscillators flipped to WEAK (guard<=-0.2, aroonosc<=-0.2), bearish
    sar/vwap, keeping trend/volatility filters + future follow-through DOWN."""
    close = _safe_close(df)
    n = len(close)
    labels = np.zeros(n, dtype=float)

    def _g(c):
        return np.nan_to_num(np.asarray(df.get(c, pd.Series(np.zeros(n))), dtype=float), nan=0.0)

    adx = _g("adx_scaled"); aroonosc = _g("aroonosc_scaled"); guard = _g("guard_metric")
    sar_ratio = _g("sar_ratio"); bb_width = _g("bb_width"); vwap_ratio = _g("vwap_ratio")
    min_future = _rolling_min_forward(close, horizon)
    future_loss = (close - min_future) / close

    votes = np.zeros(n, dtype=int)
    votes += (adx >= 0.1).astype(int)
    votes += (aroonosc <= -0.2).astype(int)
    votes += (guard <= -0.2).astype(int)
    votes += (sar_ratio <= -0.2).astype(int)
    votes += (bb_width >= 0.017).astype(int)
    votes += (vwap_ratio <= -0.2).astype(int)
    ml = min_loss if min_loss is not None else 0.0
    sell = (votes >= min_indicators_sell) & (future_loss >= ml)
    sell = np.where(np.isnan(future_loss), False, sell)
    labels[sell] = 1.0
    return pd.Series(labels, index=df.index)


# ------------------------------
# Accessors
# ------------------------------


class LabelMethod(IntEnum):
    forward_mae = 0
    triple_barrier = 1
    quantile_future = 2
    mfe_mae_ratio = 3
    local_min_followthrough = 4
    risk_adj_future = 5
    multi_horizon_vote = 6
    technical_indicators = 7
    trends = 8
    optimal_signals = 9
    future_sharpe = 10
    future_sortino = 11
    future_expectancy = 12
    local_extrema = 13
    geometry = 14
    indicators2 = 15
    indicators3 = 16
    gbb = 17
    bands = 18
    indicators4 = 19
    breakout = 20
    breakout_tb = 21
    breakout_vol = 22
    breakout_squeeze = 23
    breakout_gbb = 24
    breakout_consensus = 25


METHODS = {
    LabelMethod.forward_mae: labels_forward_return_mae_cap,
    LabelMethod.triple_barrier: labels_triple_barrier,
    LabelMethod.quantile_future: labels_quantile_future_return,
    LabelMethod.mfe_mae_ratio: labels_mfe_mae_ratio,
    LabelMethod.local_min_followthrough: labels_local_min_followthrough,
    LabelMethod.risk_adj_future: labels_risk_adjusted_future_return,
    LabelMethod.multi_horizon_vote: labels_multi_horizon_vote,
    LabelMethod.technical_indicators: labels_technical_indicators,
    LabelMethod.trends: labels_trends,
    LabelMethod.optimal_signals: labels_optimal_signals,
    LabelMethod.future_sharpe: labels_future_sharpe,
    LabelMethod.future_sortino: labels_future_sortino,
    LabelMethod.future_expectancy: labels_future_expectancy,
    LabelMethod.local_extrema: labels_local_extrema,
    LabelMethod.geometry: labels_geometry,
    LabelMethod.indicators2: labels_indicators2,
    LabelMethod.indicators3: labels_indicators3,
    LabelMethod.gbb: labels_gbb,
    LabelMethod.bands: labels_bands,
    LabelMethod.indicators4: labels_indicators4,
    LabelMethod.breakout: labels_breakout,
    LabelMethod.breakout_tb: labels_breakout_tb,
    LabelMethod.breakout_vol: labels_breakout_vol,
    LabelMethod.breakout_squeeze: labels_breakout_squeeze,
    LabelMethod.breakout_gbb: labels_breakout_gbb,
    LabelMethod.breakout_consensus: labels_breakout_consensus,
}


def available_method_ids() -> List[int]:
    return [int(m.value) for m in LabelMethod]


def available_methods() -> Dict[int, str]:
    return {int(m.value): m.name for m in LabelMethod}


def _resolve_method_id(method: Any) -> LabelMethod:
    # Accept IntEnum, int id, or legacy str name for flexibility
    if isinstance(method, LabelMethod):
        return method
    if isinstance(method, int):
        try:
            return LabelMethod(method)
        except ValueError:
            raise ValueError(
                f"Unknown method id '{method}'. Available ids: {available_methods()}"
            )
    if isinstance(method, str):
        try:
            return LabelMethod[method]
        except KeyError:
            raise ValueError(
                f"Unknown method name '{method}'. Available: {available_methods()}"
            )
    raise ValueError(f"Unsupported method type: {type(method)}")


def get_train_buy_signals(
    df: pd.DataFrame,
    method: Any = LabelMethod.forward_mae,
    params: Optional[Dict[str, Any]] = None,
) -> pd.Series:
    """
    Accessor for buy labels using selected method (int enum id or LabelMethod).
    """

    # print(f"get_train_buy_signals: method={method}, params={params}")

    method_enum = _resolve_method_id(method)
    fn = METHODS.get(method_enum)
    if fn is None:
        raise ValueError(f"Unknown method '{method}'. Available: {available_methods()}")
    return fn(df, **(params or {})).astype(float)


def get_train_sell_signals(
    df: pd.DataFrame,
    method: Any = LabelMethod.forward_mae,
    params: Optional[Dict[str, Any]] = None,
) -> pd.Series:
    """
    Accessor for sell labels. For simplicity, invert buy logic via future MIN move:
    Reuse same methods but on inverted return perspective where applicable.
    """

    # print(f"get_train_sell_signals: method={method}, params={params}")

    method_enum = _resolve_method_id(method)
    # Simple dedicated sell analog for forward_mae; others reuse buy logic by method id
    if method_enum == LabelMethod.forward_mae:
        horizon = (params or {}).get("horizon", 72)  # Match buy function default
        min_loss = (params or {}).get("min_loss", 0.02)
        max_drawdown = (params or {}).get("max_drawdown", 0.02)
        atr_scale = (params or {}).get("atr_scale", None)

        close = _safe_close(df)
        min_future = _rolling_min_forward(close, horizon)
        mfe = (close - min_future) / close  # favorable move for sell (down)
        mae = _max_adverse_excursion(close[::-1], horizon)[
            ::-1
        ]  # reusing MAE function approx

        if atr_scale is not None:
            atr = _atr(df)
            scale = (atr / close) * atr_scale
            min_move_arr = np.maximum(min_loss, scale)
            max_dd_arr = np.maximum(max_drawdown, scale)
        else:
            min_move_arr = np.full_like(mfe, min_loss)
            max_dd_arr = np.full_like(mae, max_drawdown)

        sell = (mfe >= min_move_arr) & (mae <= max_dd_arr)
        sell = np.where(np.isnan(mfe) | np.isnan(mae), 0, sell.astype(int))
        return pd.Series(sell, index=df.index, dtype=float)

    if method_enum == LabelMethod.triple_barrier:
        # Use dedicated sell variant to honor min_loss via stop loss level
        local = dict(params or {})
        # Whitelist only supported params for sell variant
        allowed = {"horizon", "pt", "sl", "atr_scale", "min_loss"}
        local = {k: v for k, v in local.items() if k in allowed}
        return labels_triple_barrier_sell(df, **local).astype(float)

    if method_enum == LabelMethod.technical_indicators:
        # Use dedicated sell variant for technical indicators
        local = dict(params or {})
        # Whitelist only supported params for sell variant
        allowed = {
            "min_loss",
            "horizon",
            "min_indicators_sell",
        }
        local = {k: v for k, v in local.items() if k in allowed}
        return labels_technical_indicators_sell(df, **local).astype(float)

    if method_enum == LabelMethod.trends:
        # Use dedicated sell variant for technical indicators
        local = dict(params or {})
        # Whitelist only supported params for sell variant
        allowed = {
            "min_loss",
        }
        local = {k: v for k, v in local.items() if k in allowed}
        return labels_trends_sell(df, **local).astype(float)

    if method_enum == LabelMethod.optimal_signals:
        # Use dedicated sell variant for optimal signals
        local = dict(params or {})
        # Whitelist only supported params for sell variant
        allowed = {"horizon", "min_threshold", "min_loss"}
        local = {k: v for k, v in local.items() if k in allowed}
        return labels_optimal_signals_sell(df, **local).astype(float)

    if method_enum == LabelMethod.future_sharpe:
        # Use dedicated sell variant for future Sharpe
        local = dict(params or {})
        allowed = {"horizon", "min_sharpe", "min_loss"}
        local = {k: v for k, v in local.items() if k in allowed}
        return labels_future_sharpe_sell(df, **local).astype(float)

    if method_enum == LabelMethod.future_sortino:
        # Use dedicated sell variant for future Sortino
        local = dict(params or {})
        allowed = {"horizon", "min_sortino", "min_loss"}
        local = {k: v for k, v in local.items() if k in allowed}
        return labels_future_sortino_sell(df, **local).astype(float)

    if method_enum == LabelMethod.future_expectancy:
        # Use dedicated sell variant for future Expectancy
        local = dict(params or {})
        allowed = {"horizon", "min_expectancy", "min_loss"}
        local = {k: v for k, v in local.items() if k in allowed}
        return labels_future_expectancy_sell(df, **local).astype(float)

    if method_enum == LabelMethod.local_extrema:
        # Use dedicated sell variant for local extrema
        local = dict(params or {})
        allowed = {"horizon", "min_loss"}
        local = {k: v for k, v in local.items() if k in allowed}
        return labels_local_extrema_sell(df, **local).astype(float)

    if method_enum == LabelMethod.geometry:
        # Use dedicated sell variant for geometry
        local = dict(params or {})
        allowed = {"min_loss", "exit_guard_threshold", "prominence", "horizon"}
        local = {k: v for k, v in local.items() if k in allowed}
        return labels_geometry_sell(df, **local).astype(float)

    if method_enum == LabelMethod.indicators2:
        # Use dedicated sell variant for indicators2
        local = dict(params or {})
        # Whitelist only supported params for sell variant
        allowed = {
            "min_loss",
            "horizon",
            "min_indicators_sell",
        }
        local = {k: v for k, v in local.items() if k in allowed}
        return labels_indicators2_sell(df, **local).astype(float)

    if method_enum == LabelMethod.indicators3:
        # Use dedicated sell variant for indicators3
        local = dict(params or {})
        # Whitelist only supported params for sell variant
        allowed = {
            "min_loss",
            "horizon",
            "min_indicators_sell",
        }
        local = {k: v for k, v in local.items() if k in allowed}
        return labels_indicators3_sell(df, **local).astype(float)

    if method_enum == LabelMethod.indicators4:
        # Use dedicated sell variant for indicators4
        local = dict(params or {})
        allowed = {
            "min_loss",
            "horizon",
            "min_indicators_sell",
        }
        local = {k: v for k, v in local.items() if k in allowed}
        return labels_indicators4_sell(df, **local).astype(float)

    if method_enum == LabelMethod.gbb:
        # Use dedicated sell variant for gbb
        local = dict(params or {})
        allowed = {
            "guard_threshold",
            "bb_width_threshold",
            "min_gain",
            "min_loss",
            "horizon",
        }
        local = {k: v for k, v in local.items() if k in allowed}
        return labels_gbb_sell(df, **local).astype(float)

    if method_enum == LabelMethod.bands:
        # Use dedicated sell variant for bands
        local = dict(params or {})
        allowed = {"min_loss", "horizon"}
        local = {k: v for k, v in local.items() if k in allowed}
        return labels_bands_sell(df, **local).astype(float)

    if method_enum == LabelMethod.breakout:
        # Breakdown sell variant (mirror of the momentum breakout buy)
        local = dict(params or {})
        allowed = {"min_loss", "horizon", "lookback"}
        local = {k: v for k, v in local.items() if k in allowed}
        return labels_breakout_sell(df, **local).astype(float)

    if method_enum == LabelMethod.breakout_gbb:
        # Inverse-gbb breakdown sell variant
        local = dict(params or {})
        allowed = {"min_loss", "horizon", "guard_threshold", "bb_width_threshold"}
        local = {k: v for k, v in local.items() if k in allowed}
        return labels_breakout_gbb_sell(df, **local).astype(float)

    if method_enum == LabelMethod.breakout_consensus:
        # Inverse-consensus breakdown sell variant
        local = dict(params or {})
        allowed = {"min_loss", "horizon", "min_indicators_sell"}
        local = {k: v for k, v in local.items() if k in allowed}
        return labels_breakout_consensus_sell(df, **local).astype(float)

    if method_enum == LabelMethod.quantile_future:
        # Use dedicated sell variant for quantile future return
        local = dict(params or {})
        # Convert top_quantile to bottom_quantile for sell (default 0.2 = bottom 20%)
        if "top_quantile" in local:
            local["bottom_quantile"] = 1.0 - local.pop("top_quantile")
        # Rename min_gain to min_loss for sell
        if "min_gain" in local:
            local["min_loss"] = local.pop("min_gain")
        allowed = {
            "horizon",
            "bottom_quantile",
            "max_drawdown",
            "atr_scale",
            "min_loss",
        }
        local = {k: v for k, v in local.items() if k in allowed}
        return labels_quantile_future_return_sell(df, **local).astype(float)

    if method_enum == LabelMethod.local_min_followthrough:
        # Use dedicated sell variant for local min followthrough
        local = dict(params or {})
        # Rename min_gain to min_loss for sell
        if "min_gain" in local:
            local["min_loss"] = local.pop("min_gain")
        allowed = {"horizon", "window_k", "min_loss", "max_drawdown"}
        local = {k: v for k, v in local.items() if k in allowed}
        return labels_local_min_followthrough_sell(df, **local).astype(float)

    # Fallback: reuse buy accessor and allow user to specify separate params per method
    return get_train_buy_signals(df, method=method_enum, params=params).astype(float)
