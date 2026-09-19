"""Strict triple-barrier labeler tests.

The triple-barrier labeler was always meant to be the STRICT form: buy=1 only
if the profit-take (upper) barrier is touched strictly before the stop-loss
(lower) barrier; if neither barrier is touched within `horizon` bars, the
label is 0 (HOLD) -- there is no directional (drift-sign) fallback. The sell
labeler is the exact mirror. A same-bar tie (both barriers first touched on
the same future bar) is treated as NOT a signal (0) for both labelers, since
we cannot tell intrabar which barrier was actually touched first from close
prices alone.
"""

import numpy as np
import pandas as pd
import pytest

from Framework.TrainingSignals import labels_triple_barrier, labels_triple_barrier_sell


def _df(close, high=None, low=None) -> pd.DataFrame:
    close = np.asarray(close, dtype=float)
    high = np.asarray(high, dtype=float) if high is not None else close
    low = np.asarray(low, dtype=float) if low is not None else close
    return pd.DataFrame({"close": close, "high": high, "low": low})


def test_pt_hit_first_is_buy_and_not_sell():
    # entry=100, pt=0.05 (upper=105), sl=0.03 (lower=97)
    # future: 101, 106 (upper touched first, idx1), 95 (lower touched, idx2), 100, 100
    close = [100, 101, 106, 95, 100, 100]
    df = _df(close)
    buy = labels_triple_barrier(df, horizon=5, pt=0.05, sl=0.03, min_gain=None)
    sell = labels_triple_barrier_sell(df, horizon=5, pt=0.05, sl=0.03)
    assert buy.iloc[0] == 1
    assert sell.iloc[0] == 0


def test_sl_hit_first_is_not_buy_and_is_sell():
    # entry=100, pt=0.05 (upper=105), sl=0.03 (lower=97)
    # future: 101, 95 (lower touched first, idx1), 106 (upper touched, idx2), 100, 100
    close = [100, 101, 95, 106, 100, 100]
    df = _df(close)
    buy = labels_triple_barrier(df, horizon=5, pt=0.05, sl=0.03, min_gain=None)
    sell = labels_triple_barrier_sell(df, horizon=5, pt=0.05, sl=0.03)
    assert buy.iloc[0] == 0
    assert sell.iloc[0] == 1


def test_no_touch_with_upward_drift_is_hold_for_both():
    """Behaviour change: the old code fell back to the sign of the final
    return when neither barrier was touched. The strict form must return 0
    (HOLD) regardless of drift direction."""
    # entry=100, pt=0.05 (upper=105), sl=0.03 (lower=97); drifts up but never
    # reaches 105, never drops to 97.
    close = [100, 100.5, 101, 101.5, 102, 102.5]
    df = _df(close)
    buy = labels_triple_barrier(df, horizon=5, pt=0.05, sl=0.03, min_gain=None)
    sell = labels_triple_barrier_sell(df, horizon=5, pt=0.05, sl=0.03)
    assert buy.iloc[0] == 0
    assert sell.iloc[0] == 0


def test_no_touch_with_downward_drift_is_hold_for_both():
    """Behaviour change: old sell labeler fell back to 1 when the final
    return was negative even without a barrier touch. Strict form must be 0."""
    # entry=100, pt=0.05 (upper=105), sl=0.03 (lower=97); drifts down but
    # never drops to 97, never reaches 105.
    close = [100, 99.5, 99, 98.5, 98, 97.5]
    df = _df(close)
    buy = labels_triple_barrier(df, horizon=5, pt=0.05, sl=0.03, min_gain=None)
    sell = labels_triple_barrier_sell(df, horizon=5, pt=0.05, sl=0.03)
    assert buy.iloc[0] == 0
    assert sell.iloc[0] == 0


def test_same_bar_tie_is_hold_for_both():
    """Degenerate barriers (pt=sl=0 -> upper=lower=entry) let a single future
    bar touch both barriers simultaneously. That tie must resolve to 0."""
    close = [100, 100, 101, 99]
    df = _df(close)
    buy = labels_triple_barrier(df, horizon=3, pt=0.0, sl=0.0, min_gain=None)
    sell = labels_triple_barrier_sell(df, horizon=3, pt=0.0, sl=0.0)
    assert buy.iloc[0] == 0
    assert sell.iloc[0] == 0


def test_horizon_boundary_touch_at_horizon_counts():
    # entry=100, horizon=3, pt=0.05 (upper=105); touch lands exactly on the
    # last in-window bar (t + horizon).
    close = [100, 101, 102, 105]
    df = _df(close)
    buy = labels_triple_barrier(df, horizon=3, pt=0.05, sl=0.03, min_gain=None)
    assert buy.iloc[0] == 1


def test_horizon_boundary_touch_after_horizon_does_not_count():
    # Same setup, but the touch is pushed one bar past the horizon window
    # (t + horizon + 1) and must NOT be counted.
    close = [100, 101, 102, 103, 105]
    df = _df(close)
    buy = labels_triple_barrier(df, horizon=3, pt=0.05, sl=0.03, min_gain=None)
    assert buy.iloc[0] == 0


def test_min_gain_overrides_pt():
    # entry=100, explicit pt=0.02 (upper=102); future touches 103 (3%).
    # With no min_gain override, pt=0.02 alone is hit -> buy=1.
    close = [100, 101, 103, 100, 100, 100]
    df = _df(close)
    buy_no_override = labels_triple_barrier(
        df, horizon=5, pt=0.02, sl=0.10, min_gain=None
    )
    assert buy_no_override.iloc[0] == 1

    # With min_gain=0.05 overriding pt to 0.05 (upper=105), the same future
    # (max 103) never touches the raised barrier -> strict no-touch -> 0.
    buy_override = labels_triple_barrier(
        df, horizon=5, pt=0.02, sl=0.10, min_gain=0.05
    )
    assert buy_override.iloc[0] == 0


def test_atr_scale_widens_barriers():
    # Constant true range of 10 (high = close+5, low = close-5) with a flat
    # close history gives an ~constant ATR of 10. At close=200, atr_scale=1
    # widens both barriers to 10/200 = 5%, well past the explicit pt=sl=0.01.
    close = [200.0] * 4 + [201.0, 204.0, 205.0]
    high = np.asarray(close) + 5.0
    low = np.asarray(close) - 5.0
    df = _df(close, high=high, low=low)
    t = 3  # entry row; future = close[4:7] = [201, 204, 205]

    # Without ATR scaling: pt=0.01 -> upper=202; future touches 204 -> buy=1.
    buy_no_atr = labels_triple_barrier(
        df, horizon=3, pt=0.01, sl=0.01, atr_scale=None, min_gain=None
    )
    assert buy_no_atr.iloc[t] == 1

    # With ATR scaling: effective pt ~ 0.05 -> upper=210; future max is 205,
    # never touched, and lower (190) is never touched either -> buy=0.
    buy_atr = labels_triple_barrier(
        df, horizon=3, pt=0.01, sl=0.01, atr_scale=1.0, min_gain=None
    )
    assert buy_atr.iloc[t] == 0


def test_no_signal_returns_0_and_1_only():
    """Sanity: output stays binary {0, 1} and index-aligned."""
    rng = np.random.default_rng(0)
    close = 100 + np.cumsum(rng.normal(0, 0.3, size=50))
    df = _df(close)
    buy = labels_triple_barrier(df, horizon=6, pt=0.02, sl=0.02, min_gain=None)
    sell = labels_triple_barrier_sell(df, horizon=6, pt=0.02, sl=0.02)
    assert set(np.unique(buy)).issubset({0.0, 1.0})
    assert set(np.unique(sell)).issubset({0.0, 1.0})
    assert list(buy.index) == list(df.index)
    assert list(sell.index) == list(df.index)
