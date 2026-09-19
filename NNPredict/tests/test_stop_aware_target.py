"""Tests for NNPredictStrategy's target_mode == "stop_aware" label.

Builds a strategy instance via ``object.__new__`` (bypassing freqtrade's
IStrategy.__init__, which needs a live config/exchange) and sets only the
class attributes get_training_labels / _stop_aware_labels actually reads,
mirroring the same stop arithmetic as Framework.BaseStrategy.custom_stoploss.
"""

import numpy as np
import pandas as pd
import pytest

from NNPredict.NNPredictStrategy import NNPredictStrategy


def make_strategy(**overrides):
    strat = object.__new__(NNPredictStrategy)
    strat.HORIZON = 4
    strat.target_mode = "stop_aware"
    strat.target_max_gain = 8.0
    strat.target_max_loss = 8.0
    strat.atr_floor = 1e-3
    strat.timeframe = "1h"
    strat.use_atr_adaptive_stoploss = True
    strat.atr_stoploss_multiplier = 2.0
    strat.atr_stoploss_floor = -0.10  # loosest allowed
    strat.atr_stoploss_cap = -0.01    # tightest allowed
    strat.stoploss = -0.05
    strat.stoploss_grace_hours = 0.0
    strat.stoploss_grace_level = -0.15
    strat.dbg_verbose = False
    for k, v in overrides.items():
        setattr(strat, k, v)
    return strat


def make_df(closes, lows, atr_pct):
    n = len(closes)
    atr = atr_pct if hasattr(atr_pct, "__len__") else [atr_pct] * n
    return pd.DataFrame(
        {
            "close": closes,
            "low": lows,
            "high": closes,
            "atr_pct": atr,
        }
    )


# With atr_pct=0.02 and atr_stoploss_multiplier=2.0, the ATR-adaptive normal
# stop is -0.04 (within [floor=-0.10, cap=-0.01], so unclipped).
CAP = 8.0


def expected_label(raw, atr_pct_endpoint=0.02, cap=CAP):
    gain = raw / atr_pct_endpoint
    gain = max(min(gain, cap), -cap) / cap
    return gain


def test_dip_through_stop_then_rally_labels_as_stop():
    """(a) A dip that breaches the stop before the endpoint, followed by a
    rally to +3% at H, must be labelled as the realised stop, not the +3%
    endpoint gain."""
    strat = make_strategy()
    closes = [100, 95, 98, 100, 103]
    lows = [100, 94, 98, 100, 103]  # bar1 low=94 breaches -4% stop (96)
    df = make_df(closes, lows, 0.02)

    labels = strat.get_training_labels(df)

    assert labels[0] == pytest.approx(expected_label(-0.04), abs=1e-4)


def test_rally_without_breach_labels_as_point_return():
    """(b) No breach anywhere in the window -> label is the plain point
    return, same as "point" mode."""
    strat = make_strategy()
    closes = [100, 99, 101, 102, 105]
    lows = [100, 99, 100, 101, 104]  # never below 96
    df = make_df(closes, lows, 0.02)

    labels = strat.get_training_labels(df)

    assert labels[0] == pytest.approx(expected_label(0.05), abs=1e-4)


def test_breach_exactly_at_stop_level_counts():
    """(c) low == close[t] * (1 + stop) counts as a breach (<=)."""
    strat = make_strategy()
    closes = [100, 95, 98, 100, 103]
    lows = [100, 96.0, 98, 100, 103]  # exactly at the -4% stop level
    df = make_df(closes, lows, 0.02)

    labels = strat.get_training_labels(df)

    assert labels[0] == pytest.approx(expected_label(-0.04), abs=1e-4)


def test_breach_one_tick_above_stop_level_does_not_count():
    """(c) low one tick above the stop level does NOT breach; falls through
    to the endpoint point return."""
    strat = make_strategy()
    closes = [100, 95, 98, 100, 103]
    lows = [100, 96.01, 98, 100, 103]  # one tick above the -4% stop level
    df = make_df(closes, lows, 0.02)

    labels = strat.get_training_labels(df)

    assert labels[0] == pytest.approx(expected_label(0.03), abs=1e-4)


def test_grace_window_ignores_normal_stop_breach():
    """(d) With grace enabled, a breach of the (tighter) normal stop inside
    the grace window does NOT stop the trade out unless it also breaches the
    (wider) grace level. timeframe=1h + stoploss_grace_hours=2.0 -> only
    offset 1 (age 1h < 2h) is inside the grace window; offsets >= 2 already
    use the normal stop."""
    strat = make_strategy(
        stoploss_grace_hours=2.0,
        stoploss_grace_level=-0.20,
    )
    closes = [100, 95, 98, 99, 102]
    # bar1 (offset1, in grace): low=95 breaches normal stop (<=96) but NOT
    # the grace level (<=80) -> no stop-out.
    # bars 2-4 (normal stop, -0.04 i.e. <=96): none breach.
    lows = [100, 95, 97, 98, 101]
    df = make_df(closes, lows, 0.02)

    labels = strat.get_training_labels(df)

    # No stop-out anywhere -> falls through to the point return (+2%).
    assert labels[0] == pytest.approx(expected_label(0.02), abs=1e-4)


def test_static_stoploss_fallback_when_atr_adaptive_disabled():
    """Bonus: when use_atr_adaptive_stoploss is False, the normal stop is the
    static self.stoploss, not an ATR-derived value."""
    strat = make_strategy(use_atr_adaptive_stoploss=False, stoploss=-0.03)
    closes = [100, 96, 98, 100, 103]
    lows = [100, 96.5, 98, 100, 103]  # breaches -3% stop (<=97) but not -4%
    df = make_df(closes, lows, 0.02)

    labels = strat.get_training_labels(df)

    assert labels[0] == pytest.approx(expected_label(-0.03), abs=1e-4)


def test_point_mode_output_unchanged():
    """(e) Regression guard: target_mode == "point" must still produce
    current_gain.shift(-H).fillna(0.0), unaffected by the stop_aware dispatch
    branch added alongside it."""
    horizon = 4
    cap = 8.0
    closes = np.array([100, 101, 99, 102, 105, 103, 107, 110], dtype=np.float64)
    atr_pct = np.full(len(closes), 0.02)

    past = np.concatenate([np.full(horizon, np.nan), closes[:-horizon]])
    gain = (closes - past) / past / np.maximum(atr_pct, 1e-3)
    gain = np.nan_to_num(gain, nan=0.0, posinf=0.0, neginf=0.0)
    current_gain = (np.clip(gain, -cap, cap) / cap).astype(np.float32)

    df = pd.DataFrame({"close": closes, "current_gain": current_gain})

    strat = make_strategy(target_mode="point", HORIZON=horizon)
    labels = strat.get_training_labels(df)

    expected = (
        pd.Series(current_gain).shift(-horizon).fillna(0.0).to_numpy(dtype=np.float32)
    )
    np.testing.assert_allclose(labels, expected)
