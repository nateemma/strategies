# pragma pylint: disable=C0103, C0114, C0115, C0116, C0301
"""`utils.ExitPolicy.custom_exit_decision` (specs/021-label-exit-coordination, FR-001's
amendment) -- the single source of truth for `BaseStrategy.custom_exit`'s profit/time
decision, now shared with `simulate_forward`.

Companion to `test_exit_policy_simulate_forward.py`: that file pins the stop/take-profit/
hold-cap state machine; this one pins the NEW branches in isolation, independent of any
price array, since `custom_exit_decision` takes `current_profit`/`age_hours` directly.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from utils.ExitPolicy import (  # noqa: E402
    CLOSE_MAX_HOLD,
    CLOSE_METRIC_OVERBOUGHT,
    CLOSE_TAKE_PROFIT,
    CLOSE_UNCLOG_12H,
    CLOSE_UNCLOG_1D,
    ExitParams,
    custom_exit_decision,
)

PARAMS = ExitParams(
    use_atr_adaptive_stoploss=False,
    take_profit=0.022,
    max_hold_days=10,
    stoploss_grace_hours=0.0,
    stoploss_grace_level=-0.15,
)


def test_metric_overbought_requires_profit_positive():
    """guard_metric > threshold alone is not enough -- current_profit must ALSO be > 0,
    matching BaseStrategy.custom_exit's nesting (`if current_profit > 0.0: ... if
    guard_metric > 0.98: ...`)."""
    assert custom_exit_decision(0.0, age_hours=1.0, guard_metric=0.99, params=PARAMS) is None
    assert custom_exit_decision(-0.01, age_hours=1.0, guard_metric=0.99, params=PARAMS) is None


def test_metric_overbought_fires_on_any_profit():
    """Unlike take_profit, metric_overbought has NO profit-magnitude threshold -- it fires
    on the smallest positive profit once guard_metric crosses 0.98."""
    assert custom_exit_decision(
        0.0001, age_hours=1.0, guard_metric=0.981, params=PARAMS
    ) == CLOSE_METRIC_OVERBOUGHT


def test_metric_overbought_precedes_take_profit():
    """Both conditions true on the same call -- metric_overbought wins, matching
    BaseStrategy.custom_exit's source order (guard_metric checked before take_profit)."""
    assert custom_exit_decision(
        0.05, age_hours=1.0, guard_metric=0.99, params=PARAMS
    ) == CLOSE_METRIC_OVERBOUGHT


def test_metric_overbought_none_guard_metric_never_fires():
    """None means 'not modelled' (consistent with rvol=None elsewhere in this module) --
    required for `simulate_forward` callers that do not supply a guard_metric array."""
    assert custom_exit_decision(
        0.05, age_hours=1.0, guard_metric=None, params=PARAMS
    ) == CLOSE_TAKE_PROFIT


def test_take_profit_in_isolation():
    assert custom_exit_decision(
        0.023, age_hours=1.0, guard_metric=0.5, params=PARAMS
    ) == CLOSE_TAKE_PROFIT
    assert custom_exit_decision(
        0.021, age_hours=1.0, guard_metric=0.5, params=PARAMS
    ) is None


def test_unclog_12h_requires_age_and_profit():
    assert custom_exit_decision(0.006, age_hours=11.9, guard_metric=0.5, params=PARAMS) is None
    assert custom_exit_decision(0.004, age_hours=12.1, guard_metric=0.5, params=PARAMS) is None
    assert custom_exit_decision(
        0.006, age_hours=12.1, guard_metric=0.5, params=PARAMS
    ) == CLOSE_UNCLOG_12H


def test_unclog_1d_requires_age_and_nonnegative_profit():
    assert custom_exit_decision(-0.001, age_hours=24.1, guard_metric=0.5, params=PARAMS) is None
    assert custom_exit_decision(
        0.0, age_hours=24.1, guard_metric=0.5, params=PARAMS
    ) == CLOSE_UNCLOG_1D


def test_max_hold_fires_regardless_of_profit_sign():
    assert custom_exit_decision(
        -0.03, age_hours=240.1, guard_metric=0.5, params=PARAMS
    ) == CLOSE_MAX_HOLD


def test_time_checks_are_not_gated_by_cexit_enable_profit_checks():
    """unclog_12h/1d/max_hold sit AFTER, not inside, the `cexit_enable_profit_checks` block
    in BaseStrategy.custom_exit -- disabling profit checks must not disable them."""
    params = ExitParams(
        use_atr_adaptive_stoploss=False, take_profit=0.022, max_hold_days=10,
        stoploss_grace_hours=0.0, stoploss_grace_level=-0.15,
        cexit_enable_profit_checks=False,
    )
    assert custom_exit_decision(
        0.006, age_hours=12.1, guard_metric=0.5, params=params
    ) == CLOSE_UNCLOG_12H
    # But metric_overbought/take_profit ARE gated by it.
    assert custom_exit_decision(0.05, age_hours=1.0, guard_metric=0.99, params=params) is None


def test_nothing_fires_when_no_condition_met():
    assert custom_exit_decision(0.001, age_hours=1.0, guard_metric=0.5, params=PARAMS) is None
