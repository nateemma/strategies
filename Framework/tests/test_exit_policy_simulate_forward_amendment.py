# pragma pylint: disable=C0103, C0114, C0115, C0116, C0301
"""`simulate_forward`'s FR-001 amendment (specs/021-label-exit-coordination, T021 follow-up):
`open_`, `guard_metric` and `ExitParams.roi`. Companion to
`test_exit_policy_simulate_forward.py`, which pins the pre-amendment stop/take-profit/
hold-cap state machine and is left untouched by this file's cases (none of them supply
`open_`/`guard_metric`/`roi`, so that file's geometry is undisturbed).
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from utils.ExitPolicy import (  # noqa: E402
    CLOSE_METRIC_OVERBOUGHT,
    CLOSE_ROI,
    CLOSE_STOP,
    CLOSE_TAKE_PROFIT,
    CLOSE_UNCLOG_12H,
    CLOSE_UNCLOG_1D,
    ExitParams,
    simulate_forward,
)

ENTRY = 100.0
#: One bar == one hour, ten days of bars, so age-in-hours == bar index - entry index.
PARAMS = ExitParams(
    static_stoploss=-0.05,
    use_atr_adaptive_stoploss=False,
    take_profit=0.022,
    max_hold_days=10,
    timeframe_minutes=60,
    fee_per_side=0.001,
    stoploss_grace_hours=0.0,
    stoploss_grace_level=-0.15,
)
ROUND_TRIP_FEE = 2.0 * PARAMS.fee_per_side


def _flat(n: int, price: float = ENTRY):
    return [price] * n


def test_open_based_take_profit_fires_before_stop_and_before_the_legacy_high_check():
    """Real precedence (this module's docstring, citing interface.py/backtesting.py):
    custom_exit's own decision -- here, take_profit on the OPEN -- fires BEFORE the native
    stop is even checked. Bar 1's open alone clears the 2.2% threshold; its low ALSO
    breaches the stop and its high reaches neither (controls for the legacy TP path)."""
    n = 5
    close = _flat(n)
    high = _flat(n)
    low = [ENTRY] + [80.0] * (n - 1)     # breaches any plausible stop from bar 1 on
    open_ = [ENTRY] + [103.0] * (n - 1)  # 3% open -- clears take_profit

    profit, hold_bars, reason = simulate_forward(
        high, low, close, atr_pct=[0.0] * n, rvol=None, start_index=0, params=PARAMS,
        open_=open_,
    )
    assert reason == CLOSE_TAKE_PROFIT
    assert hold_bars == 1
    assert abs(profit - (0.03 - ROUND_TRIP_FEE)) < 1e-12


def test_metric_overbought_fires_via_open_and_guard_metric():
    n = 3
    close = _flat(n)
    high = _flat(n)
    low = _flat(n)
    open_ = [ENTRY, 100.5, 100.5]   # tiny positive profit -- below take_profit
    guard_metric = [0.5, 0.99, 0.99]

    profit, hold_bars, reason = simulate_forward(
        high, low, close, atr_pct=[0.0] * n, rvol=None, start_index=0, params=PARAMS,
        open_=open_, guard_metric=guard_metric,
    )
    assert reason == CLOSE_METRIC_OVERBOUGHT
    assert hold_bars == 1
    expected = (100.5 / ENTRY - 1.0) - ROUND_TRIP_FEE
    assert abs(profit - expected) < 1e-12


def test_unclog_12h_fires_via_open_before_stop_and_take_profit():
    """One bar == one hour; bar 13 is the first past the 12h boundary."""
    n = 15
    close = _flat(n)
    high = _flat(n)
    low = _flat(n)   # never breaches any stop
    open_ = list(_flat(n))
    open_[13] = 100.6   # 0.6% profit -- above unclog_12h's 0.5%, below take_profit's 2.2%

    profit, hold_bars, reason = simulate_forward(
        high, low, close, atr_pct=[0.0] * n, rvol=None, start_index=0, params=PARAMS,
        open_=open_,
    )
    assert reason == CLOSE_UNCLOG_12H
    assert hold_bars == 13


def test_unclog_1d_fires_via_open_on_zero_profit():
    """One bar == one hour; bar 24 is the first bar whose age (24h) reaches the unclog_1d
    boundary, and `open_` is flat throughout (profit exactly 0, satisfying `>= 0.0`)."""
    n = 26
    close = _flat(n)
    high = _flat(n)
    low = _flat(n)
    open_ = _flat(n)   # flat profit (0.0) at every bar

    profit, hold_bars, reason = simulate_forward(
        high, low, close, atr_pct=[0.0] * n, rvol=None, start_index=0, params=PARAMS,
        open_=open_,
    )
    assert reason == CLOSE_UNCLOG_1D
    assert hold_bars == 24


def test_native_roi_fires_on_high_after_stop_and_legacy_take_profit_checked():
    """ROI is the LOWEST-precedence of the amended exits: this bar's low does not breach
    the stop, and `open_` is NOT supplied (so the legacy high-based take_profit path IS
    active) -- but take_profit (2.2%) and ROI (5%) are set far enough apart that only ROI's
    own threshold is crossed."""
    params = ExitParams(
        static_stoploss=-0.05, use_atr_adaptive_stoploss=False, take_profit=0.30,
        max_hold_days=10, timeframe_minutes=60, fee_per_side=0.001,
        stoploss_grace_hours=0.0, stoploss_grace_level=-0.15, roi=0.05,
    )
    n = 3
    close = [ENTRY, 104.0, 104.0]
    high = [ENTRY, 96.0, 106.0]   # bar 1: no roi; bar 2: clears the 5% roi level
    low = [ENTRY, 95.5, 103.0]    # never breaches the -5% static stop (95.0)

    profit, hold_bars, reason = simulate_forward(
        high, low, close, atr_pct=[0.0] * n, rvol=None, start_index=0, params=params,
    )
    assert reason == CLOSE_ROI
    assert hold_bars == 2
    assert abs(profit - (0.05 - ROUND_TRIP_FEE)) < 1e-12


def test_roi_none_means_not_modelled():
    """The pre-amendment default: a high that clears what WOULD be an roi level, with
    `params.roi` left at its None default, must not produce a roi exit."""
    n = 3
    close = [ENTRY, 104.0, 104.0]
    high = [ENTRY, 96.0, 200.0]
    low = [ENTRY, 95.5, 103.0]
    params = ExitParams(
        static_stoploss=-0.05, use_atr_adaptive_stoploss=False, take_profit=0.30,
        max_hold_days=10, timeframe_minutes=60, fee_per_side=0.001,
        stoploss_grace_hours=0.0, stoploss_grace_level=-0.15,
    )
    assert params.roi is None

    profit, hold_bars, reason = simulate_forward(
        high, low, close, atr_pct=[0.0] * n, rvol=None, start_index=0, params=params,
    )
    assert reason != CLOSE_ROI


def test_stop_still_wins_when_open_is_supplied_but_no_custom_exit_condition_fires():
    """`open_` being present must not disable the native stop when none of
    metric_overbought/take_profit/unclog fire first."""
    n = 3
    close = [ENTRY, 99.0, 99.0]
    high = [ENTRY, 100.0, 100.0]
    low = [ENTRY, 94.0, 94.0]     # breaches the -5% static stop (95.0)
    open_ = [ENTRY, 100.0, 100.0]  # flat -- no custom_exit condition fires

    profit, hold_bars, reason = simulate_forward(
        high, low, close, atr_pct=[0.0] * n, rvol=None, start_index=0, params=PARAMS,
        open_=open_,
    )
    assert reason == CLOSE_STOP
    assert hold_bars == 1
