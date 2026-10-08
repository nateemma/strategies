# pragma pylint: disable=C0103, C0114, C0115, C0116, C0301
"""`simulate_forward`'s FR-001 amendment (specs/021-label-exit-coordination, T021 follow-up):
`open_`, `guard_metric` and `ExitParams.roi`. Companion to
`test_exit_policy_simulate_forward.py`, which pins the pre-amendment stop/take-profit/
hold-cap state machine and is left untouched by this file's cases (none of them supply
`open_`/`guard_metric`/`roi`, so that file's geometry is undisturbed).

AMENDED 2026-10-08 (fee double-count fix, `utils/ExitPolicy.py`'s module docstring has the
freqtrade source citations). The `unclog_12h` / `unclog_1d` / `roi` cases below were updated
to compare and assert against NET profit (gross minus `round_trip_fee`), not gross -- the
pre-fix values they used to pin were the double-counted bug this fix removes, not a
behaviour worth preserving. `take_profit` and `metric_overbought` happen to need no case
changes: their trigger margins are wide enough that netting the fee out before the
comparison does not change which bar fires or the arithmetic identity of the returned
value.

UPDATED 2026-10-08 (entry-price convention fix, `utils/ExitPolicy.py`'s module docstring).
Every case in this file supplies `open_`, so every case now enters at `open_[0]` instead of
`close[0]` and replays from bar 0 itself instead of bar 1 (`simulate_forward`'s docstring).
In every case except one, `open_[0]` already equalled `close[0]` (== `ENTRY`) by
construction, and bar 0 is a provable no-op for the open-based custom_exit_decision check
(entry == open_[0] by definition, so its own profit is exactly `-round_trip_fee` and its age
is 0), so no assertion values changed. The one exception --
`test_unclog_1d_fires_via_open_on_zero_profit`, whose `open_` array was flat at a
NON-`ENTRY` value at EVERY index including 0 -- needed a one-line setup fix (`open_[0] =
ENTRY`); see that test for why.
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
    """One bar == one hour; bar 13 is the first past the 12h boundary. The comparison is
    NET (fee double-count fix): `unclog_12h_min_profit` (0.005) is compared against GROSS
    profit minus `round_trip_fee` (0.002), so the open must clear 0.7%, not 0.5%, to fire --
    0.6% gross (net 0.4%) would NOT fire, unlike the pre-fix (gross-compared) code."""
    n = 15
    close = _flat(n)
    high = _flat(n)
    low = _flat(n)   # never breaches any stop
    open_ = list(_flat(n))
    open_[13] = 100.8   # 0.8% gross -> 0.6% net -- above unclog_12h's 0.5% net threshold,
                        # below take_profit's 2.2% net threshold

    profit, hold_bars, reason = simulate_forward(
        high, low, close, atr_pct=[0.0] * n, rvol=None, start_index=0, params=PARAMS,
        open_=open_,
    )
    assert reason == CLOSE_UNCLOG_12H
    assert hold_bars == 13
    expected = (100.8 / ENTRY - 1.0) - ROUND_TRIP_FEE
    assert abs(profit - expected) < 1e-12


def test_unclog_1d_fires_via_open_on_zero_profit():
    """One bar == one hour; bar 24 is the first bar whose age (24h) reaches the unclog_1d
    boundary. The comparison is NET (fee double-count fix): `open_` is flat at a GROSS
    profit exactly equal to `round_trip_fee` (0.2%), so NET profit is exactly 0.0,
    satisfying `>= 0.0` -- the pre-fix code would have required gross to be used directly
    and fired on a literally-flat open instead.

    UPDATED 2026-10-08 (entry-price convention fix, `utils/ExitPolicy.py`'s module
    docstring): entry is now `open_[0]`, not `close[0]`, so `open_[0]` must stay at
    `ENTRY` -- SETUP ONLY, no assertion value changed. Before this fix `open_` was flat
    at `ENTRY * (1 + ROUND_TRIP_FEE)` at EVERY index including 0, which was harmless when
    entry came from `close[0]`; now that same flat value at index 0 would silently become
    the entry price itself, changing what "zero net profit" even means at bar 24."""
    n = 26
    close = _flat(n)
    high = _flat(n)
    low = _flat(n)
    open_ = _flat(n, price=ENTRY * (1.0 + ROUND_TRIP_FEE))   # net profit exactly 0.0
    open_[0] = ENTRY  # entry-price convention fix: open_[start_index] IS entry now.

    profit, hold_bars, reason = simulate_forward(
        high, low, close, atr_pct=[0.0] * n, rvol=None, start_index=0, params=PARAMS,
        open_=open_,
    )
    assert reason == CLOSE_UNCLOG_1D
    assert hold_bars == 24
    assert abs(profit - 0.0) < 1e-12


def test_native_roi_fires_on_high_after_stop_and_legacy_take_profit_checked():
    """ROI is the LOWEST-precedence of the amended exits: this bar's low does not breach
    the stop, and `open_` is NOT supplied (so the legacy high-based take_profit path IS
    active) -- but take_profit (2.2%) and ROI (5%) are set far enough apart that only ROI's
    own threshold is crossed.

    The comparison and the return are NET (fee double-count fix): bar 2's high clears 6%
    gross, 5.8% net -- past the 5% ROI threshold -- and the returned profit is that same
    5.8% net figure, NOT the threshold minus a fee (the pre-fix bug returned a fixed
    `roi - round_trip_fee` regardless of how far the high actually moved)."""
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
    expected = (106.0 / ENTRY - 1.0) - ROUND_TRIP_FEE
    assert abs(profit - expected) < 1e-12


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


def test_stop_fills_at_the_stop_level_when_the_open_is_above_it():
    """The ORDINARY stop-fill case (T021 reconciliation follow-up, "grace cliff" fix):
    the stop price (95.0) sits strictly between this bar's open (100.0) and low (94.0),
    i.e. reachable from the open -- matching freqtrade's `_get_close_rate_for_stoploss`
    "stop within candle range" branch. The fill must still be AT the stop level, not the
    open, so this fix does not silently change the pre-existing (correct) case."""
    n = 3
    close = [ENTRY, 99.0, 99.0]
    high = [ENTRY, 100.0, 100.0]
    low = [ENTRY, 94.0, 94.0]       # breaches the -5% static stop (95.0)
    open_ = [ENTRY, 100.0, 100.0]   # ABOVE the stop price -- no gap

    profit, hold_bars, reason = simulate_forward(
        high, low, close, atr_pct=[0.0] * n, rvol=None, start_index=0, params=PARAMS,
        open_=open_,
    )
    assert reason == CLOSE_STOP
    assert hold_bars == 1
    assert abs(profit - (PARAMS.static_stoploss - ROUND_TRIP_FEE)) < 1e-12


def test_stop_fills_at_the_open_when_the_open_has_already_gapped_below_the_stop():
    """THE BUG this fix closes. The "grace cliff": during
    `stoploss_grace_hours` the stop is held at the wide `stoploss_grace_level`
    (-0.15 here); once grace expires (bar 1, age 1h >= `stoploss_grace_hours=0.0`) the
    stop snaps to the normal -5% static level (95.0) -- but the bar's OPEN (80.0) is
    already below that level. freqtrade's `_get_close_rate_for_stoploss` fills this at
    the bar's OPEN (`stoploss_value > row[HIGH_IDX] -> return row[OPEN_IDX]`), not the
    stop price: the stop price was never actually tradeable. The realised loss must
    equal the open-based fill (80/100 - 1 = -0.20) net of fees, NOT the shallower
    stop-level loss (-0.05) the pre-fix code would have returned."""
    n = 3
    close = [ENTRY, 80.0, 80.0]
    high = [ENTRY, 82.0, 82.0]
    low = [ENTRY, 78.0, 78.0]      # breaches the -5% static stop (95.0)
    open_ = [ENTRY, 80.0, 80.0]    # already BELOW the stop price -- a gap

    profit, hold_bars, reason = simulate_forward(
        high, low, close, atr_pct=[0.0] * n, rvol=None, start_index=0, params=PARAMS,
        open_=open_,
    )
    assert reason == CLOSE_STOP
    assert hold_bars == 1
    expected = (80.0 / ENTRY - 1.0) - ROUND_TRIP_FEE
    assert abs(profit - expected) < 1e-12
    # The regression this guards: a fill AT the stop level would be far shallower.
    assert profit < PARAMS.static_stoploss - ROUND_TRIP_FEE


def test_stop_fallback_without_open_still_fills_at_the_stop_level():
    """Pre-amendment callers that cannot supply `open_` get the old, optimistic
    behaviour verbatim (documented in `simulate_forward`'s docstring): without a
    per-bar open there is no signal to detect the gap, so the fallback still fills at
    `stop_frac`, even on a bar whose low plunges far past the stop (mirroring the
    gap scenario above, minus the open array)."""
    n = 3
    close = [ENTRY, 80.0, 80.0]
    high = [ENTRY, 82.0, 82.0]
    low = [ENTRY, 78.0, 78.0]      # breaches the -5% static stop (95.0)

    profit, hold_bars, reason = simulate_forward(
        high, low, close, atr_pct=[0.0] * n, rvol=None, start_index=0, params=PARAMS,
    )
    assert reason == CLOSE_STOP
    assert hold_bars == 1
    assert abs(profit - (PARAMS.static_stoploss - ROUND_TRIP_FEE)) < 1e-12
