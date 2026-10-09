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

AMENDED 2026-10-08 (ROI fill-price fix + exit-sequence audit, `utils/ExitPolicy.py`'s
module docstring, "AMENDED 2026-10-08 (ROI realises its TARGET, not the triggering
spike)"). `test_native_roi_fires_on_high_after_stop_and_legacy_take_profit_checked`'s
expected value changed from the measured high-based net profit to `params.roi` itself --
the pre-fix value it pinned was the +0.012649-mean-biased behaviour T021's reconciliation
found, not a behaviour worth preserving; see that test's updated docstring. Two cases were
added: `test_roi_realises_the_target_not_an_extreme_spike` (an 11x-magnitude spike, pinning
that the fill stays at the target regardless of overshoot) and
`test_custom_exit_beats_both_stop_and_roi_on_a_simultaneously_satisfiable_bar` (a single
bar where open-based take_profit, low-based stop, AND high-based roi are all individually
satisfiable, pinning that `custom_exit_decision` -- not the stop, not roi -- wins; this is
the audit FIX 2 asked for, and found the pre-existing order already correct, so this test
exists to pin it rather than to fix anything)."""
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
    STOP_AT_MARKET_RATIO,
    price_ratio_for_net,
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

def _ft_net(gross_fraction: float, fee_per_side: float = 0.001) -> float:
    """freqtrade's own fee-inclusive profit ratio for a SPOT long, from a GROSS fraction.

    Written out from `freqtrade/persistence/trade_model.py`'s formulas DELIBERATELY, rather
    than by calling `utils.ExitPolicy.net_from_price_ratio`, so these tests stay an
    INDEPENDENT check of that helper's arithmetic instead of a restatement of it. If both
    were the same expression, a wrong fee model would satisfy every assertion here.

        open_trade_value  = amount * open_rate * (1 + fee_open)     # _calc_open_trade_value
        close_trade_value = amount * rate      * (1 - fee_close)    # _calc_base_close, SPOT
        profit_ratio      = close_trade_value / open_trade_value - 1   # calc_profit_ratio

    Replaced `gross - ROUND_TRIP_FEE` across this file on 2026-10-08: that additive form was
    only the first-order expansion of this one, and differed by about `-2 * fee * r` -- see
    `utils/ExitPolicy.py`'s module docstring, "AMENDED 2026-10-08 (freqtrade's fee model is
    MULTIPLICATIVE, not additive)". `test_exit_policy_fee_netting.py` pins this helper
    against a REAL freqtrade `Trade` object, so the restatement cannot drift silently.
    """
    amount, open_rate = 1.0, 100.0
    close_rate = open_rate * (1.0 + gross_fraction)
    open_trade_value = amount * open_rate * (1.0 + fee_per_side)
    close_trade_value = amount * close_rate * (1.0 - fee_per_side)
    return close_trade_value / open_trade_value - 1.0



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
    assert abs(profit - _ft_net(0.03)) < 1e-12


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
    expected = _ft_net(100.5 / ENTRY - 1.0)
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
    expected = _ft_net(100.8 / ENTRY - 1.0)
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
    # The price that nets EXACTLY 0.0 is the inverse of a 0.0 net target through
    # freqtrade's fee model, NOT `1 + 2 * fee`: under the exact multiplicative model
    # `ENTRY * (1 + 2 * fee)` nets -2e-6, which FAILS unclog_1d's `>= 0.0` and let the
    # trade run to end_of_data. This setup was calibrated to the additive model.
    open_ = _flat(n, price=ENTRY * price_ratio_for_net(0.0, PARAMS.fee_per_side))
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

    AMENDED 2026-10-08 (ROI fill-price fix, `utils/ExitPolicy.py`'s module docstring,
    "AMENDED 2026-10-08 (ROI realises its TARGET, not the triggering spike)"). DETECTION
    stays NET-high-based and unchanged -- bar 2's high still clears 6% gross / 5.8% net,
    past the 5% ROI threshold, so ROI still fires on the same bar. But the RETURNED value
    changes: freqtrade's own `_get_close_rate_for_roi` fills a ROI exit at the price that
    realises the ROI target (`trade.calc_close_rate_for_roi(roi)`), not at the bar's high,
    so the pre-fix expectation here (the measured 5.8% net figure) was pinning the bug
    T021's reconciliation found (mean +0.012649 on 31 real `roi` trades, max divergence
    +11.36913 on one). The correct, now-asserted value is `params.roi` itself (0.05) --
    already a NET figure by construction, so no further fee subtraction applies."""
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
    assert abs(profit - params.roi) < 1e-12


def test_roi_realises_the_target_not_an_extreme_spike():
    """The exact failure mode T021's reconciliation measured: a real `roi` trade whose
    high-based excursion reached +11.36913 against a 0.05 target. Deliberately reproduce
    that magnitude here -- the realised profit must stay pinned at the ROI target
    regardless of how far past it the high overshoots."""
    params = ExitParams(
        static_stoploss=-0.5, use_atr_adaptive_stoploss=False, take_profit=0.5,
        max_hold_days=10, timeframe_minutes=60, fee_per_side=0.001,
        stoploss_grace_hours=0.0, stoploss_grace_level=-0.5, roi=0.05,
    )
    n = 3
    close = [ENTRY, ENTRY, ENTRY]
    low = [ENTRY, ENTRY, ENTRY]            # never breaches the -50% static stop
    high = [ENTRY, ENTRY, ENTRY * 12.5]    # a +1150% high -- far past the 5% roi level
    # `open_` supplied and flat at ENTRY: disables the legacy HIGH-based take_profit
    # fallback (step 2, only active when `open_` is None) so this bar's huge high can
    # only be reached by the ROI branch (step 3) -- the real caller (table.py) always
    # supplies `open_`, so this is the realistic configuration.
    open_ = [ENTRY, ENTRY, ENTRY]

    profit, hold_bars, reason = simulate_forward(
        high, low, close, atr_pct=[0.0] * n, rvol=None, start_index=0, params=params,
        open_=open_,
    )
    assert reason == CLOSE_ROI
    assert hold_bars == 2
    assert abs(profit - params.roi) < 1e-12


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
    assert abs(profit - _ft_net(PARAMS.static_stoploss)) < 1e-12


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
    expected = _ft_net(80.0 / ENTRY - 1.0)
    assert abs(profit - expected) < 1e-12
    # The regression this guards: a fill AT the stop level would be far shallower.
    assert profit < _ft_net(PARAMS.static_stoploss)


def test_custom_exit_beats_both_stop_and_roi_on_a_simultaneously_satisfiable_bar():
    """FIX 2's exit-sequence audit (T021 follow-up): pin the full precedence order --
    custom_exit's own decision, THEN the native stop, THEN the native ROI -- on a SINGLE
    bar where all three are independently satisfiable: the open clears take_profit (2.2%),
    the low breaches the static stop (-5%), AND the high clears the roi level (5%). Per
    `should_exit`'s own `exits` list order (`interface.py:1504-1520`: custom_exit appended
    before the stoploss, the stoploss before ROI) and `_check_trade_exit`'s
    first-fill-wins iteration (`backtesting.py:1001-1004`), `custom_exit`'s `take_profit`
    must win."""
    params = ExitParams(
        static_stoploss=-0.05, use_atr_adaptive_stoploss=False, take_profit=0.022,
        max_hold_days=10, timeframe_minutes=60, fee_per_side=0.001,
        stoploss_grace_hours=0.0, stoploss_grace_level=-0.15, roi=0.05,
    )
    n = 2
    close = [ENTRY, 103.0]
    high = [ENTRY, 200.0]   # clears the 5% roi level by a wide margin
    low = [ENTRY, 80.0]     # breaches the -5% static stop
    open_ = [ENTRY, 103.0]  # clears the 2.2% take_profit level

    profit, hold_bars, reason = simulate_forward(
        high, low, close, atr_pct=[0.0] * n, rvol=None, start_index=0, params=params,
        open_=open_,
    )
    assert reason == CLOSE_TAKE_PROFIT
    assert hold_bars == 1
    expected = _ft_net(103.0 / ENTRY - 1.0)
    assert abs(profit - expected) < 1e-12


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
    assert abs(profit - _ft_net(PARAMS.static_stoploss)) < 1e-12


# =============================================================================================
# `confirm_trade_exit`'s 1-hour discretionary veto, inside simulate_forward (2026-10-08).
#
# The live rule is characterised in `test_confirm_trade_exit_characterisation.py`; these pin
# that `simulate_forward` APPLIES it, on a 15m grid where it actually bites. Note the tests
# above run a 60-MINUTE bar, where bar 1 is already 1.0h old and the veto is a no-op -- which
# is exactly why it went unnoticed and needs its own coverage here.
# =============================================================================================

_P15 = ExitParams(
    atr_stoploss_multiplier=2.5, atr_stoploss_cap=-0.02, atr_stoploss_floor=-0.04,
    static_stoploss=-0.05, use_atr_adaptive_stoploss=False,
    use_volume_confirmation_stoploss=False,
    stoploss_grace_hours=6.5, stoploss_grace_level=-0.27,
    take_profit=0.022, max_hold_days=10, fee_per_side=0.001, timeframe_minutes=15,
    roi=0.05,
)


def test_take_profit_is_vetoed_below_one_hour_and_taken_at_the_first_allowed_bar():
    """A 15m bar 1 is 0.25h old, so an open-based take-profit there is VETOED; the same
    condition at bar 4 (exactly 1.0h) is taken. This is the mechanism behind
    reconciliation #6's 37-trade `take_profit`->`roi` class."""
    n = 12
    # Flat and comfortably profitable from bar 1 onward: eligible every bar, so the ONLY
    # thing that can delay the exit is the veto.
    px = ENTRY * (1.0 + 0.03)
    open_ = [ENTRY] + [px] * (n - 1)
    high = [ENTRY] * n          # never reaches roi (0.05), so roi cannot pre-empt
    low = [ENTRY] * n           # never reaches the stop
    close = [ENTRY] * n

    profit, hold_bars, reason = simulate_forward(
        high, low, close, atr_pct=[0.0] * n, rvol=None, start_index=0, params=_P15,
        open_=open_,
    )
    assert reason == CLOSE_TAKE_PROFIT
    assert hold_bars == 4, "0.25h/0.50h/0.75h must be vetoed; 1.00h is the first allowed bar"


def test_roi_is_vetoed_below_one_hour():
    """Same veto, native-ROI branch. `roi` is NOT in `EXITS_EXEMPT_FROM_MIN_HOLD`."""
    n = 12
    high = [ENTRY] + [ENTRY * 1.10] * (n - 1)   # roi reachable from bar 1
    open_ = [ENTRY] * n                          # no open-based exit
    low = [ENTRY] * n
    close = [ENTRY] * n

    profit, hold_bars, reason = simulate_forward(
        high, low, close, atr_pct=[0.0] * n, rvol=None, start_index=0, params=_P15,
        open_=open_,
    )
    assert reason == CLOSE_ROI
    assert hold_bars == 4


def test_the_stop_is_NOT_vetoed_below_one_hour():
    """The exemption that matters: "never hold a position past its stop just because the
    trade is young". A stop reachable at bar 1 must fire at bar 1."""
    n = 12
    low = [ENTRY] + [ENTRY * (1.0 + _P15.stoploss_grace_level - 0.01)] + [ENTRY] * (n - 2)
    open_ = [ENTRY] * n
    high = [ENTRY] * n
    close = [ENTRY] * n

    profit, hold_bars, reason = simulate_forward(
        high, low, close, atr_pct=[0.0] * n, rvol=None, start_index=0, params=_P15,
        open_=open_,
    )
    assert reason == CLOSE_STOP
    assert hold_bars == 1


def test_a_vetoed_take_profit_is_LOST_not_deferred():
    """The second-order effect, and why reconciliation #6's disagreements ran in BOTH
    directions: the veto does not queue the exit. A take-profit that is eligible only at
    bar 1 (vetoed) and never again does not fire at bar 4 -- the trade runs on and stops
    out instead. Without this, a vetoed exit would merely be late; with it, the arm can
    end up taking a LOSS where the table booked a profit."""
    n = 12
    spike = ENTRY * (1.0 + 0.03)
    open_ = [ENTRY, spike] + [ENTRY] * (n - 2)      # profitable ONLY at bar 1
    high = [ENTRY] * n
    low = [ENTRY] * 5 + [ENTRY * (1.0 + _P15.stoploss_grace_level - 0.01)] + [ENTRY] * (n - 6)
    close = [ENTRY] * n

    profit, hold_bars, reason = simulate_forward(
        high, low, close, atr_pct=[0.0] * n, rvol=None, start_index=0, params=_P15,
        open_=open_,
    )
    assert reason == CLOSE_STOP, "the vetoed take-profit must NOT be deferred to bar 4"
    assert hold_bars == 5
    assert profit < 0.0


def test_post_grace_stop_is_clamped_to_the_market_not_filled_at_the_open():
    """The table side of the live post-grace sign fix (`STOP_AT_MARKET_RATIO`).

    Grace is ENABLED (6.5h) and the trade is deeply underwater when it expires, so the
    entry-anchored -5% stop (95.0) sits above the bar entirely. Live, freqtrade cannot
    place a stop above the market, so it is held AT the market (the bar's high, which is
    what the backtester passes as `current_rate`) and fires there. The table must model
    that fill, NOT the open-based gap fill -- the gap fill applies only to a genuine gap
    through the GRACE stop, which IS expressible live.

    Contrast `test_stop_fills_at_the_open_when_the_open_has_already_gapped_below_the_stop`
    above, which runs grace DISABLED: there the stop really is fixed at entry and a gap
    through it really does fill at the open."""
    n = 40
    # Bars 0..25 (0-6.25h) sit inside grace, above the -27% grace stop so nothing fires.
    # Bar 26 is 6.5h -- grace expires, and the whole bar is far below the -5% stop.
    open_ = [ENTRY] * 26 + [80.0] * (n - 26)
    high = [ENTRY] * 26 + [82.0] * (n - 26)
    low = [ENTRY] * 26 + [78.0] * (n - 26)
    close = [ENTRY] * 26 + [80.0] * (n - 26)

    params = ExitParams(
        static_stoploss=-0.05, use_atr_adaptive_stoploss=False,
        use_volume_confirmation_stoploss=False,
        take_profit=0.022, max_hold_days=10, timeframe_minutes=15, fee_per_side=0.001,
        stoploss_grace_hours=6.5, stoploss_grace_level=-0.27,
    )
    profit, hold_bars, reason = simulate_forward(
        high, low, close, atr_pct=[0.0] * n, rvol=None, start_index=0, params=params,
        open_=open_,
    )
    assert reason == CLOSE_STOP
    assert hold_bars == 26, "the stop must fire on the first post-grace bar"
    clamped = 82.0 * (1.0 + STOP_AT_MARKET_RATIO)
    assert abs(profit - _ft_net(clamped / ENTRY - 1.0)) < 1e-9
    # Not the open-based gap fill, and not the un-clamped -5% level either.
    assert abs(profit - _ft_net(80.0 / ENTRY - 1.0)) > 1e-3
    assert profit < _ft_net(params.static_stoploss)


def test_inside_grace_a_gap_through_the_grace_stop_still_fills_at_the_open():
    """The clamp must not reach into the grace window. The grace stop is set ONCE, at
    after_fill, with `allow_refresh=True`, so it is NOT re-derived from `current_rate`
    and CAN sit above a later bar -- making freqtrade's gap rule genuinely reachable
    there."""
    n = 10
    open_ = [ENTRY, 60.0] + [60.0] * (n - 2)   # gaps through the -27% grace stop (73.0)
    high = [ENTRY, 62.0] + [62.0] * (n - 2)
    low = [ENTRY, 58.0] + [58.0] * (n - 2)
    close = [ENTRY, 60.0] + [60.0] * (n - 2)

    params = ExitParams(
        static_stoploss=-0.05, use_atr_adaptive_stoploss=False,
        use_volume_confirmation_stoploss=False,
        take_profit=0.022, max_hold_days=10, timeframe_minutes=15, fee_per_side=0.001,
        stoploss_grace_hours=6.5, stoploss_grace_level=-0.27,
    )
    profit, hold_bars, reason = simulate_forward(
        high, low, close, atr_pct=[0.0] * n, rvol=None, start_index=0, params=params,
        open_=open_,
    )
    assert reason == CLOSE_STOP
    assert hold_bars == 1
    assert abs(profit - _ft_net(60.0 / ENTRY - 1.0)) < 1e-12, "open fill, not the clamp"
