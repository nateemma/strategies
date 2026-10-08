# pragma pylint: disable=C0103, C0114, C0115, C0116, C0301
"""Net-vs-gross fee accounting, one test per exit type (2026-10-08 fee double-count fix,
specs/021-label-exit-coordination). Companion to `test_exit_policy_simulate_forward.py`
(the pre-amendment stop/take-profit/hold-cap pin) and
`test_exit_policy_simulate_forward_amendment.py` (the FR-001 amendment's own cases, updated
in this same change to compare/return NET values for the exits below).

THE ASYMMETRY THIS FIXTURE PINS, per `utils/ExitPolicy.py`'s module docstring (freqtrade
source citations live there, not repeated here):

  * `stop` / `hold_cap` / `end_of_data` close at a traded PRICE. The comparison itself has
    no fee in it (a price level, or a bar's open/close); the fee is realised exactly once,
    subtracted from the price-based fraction AFTER the fill is decided.
  * `metric_overbought` / `take_profit` / `unclog_12h` / `unclog_1d` (the `open_`-based
    `custom_exit_decision` branches) and `roi` compare against thresholds that are
    themselves NET live (`trade.calc_profit_ratio`, fee-inclusive). So both the COMPARISON
    and the RETURN must be net, computed once -- not gross-compared-then-fee-subtracted
    (double-counts) and not gross-compared-then-returned-raw (under-counts on the other
    side).

CHECKED, NOT CHANGED, 2026-10-08 (entry-price convention fix, `utils/ExitPolicy.py`'s
module docstring): every case that supplies `open_` here already has `open_[0] == ENTRY ==
close[0]`, so entering at `open_[0]` instead of `close[0]` is a no-op, and bar 0 joining the
replay loop is a provable no-op too (same reasoning as
`test_exit_policy_simulate_forward_amendment.py`'s module docstring). No setup or assertion
needed changing in this file.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from utils.ExitPolicy import (  # noqa: E402
    CLOSE_END_OF_DATA,
    CLOSE_HOLD_CAP,
    CLOSE_ROI,
    CLOSE_STOP,
    CLOSE_TAKE_PROFIT,
    CLOSE_UNCLOG_12H,
    ExitParams,
    custom_exit_decision,
    simulate_forward,
)

ENTRY = 100.0
FEE_PER_SIDE = 0.001
ROUND_TRIP_FEE = 2.0 * FEE_PER_SIDE  # 0.002

BASE = dict(
    static_stoploss=-0.05,
    use_atr_adaptive_stoploss=False,
    take_profit=0.022,
    max_hold_days=10,
    timeframe_minutes=60,
    fee_per_side=FEE_PER_SIDE,
    stoploss_grace_hours=0.0,
    stoploss_grace_level=-0.15,
)


def _flat(n: int, price: float = ENTRY):
    return [price] * n


# =============================================================================================
# custom_exit_decision itself -- the function's contract is "current_profit is already net"
# (unchanged by this fix; `test_exit_policy_custom_exit_decision.py` pins its branches). These
# two cases exist only to make the NET-IN/NET-OUT contract explicit at the unit level.
# =============================================================================================


def test_custom_exit_decision_take_profit_boundary_is_evaluated_on_whatever_it_is_given():
    """`custom_exit_decision` performs no fee arithmetic of its own -- it is the CALLER's
    job (`simulate_forward` for the offline path, `should_exit` for the live path) to pass
    a value that is already net. A gross value that merely LOOKS like it clears the
    threshold is not this function's problem to detect."""
    params = ExitParams(**BASE)
    gross = 0.023  # clears 0.022 if compared as-is
    net = gross - ROUND_TRIP_FEE  # 0.021 -- does NOT clear 0.022
    assert custom_exit_decision(gross, age_hours=1.0, guard_metric=None, params=params) == (
        CLOSE_TAKE_PROFIT
    )
    assert custom_exit_decision(net, age_hours=1.0, guard_metric=None, params=params) is None


# =============================================================================================
# simulate_forward -- stop / hold_cap / end_of_data: fee subtracted ONCE, after a price fill.
# =============================================================================================


def test_stop_fee_is_subtracted_once_from_the_price_based_fill():
    """The stop is a PRICE level; the comparison (`low[k] <= stop_price`) has no fee in it
    at all. The realised fraction is the price-based fill (here, fill AT the stop, entry
    100 -> stop 95) minus `round_trip_fee` exactly once -- not derived from any NET-vs-NET
    comparison."""
    params = ExitParams(**BASE)
    n = 3
    close = [ENTRY, 99.0, 99.0]
    high = [ENTRY, 100.0, 100.0]
    low = [ENTRY, 94.0, 94.0]       # breaches the -5% static stop (95.0)
    open_ = [ENTRY, 100.0, 100.0]   # above the stop price -- fills AT the stop, no gap

    profit, hold_bars, reason = simulate_forward(
        high, low, close, atr_pct=[0.0] * n, rvol=None, start_index=0, params=params,
        open_=open_,
    )
    assert reason == CLOSE_STOP
    assert abs(profit - (params.static_stoploss - ROUND_TRIP_FEE)) < 1e-12


def test_hold_cap_fee_is_subtracted_once_from_the_close_based_fill():
    """`hold_cap` closes at the final bar's CLOSE -- a price, not a net-profit comparison --
    so the fee is subtracted exactly once from that price ratio, same discipline as the
    stop."""
    params = ExitParams(
        static_stoploss=-0.5, use_atr_adaptive_stoploss=False, take_profit=0.5,
        max_hold_days=1, timeframe_minutes=1440, fee_per_side=FEE_PER_SIDE,
        stoploss_grace_hours=0.0, stoploss_grace_level=-0.5,
    )
    n = 2
    close = [ENTRY, 101.0]
    high = _flat(n, 101.0)
    low = _flat(n, 99.0)

    profit, hold_bars, reason = simulate_forward(
        high, low, close, atr_pct=[0.0] * n, rvol=None, start_index=0, params=params,
    )
    assert reason == CLOSE_HOLD_CAP
    expected = (101.0 / ENTRY - 1.0) - ROUND_TRIP_FEE
    assert abs(profit - expected) < 1e-12


def test_end_of_data_has_no_profit_and_no_fee_to_net():
    """`end_of_data` is the "ran out of bars with nothing to close at" case -- 0.0 profit,
    not a fee-netted figure at all."""
    params = ExitParams(**BASE)
    n = 1  # start_index is the last bar: no bars left to replay forward
    close = [ENTRY]
    high = [ENTRY]
    low = [ENTRY]

    profit, hold_bars, reason = simulate_forward(
        high, low, close, atr_pct=[0.0] * n, rvol=None, start_index=0, params=params,
    )
    assert reason == CLOSE_END_OF_DATA
    assert profit == 0.0


# =============================================================================================
# simulate_forward -- custom_exit_decision branches (open-based) and roi: compare AND return
# NET, computed once.
# =============================================================================================


def test_take_profit_open_based_requires_the_open_to_clear_the_threshold_net_of_fees():
    """A GROSS open move that would clear `take_profit` (0.022) but whose NET (minus
    `round_trip_fee`) does not must NOT fire -- the exact double-count direction the fix
    closes. 0.023 gross -> 0.021 net, short of 0.022."""
    params = ExitParams(**BASE)
    n = 2
    close = _flat(n)
    high = _flat(n)
    low = _flat(n)
    open_ = [ENTRY, ENTRY * 1.023]

    profit, hold_bars, reason = simulate_forward(
        high, low, close, atr_pct=[0.0] * n, rvol=None, start_index=0, params=params,
        open_=open_,
    )
    assert reason != CLOSE_TAKE_PROFIT


def test_take_profit_open_based_fires_and_returns_the_net_figure_once_net_clears():
    """Once the NET open-based profit clears the threshold, the function both fires AND
    returns that same net figure -- no further fee subtraction on top."""
    params = ExitParams(**BASE)
    n = 2
    close = _flat(n)
    high = _flat(n)
    low = _flat(n)
    gross = 0.022 + ROUND_TRIP_FEE + 0.001  # net clears 0.022 by 0.001
    open_ = [ENTRY, ENTRY * (1.0 + gross)]

    profit, hold_bars, reason = simulate_forward(
        high, low, close, atr_pct=[0.0] * n, rvol=None, start_index=0, params=params,
        open_=open_,
    )
    assert reason == CLOSE_TAKE_PROFIT
    expected_net = gross - ROUND_TRIP_FEE
    assert abs(profit - expected_net) < 1e-12


def test_unclog_12h_requires_net_not_gross_to_clear_its_threshold():
    """Same double-count direction as take_profit, for `unclog_12h_min_profit` (0.005):
    0.006 gross nets to 0.004, short of 0.005, so it must NOT fire at the 12h boundary."""
    params = ExitParams(**BASE)
    n = 15
    close = _flat(n)
    high = _flat(n)
    low = _flat(n)
    open_ = list(_flat(n))
    open_[13] = ENTRY * 1.006  # 0.6% gross -> 0.4% net

    profit, hold_bars, reason = simulate_forward(
        high, low, close, atr_pct=[0.0] * n, rvol=None, start_index=0, params=params,
        open_=open_,
    )
    assert reason != CLOSE_UNCLOG_12H


def test_roi_compares_net_and_returns_the_net_high_not_the_fixed_threshold_minus_fee():
    """A high that clears the ROI threshold GROSS but not NET must not fire (same
    double-count direction); one that clears NET fires and returns that measured net
    figure, not a fixed `roi - round_trip_fee` regardless of how far price moved."""
    params = ExitParams(
        static_stoploss=-0.5, use_atr_adaptive_stoploss=False, take_profit=0.5,
        max_hold_days=10, timeframe_minutes=60, fee_per_side=FEE_PER_SIDE,
        stoploss_grace_hours=0.0, stoploss_grace_level=-0.5, roi=0.05,
    )
    n = 3
    close = [ENTRY, ENTRY, ENTRY]
    low = [ENTRY, ENTRY * 0.99, ENTRY * 0.99]

    # Bar 1: 5.1% gross, 4.9% net -- short of the 5% ROI level net -- must not fire.
    high_short = [ENTRY, ENTRY * 1.051, ENTRY * 1.051]
    profit, hold_bars, reason = simulate_forward(
        high_short, low, close, atr_pct=[0.0] * n, rvol=None, start_index=0, params=params,
    )
    assert reason != CLOSE_ROI

    # Bar 2 (fresh replay): 6% gross, 5.8% net -- clears the 5% ROI level net.
    high_clear = [ENTRY, ENTRY * 0.98, ENTRY * 1.06]
    profit, hold_bars, reason = simulate_forward(
        high_clear, low, close, atr_pct=[0.0] * n, rvol=None, start_index=0, params=params,
    )
    assert reason == CLOSE_ROI
    expected_net = (1.06 - 1.0) - ROUND_TRIP_FEE
    assert abs(profit - expected_net) < 1e-12
