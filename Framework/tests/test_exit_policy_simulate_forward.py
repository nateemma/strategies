# pragma pylint: disable=C0103, C0114, C0115, C0116, C0301
"""T009 (specs/021-label-exit-coordination) — `simulate_forward`'s state machine.

WHY THIS EXISTS. `simulate_forward` is the one place an offline outcome table asks
"what would the live exit have done here?" (contracts/exit-policy.md). Its rule
precedence is deliberately fixed: on a bar whose `low` breaches the stop AND whose
`high` reaches the take-profit in the SAME bar, OHLC alone cannot order the two
intra-bar events, and the policy picks `stop` because the opposite choice would let
an outcome table claim a profit the live path would never have collected -- the
phantom-return failure mode Constitution Principle V exists to prevent (see the
module docstring in `utils/ExitPolicy.py`).

This file pins that precedence with an EXACT tie (not "close enough") and pins each
of the four close reasons in isolation, so the precedence test is not the only thing
keeping the state machine honest.

All cases share one hand-built two-bar world: `close[0] = 100` is the entry, and
`max_hold_days=1` with `timeframe_minutes=1440` (one bar == one day) makes
`max_bars == 1`, so bar index 1 is the only bar the loop ever inspects. That keeps
every array trivial to read by eye while still exercising the real loop (as opposed
to a 1-bar market that never enters the loop at all -- reserved for the
`end_of_data` case below).

NOTE 2026-10-08 (entry-price convention fix, `utils/ExitPolicy.py`'s module
docstring). None of these cases supply `open_`, so they exercise the function's
LEGACY path -- entry at `close[start_index]`, replay starting at `start_index + 1`,
the HIGH-based `take_profit` fallback -- UNCHANGED and bit-for-bit pinned by this
file, exactly as before this fix. The fix only changes behaviour for callers that DO
supply `open_` (the real caller, `research/outcomes/table.py`, always does); see
`test_exit_policy_simulate_forward_amendment.py` for cases covering that path. This
file needed NO changes for this fix.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from utils.ExitPolicy import (  # noqa: E402
    CLOSE_END_OF_DATA,
    CLOSE_HOLD_CAP,
    CLOSE_STOP,
    CLOSE_TAKE_PROFIT,
    ExitParams,
    simulate_forward,
)

#: Static stop (ATR adaptation off), so `atr_pct` is irrelevant to every case below
#: and can be a dummy zero array -- the arithmetic under test is the bar-by-bar
#: STATE MACHINE, not the stop *level*, which is already covered by
#: test_exit_policy_characterisation.py and the effective_stop_fraction tests.
PARAMS = ExitParams(
    static_stoploss=-0.05,
    use_atr_adaptive_stoploss=False,
    take_profit=0.05,
    max_hold_days=1,
    timeframe_minutes=1440,  # => bars_per_hour = 1/24, max_bars = 1
    fee_per_side=0.001,
    # Both grace fields stated EXPLICITLY. They have no default (see ExitParams), and
    # 0.0 here is a deliberate choice -- these tests pin the stop/take-profit/hold-cap
    # state machine, so a grace window would mask the very transitions under test.
    stoploss_grace_hours=0.0,
    stoploss_grace_level=-0.15,
)
ENTRY = 100.0
STOP_PRICE = ENTRY * (1.0 + PARAMS.static_stoploss)   # 95.0
TP_PRICE = ENTRY * (1.0 + PARAMS.take_profit)          # 105.0
ROUND_TRIP_FEE = 2.0 * PARAMS.fee_per_side             # 0.002


def _run(high, low, close):
    atr_pct = [0.0] * len(close)
    return simulate_forward(high, low, close, atr_pct, rvol=None, start_index=0, params=PARAMS)


def test_same_bar_tie_stop_wins_over_take_profit():
    """THE precedence case: bar 1's low breaches the stop AND its high reaches the
    take-profit, with an EXACT tie (low == stop_price, high == tp_price) so there is
    no ambiguity about which threshold was "really" crossed first -- both are hit to
    the cent. The policy must still report `stop`."""
    close = [ENTRY, 102.0]
    low = [ENTRY, STOP_PRICE]       # exactly at the stop
    high = [ENTRY, TP_PRICE]        # exactly at the take-profit, same bar

    profit, hold_bars, reason = _run(high, low, close)

    assert reason == CLOSE_STOP
    assert hold_bars == 1
    assert profit == PARAMS.static_stoploss - ROUND_TRIP_FEE
    assert abs(profit - (-0.052)) < 1e-12


def test_stop_in_isolation():
    close = [ENTRY, 99.0]
    low = [ENTRY, 94.0]             # breaches the 95.0 stop
    high = [ENTRY, 100.0]           # nowhere near the 105.0 take-profit

    profit, hold_bars, reason = _run(high, low, close)

    assert reason == CLOSE_STOP
    assert hold_bars == 1
    assert profit == PARAMS.static_stoploss - ROUND_TRIP_FEE


def test_take_profit_in_isolation():
    close = [ENTRY, 104.0]
    low = [ENTRY, 96.0]             # does not breach the 95.0 stop
    high = [ENTRY, 106.0]           # clears the 105.0 take-profit

    profit, hold_bars, reason = _run(high, low, close)

    assert reason == CLOSE_TAKE_PROFIT
    assert hold_bars == 1
    assert profit == PARAMS.take_profit - ROUND_TRIP_FEE


def test_hold_cap_in_isolation():
    """Neither threshold fires on the only bar the 1-bar `max_hold_days` window
    permits, so the position closes at that bar's `close`, net of the round-trip
    fee -- NOT the raw close-to-close return."""
    close = [ENTRY, 102.0]
    low = [ENTRY, 96.0]             # does not breach the stop
    high = [ENTRY, 100.0]           # does not reach the take-profit

    profit, hold_bars, reason = _run(high, low, close)

    assert reason == CLOSE_HOLD_CAP
    assert hold_bars == 1
    expected = (close[1] / ENTRY) - 1.0 - ROUND_TRIP_FEE
    assert abs(profit - expected) < 1e-12
    assert abs(profit - 0.018) < 1e-12


def test_end_of_data_in_isolation():
    """No bar exists after the entry bar at all (a 1-element market), so the loop
    never runs and the position cannot be said to have closed by the hold cap --
    `end_of_data` must be distinguishable from a real exit, and its profit must be
    exactly 0.0 (no fee charged on a position that was never closed)."""
    close = [ENTRY]
    low = [ENTRY]
    high = [ENTRY]

    profit, hold_bars, reason = _run(high, low, close)

    assert reason == CLOSE_END_OF_DATA
    assert hold_bars == 0
    assert profit == 0.0
