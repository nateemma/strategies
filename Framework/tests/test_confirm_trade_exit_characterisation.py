"""Characterisation of `BaseStrategy.confirm_trade_exit`'s 1-hour discretionary-exit veto.

WHY THIS EXISTS, and why it was written BEFORE the rule was extracted (the T004
discipline): this veto is a LIVE exit rule that feature 021's FR-001 exit-surface
enumeration MISSED, and it is the single largest cause of the outcome table diverging
from the walk-forward arm. Reconciliation #6 localised the entire remaining residual to
91 trades (6.5%) where the table and the arm chose different exit rules; in every class
the table fired the EARLIEST eligible rule and the arm fired later. The cause is here:

    `confirm_trade_exit` returns False -- vetoing the exit -- for ANY exit whose reason is
    not risk-management, while the trade is younger than 1 hour (4 bars at 15m).

freqtrade then skips that exit and falls through to the next candidate on the SAME bar
(`Backtesting._get_exit_for_signal` returns None at optimize/backtesting.py:911-924, and
`_check_trade_exit`'s `for exit_ in exits: if t: return t` loop continues,
:1001-1004). So the veto does not merely DELAY a discretionary exit -- if its condition
stops holding before the trade turns 1 hour old, that exit is LOST, and the trade runs on
to whatever fires later (very often the stop). That second-order effect is why the
disagreements run in BOTH directions.

These expectations were read off the UNMODIFIED `confirm_trade_exit` and seen green
BEFORE the rule was extracted into `utils/ExitPolicy.py`. A fixture captured after an
extraction proves nothing about what the live code used to do.
"""
from __future__ import annotations

import sys
import types
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
for p in (str(ROOT), str(ROOT / "utils"), str(ROOT / "NNNC")):
    if p not in sys.path:
        sys.path.insert(0, p)

from Framework.BaseStrategy import BaseStrategy  # noqa: E402

#: Exempt from the veto -- risk management and forced exits are ALWAYS honoured, because
#: (the live comment's words) you must "never hold a position past its stop just because
#: the trade is young". Verbatim from the live list.
ALWAYS_HONOURED = [
    "force_exit", "emergency_exit", "stop_loss", "trailing_stop_loss",
    "stoploss_on_exchange", "liquidation",
]
#: Everything else is DISCRETIONARY and vetoed below 1 hour. `trailing_stop_loss` above is
#: freqtrade's LABEL for a `custom_stoploss` exit in this strategy (`trailing_stop` is
#: False), so the real native stop is exempt under that name.
DISCRETIONARY = [
    "roi", "take_profit", "metric_overbought", "unclog_12h", "unclog_1d",
    "exit_signal", "max_hold", "",
]


class _StubDp:
    class _RunMode:
        value = "backtest"
    runmode = _RunMode()


class _StubTrade:
    def __init__(self, open_date: datetime):
        self.open_date_utc = open_date
        self.pair = "SOL/USDT"
        self.open_rate = 100.0

    def calc_profit_ratio(self, rate: float) -> float:
        return rate / self.open_rate - 1.0


class _StubStrategy:
    def __init__(self):
        self.dp = _StubDp()
        self.confirm_trade_exit = types.MethodType(BaseStrategy.confirm_trade_exit, self)


def _call(exit_reason: str, age_hours: float) -> bool:
    now = datetime(2026, 1, 1, 12, 0, tzinfo=timezone.utc)
    trade = _StubTrade(now - timedelta(hours=age_hours))
    return BaseStrategy.confirm_trade_exit(
        _StubStrategy(), pair="SOL/USDT", trade=trade, order_type="market",
        amount=1.0, rate=100.0, time_in_force="gtc",
        exit_reason=exit_reason, current_time=now,
    )


@pytest.mark.parametrize("reason", ALWAYS_HONOURED)
@pytest.mark.parametrize("age_hours", [0.0, 0.25, 0.5, 0.75, 0.9999, 1.0, 2.0, 24.0])
def test_risk_exits_are_never_vetoed_at_any_age(reason, age_hours):
    """A stop must fire the instant it is reached, however young the trade."""
    assert _call(reason, age_hours) is True


@pytest.mark.parametrize("reason", DISCRETIONARY)
@pytest.mark.parametrize("age_hours", [0.0, 0.25, 0.5, 0.75, 0.9999])
def test_discretionary_exits_are_vetoed_below_one_hour(reason, age_hours):
    """THE defect's mechanism: the table lets these fire from bar 0, live cannot."""
    assert _call(reason, age_hours) is False


@pytest.mark.parametrize("reason", DISCRETIONARY)
@pytest.mark.parametrize("age_hours", [1.0, 1.0001, 2.0, 24.0])
def test_discretionary_exits_are_allowed_from_exactly_one_hour(reason, age_hours):
    """The boundary is INCLUSIVE: the live test is `duration_hours < 1.0`, so a trade
    exactly 1.0h old (bar 4 at 15m) may take a discretionary exit. Three of the first
    `take_profit`->`roi` divergences closed at exactly bar 4, which is what put this
    rule in the frame at all."""
    assert _call(reason, age_hours) is True


def test_veto_boundary_is_one_hour_not_a_bar_count():
    """Pinned as HOURS, not bars: the same rule applied to a 5m or 1h timeframe gives a
    different bar count, so `simulate_forward` must convert, not hardcode 4."""
    assert _call("roi", 59.9 / 60.0) is False
    assert _call("roi", 60.0 / 60.0) is True
