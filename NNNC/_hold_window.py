"""Per-strategy hold-window extension for feature 003 (T022).

`unclog_12h` is hardcoded inside `BaseStrategy.custom_exit`
(Framework/BaseStrategy.py:1040) as

    if (num_hours >= 12) & (current_profit > 0.005):
        return "unclog_12h"

It is not a parameter, so a strategy whose label peaks after 12h is force-exited
before its payoff arrives. research.md R4 chose a per-strategy override over an
edit-measure-revert on BaseStrategy: a live dry run (MomentumRegimeBasket15mFast)
imports that file, and an override needs no revert at all.

WHY DEFERRAL IS FAITHFUL HERE. The mixin lets `custom_exit` run in full and only
suppresses a time-based unclog that fires before the strategy's own hold window
has elapsed. Suppressing it does skip the `strategy_custom_exit` hook that sits
after the time checks -- but no class in this family implements that hook
(verified: neither BaseNNStrategy nor any NNNC strategy defines it), so the
default no-op is all that is skipped. Every profit-based exit (rsi_*,
metric_overbought, take_profit) is evaluated BEFORE the time checks and is
therefore untouched, as are the ATR stoploss and `max_hold`.

Only `unclog_12h` is deferred. `unclog_1d` (24h, any profit) and `max_hold`
(cexit_max_days) are left alone, so a trade still cannot run unbounded -- the
window simply widens from 12h to HOLD_WINDOW_HOURS.
"""
from typing import Optional


class HoldWindowMixin:
    """Defer `unclog_12h` until HOLD_WINDOW_HOURS have elapsed."""

    # Set to the label's own horizon by the arm, never tuned. None = inherit
    # base behaviour unchanged.
    HOLD_WINDOW_HOURS: Optional[float] = None

    # Only the 12h rule is deferred; unclog_1d and max_hold still bound the trade.
    _DEFERRABLE_EXITS = ("unclog_12h",)

    # Feature 030: the model key (Framework/ModelKey.py). This class declares the names it
    # owns; the lists merge along the MRO. ATTRIBUTES/METHODS shape trained weights;
    # IGNORED act only after the model. The audit test keeps every name classified.
    MODEL_KEY_IGNORED = (
        "HOLD_WINDOW_HOURS", "_DEFERRABLE_EXITS",
    )

    def custom_exit(
        self, pair, trade, current_time, current_rate, current_profit, **kwargs
    ) -> Optional[str]:
        reason = super().custom_exit(
            pair, trade, current_time, current_rate, current_profit, **kwargs
        )
        if self.HOLD_WINDOW_HOURS and reason in self._DEFERRABLE_EXITS:
            hours = (current_time - trade.open_date_utc).total_seconds() / 3600.0
            if hours < self.HOLD_WINDOW_HOURS:
                return None
        return reason
