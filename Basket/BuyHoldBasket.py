"""
BuyHoldBasket — the passive benchmark.

Buys each whitelist coin ONCE, up front, in equal DOLLAR amounts (1/n of the
starting balance) and then holds forever: no rebalancing, no skim, no CPPI
floor. This is the "just hold the basket" baseline every active variant is
measured against — it tracks the equal-weight market return (≈ freqtrade's
`market change`), but as a first-class strategy you can backtest and
walk-forward directly.

Why override the base's entry + sizing: the base fills gradually on dips and
sizes each entry as a % of CURRENT portfolio value. With no rebalancing that
over-weights coins that happen to enter after early winners have inflated the
book (a single winner's per-trade ratio blew up to 40000%). A clean benchmark
needs equal INITIAL DOLLARS bought at the same time, which is exactly what a
passive equal-weight hold is.
"""

from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path

from pandas import DataFrame

sys.path.append(str(Path(__file__).parent))
from BasketStrategy import BasketStrategy


class BuyHoldBasket(BasketStrategy):
    # Truly exit-proof: a passive hold must never take an ROI/stop exit. The
    # base's 10 000% ROI is NOT enough — a coin bought at its bottom can spike
    # past it (ZEC's bad 2355 high tick = a 103x move that tripped ROI and
    # re-opened the trade). Set both far beyond any real or bad-tick move.
    minimal_roi = {"0": 100000.0}
    stoploss = -1.0

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        # Enter as soon as data is available (first post-startup candle) and
        # hold — no dip timing. One position per coin; it never closes.
        dataframe["enter_long"] = 1
        return dataframe

    def custom_stake_amount(
        self,
        pair: str,
        current_time: datetime,
        current_rate: float,
        proposed_stake: float,
        min_stake: float | None,
        max_stake: float,
        leverage: float,
        entry_tag: str | None,
        side: str,
        **kwargs,
    ) -> float:
        # Equal DOLLAR per coin off the STARTING balance, so the held basket
        # tracks the equal-weight market return rather than a value-scaled weight.
        start = float(self.config.get("dry_run_wallet") or 0.0)
        # 0.99 leaves headroom for entry fees on the earlier coins so the LAST
        # coin still has ~1/n cash to enter (otherwise fees squeeze it out).
        stake = min(start / self._n_coins() * 0.99, max_stake)
        if min_stake and stake < min_stake:
            return 0.0
        return max(stake, 0.0)

    def adjust_trade_position(self, *args, **kwargs) -> float | None:
        # Passive hold: never rebalance, trim, or skim.
        return None
