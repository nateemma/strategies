"""
ConstantMixBasket — constant-weight (constant-mix) rebalancing.

The CONTRARIAN variant: trim coins that rise above their target weight, add
to coins that fall below, always restoring toward equal weights. Sells
strength, buys weakness. Mean-reverting; outperforms in choppy / range-bound
markets and underperforms a strong sustained trend (it keeps trimming the
winner).
"""

from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path

from pandas import DataFrame
from freqtrade.strategy import BooleanParameter, DecimalParameter

sys.path.append(str(Path(__file__).parent))
from BasketStrategy import BasketStrategy


class ConstantMixBasket(BasketStrategy):

    # =====================================================================
    # TUNABLE PARAMETERS  (constant-mix specific; shared ones are in base)
    # =====================================================================

    # KEY BEHAVIORAL CHOICE ------------------------------------------------
    # runaway_trend_override: default OFF.
    #   OFF (default): the BB timing gate ALWAYS applies. A runaway coin that
    #     keeps rising simply rides — we only trim it on pullbacks above
    #     mid-BB. Purest constant-mix stance; a strong trender can stay
    #     overweight for a long time.
    #   ON: a coin whose drift exceeds `hard_drift_ceiling` gets trimmed
    #     REGARDLESS of the BB gate, so no single coin can dominate the
    #     basket indefinitely. Trades off some contrarian purity for a hard
    #     cap on concentration.
    runaway_trend_override = BooleanParameter(default=False, space="buy")
    # only consulted when the override is ON
    hard_drift_ceiling = DecimalParameter(0.08, 0.30, default=0.15, decimals=2, space="buy")
    # ---------------------------------------------------------------------

    def get_target_weight(
        self, pair: str, current_time: datetime, dataframe: DataFrame
    ) -> float:
        # Fixed equal-weight target (or target_weight_per_coin if set). Cash
        # is the fixed residual cash_target_weight.
        return self._equal_coin_weight()

    def _bb_gate_allows(
        self, adding: bool, current_rate: float, bb_mid: float, pair: str, drift: float
    ) -> bool:
        # Runaway override: an over-weight coin past the hard ceiling is
        # trimmed even if price is below mid-BB (i.e. gate bypassed). Adds and
        # normal trims still respect the BB gate.
        if (
            self.runaway_trend_override.value
            and not adding
            and drift > self.hard_drift_ceiling.value
        ):
            return True
        return super()._bb_gate_allows(adding, current_rate, bb_mid, pair, drift)
