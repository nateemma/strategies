"""
VolTargetBasket — volatility-targeting overlay on an equal-weight basket.

Scale TOTAL deployed exposure so the basket's realised volatility tracks a
target: cut exposure when vol spikes, add it back when markets calm. Same
risk-management spirit as CPPI, but driven by realised volatility rather than a
drawdown floor. Within the deployed sleeve, coins are equal-weight; the
remainder is cash.

Cash is a DYNAMIC residual (1 - exposure), so — like CPPI — this class fixes
`cash_target_weight` (plain attr, not hyperopt) instead of treating it as a
fixed reserve.
"""

from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path

from pandas import DataFrame
from freqtrade.strategy import DecimalParameter, IntParameter

sys.path.append(str(Path(__file__).parent))
from BasketStrategy import BasketStrategy


class VolTargetBasket(BasketStrategy):

    startup_candle_count = 150  # covers vol_lookback (max 100) + BB window

    # Cash is the dynamic residual of the vol-target exposure → not a reserve.
    cash_target_weight = 0.0

    vol_lookback = IntParameter(10, 100, default=30, space="buy")
    # Annualised portfolio volatility target (e.g. 0.60 = 60%/yr).
    target_vol = DecimalParameter(0.2, 1.5, default=0.6, decimals=2, space="buy")

    def _exposure(self, current_time: datetime) -> float:
        rm = self._return_matrix(self.vol_lookback.value, current_time)
        if rm is None or rm.empty or rm.shape[1] == 0:
            return 0.0
        # Equal-weight portfolio return series → realised (annualised) vol.
        port = rm.mean(axis=1)
        per_period = float(port.std())
        if per_period <= 0:
            return 1.0
        annualised = per_period * (self._periods_per_year() ** 0.5)
        return min(1.0, max(0.0, self.target_vol.value / annualised))

    def get_target_weight(
        self, pair: str, current_time: datetime, dataframe: DataFrame
    ) -> float:
        # Exposure is a portfolio-level scalar; split equally across coins.
        return self._exposure(current_time) / self._n_coins()
