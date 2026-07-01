"""
InverseVolBasket — inverse-volatility (risk-parity lite) weighting.

Weight each coin inversely to its trailing volatility: calm coins get more,
volatile coins less, so each contributes roughly equal RISK rather than equal
dollars. Risk-based cousin of constant-mix; tends to lower drawdown than
equal-weight. Contrarian on volatility (adds to whatever has calmed down).
"""

from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path

from pandas import DataFrame
from freqtrade.strategy import IntParameter

sys.path.append(str(Path(__file__).parent))
from BasketStrategy import BasketStrategy


class InverseVolBasket(BasketStrategy):

    # Must cover vol_lookback (max 100) + BB window.
    startup_candle_count = 150

    # Trailing window (candles) for the volatility estimate.
    vol_lookback = IntParameter(10, 100, default=30, space="buy")

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe = super().populate_indicators(dataframe, metadata)
        # Causal trailing volatility = stdev of simple returns.
        dataframe["vol"] = (
            dataframe["close"].pct_change()
            .rolling(self.vol_lookback.value, min_periods=self.vol_lookback.value)
            .std()
        )
        return dataframe

    def get_target_weight(
        self, pair: str, current_time: datetime, dataframe: DataFrame
    ) -> float:
        vols = self._cross_section("vol", current_time)
        inv = {p: 1.0 / v for p, v in vols.items() if v and v > 0}
        total = sum(inv.values())
        if not total or pair not in inv:
            return 0.0
        # Inverse-vol weights spread over the deployable (non-cash) fraction.
        return (1.0 - self.cash_target_weight.value) * inv[pair] / total
