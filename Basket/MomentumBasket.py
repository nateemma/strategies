"""
MomentumBasket — cross-sectional (relative) momentum with an absolute gate.

Overweight recent winners, drop losers. Trend-following like CPPI, but
RELATIVE (which coins are strongest) rather than absolute (CPPI = whole
portfolio vs a floor). "Dual momentum": a coin only qualifies if its trailing
return clears `mom_threshold` (absolute momentum) — otherwise its share goes to
cash — and among qualifiers weight is proportional to excess momentum.

Cash here is partly DYNAMIC: `cash_target_weight` is a fixed floor, but the
basket holds MORE cash whenever few coins clear the threshold (e.g. a broad
downtrend), which is the intended defensive behaviour.
"""

from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path

from pandas import DataFrame
from freqtrade.strategy import DecimalParameter, IntParameter

sys.path.append(str(Path(__file__).parent))
from BasketStrategy import BasketStrategy


class MomentumBasket(BasketStrategy):

    startup_candle_count = 250  # covers mom_lookback (max 200) + BB window

    # Trailing window (candles) for the momentum (return) measurement.
    mom_lookback = IntParameter(20, 200, default=60, space="buy")
    # Absolute-momentum floor: coins below this trailing return go to cash.
    mom_threshold = DecimalParameter(-0.10, 0.20, default=0.0, decimals=2, space="buy")

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe = super().populate_indicators(dataframe, metadata)
        # Causal trailing return over mom_lookback candles.
        dataframe["mom"] = dataframe["close"].pct_change(self.mom_lookback.value)
        return dataframe

    def get_target_weight(
        self, pair: str, current_time: datetime, dataframe: DataFrame
    ) -> float:
        moms = self._cross_section("mom", current_time)
        thr = self.mom_threshold.value
        # Only coins above the absolute-momentum floor qualify; weight ∝ excess.
        strong = {p: (m - thr) for p, m in moms.items() if m > thr}
        total = sum(strong.values())
        if not total or pair not in strong:
            return 0.0
        return (1.0 - self.cash_target_weight.value) * strong[pair] / total
