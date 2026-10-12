# pragma pylint: disable=C0103, C0114, C0115, C0116, C0301
# type: ignore
# pylint: disable=import-error
"""027 EXPLORATORY (post-hoc, not out-of-sample): gated reversion whose take-profit is
k x atr_pct_roll of the trade's signal bar, clamped to [1%, 5%] net -- coupling C2 at its
early-chosen k (specs/027-target-aware-exits/feasibility.json).

The signal bar is the candle before the entry candle (open_date - 1 timeframe), read from the
analysed dataframe; the value is cached per trade. Implemented in the arm, not Framework/:
exploratory, and 027 built no framework switch."""

import dataclasses
import sys
from datetime import timedelta
from pathlib import Path


sys.path.append(str(Path(__file__).parent.parent))

from freqtrade.exchange import timeframe_to_minutes
from SimpleTrigger.SimpleTrigger_RevOnlyVolreg2 import SimpleTrigger_RevOnlyVolreg2


class SimpleTrigger_RevOnlyVolreg2_XC2(SimpleTrigger_RevOnlyVolreg2):
    TP_ATR_MULT = 2.8970711921574543
    TP_MIN, TP_MAX = 0.01, 0.05

    def _exit_params(self, trade=None):
        params = super()._exit_params(trade)
        if trade is None or isinstance(self, type) or getattr(self, "dp", None) is None:
            return params
        cache = self.__dict__.setdefault("_xc2_tp", {})
        key = getattr(trade, "id", None) or (trade.pair, trade.open_date_utc)
        if key not in cache:
            df, _ = self.dp.get_analyzed_dataframe(trade.pair, self.timeframe)
            signal = trade.open_date_utc - timedelta(minutes=timeframe_to_minutes(self.timeframe))
            row = df.loc[df["date"] == signal, "atr_pct_roll"]
            if row.empty:
                return params  # signal bar not visible: keep the preset take-profit
            atr = float(row.iloc[-1])
            cache[key] = min(max(self.TP_ATR_MULT * atr, self.TP_MIN), self.TP_MAX)
        return dataclasses.replace(params, take_profit=cache[key])
