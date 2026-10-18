# pragma pylint: disable=C0103, C0114, C0115, C0116, C0301
# type: ignore
"""Take-profit at k x ATR(14)/close of the signal bar, unclamped (research/tb_model/
preregistration_tp2.json; the exit the replay `tp_grace_2` measured).

028's C2 switch (`take_profit_atr_mult`) resolves k x `atr_pct_roll` (a 20-bar mean of ATR(30)/close),
clamped to [1%, 5%]. This mixin keeps that switch and its per-trade caching but resolves it from
the triple-barrier label's own ATR (Framework.TrainingSignals._atr: Wilder smoothing, alpha 1/14)
/ close of the signal bar, computed here from the analysed frame's high/low/close, with no clamp.
Computed in the resolver, not populate_indicators, which is a model-key method (the arm keeps the
width book's weights). Stops, grace and every other exit are unchanged. Nothing in Framework/ changes.
"""

from datetime import timedelta

import numpy as np

from Framework import TrainingSignals


class Atr14TakeProfitMixin:
    def _apply_atr_take_profit(self, params, trade):
        k = params.take_profit_atr_mult
        if k is None or trade is None or getattr(self, "dp", None) is None:
            return params
        from dataclasses import replace

        from freqtrade.exchange import timeframe_to_minutes

        cache = self.__dict__.setdefault("_atr14_tp_cache", {})
        start = self.trade_clock_start(trade)
        key = (getattr(trade, "id", None) or (trade.pair, trade.open_date_utc), start, k)
        if key not in cache:
            df, _ = self.dp.get_analyzed_dataframe(trade.pair, self.timeframe)
            signal = start - timedelta(minutes=timeframe_to_minutes(self.timeframe))
            i = np.flatnonzero((df["date"] == signal).to_numpy())
            if not len(i):
                return params
            atr = TrainingSignals._atr(df.iloc[: i[-1] + 1])[-1] / max(float(df["close"].iloc[i[-1]]), 1e-12)
            if not np.isfinite(atr):
                return params
            cache[key] = float(k * atr)
        return replace(params, take_profit=cache[key])
