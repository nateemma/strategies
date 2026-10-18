"""Pair-relative liquidity entry guard (research/tb_model/preregistration_liq.json).

OFF by default (`LIQUIDITY_GUARD = None`): `get_entry_conditions` is exactly the parent's.

When ON (`{"window": bars, "min_ratio": r}`), an entry signal is kept only where the candle's quote
volume (volume x close) exceeds r x the pair's own trailing median quote volume over `window` bars,
current candle included (causal). Rows without a full window are refused, so the strategy's
`startup_candle_count` must cover the window. The measure is relative to each pair's own history:
no pair-specific values.

`"nonzero": True` (research/tb_model/preregistration_liqnz.json) takes the median over the window's
non-zero-volume candles only. On binance.us a thin pair has 46-63% zero-volume 15m candles, so the
all-candle median is ~0 and the guard refused 2.8% of signals; the non-zero median is the pair's
typical TRADED candle. A window with no traded candle has no median (refused).
"""

from __future__ import annotations

import numpy as np


class LiquidityGuardMixin:
    LIQUIDITY_GUARD: dict | None = None
    MODEL_KEY_IGNORED = ("LIQUIDITY_GUARD",)
    # an entry filter applied after the model; it does not shape the weights
    MODEL_KEY_METHODS_IGNORED = ("get_entry_conditions",)

    def get_entry_conditions(self, dataframe):
        mask = super().get_entry_conditions(dataframe)
        g = self.LIQUIDITY_GUARD
        if g is None or mask is None:
            return mask
        window = int(g["window"])
        if int(self.startup_candle_count) < window:
            raise ValueError(f"{type(self).__name__}: LIQUIDITY_GUARD window {window} needs "
                             f"startup_candle_count >= {window} (got {self.startup_candle_count})")
        qv = dataframe["volume"] * dataframe["close"]
        if g.get("nonzero"):
            med = qv.where(qv > 0).rolling(window, min_periods=1).median()
            med.iloc[: window - 1] = np.nan  # a full window of bars, as below
        else:
            med = qv.rolling(window, min_periods=window).median()
        ok = (qv > float(g["min_ratio"]) * med).to_numpy()  # NaN median -> False
        keep = np.asarray(mask, dtype=bool) & ok
        tags = getattr(self, "_st_tags", None)
        if tags is not None and len(tags) == len(keep):
            tags = tags.copy()
            tags[~keep] = ""
            self._st_tags = tags
        return keep
