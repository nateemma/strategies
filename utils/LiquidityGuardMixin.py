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

Signal-rank mode (`{"signals": K, "quantile": q}`, research/tb_model/preregistration_liqsr.json): a
signal is kept only where its candle's quote volume exceeds the q-quantile of the pair's previous K
TRADED signal candles (rows of this mask with quote volume > 0, current row excluded) -- the causal
form of "the pair's top (1 - q) of its own signal candles". A signal with fewer than K traded signals
before it is kept (no history, no filter). `"observe": True` filters nothing. With the environment
variable FT_LIQ_GUARD_DUMP set to a directory, each call writes the signal rows (date, qv, thr, n,
keep) to <dir>/<pair>_<first date>.csv for the instrument.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd


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
        if "signals" in g:
            keep = np.asarray(mask, dtype=bool) & self._signal_rank_ok(dataframe, np.asarray(mask, dtype=bool), g)
            return self._clear_tags(keep)
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
        return self._clear_tags(np.asarray(mask, dtype=bool) & ok)

    def _signal_rank_ok(self, dataframe, mask: np.ndarray, g: dict) -> np.ndarray:
        k, q = int(g["signals"]), float(g["quantile"])
        qv = (dataframe["volume"] * dataframe["close"]).to_numpy(float)
        traded = np.flatnonzero(mask & (qv > 0))
        hist = pd.Series(qv[traded])
        # threshold for row i = q-quantile of the K traded signals before it
        thr_t = hist.rolling(k, min_periods=k).quantile(q).to_numpy()
        n_t = np.minimum(np.arange(1, len(traded) + 1), k)
        rows = np.flatnonzero(mask)
        pos = np.searchsorted(traded, rows) - 1  # last traded signal strictly before the row
        has = pos >= 0  # no traded signal yet (or none at all): no threshold
        thr = np.full(len(rows), np.nan)
        thr[has] = thr_t[pos[has]]
        n = np.zeros(len(rows), dtype=int)
        n[has] = n_t[pos[has]]
        row_ok = np.isnan(thr) | (qv[rows] > thr)
        if g.get("observe"):
            row_ok = np.ones(len(rows), dtype=bool)
        ok = np.ones(len(mask), dtype=bool)
        ok[rows] = row_ok
        dump = os.environ.get("FT_LIQ_GUARD_DUMP")
        if dump and len(rows):
            pair = str(getattr(self, "curr_pair", "") or "pair").replace("/", "_")
            dates = pd.DatetimeIndex(dataframe["date"])
            pd.DataFrame({"date": dates[rows], "qv": qv[rows], "thr": thr, "n": n, "keep": row_ok}).to_csv(
                Path(dump) / f"{pair}_{dates[0]:%Y%m%d%H%M}.csv", index=False)
        return ok

    def _clear_tags(self, keep: np.ndarray) -> np.ndarray:
        tags = getattr(self, "_st_tags", None)
        if tags is not None and len(tags) == len(keep):
            tags = tags.copy()
            tags[~keep] = ""
            self._st_tags = tags
        return keep
