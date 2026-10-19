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

`"runs": True` (research/tb_model/preregistration_liqsr2.json): one decision per signal run (consecutive
mask rows), taken at the run's first traded row, so a refused run cannot re-enter a few bars later on a
louder candle (liqsr_decomp.py: refused reversion entries came back a median 2 bars later).

`"size": {"share": s, "weights": (w_hi, w_lo)}` (same pre-registration): refuse nothing; tag each signal
":liqhi" (the rank test passes) or ":liqlo" and scale the parent's `stake_weight` by
w / (s x w_hi + (1 - s) x w_lo), capital-neutral at the pre-registered hi share s. Needs a parent with
`stake_weight` (the NNNC sizer books).
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
    MODEL_KEY_METHODS_IGNORED = ("get_entry_conditions", "stake_weight")
    SIZE_TAGS = (":liqhi", ":liqlo")

    def get_entry_conditions(self, dataframe):
        mask = super().get_entry_conditions(dataframe)
        g = self.LIQUIDITY_GUARD
        if g is None or mask is None:
            return mask
        if "signals" in g:
            m = np.asarray(mask, dtype=bool)
            ok = self._signal_rank_ok(dataframe, m, g)
            if g.get("size"):
                tags = self._st_tags.copy()
                for sel, sfx in ((m & ok, self.SIZE_TAGS[0]), (m & ~ok, self.SIZE_TAGS[1])):
                    tags[sel] = [f"{t}{sfx}" for t in tags[sel]]
                self._st_tags = tags
                return mask
            return self._clear_tags(m & ok)
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
        if g.get("runs") and len(rows):
            start = np.r_[True, np.diff(rows) != 1]  # first row of each run of consecutive signals
            run = np.cumsum(start) - 1
            traded_row = qv[rows] > 0
            first = np.full(run[-1] + 1, -1)
            for i in np.flatnonzero(traded_row)[::-1]:  # reversed: the earliest traded row wins
                first[run[i]] = i
            has_t = first[run] >= 0
            row_ok = np.where(has_t, row_ok[np.maximum(first[run], 0)], row_ok)
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

    def stake_weight(self, entry_tag) -> float:
        tag = str(entry_tag or "")
        for sfx, i in zip(self.SIZE_TAGS, (0, 1)):
            if tag.endswith(sfx):
                sz = self.LIQUIDITY_GUARD["size"]
                s, (wh, wl) = float(sz["share"]), sz["weights"]
                return super().stake_weight(tag[: -len(sfx)]) * (wh, wl)[i] / (s * wh + (1.0 - s) * wl)
        return super().stake_weight(entry_tag)

    def _clear_tags(self, keep: np.ndarray) -> np.ndarray:
        tags = getattr(self, "_st_tags", None)
        if tags is not None and len(tags) == len(keep):
            tags = tags.copy()
            tags[~keep] = ""
            self._st_tags = tags
        return keep
