"""A per-pair close history longer than the bot's candle buffer (generic; no strategy code here).

Dry-run and live bots get only ~920 bars of startup history, but some trailing instruments need more
(a 2880-bar volatility window, a 1000-bar data-quality share). The store holds, per pair, the
gap-filled candle grid with two columns:

  close   the candle close (gap-filled bars repeat the previous close, as freqtrade fills them)
  has     1 if the candle exists in the raw data, 0 if it was gap-filled

Since 2026-10-24 the grid also holds the candle's open, high, low and volume (gap-filled bars: open = high
= low = close = the previous close, volume 0, exactly as freqtrade fills them), so a consumer can compute
indicators on a history longer than the live buffer (the transition stand-in, v2).

It is SEEDED from the local OHLCV files (`download-data` output, the same files backtests read) and
APPENDED from the bot's own dataframe on every call to `update`, so after startup the long windows stay
exact. Appended candles count as present (`has` 1): see `PRESENCE_NOTE`.

`series(pair, first_needed)` returns the history only if it is CONTIGUOUS through the bot's dataframe:
the local data must reach the dataframe's first candle (no missing seam between the download and the
live buffer). Otherwise it returns None and the caller must treat its instrument as unavailable, never
as a value. In a backtest the local data covers the whole range, so the seam check is trivially met.

Appended live candles are persisted under `persist_dir` (optional) so a restart between downloads keeps
them; persisted rows are merged on seed, the raw files win where both exist.
"""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

PRESENCE_NOTE = ("appended (live) candles count as present. Measured on binanceus 15m (2026-10-24): the raw files "
                 "hold MANY real zero-volume candles (ZEC 88,609, DOT 67,233, SOL 6,994 of ~170-200k), so volume > 0 "
                 "is not presence (agreement 0.47-0.97); the candles freqtrade fills are rare (65-110 per pair over "
                 "5.7 years, ~0.05%). So `has` 1 is wrong only on those rare filled bars; a seam hole is marked 0.")


OHLCV = ("open", "high", "low", "close", "volume")


def _grid(raw: pd.DataFrame, timeframe_min: int) -> pd.DataFrame:
    """raw candles (date + close, optionally open/high/low/volume) -> the gap-filled grid (date, OHLCV
    present in the input, has), freqtrade-style fill (filled bars: o=h=l=c=previous close, volume 0)."""
    cols = [c for c in OHLCV if c in raw.columns]
    raw = raw[["date", *cols]].drop_duplicates("date").sort_values("date")
    if raw.empty:
        return pd.DataFrame({"date": pd.Series(dtype="datetime64[ns, UTC]"), **{c: [] for c in cols}, "has": []})
    idx = pd.date_range(raw["date"].iloc[0], raw["date"].iloc[-1], freq=f"{timeframe_min}min", tz="UTC")
    g = raw.set_index("date").reindex(idx)
    has = g["close"].notna().astype(np.int8)
    close = g["close"].ffill()
    out = {"date": idx}
    for c in cols:
        if c == "close":
            out[c] = close.to_numpy(float)
        elif c == "volume":
            out[c] = g[c].fillna(0.0).to_numpy(float)
        else:
            out[c] = g[c].fillna(close).to_numpy(float)
    out["has"] = has.to_numpy(np.int8)
    return pd.DataFrame(out)


class CloseHistoryStore:
    def __init__(self, datadir: Path, timeframe: str = "15m", data_format: str = "feather",
                 candle_type: str = "spot", persist_dir: Path | None = None, keep_bars: int | None = None):
        self.datadir = Path(datadir)
        self.timeframe = timeframe
        self.tf_min = int(pd.Timedelta(timeframe).total_seconds() // 60)
        self.data_format = data_format
        self.candle_type = candle_type
        self.persist_dir = Path(persist_dir) if persist_dir else None
        self.keep_bars = keep_bars
        self._h: dict[str, pd.DataFrame] = {}

    # ------------------------------------------------------------------ seed / persist
    def _load_raw(self, pair: str) -> pd.DataFrame:
        from freqtrade.data import history

        df = history.load_pair_history(pair=pair, timeframe=self.timeframe, datadir=self.datadir,
                                       fill_up_missing=False, data_format=self.data_format,
                                       candle_type=self.candle_type)
        return df[["date", *OHLCV]] if not df.empty else pd.DataFrame(columns=["date", *OHLCV])

    def _persist_path(self, pair: str) -> Path | None:
        return self.persist_dir / f"{pair.replace('/', '_')}.parquet" if self.persist_dir else None

    def _seed(self, pair: str) -> pd.DataFrame:
        raw = self._load_raw(pair)
        g = _grid(raw, self.tf_min)
        p = self._persist_path(pair)
        if p is not None and p.is_file():
            extra = pd.read_parquet(p)
            extra = extra[extra["date"] > (g["date"].iloc[-1] if len(g) else pd.Timestamp.min.tz_localize("UTC"))]
            if len(extra):
                g = self._append(g, extra)
        return g

    def _append(self, g: pd.DataFrame, new: pd.DataFrame) -> pd.DataFrame:
        new = new[["date", *[c for c in OHLCV if c in g.columns or len(g) == 0], "has"]].sort_values("date")
        if len(g) == 0:
            return new.reset_index(drop=True)
        last = g["date"].iloc[-1]
        new = new[new["date"] > last]
        if new.empty:
            return g
        first = new["date"].iloc[0]
        if first != last + pd.Timedelta(minutes=self.tf_min):
            # a hole between the history and the new candles: keep the grid contiguous but mark it filled
            gap = pd.date_range(last + pd.Timedelta(minutes=self.tf_min), first - pd.Timedelta(minutes=self.tf_min),
                                freq=f"{self.tf_min}min", tz="UTC")
            last_c = g["close"].iloc[-1]
            fill = {"date": gap, **{c: (0.0 if c == "volume" else last_c) for c in OHLCV if c in g.columns}, "has": np.int8(0)}
            g = pd.concat([g, pd.DataFrame(fill)], ignore_index=True)
        out = pd.concat([g, new], ignore_index=True)
        if self.keep_bars:
            out = out.iloc[-self.keep_bars:].reset_index(drop=True)
        return out

    # ------------------------------------------------------------------ public
    def update(self, pair: str, dataframe: pd.DataFrame) -> None:
        """Append the dataframe's candles newer than the history (present: has 1; see PRESENCE_NOTE)."""
        g = self._h.get(pair)
        if g is None:
            g = self._seed(pair)
        new = pd.DataFrame({"date": pd.to_datetime(dataframe["date"], utc=True),
                            **{c: dataframe[c].astype(float).to_numpy() for c in OHLCV if c in dataframe.columns},
                            "has": np.ones(len(dataframe), dtype=np.int8)})
        before = len(g)
        g = self._append(g, new)
        self._h[pair] = g
        p = self._persist_path(pair)
        if p is not None and len(g) > before:
            p.parent.mkdir(parents=True, exist_ok=True)
            tail = g.iloc[before:] if self.keep_bars is None else g[g["date"] > new["date"].min() - pd.Timedelta(days=1)]
            old = pd.read_parquet(p) if p.is_file() else None
            merged = tail if old is None else pd.concat([old, tail]).drop_duplicates("date", keep="last")
            merged.sort_values("date").to_parquet(p, index=False)

    def series(self, pair: str, first_needed: pd.Timestamp) -> pd.DataFrame | None:
        """The contiguous (date, close, has) history, or None if it does not reach `first_needed` from
        the local data side (a seam between the download and the bot's buffer)."""
        g = self._h.get(pair)
        if g is None:
            g = self._seed(pair)
            self._h[pair] = g
        if g.empty or g["date"].iloc[0] > first_needed:
            return None
        return g
