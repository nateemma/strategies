"""Market-wide regime instruments (feature 026, contract C1, research R1).

Each instrument maps {pair: OHLCV frame with `date`, `close`} (plus a BTC frame for `btctrend`) to a
date-indexed float series on the 15m grid. All are CAUSAL -- every value at bar t uses bars <= t
(backward rolling windows and adjust=False EWMs; verified by truncation in
`utils/tests/test_regime_state.py`) -- and MARKET-WIDE: one value per timestamp, identical for every
pair. Pair names appear only as dict keys (constitution X).

    breadth   share of pairs with close > EMA(close, 768)                      (768 bars = 8 days)
    volreg    cross-pair median of std(log-returns, 96), over its own trailing-2880 median
    persist   cross-pair median variance ratio VR(16) = var(16-bar ret) / (16 * var(1-bar ret)),
              each variance over a trailing 2880-bar window; > 1 trending, < 1 mean-reverting
    btctrend  BTC close / EMA(BTC close, 768) - 1

A cross-pair value needs at least `min_pairs` pairs defined at that bar, else NaN. NaN means "state
unknown"; a gate treats it as "on" (reversion allowed), the 024 behaviour.

Lengths are fixed by research R1 and are not tuning knobs.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


EMA_BARS = 768
VOL_BARS = 96
TRAIL_BARS = 2880
VR_Q = 16
MIN_PAIRS = 6


def _close(frame: pd.DataFrame) -> pd.Series:
    s = pd.Series(frame["close"].to_numpy(dtype=float), index=pd.DatetimeIndex(frame["date"]))
    return s[~s.index.duplicated(keep="last")].sort_index()


def _wide(frames: dict, per_pair) -> pd.DataFrame:
    """Apply `per_pair(close)` to each pair on its OWN bars, then align on the union of dates."""
    cols = {k: per_pair(_close(f)) for k, f in frames.items() if len(f)}
    if not cols:
        return pd.DataFrame()
    return pd.DataFrame(cols).sort_index()


def _cross(wide: pd.DataFrame, agg: str, min_pairs: int) -> pd.Series:
    if wide.empty:
        return pd.Series(dtype=float)
    n = wide.notna().sum(axis=1)
    out = getattr(wide, agg)(axis=1, skipna=True)
    return out.where(n >= min_pairs)


def breadth(frames: dict, ema: int = EMA_BARS, min_pairs: int = MIN_PAIRS) -> pd.Series:
    def above(c):
        e = c.ewm(span=ema, adjust=False, min_periods=ema).mean()
        return (c > e).astype(float).where(e.notna())

    return _cross(_wide(frames, above), "mean", min_pairs).rename("breadth")


def volreg(
    frames: dict, win: int = VOL_BARS, trail: int = TRAIL_BARS, min_pairs: int = MIN_PAIRS
) -> pd.Series:
    def vol(c):
        return np.log(c).diff().rolling(win, min_periods=win).std()

    m = _cross(_wide(frames, vol), "median", min_pairs)
    base = m.rolling(trail, min_periods=trail).median()
    return (m / base).rename("volreg")


def persist(
    frames: dict, q: int = VR_Q, win: int = TRAIL_BARS, min_pairs: int = MIN_PAIRS
) -> pd.Series:
    def vr(c):
        lc = np.log(c)
        v1 = lc.diff().rolling(win, min_periods=win).var()
        vq = (lc - lc.shift(q)).rolling(win, min_periods=win).var()
        return vq / (q * v1)

    return _cross(_wide(frames, vr), "median", min_pairs).rename("persist")


def btctrend(btc: pd.DataFrame, ema: int = EMA_BARS) -> pd.Series:
    c = _close(btc)
    e = c.ewm(span=ema, adjust=False, min_periods=ema).mean()
    return (c / e - 1.0).rename("btctrend")


INSTRUMENTS = {
    "breadth": lambda frames, btc: breadth(frames),
    "volreg": lambda frames, btc: volreg(frames),
    "persist": lambda frames, btc: persist(frames),
    "btctrend": lambda frames, btc: btctrend(btc),
}


def compute(instrument: str, frames: dict, btc: pd.DataFrame | None) -> pd.Series:
    if instrument not in INSTRUMENTS:
        raise KeyError(f"unknown regime instrument {instrument!r}; one of {sorted(INSTRUMENTS)}")
    if instrument == "btctrend" and btc is None:
        raise ValueError("btctrend needs a BTC frame")
    return INSTRUMENTS[instrument](frames, btc)
