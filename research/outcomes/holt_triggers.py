"""038 multi-Holt vote as SimpleTrigger entry triggers (research/holt_trigger/preregistration.json).

At every 16th candle (a fixed date grid: epoch candle index % 16 == 0), damped Holt is fitted to the
trailing 32 / 64 / 128 / 256 log closes and forecasts close 48 candles (12 h) ahead -- 038's vote C:
- `holt_vote_up` (momentum): at least 3 of the 4 forecast a log return > +0.003 (the 30 bp round trip);
- `holt_vote_fade` (reversion): at least 3 of the 4 forecast < -0.003. The long-only mirror: buy after
  the trend models call a fall (on spot the up-signal itself cannot be faded).
A candle without 256 closes of history does not vote.

Forecasts are cached by their DATA, not by pair: key (window, candle date, log close at t, log close at
t - w + 1). The trigger therefore never reads the pair (Constitution X) and the same OHLCV always gives
the same output; the cache only saves refits. Shards live in user_data/data/forecast_038/trigger_cache/
(one per process, merged on load) and are seeded from 038's saved Holt forecasts by
research/holt_trigger/seed_cache.py.
"""
from __future__ import annotations

import os
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from research.outcomes.triggers import Trigger, register

STRAT = Path(__file__).resolve().parents[2]
CACHE_DIR = STRAT.parents[1] / "user_data" / "data" / "forecast_038" / "trigger_cache"
WINDOWS = (32, 64, 128, 256)
HORIZON = 48
NEED = 3
THRESHOLD = 0.003
STRIDE = 16
BAR = pd.Timedelta(minutes=15)
EPOCH = pd.Timestamp(0, tz="UTC")

_CACHE: dict | None = None
_NEW: dict = {}


def _key(w: int, date_ns: int, a: float, b: float) -> tuple:
    return (w, date_ns, round(a, 10), round(b, 10))


def _cache_on() -> bool:
    """HOLT_TRIGGER_NOCACHE=1 refits everything. Causality checks MUST run with it: a cached value
    fitted on the full history would be handed to a truncated prefix and hide any forward read."""
    return os.environ.get("HOLT_TRIGGER_NOCACHE", "") != "1"


def _load() -> dict:
    global _CACHE
    if not _cache_on():
        return {}
    if _CACHE is None:
        _CACHE = {}
        if CACHE_DIR.is_dir():
            for f in sorted(CACHE_DIR.glob("*.feather")):
                try:
                    d = pd.read_feather(f)
                except Exception:  # a shard being written by another process: skip it
                    continue
                for w, t, a, b, fc in d[["w", "date_ns", "a", "b", "fc"]].itertuples(index=False):
                    _CACHE[_key(int(w), int(t), a, b)] = fc
    return _CACHE


def _save() -> None:
    if not _NEW or not _cache_on():
        _NEW.clear()
        return
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    shard = CACHE_DIR / f"run_{os.getpid()}.feather"
    old = pd.read_feather(shard) if shard.exists() else None
    new = pd.DataFrame([(k[0], k[1], k[2], k[3], v) for k, v in _NEW.items()], columns=["w", "date_ns", "a", "b", "fc"])
    out = new if old is None else pd.concat([old, new], ignore_index=True)
    tmp = shard.with_suffix(".tmp")
    out.to_feather(tmp)
    tmp.replace(shard)
    _NEW.clear()


def _holt(x: np.ndarray, n: int) -> float:
    sys.path.append(str(STRAT / "utils"))
    import Forecasters as FC

    f = FC.make_forecaster(FC.ForecasterType.HOLT)
    f.set_detrend(False)
    f.set_smooth(False)
    try:
        return float(np.asarray(f.forecast(x.copy(), n), float)[-1])  # [-1] = exactly n ahead
    except Exception:
        return float("nan")


def votes(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """(ups, downs) per candle: how many windows forecast beyond +/- THRESHOLD (0 off the grid)."""
    cache = _load()
    lc = np.log(df["close"].to_numpy(float))
    dates = pd.DatetimeIndex(pd.to_datetime(df["date"], utc=True))
    k = ((dates - EPOCH) // BAR).to_numpy()
    rows = np.flatnonzero((k % STRIDE == 0) & (np.arange(len(lc)) >= max(WINDOWS) - 1))
    ups = np.zeros(len(lc), dtype=int)
    downs = np.zeros(len(lc), dtype=int)
    date_ns = dates.asi8
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for t in rows:
            for w in WINDOWS:
                key = _key(w, int(date_ns[t]), lc[t], lc[t - w + 1])
                fc = cache.get(key)
                if fc is None:
                    fc = _holt(lc[t - w + 1: t + 1], HORIZON) - lc[t]
                    cache[key] = fc
                    if _cache_on():
                        _NEW[key] = fc
                if fc > THRESHOLD:
                    ups[t] += 1
                elif fc < -THRESHOLD:
                    downs[t] += 1
    _save()
    return ups, downs


def _up(df: pd.DataFrame) -> pd.Series:
    return pd.Series(votes(df)[0] >= NEED, index=df.index)


def _fade(df: pd.DataFrame) -> pd.Series:
    return pd.Series(votes(df)[1] >= NEED, index=df.index)


def register_holt_triggers(*, replace: bool = False) -> list[Trigger]:
    src = "038 vote C (research/forecast, preregistration_vote.json; research/holt_trigger)"
    return [
        register(Trigger(name="holt_vote_up", evaluate=_up, required_columns=("date", "close"), source=src,
                         description="3 of 4 damped-Holt windows (32/64/128/256) forecast > +30 bp over 48 candles",
                         family="momentum",
                         family_reason="the forecasts extend the recent slope up: buys strength after a rise"),
                 replace=replace),
        register(Trigger(name="holt_vote_fade", evaluate=_fade, required_columns=("date", "close"), source=src,
                         description="3 of 4 damped-Holt windows forecast < -30 bp over 48 candles (long-only fade)",
                         family="reversion",
                         family_reason="the forecasts extend a recent fall: buys weakness after a drop"),
                 replace=replace),
    ]
