"""Data path: feathers -> indicators -> raw8 -> labels, per pair, continuous (research R1).

Every function here operates on a pair's continuous series; row selection by date /
window happens downstream (splits), never inside this module, so warm-up and the label
look-ahead stay correct at every window boundary.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd

from research.clfstudy import CACHE_DIR, DATA_DIR, DATA_START, HARNESS_VERSION, PAIRS
from research.clfstudy import labels as labels_mod
from utils.DataframePopulator import DataframePopulator, DatasetType

_ROLL_WINDOW = 96


def _feather_path(pair: str) -> Path:
    base = pair.split("/")[0]
    return DATA_DIR / f"{base}_USDT-15m.feather"


def load_pair(pair: str) -> pd.DataFrame:
    """Read `<BASE>_USDT-15m.feather` for `pair`, filtered to `date >= DATA_START`.

    Returns a fresh, index-reset DataFrame with columns date/open/high/low/close/volume.
    """
    path = _feather_path(pair)
    df = pd.read_feather(path)
    df = df[df["date"] >= pd.Timestamp(DATA_START, tz="UTC")].reset_index(drop=True)
    return df


def populate(df: pd.DataFrame) -> pd.DataFrame:
    """Add the production `ind24` indicator set (MINIMAL) exactly as
    `BaseNNStrategy.populate_indicators` does (Framework/BaseNNStrategy.py:1390)."""
    df = df.copy()
    return DataframePopulator().add_indicators(df, dataset_type=DatasetType.MINIMAL)


def raw8(df: pd.DataFrame) -> pd.DataFrame:
    """Add the eight causal `raw8` columns (research R11) to a copy of `df`."""
    df = df.copy()
    close = df["close"]
    df["ret1"] = np.log(close / close.shift(1))
    df["ret4"] = np.log(close / close.shift(4))
    df["log_hc"] = np.log(df["high"] / close)
    df["log_lc"] = np.log(df["low"] / close)
    df["log_oc"] = np.log(df["open"] / close)
    df["range"] = (df["high"] - df["low"]) / close

    logvol = np.log(df["volume"] + 1.0)
    roll_mean = logvol.rolling(_ROLL_WINDOW, min_periods=_ROLL_WINDOW).mean()
    roll_std = logvol.rolling(_ROLL_WINDOW, min_periods=_ROLL_WINDOW).std()
    df["logvol_z"] = (logvol - roll_mean) / roll_std

    vol_mean = df["volume"].rolling(_ROLL_WINDOW, min_periods=_ROLL_WINDOW).mean()
    df["vol_ratio"] = np.log((df["volume"] + 1.0) / (vol_mean + 1.0))

    raw8_cols = [
        "ret1", "ret4", "log_hc", "log_lc", "log_oc", "range", "logvol_z", "vol_ratio",
    ]
    df[raw8_cols] = df[raw8_cols].fillna(0.0)
    return df


def _cache_path(pair: str, label_key: str) -> Path:
    base = pair.split("/")[0]
    return CACHE_DIR / "frames" / f"{base}_{HARNESS_VERSION}_{label_key}.feather"


def load_all(
    pairs: tuple[str, ...] = PAIRS, label: Optional[Dict[str, Any]] = None
) -> dict[str, pd.DataFrame]:
    """Load, populate, raw8 and label every pair, caching the result per pair.

    Cache key is `(pair, HARNESS_VERSION, label_key)`: bumping `HARNESS_VERSION`, or
    passing a different `label` (see `labels.label_key`), invalidates that combination's
    cache without touching any other pair/label's cached frame -- different labels never
    share a cache file. The cached frame carries an int8 `label` column
    (SELL=0/HOLD=1/BUY=2) computed for the requested `label`.

    `label=None` (default) reproduces the exact gbb-label behaviour this function has
    always had; the cache filename gains a `_gbb` suffix, a one-time cache miss.
    """
    lkey = labels_mod.label_key(label)
    frames: dict[str, pd.DataFrame] = {}
    cache_dir = CACHE_DIR / "frames"
    cache_dir.mkdir(parents=True, exist_ok=True)

    for pair in pairs:
        cpath = _cache_path(pair, lkey)
        if cpath.exists():
            frames[pair] = pd.read_feather(cpath)
            continue

        df = raw8(populate(load_pair(pair)))
        df["label"] = labels_mod.ternary_labels(df, label=label).astype(np.int8)
        df.to_feather(cpath)
        frames[pair] = df

    return frames
