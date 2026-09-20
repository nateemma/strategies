"""Data path: feathers -> indicators -> raw8 -> labels, per pair, continuous (research R1).

Every function here operates on a pair's continuous series; row selection by date /
window happens downstream (splits), never inside this module, so warm-up and the label
look-ahead stay correct at every window boundary.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Dict, Optional, Sequence

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


def zero_volume_share(df: pd.DataFrame) -> float:
    """Fraction of `df`'s volume-known rows whose `volume` is exactly 0.0.

    freqtrade fills a missing candle with open=high=low=close(prev) and volume=0 --
    this is the resulting contamination measure (research: FIX for "contamination must be
    visible"). Rows with a NaN `volume` (e.g. a caller-supplied frame with no real volume
    data) are excluded from both the numerator and denominator, so NaN input never gets
    silently counted as "not zero"; NaN is returned if every row (or the frame itself) is
    empty of volume information.
    """
    if len(df) == 0:
        return float("nan")
    vol = df["volume"].to_numpy(dtype=np.float64)
    known = ~np.isnan(vol)
    if not known.any():
        return float("nan")
    return float((vol[known] == 0.0).mean())


def ohlc_equal_share(df: pd.DataFrame) -> float:
    """Fraction of `df`'s rows where open==high==low==close (a strict superset of the
    gap-fill rows `zero_volume_share` measures -- a real bar can also happen to be flat).
    NaN for an empty frame."""
    if len(df) == 0:
        return float("nan")
    equal = (
        (df["open"] == df["high"]) & (df["high"] == df["low"]) & (df["low"] == df["close"])
    )
    return float(equal.to_numpy().mean())


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


# ---------------------------------------------------------------------------
# `--inventory` CLI (research: FIX for "contamination must be visible ... in the
# inventory"). Reads raw feathers only (`load_pair`), never `populate`/`raw8`/labels, so
# it stays light and never touches the indicator/label caches.
# ---------------------------------------------------------------------------


def _inventory_row(pair: str) -> Dict[str, Any]:
    df = load_pair(pair)
    return {
        "pair": pair,
        "rows": len(df),
        "first_date": df["date"].min(),
        "last_date": df["date"].max(),
        "zero_vol_share": zero_volume_share(df),
        "ohlc_equal_share": ohlc_equal_share(df),
    }


def _print_inventory_markdown(rows: Sequence[Dict[str, Any]]) -> None:
    header = ["pair", "rows", "first_date", "last_date", "zero_vol_share", "ohlc_equal_share"]
    print("| " + " | ".join(header) + " |")
    print("|" + "|".join(["---"] * len(header)) + "|")
    for row in rows:
        first_date = pd.Timestamp(row["first_date"]).strftime("%Y-%m-%d")
        last_date = pd.Timestamp(row["last_date"]).strftime("%Y-%m-%d")
        print(
            f"| {row['pair']} | {row['rows']} | {first_date} | {last_date} | "
            f"{row['zero_vol_share']:.1%} | {row['ohlc_equal_share']:.1%} |"
        )


def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = argparse.ArgumentParser(prog="python -m research.clfstudy.data")
    parser.add_argument(
        "--inventory", action="store_true", help="print a per-pair contamination inventory"
    )
    parser.add_argument(
        "--pairs", nargs="+", default=None, help="pairs to inventory (default: PAIRS)"
    )
    args = parser.parse_args(argv)

    if not args.inventory:
        parser.error("no command given; use --inventory")

    pairs = tuple(args.pairs) if args.pairs else PAIRS
    rows = [_inventory_row(pair) for pair in pairs]
    _print_inventory_markdown(rows)


if __name__ == "__main__":
    main()
