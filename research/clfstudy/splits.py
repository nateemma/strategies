# pragma pylint: disable=C0103, C0114, C0115, C0116, C0301, C0303, C0411, C0413
# pylint: disable=import-error
# flake8: noqa: F401, E402

"""
Split geometry for the NNC classifier study (research R3, data-model.md §SplitSet).

A `SplitSet` is three anchored, expanding-window `Window`s (A, B, C). Every window shares
`train_start = 2022-01-01` and differs only in its evaluation (`test`) span; the training
span grows to stay 96 bars (`EMBARGO_BARS`, 2x `HORIZON`) clear of it.

Role timeline for one window, oldest to newest (see `assign_roles`)::

    train_start                                                    train_end   test_start   test_end
        |---------------- train -----------------|--embargo--|-- val --|--embargo--|-- test --|
                                        val_start-EMBARGO   val_start

    * train   = [train_start, val_start - EMBARGO)
    * embargo = [val_start - EMBARGO, val_start) U [train_end, test_start)   (belongs to neither
                train nor val; also not test) -- purge for the label horizon (HORIZON bars) plus
                slack, so no row used for early-stopping or evaluation shares a label window with
                a row the model was fitted on.
    * val     = [val_start, train_end)     (val_end == train_end)
    * test    = [test_start, test_end]     (closed interval; test_end is the last bar)
    * none    = everything outside [train_start, test_end]

val_start is placed so that the val slice is the last VAL_FRACTION (by wall-clock time, not row
count) of the [train_start, train_end) span::

    val_start = train_end - VAL_FRACTION * (train_end - train_start)

This module computes no features and reads no market data by itself; `included_pairs` only
consumes a caller-supplied `first_dates` mapping (see `splits build` for the one place that
reads a feather's `date` column, defensively).
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np
import pandas as pd

from research.clfstudy import (
    EMBARGO_BARS,
    HORIZON,
    MIN_HISTORY_DAYS,
    PAIRS,
    SPLITS_DIR,
    TIMEFRAME,
    VAL_FRACTION,
)

ROLES = ("train", "embargo", "val", "test", "none")

_BAR = pd.Timedelta(minutes=15)  # TIMEFRAME == "15m"; asserted below
assert TIMEFRAME == "15m", "splits.py hard-codes the 15m bar duration; update _BAR if this changes"


def _ts(s: str) -> pd.Timestamp:
    return pd.Timestamp(s, tz="UTC")


@dataclass(frozen=True)
class Window:
    window_id: str
    train_start: pd.Timestamp
    val_start: pd.Timestamp
    train_end: pd.Timestamp
    test_start: pd.Timestamp
    test_end: pd.Timestamp
    distant: bool


@dataclass(frozen=True)
class SplitSet:
    split_set: str
    embargo_bars: int
    val_fraction: float
    min_history_days: int
    timeframe: str
    pairs: List[str]
    windows: List[Window] = field(default_factory=list)


def _make_window(window_id: str, train_start: pd.Timestamp, test_start: pd.Timestamp,
                  test_end: pd.Timestamp, distant: bool) -> Window:
    embargo_delta = EMBARGO_BARS * _BAR
    train_end = test_start - embargo_delta
    val_start = train_end - VAL_FRACTION * (train_end - train_start)
    return Window(
        window_id=window_id,
        train_start=train_start,
        val_start=val_start,
        train_end=train_end,
        test_start=test_start,
        test_end=test_end,
        distant=distant,
    )


def build(set_id: str = "v1") -> SplitSet:
    """Build split set `set_id` (only "v1" is defined; research R3)."""
    if set_id != "v1":
        raise ValueError(f"unknown split set {set_id!r}; only 'v1' is defined")

    train_start = _ts("2022-01-01T00:00:00Z")
    windows = [
        _make_window("A", train_start, _ts("2023-07-01T00:00:00Z"), _ts("2023-12-31T23:45:00Z"), True),
        _make_window("B", train_start, _ts("2025-01-01T00:00:00Z"), _ts("2025-06-30T23:45:00Z"), False),
        _make_window("C", train_start, _ts("2026-01-01T00:00:00Z"), _ts("2026-06-30T23:45:00Z"), False),
    ]
    return SplitSet(
        split_set=set_id,
        embargo_bars=EMBARGO_BARS,
        val_fraction=VAL_FRACTION,
        min_history_days=MIN_HISTORY_DAYS,
        timeframe=TIMEFRAME,
        pairs=list(PAIRS),
        windows=windows,
    )


def assign_roles(dates: "pd.Series | pd.DatetimeIndex", w: Window) -> np.ndarray:
    """Vectorised role assignment for `dates` under window `w`. See module docstring.

    Returns an array of dtype '<U7' with values in {"train","embargo","val","test","none"}.
    Each row gets exactly one role.
    """
    idx = pd.DatetimeIndex(dates)
    if idx.tz is None:
        idx = idx.tz_localize("UTC")
    else:
        idx = idx.tz_convert("UTC")

    embargo_delta = EMBARGO_BARS * _BAR
    pre_val_embargo_start = w.val_start - embargo_delta

    roles = np.full(len(idx), "none", dtype="<U7")

    train_mask = (idx >= w.train_start) & (idx < pre_val_embargo_start)
    pre_val_embargo_mask = (idx >= pre_val_embargo_start) & (idx < w.val_start)
    val_mask = (idx >= w.val_start) & (idx < w.train_end)
    pre_test_embargo_mask = (idx >= w.train_end) & (idx < w.test_start)
    test_mask = (idx >= w.test_start) & (idx <= w.test_end)

    roles[train_mask] = "train"
    roles[pre_val_embargo_mask] = "embargo"
    roles[val_mask] = "val"
    roles[pre_test_embargo_mask] = "embargo"
    roles[test_mask] = "test"
    return roles


def included_pairs(w: Window, first_dates: Dict[str, pd.Timestamp]) -> List[str]:
    """Pairs with >= MIN_HISTORY_DAYS of history before `w.val_start`, in `first_dates` order."""
    included = []
    horizon = pd.Timedelta(days=MIN_HISTORY_DAYS)
    for pair, first_date in first_dates.items():
        fd = pd.Timestamp(first_date)
        if fd.tz is None:
            fd = fd.tz_localize("UTC")
        else:
            fd = fd.tz_convert("UTC")
        if fd + horizon <= w.val_start:
            included.append(pair)
    return included


def validate(ss: SplitSet) -> None:
    """Raise ValueError if any invariant in data-model.md §SplitSet or research R3 is broken."""
    if not ss.windows:
        raise ValueError("SplitSet has no windows")

    embargo_delta = ss.embargo_bars * _BAR
    if embargo_delta < HORIZON * _BAR:
        raise ValueError(f"embargo ({ss.embargo_bars} bars) must be >= HORIZON ({HORIZON} bars)")

    if not any(w.distant for w in ss.windows):
        raise ValueError("SplitSet has no distant window (Constitution II)")

    ordered = sorted(ss.windows, key=lambda w: w.test_start)
    for prev, cur in zip(ordered, ordered[1:]):
        if prev.test_end >= cur.test_start:
            raise ValueError(
                f"windows {prev.window_id} and {cur.window_id} are not disjoint/ordered "
                f"({prev.test_end} >= {cur.test_start})"
            )

    for w in ss.windows:
        if not (w.train_start < w.val_start < w.train_end < w.test_start < w.test_end):
            raise ValueError(f"window {w.window_id}: dates out of order {w}")

        if w.train_end + embargo_delta > w.test_start:
            raise ValueError(
                f"window {w.window_id}: train_end + embargo ({w.train_end + embargo_delta}) "
                f"> test_start ({w.test_start})"
            )

        expected_val_start = w.train_end - ss.val_fraction * (w.train_end - w.train_start)
        if abs((w.val_start - expected_val_start).total_seconds()) > 1e-6:
            raise ValueError(f"window {w.window_id}: val_start does not match the val_fraction rule")

        pre_val_embargo_start = w.val_start - embargo_delta
        if pre_val_embargo_start < w.train_start:
            raise ValueError(
                f"window {w.window_id}: training span too short for a full {ss.embargo_bars}-bar "
                f"pre-val embargo (pre_val_embargo_start {pre_val_embargo_start} < train_start "
                f"{w.train_start})"
            )


def _window_to_json(w: Window) -> dict:
    def fmt(ts: pd.Timestamp) -> str:
        return ts.tz_convert("UTC").strftime("%Y-%m-%dT%H:%M:%SZ")

    return {
        "window_id": w.window_id,
        "train_start": fmt(w.train_start),
        "val_start": fmt(w.val_start),
        "train_end": fmt(w.train_end),
        "test_start": fmt(w.test_start),
        "test_end": fmt(w.test_end),
        "distant": w.distant,
    }


def _window_from_json(d: dict) -> Window:
    return Window(
        window_id=d["window_id"],
        train_start=_ts(d["train_start"]),
        val_start=_ts(d["val_start"]),
        train_end=_ts(d["train_end"]),
        test_start=_ts(d["test_start"]),
        test_end=_ts(d["test_end"]),
        distant=bool(d["distant"]),
    )


def save(ss: SplitSet, path: "str | Path") -> None:
    """Write `ss` as JSON per contracts/schemas.md (SplitSet file)."""
    payload = {
        "split_set": ss.split_set,
        "embargo_bars": ss.embargo_bars,
        "val_fraction": ss.val_fraction,
        "min_history_days": ss.min_history_days,
        "timeframe": ss.timeframe,
        "pairs": ss.pairs,
        "windows": [_window_to_json(w) for w in ss.windows],
    }
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n")


def load(path: "str | Path") -> SplitSet:
    payload = json.loads(Path(path).read_text())
    return SplitSet(
        split_set=payload["split_set"],
        embargo_bars=payload["embargo_bars"],
        val_fraction=payload["val_fraction"],
        min_history_days=payload["min_history_days"],
        timeframe=payload["timeframe"],
        pairs=list(payload["pairs"]),
        windows=[_window_from_json(d) for d in payload["windows"]],
    )


def _print_window_summary(ss: SplitSet) -> None:
    for w in ss.windows:
        embargo_delta = ss.embargo_bars * _BAR
        print(
            f"window {w.window_id} (distant={w.distant}):\n"
            f"  train   [{w.train_start}, {w.val_start - embargo_delta})\n"
            f"  embargo [{w.val_start - embargo_delta}, {w.val_start})"
            f"  U  [{w.train_end}, {w.test_start})\n"
            f"  val     [{w.val_start}, {w.train_end})\n"
            f"  test    [{w.test_start}, {w.test_end}]"
        )


def _print_pair_inclusion(ss: SplitSet) -> None:
    from research.clfstudy import DATA_DIR

    try:
        first_dates: Dict[str, pd.Timestamp] = {}
        for pair in ss.pairs:
            base = pair.split("/")[0]
            feather_path = DATA_DIR / f"{base}_USDT-15m.feather"
            df = pd.read_feather(feather_path, columns=["date"])
            first_dates[pair] = pd.Timestamp(df["date"].iloc[0])

        for w in ss.windows:
            included = included_pairs(w, first_dates)
            excluded = [p for p in ss.pairs if p not in included]
            print(f"window {w.window_id}: included={included} excluded={excluded}")
    except Exception as exc:  # pragma: no cover - data may be absent in CI
        print(f"(skipping per-pair inclusion: {exc})")


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="python -m research.clfstudy.splits")
    sub = parser.add_subparsers(dest="command", required=True)

    build_p = sub.add_parser("build")
    build_p.add_argument("--set", dest="set_id", default="v1")
    build_p.add_argument("--out", dest="out", default=None)

    args = parser.parse_args(argv)

    if args.command == "build":
        ss = build(args.set_id)
        validate(ss)
        out = Path(args.out) if args.out else (SPLITS_DIR / f"{args.set_id}.json")
        save(ss, out)
        print(f"built and validated split set {args.set_id!r}; saved to {out}")
        _print_window_summary(ss)
        _print_pair_inclusion(ss)


if __name__ == "__main__":
    main()
