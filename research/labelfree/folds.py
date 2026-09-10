"""Windows, anchored folds and per-fold pair universe (T005).

Wraps the existing instruments rather than reimplementing them: pinned windows come
from `regime/regimedata.py`, the anchored plan from `regime/wf/foldplan.py`.

FR-015  one policy per fold, trained on that fold's training range only
FR-015a single-window runs keep the 730-day minimum; the universe shrinks in the
        distant window and its size MUST be reported per window
FR-018  purge the maximum horizon and embargo one horizon at fold boundaries
"""
from __future__ import annotations

import json
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from regime import regimedata as rd            # noqa: E402
from regime.wf import foldplan                 # noqa: E402

WF_CONFIG = ROOT / "config" / "config_wf.json"
MIN_TRAIN_DAYS = 730
TEST_DAYS = 90


@dataclass(frozen=True)
class Fold:
    fold_id: str
    train_start: str
    train_end: str
    test_start: str
    test_end: str
    test_days: int
    pair_universe: tuple[str, ...] = field(default=())
    purge_bars: int = 0
    embargo_bars: int = 0


def whitelist() -> list[str]:
    cfg = json.loads(re.sub(r"//.*", "", WF_CONFIG.read_text()))
    return cfg["exchange"]["pair_whitelist"]


# A pair needs real training data in the range, not merely a start date before it.
# Measured 2026-09-10: XRP/USDT starts 2021-01-01 but carries a 911-day gap (the
# Binance.US delisting, Dec 2020 -> Jul 2023), leaving 12 days inside the W3 training
# range. A `first_bar <= train_start` test calls that "covered" and would train on
# almost nothing while reporting a full universe. Count bars in range instead.
MIN_TRAIN_DAYS_IN_RANGE = 180
BARS_PER_DAY = 96


def _bars_in_range(pair: str, lo: pd.Timestamp, hi: pd.Timestamp) -> int:
    try:
        idx = rd.load_ohlcv(pair.split("/")[0]).index
    except Exception:
        return 0
    return int(((idx >= lo) & (idx <= hi)).sum())


def coverage_days(train_start: str, train_end: str,
                  pairs: list[str] | None = None) -> dict[str, float]:
    """Days of actual training data each pair has inside the range."""
    lo = pd.Timestamp(train_start, tz="UTC")
    hi = pd.Timestamp(train_end, tz="UTC")
    return {p: _bars_in_range(p, lo, hi) / BARS_PER_DAY for p in (pairs or whitelist())}


def universe_for(train_start: str, train_end: str | None = None,
                 pairs: list[str] | None = None,
                 min_days: int = MIN_TRAIN_DAYS_IN_RANGE) -> list[str]:
    """Pairs with at least `min_days` of actual data inside the training range.

    This is the whole of FR-015a. The control MUST be run on whatever this returns for
    the same fold, or the comparison is not paired.
    """
    if train_end is None:
        train_end = (pd.Timestamp(train_start) + pd.Timedelta(days=MIN_TRAIN_DAYS)).strftime("%Y-%m-%d")
    cov = coverage_days(train_start, train_end, pairs)
    return [p for p, d in cov.items() if d >= min_days]


def folds(horizon_bars: int = 0) -> list[Fold]:
    """The anchored plan, each fold carrying the universe it can actually train on."""
    pairs = whitelist()
    idx = {p: rd.load_ohlcv(p.split("/")[0]).index for p in pairs}
    data_start = min(i[0] for i in idx.values())
    data_end = max(i[-1] for i in idx.values())
    raw = foldplan.build("NNDP", data_start.strftime("%Y-%m-%d"),
                         data_end.strftime("%Y-%m-%d"),
                         min_train_days=MIN_TRAIN_DAYS, test_days=TEST_DAYS)
    out = []
    for f in raw:
        uni = tuple(universe_for(f["train_start"], f["train_end"], pairs))
        out.append(Fold(f["fold_id"], f["train_start"], f["train_end"],
                        f["test_start"], f["test_end"], f["test_days"],
                        uni, horizon_bars, horizon_bars))
    return out


def window_universe(window: str) -> dict:
    """Universe for a single-window run, keeping the 730-day training minimum (FR-015a)."""
    start, _ = rd.WINDOWS[window]
    train_start = (pd.Timestamp(start) - pd.Timedelta(days=MIN_TRAIN_DAYS)).strftime("%Y-%m-%d")
    train_end = (pd.Timestamp(start) - pd.Timedelta(days=1)).strftime("%Y-%m-%d")
    cov = coverage_days(train_start, train_end)
    uni = universe_for(train_start, train_end)
    all_pairs = whitelist()
    return {
        "window": window,
        "train_start": train_start,
        "train_end": train_end,
        "pair_universe": uni,
        "n_pairs": len(uni),
        "n_total": len(all_pairs),
        "excluded": sorted(set(all_pairs) - set(uni)),
        "coverage_days": {p: round(d) for p, d in sorted(cov.items(), key=lambda kv: -kv[1])},
    }


if __name__ == "__main__":
    for w in ("W1", "W2", "W3"):
        u = window_universe(w)
        print(f"{w}: {u['n_pairs']}/{u['n_total']} pairs "
              f"(train from {u['train_start']}) excluded={u['excluded']}")
