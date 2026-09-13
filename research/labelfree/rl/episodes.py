"""Episode sampling (T019-T022). Training range ONLY.

THE TWO TRAILING BARS (research R2) are the part that leaks silently. The reward at the last
decision bar i needs open[i+1] and open[i+2]; if those come from the test range the episode
leaks by exactly two bars, and no other check would report it. Validation therefore tests
`start + length + EXEC_LAG_BARS`, not `start + length`.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from research.labelfree import folds as F                     # noqa: E402
from research.labelfree import rl as RL                       # noqa: E402


def fold(fold_id: str):
    return {f.fold_id: f for f in F.folds()}[fold_id]


def sample(fold_id: str, n: int = 1000, seed: int = 0,
           pairs: list[str] | None = None) -> list[dict]:
    """n episodes of 30-90 days, uniformly over (pair, start) in the TRAINING range."""
    fd = fold(fold_id)
    pairs = pairs or list(fd.pair_universe)
    rng = np.random.default_rng(seed)
    lo = pd.Timestamp(fd.train_start, tz="UTC")
    hi = pd.Timestamp(fd.train_end, tz="UTC")
    lo_d, hi_d = RL.EPISODE_DAYS
    out = []
    for _ in range(n):
        pair = pairs[int(rng.integers(len(pairs)))]
        days = int(rng.integers(lo_d, hi_d + 1))
        # The window PLUS its two trailing bars must end at or before train_end.
        span = pd.Timedelta(days=days) + pd.Timedelta(minutes=15 * RL.EXEC_LAG_BARS)
        latest = hi - span
        if latest <= lo:
            continue
        # Snap to the 15-minute bar grid. A free-floating Timestamp carries sub-bar
        # precision that pandas cannot losslessly place in the panel's index, and an
        # episode that does not start on a bar is not a decision point anyway.
        n_bars = int((latest - lo).total_seconds() // (15 * 60))
        if n_bars <= 0:
            continue
        start = lo + pd.Timedelta(minutes=15 * int(rng.integers(n_bars)))
        out.append({"pair": pair, "start": start.isoformat(), "days": days,
                    "fold_id": fold_id,
                    "end_inclusive": (start + span).isoformat()})
    return out


def validate(episodes: list[dict], fold_id: str) -> list[str]:
    """Every violation, not the first: one leaked episode and a systematic leak look the
    same in a `raise`, and they are very different findings."""
    fd = fold(fold_id)
    train_end = pd.Timestamp(fd.train_end, tz="UTC")
    test_start = pd.Timestamp(fd.test_start, tz="UTC")
    bad = []
    for e in episodes:
        end = pd.Timestamp(e["end_inclusive"])
        if end > train_end:
            bad.append(f"{e['pair']} {e['start'][:16]}: window+{RL.EXEC_LAG_BARS} bars ends "
                       f"{end.isoformat()[:16]} > train_end {fd.train_end}")
        if end >= test_start:
            bad.append(f"{e['pair']} {e['start'][:16]}: reaches the TEST range")
        if e["fold_id"] != fold_id:
            bad.append(f"{e['pair']}: wrong fold {e['fold_id']}")
    return bad
