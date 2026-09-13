"""Fold panel loading and episode slicing (supports T019-T023).

Loads one fold's training panel once and slices episode windows from it, rather than
re-loading per episode: the training range is ~130K bars x 10 pairs and an episode is a few
thousand bars, so per-episode loading would dominate the run.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from research.labelfree import rl as RL                       # noqa: E402
from research.labelfree.rl.episodes import fold               # noqa: E402


CACHE_DIR = Path("/tmp/lf/rl_panels")


def load_fold_panel(fold_id: str, train: bool = True, pairs: list[str] | None = None,
                    use_cache: bool = True):
    """(features, open_px, cost, volume, index) per pair, from brief 1's own loader.

    CACHED because the load is ~60s for 10 pairs over a fold's training range, and the screen
    invokes this once per run. Measured on the SC-011 probe: the load was ~40% of wall-clock
    and recurred on every invocation. A cache key must therefore cover everything that
    changes the panel -- fold, side and pair list -- or a stale panel is silently reused for
    a different universe.
    """
    import hashlib
    import pickle

    from research.labelfree.arms import load_panel
    fd = fold(fold_id)
    pairs = pairs or list(fd.pair_universe)
    key = hashlib.sha1(
        f"{fold_id}|{'train' if train else 'test'}|{','.join(pairs)}".encode()).hexdigest()[:16]
    cache = CACHE_DIR / f"{key}.pkl"
    if use_cache and cache.exists():
        with cache.open("rb") as fh:
            return pickle.load(fh)

    X, P, C, V, M, _Y, _Vol, idx, _leak = load_panel(fd, pairs, train=train,
                                                     label_threshold=None)
    out = {"pairs": pairs, "X": X, "P": P, "C": C, "V": V, "M": M, "index": idx}
    if use_cache:
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        with cache.open("wb") as fh:
            pickle.dump(out, fh, protocol=pickle.HIGHEST_PROTOCOL)
    return out


def slice_episode(panel: dict, pair: str, start_iso: str, days: int):
    """One episode window. Returns (features, open_px, cost, volume) or None if unusable.

    The window carries EXEC_LAG_BARS extra bars of price: the reward at the last decision
    needs open[i+1] and open[i+2] (research R2).
    """
    pi = panel["pairs"].index(pair)
    idx = panel["index"]
    start = pd.Timestamp(start_iso)
    n_dec = days * RL.BARS_PER_DAY
    pos = int(idx.searchsorted(start, side="left"))
    end = pos + n_dec + RL.EXEC_LAG_BARS
    if pos < 0 or end > len(idx):
        return None
    X = panel["X"][pi, pos:pos + n_dec]
    P = panel["P"][pi, pos:end]
    C = panel["C"][pi, pos:pos + n_dec]
    V = panel["V"][pi, pos:pos + n_dec]
    M = panel["M"][pi, pos:pos + n_dec]
    # An episode containing a data gap is REJECTED and counted, never interpolated (T022).
    if M.mean() < 0.99 or not np.isfinite(P).all() or (P <= 0).any():
        return None
    return X, P, C, V
