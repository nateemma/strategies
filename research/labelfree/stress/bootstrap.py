"""Stationary block bootstrap over real per-pair panels (T016-T018).

Politis & Romano (1994). A block starts at a uniformly random index; each subsequent bar
continues the block with probability 1 - 1/L and starts a new one otherwise, wrapping at the
series end.

WHY GEOMETRIC AND NOT FIXED BLOCKS. With fixed-length blocks, a bar's probability of starting a
block depends on its position modulo L, which makes the resampled series non-stationary. The
geometric variant removes that artefact for one extra line.

ONE BLOCK INDEX IS SHARED ACROSS ALL PAIRS (revised by the pre-implementation analyze pass).
The fold panels are date-aligned, so applying the same index sequence to every pair keeps the
cross-section intact: when the index selects a moment, each pair contributes ITS OWN bar from
that moment, and pairs that crashed together still crash together.

The original design resampled pairs independently. That would have destroyed cross-pair
co-movement and UNDERSTATED JOINT DRAWDOWN -- and the drawdown ratio is one of the three
calibration statistics, so the method's known bias would have pointed straight at a gate it had
to pass, in the direction the tolerance band was most permissive about. Fixing the cause beats
widening the band to accommodate it.

THE COST, recorded rather than discovered: paths are less diverse than independent resampling
would give, because the cross-section is tied together. That is the correct trade. A harness
that understates joint drawdown is worse than one with fewer distinct paths, because joint
drawdown is what a portfolio actually experiences.

EVERY COLUMN IS RESAMPLED TOGETHER by source index -- prices, indicators, cost AND the model's
own class probabilities. Inside a block every column is real and mutually consistent BECAUSE IT
IS REAL, which is what removes the need to re-run the model on synthetic features. Only the
joins between blocks are synthetic. A second inference path is exactly what feature 013 weighed
and rejected, the repo having a case where predict-time scaling silently diverged.
"""
from __future__ import annotations

import numpy as np

# Columns carried through a resample. Every one comes from the same source bar.
PANEL_COLUMNS = ("open", "high", "low", "close", "volume", "rvol_stop", "cost",
                 "p_sell", "p_hold", "p_buy",
                 "rvol", "atr_pct_roll", "guard_metric", "close_norm", "adx", "bb_width")

BLOCK_LENGTHS = (96, 144, 192, 240, 288)      # 1-3 days at 15-minute bars (FR-003a)


def block_index(n: int, length: int, block_len: int, rng: np.random.Generator) -> np.ndarray:
    """Stationary-bootstrap source indices for one path of `length` bars over `n` source bars.

    Returns the index vector itself rather than the resampled data, because the SAME vector is
    applied to every pair — that is what preserves the cross-section.
    """
    if n < 2:
        raise ValueError(f"need at least 2 source bars, got {n}")
    p_new = 1.0 / max(1, block_len)
    idx = np.empty(length, dtype=np.int64)
    cur = int(rng.integers(0, n))
    for i in range(length):
        idx[i] = cur
        if rng.random() < p_new:
            cur = int(rng.integers(0, n))          # start a new block
        else:
            cur = (cur + 1) % n                    # continue it, wrapping at the end
    return idx


def resample_panel(panel: dict, idx: np.ndarray) -> dict:
    """Apply ONE index vector to every pair, preserving the cross-section."""
    out = {"fold_id": panel.get("fold_id"), "which": panel.get("which"),
           "pairs": {}, "purge_bars_applied": panel.get("purge_bars_applied", 0),
           "leakage_control_called": True}
    for pair, d in panel["pairs"].items():
        n = len(d["open"])
        # Each pair may be shorter than the longest; take the index modulo its own length so
        # every pair contributes a real bar and none is padded.
        j = idx % n
        out["pairs"][pair] = {k: (np.asarray(d[k])[j] if k in d else None)
                              for k in PANEL_COLUMNS if k in d}
        out["pairs"][pair]["index"] = None          # synthetic paths have no calendar
    out["span_years"] = len(idx) / (365.25 * 96)
    return out


def sample_paths(panel: dict, n_paths: int, block_len: int, seed: int,
                 length: int | None = None):
    """Yield `n_paths` resampled panels. Deterministic under `seed`."""
    rng = np.random.default_rng(seed)
    n = max(len(d["open"]) for d in panel["pairs"].values())
    length = length or n
    for _ in range(n_paths):
        yield resample_panel(panel, block_index(n, length, block_len, rng))


def realised_block_lengths(idx: np.ndarray) -> np.ndarray:
    """Lengths of the runs of consecutive source indices — for the geometric-mean test."""
    if idx.size == 0:
        return np.array([], dtype=int)
    breaks = np.flatnonzero(np.diff(idx) != 1)
    bounds = np.concatenate(([-1], breaks, [idx.size - 1]))
    return np.diff(bounds)
