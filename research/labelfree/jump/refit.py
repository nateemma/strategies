"""90-day expanding-window refits and the StateSequence artefact (T011, T012, T013).

CLARIFICATION Q2=A: refits CONTINUE THROUGH a fold's test range, each using only bars strictly
before its own refit point. That is what a deployed system does -- it keeps re-learning -- and
each fit stays causal. Freezing at the fold boundary would leave the gate running on a model up
to a year stale by fold 15, which measures staleness rather than the method.

STANDARDISATION IS PART OF THE FIT, NOT PREPROCESSING. Means and scales come from the expanding
window up to the refit point only. Recorded failure (brief 07): statistics fitted over the full
series including the scoring split was a leak that survived two other controls.

THE ARTEFACT WRITTEN HERE IS WHAT THE BACKTEST READS, and therefore what the deletion test must
target. Pointing the causality test at a separate in-memory recomputation is precisely how a
leak survives a passing test.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import K_STATES, OUT, SEEDS
from . import features as FE
from . import model as M
from .online import filtered_states, smoothed_states

REFIT_DAYS = 90
FIT_STRIDE = 4          # fit on hourly bars, infer on all of them
FIT_N_INIT = 3

# WHY FITTING IS SUBSAMPLED AND INFERENCE IS NOT.
# The model's own calibration target is roughly six regime switches a year -- a mean regime
# length near two months. Estimating centroids and feature weights for an object that slow does
# not need every 15-minute bar, and the DP is O(n) in a Python loop, so the fit dominates the
# cost. States are still produced for EVERY bar: only the parameter estimation sees every fourth
# one. This is a cost decision about the fit, never about the output, and it cannot affect
# causality -- a subsample of the past is still the past.


def schedule(index: pd.DatetimeIndex, first_fit_end: pd.Timestamp,
             period_days: int = REFIT_DAYS) -> list[pd.Timestamp]:
    """Refit points every `period_days` from `first_fit_end` to the end of `index`."""
    out, t = [], pd.Timestamp(first_fit_end)
    last = index[-1]
    while t <= last:
        out.append(t)
        t = t + pd.Timedelta(days=period_days)
    return out


def _fit_at(x: np.ndarray, seed: int, lam: float, k: int) -> M.JumpFit:
    return M.fit(x[::FIT_STRIDE], k, lam, seed=seed, vol_col=0, n_init=FIT_N_INIT)


def run(feats: pd.DataFrame, first_fit_end: pd.Timestamp, lam: float,
        *, k: int = K_STATES, seed: int = 0, mode: str = "filtered",
        period_days: int = REFIT_DAYS) -> pd.DataFrame:
    """Produce one pair's StateSequence.

    `mode='filtered'` is the deployed path. `mode='smoothed'` is the LEAK REFERENCE and must
    never reach a deployed run -- `gate.py` refuses it.
    """
    if mode not in ("filtered", "smoothed"):
        raise ValueError(f"mode must be filtered|smoothed, got {mode!r}")
    cols = [c for c in feats.columns]
    if cols[0] != "rvol_realised":
        # model.fit ranks states by centroids[:, vol_col=0]; if column 0 stopped being
        # realised volatility, "high vol" would silently invert with nothing looking wrong.
        raise ValueError(f"column 0 must be rvol_realised, got {cols[0]!r}")

    x = feats.to_numpy(float)
    idx = feats.index
    points = schedule(idx, first_fit_end, period_days)

    state = np.full(len(idx), -1, dtype=int)
    refit_id = np.full(len(idx), -1, dtype=int)

    for rid, tau in enumerate(points):
        past = idx < tau                                   # STRICTLY before the refit point
        if past.sum() < 500:
            continue
        fit = _fit_at(x[past], seed, lam, k)

        hi = points[rid + 1] if rid + 1 < len(points) else idx[-1] + pd.Timedelta(seconds=1)
        live = (idx >= tau) & (idx < hi)
        if not live.any():
            continue

        # Inference consumes history up to the end of this block so the recursion is warm,
        # then only the block's own rows are kept. Standardisation uses the FIT's statistics.
        upto = idx < hi
        xs = (x[upto] - fit.scaler_mean) / fit.scaler_scale
        seq = filtered_states(xs, fit) if mode == "filtered" else smoothed_states(xs, fit)
        state[live] = seq[live[upto]]
        refit_id[live] = rid

    out = pd.DataFrame({"timestamp": idx, "state": state.astype(np.int8),
                        "refit_id": refit_id}, index=None)
    out["mode"] = mode
    return out[out.state >= 0].reset_index(drop=True)


def build_fold(pairs: list[str], lo: pd.Timestamp, hi: pd.Timestamp,
               first_fit_end: pd.Timestamp, lam: float, *, k: int = K_STATES,
               seed: int = 0, mode: str = "filtered") -> pd.DataFrame:
    frames = []
    for pair in pairs:
        f = FE.build(pair, lo, hi)
        if len(f) < 1000:
            continue
        d = run(f, first_fit_end, lam, k=k, seed=seed, mode=mode)
        d.insert(0, "pair", pair)
        frames.append(d)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def write(df: pd.DataFrame, fold_id: str, mode: str, seed: int) -> "object":
    OUT.mkdir(parents=True, exist_ok=True)
    p = OUT / f"states_{fold_id}_{mode}_s{seed}.feather"
    df.reset_index(drop=True).to_feather(p)
    return p
