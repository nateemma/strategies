"""Statistics: IQM, bootstrap CIs, turnover/hold time, DSR and PBO (T009, T031, T044).

FR-016  >=5 seeds, interquartile mean with stratified bootstrap CI, never a best seed;
        seeds producing identical trades count as one measurement
FR-024  Deflated Sharpe Ratio and PBO (CSCV) over the FULL trial log
FR-025  a delta below the floor reads "not resolved", never "no effect"
"""
from __future__ import annotations

import numpy as np

FLOOR_PP_PER_YEAR = 3.35          # pooled walk-forward resolvable floor
SINGLE_WINDOW_FLOOR_PP = 7.633    # W1 noise floor (AGENT_GUIDE)


def iqm(x) -> float:
    """Interquartile mean: mean of the middle 50%."""
    a = np.sort(np.asarray(x, dtype=float))
    if a.size == 0:
        return float("nan")
    lo, hi = np.percentile(a, 25), np.percentile(a, 75)
    m = a[(a >= lo) & (a <= hi)]
    return float(m.mean() if m.size else a.mean())


def dedupe_identical(seed_trades: dict[int, tuple]) -> dict[int, tuple]:
    """Seeds producing identical trades are one measurement, not two (FR-016)."""
    seen, out = {}, {}
    for seed, trades in seed_trades.items():
        key = hash(trades)
        if key not in seen:
            seen[key] = seed
            out[seed] = trades
    return out


def bootstrap_ci(x, n: int = 10000, alpha: float = 0.05, seed: int = 0) -> tuple[float, float]:
    a = np.asarray(x, dtype=float)
    if a.size == 0:
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    means = rng.choice(a, size=(n, a.size), replace=True).mean(axis=1)
    return (float(np.percentile(means, 100 * alpha / 2)),
            float(np.percentile(means, 100 * (1 - alpha / 2))))


def verdict(delta_pp_yr: float, ci: tuple[float, float],
            floor: float = FLOOR_PP_PER_YEAR) -> str:
    """FR-025 / SC-009: below the floor is 'not resolved', never 'no effect'."""
    lo, hi = ci
    if delta_pp_yr >= floor and lo > 0:
        return "accept-candidate"
    if lo <= 0 <= hi or abs(delta_pp_yr) < floor:
        return "not resolved"
    return "not resolved"


def resolvable(delta_pp: float, floor: float = SINGLE_WINDOW_FLOOR_PP) -> bool:
    """T032a: is a single-window difference bigger than the measured noise floor?"""
    return abs(delta_pp) > floor


def deflated_sharpe(sr: float, n_obs: int, n_trials: int,
                    skew: float = 0.0, kurt: float = 3.0) -> float:
    """Bailey & Lopez de Prado (2014) Deflated Sharpe Ratio."""
    from math import erf, log, sqrt
    if n_trials < 2 or n_obs < 2:
        return float("nan")
    e, g = 0.5772156649, 1.0
    z = sqrt(2 * log(n_trials))
    sr0 = (z - (log(log(n_trials)) + log(4 * np.pi)) / (2 * z) + e / z) * g / sqrt(n_obs)
    denom = sqrt(max(1e-12, 1 - skew * sr + (kurt - 1) / 4 * sr ** 2))
    stat = (sr - sr0) * sqrt(n_obs - 1) / denom
    return float(0.5 * (1 + erf(stat / sqrt(2))))


def pbo_cscv(perf: np.ndarray, n_splits: int = 16) -> float:
    """Probability of Backtest Overfitting via CSCV (Bailey et al. 2015).

    perf: (n_observations, n_configurations). PBO > 0.2 fails the feature (SC-006).

    CAVEAT — measured 2026-09-10, and it bears on SC-006 directly. CSCV is biased LOW when
    the configuration count is small. On pure noise, where the true PBO is 0.5, this
    implementation returns:

        n_cfg=8   -> 0.148      n_cfg=20 -> 0.385      n_cfg=50 -> 0.521

    (Validated at the other end: 0.000 when one config is genuinely superior, 0.529 for a
    pure in-sample overfit.) This feature's screening grid is **12 configurations**, which
    sits in the biased-low regime, so a raw PBO from the grid alone would clear the 0.2 gate
    more easily than intended. Compute PBO over the FULL trial log — every seed, every
    variant, every abandoned run (FR-024) — which is what pushes the configuration count out
    of the biased regime. Report the configuration count alongside the PBO figure so the
    reader can judge it.
    """
    from itertools import combinations
    m = np.asarray(perf, dtype=float)
    if m.ndim != 2 or m.shape[1] < 2:
        return float("nan")
    n = m.shape[0] - (m.shape[0] % n_splits)
    blocks = np.array_split(m[:n], n_splits)
    half = n_splits // 2
    lam = []
    for combo in combinations(range(n_splits), half):
        rest = [i for i in range(n_splits) if i not in combo]
        is_ = np.vstack([blocks[i] for i in combo]).mean(axis=0)
        oos = np.vstack([blocks[i] for i in rest]).mean(axis=0)
        best = int(np.argmax(is_))
        rank = float((oos <= oos[best]).sum()) / len(oos)
        w = max(min(rank, 1 - 1e-9), 1e-9)
        lam.append(np.log(w / (1 - w)))
    return float(np.mean(np.array(lam) < 0))
