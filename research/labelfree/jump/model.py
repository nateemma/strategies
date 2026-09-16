"""Sparse jump model: three-block coordinate descent (T006, T007, T008, T008a).

    minimise  sum_t sum_j w_j (x_tj - mu_{s_t,j})^2  +  lam * sum_t 1[s_t != s_{t-1}]
    s.t.      w on the simplex under an L1 budget  (the "sparse" part)

Bemporad-Breschi-Piga-Boyd (2018), the Nystrup-Kolm-Lindstrom (2020) financial framing, and
the Shu-Yu-Mulvey (2024) sparse feature weighting the brief names. The switching penalty `lam`
is the WHOLE mechanical difference from the eight instruments `regime/VERDICT.md` rejected.

FITTING USES THE SMOOTHED DP, AND THAT IS LEGITIMATE. A refit at time tau sees only bars
before tau, so the fit is causal at the refit granularity. What must never happen is the
DEPLOYED state series being produced by a traceback -- see `online.py`.

THE FIT IS STOCHASTIC. `init_centroids` is k-means++ and therefore seed-dependent, so every
blocked bar downstream inherits that randomness even though the backtests are deterministic.
Callers MUST use `fit_ensemble` (FR-022), never a bare `fit`.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from . import MIN_STATE_SHARE, SEEDS, SEED_AGREEMENT_FLOOR
from .online import loss_matrix, _values

N_INIT = 10          # k-means++ restarts per fit; matches jumpmodels' n_init_jm default


@dataclass(frozen=True)
class JumpFit:
    centroids: np.ndarray          # (k, d), in volatility-sorted order
    weights: np.ndarray            # (d,)
    scaler_mean: np.ndarray        # (d,)
    scaler_scale: np.ndarray       # (d,)
    lam: float
    k: int
    seed: int
    state_order: tuple[int, ...]   # raw index -> volatility rank

    def remap(self, raw: np.ndarray) -> np.ndarray:
        """Raw fitter labels -> volatility rank, so `high_vol_state` always names one thing.

        Without this, "block the high-vol state" silently INVERTS whenever a refit happens to
        permute its labels, and nothing in the output would look wrong.
        """
        lut = np.empty(self.k, dtype=int)
        for rank, raw_idx in enumerate(self.state_order):
            lut[raw_idx] = rank
        return lut[raw]

    @property
    def high_vol_state(self) -> int:
        return self.k - 1          # ranks ascend in volatility


def standardise(x: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Fit-window statistics only. Recorded failure (brief 07): standardisation fitted over the
    full series including the scoring split was a leak that survived two other controls."""
    mu = x.mean(axis=0)
    sd = x.std(axis=0)
    sd[sd < 1e-12] = 1.0
    return (x - mu) / sd, mu, sd


def init_centroids(x: np.ndarray, k: int, rng: np.random.Generator) -> np.ndarray:
    """k-means++ seeding. THIS is where the stochasticity enters (FR-022)."""
    n = len(x)
    centres = [x[rng.integers(n)]]
    for _ in range(1, k):
        d2 = np.min(((x[:, None, :] - np.array(centres)[None, :, :]) ** 2).sum(-1), axis=1)
        tot = d2.sum()
        probs = np.full(n, 1.0 / n) if tot <= 0 else d2 / tot
        centres.append(x[rng.choice(n, p=probs)])
    return np.array(centres)


def _sparse_weights(x: np.ndarray, s: np.ndarray, k: int, kappa: float) -> np.ndarray:
    """Witten & Tibshirani (2010) sparse k-means weight update.

    a_j is feature j's between-cluster sum of squares. Soft-threshold a, project to unit L2,
    and bisect the threshold until the L1 norm meets the budget `kappa`.
    """
    tss = ((x - x.mean(0)) ** 2).sum(0)
    wss = np.zeros_like(tss)
    for c in range(k):
        m = s == c
        if m.sum() > 0:
            wss += ((x[m] - x[m].mean(0)) ** 2).sum(0)
    a = np.maximum(tss - wss, 0.0)

    def w_of(delta: float) -> np.ndarray:
        v = np.maximum(a - delta, 0.0)
        nrm = np.linalg.norm(v)
        return v / nrm if nrm > 0 else np.full_like(a, 1.0 / np.sqrt(len(a)))

    if np.abs(w_of(0.0)).sum() <= kappa:
        return w_of(0.0)
    lo, hi = 0.0, a.max()
    for _ in range(60):                       # bisection on the L1 norm, monotone in delta
        mid = 0.5 * (lo + hi)
        if np.abs(w_of(mid)).sum() > kappa:
            lo = mid
        else:
            hi = mid
    return w_of(hi)


def _assign(x: np.ndarray, centroids: np.ndarray, weights: np.ndarray, lam: float) -> np.ndarray:
    """Smoothed DP assignment -- correct for FITTING, never for deployment."""
    loss = loss_matrix(x, centroids, weights)
    v = _values(loss, lam)
    n, k = loss.shape
    pen = lam * (1.0 - np.eye(k))
    s = np.empty(n, dtype=int)
    s[-1] = int(v[-1].argmin())
    for t in range(n - 1, 0, -1):
        s[t - 1] = int((v[t - 1] + pen[:, s[t]]).argmin())
    return s


def default_kappa(d: int) -> float:
    """L1 budget, PRE-REGISTERED and frozen.

    The admissible range is [1, sqrt(d)] -- 1 is maximally sparse (one feature), sqrt(d) is
    uniform weighting with no sparsity at all. The midpoint is taken because it is the one
    choice in that interval that is not a tuning decision. With d = 6 this is ~1.72.
    """
    return 0.5 * (1.0 + np.sqrt(d))


def _objective(xs: np.ndarray, s: np.ndarray, centroids: np.ndarray,
               weights: np.ndarray, lam: float) -> float:
    fit_loss = float(loss_matrix(xs, centroids, weights)[np.arange(len(s)), s].sum())
    return fit_loss + lam * float((np.diff(s) != 0).sum())


def _one_start(xs: np.ndarray, k: int, lam: float, kappa: float,
               rng: np.random.Generator, max_iter: int, warm_iter: int = 10):
    """One restart, WARM-STARTED from the unpenalised (lam=0) clustering.

    Why the warm start is not optional, measured 2026-09-15 on SOL/USDT. Initialising
    centroids at random and immediately running the DP at the calibrated lam=785 made the fit
    collapse: across five seeds the high-volatility share came out
    [0.004, 0.999, 0.996, 0.004, 0.005] -- the model flipping between "everything is calm" and
    "everything is turbulent" purely on the seed, with a 4% spread in the objective. A large
    jump penalty makes the DP sticky, so a start that puts most bars in one cluster can never
    pay the penalty to leave it.

    Running the first `warm_iter` sweeps at lam=0 finds the clustering first and only then
    imposes the switching cost, which is the standard annealing order for this model family.
    """
    d = xs.shape[1]
    centroids = init_centroids(xs, k, rng)
    weights = np.full(d, 1.0 / np.sqrt(d))
    s = np.zeros(len(xs), dtype=int)

    # Phase 1 -- unpenalised: find the clusters before charging for switching between them.
    for _ in range(warm_iter):
        s_w = _assign(xs, centroids, weights, 0.0)
        for c in range(k):
            m = s_w == c
            if m.sum() > 0:
                centroids[c] = xs[m].mean(0)
        weights = _sparse_weights(xs, s_w, k, kappa)

    # Phase 2 -- the penalised objective proper.
    for _ in range(max_iter):
        s_new = _assign(xs, centroids, weights, lam)
        for c in range(k):
            m = s_new == c
            if m.sum() > 0:
                centroids[c] = xs[m].mean(0)
        weights = _sparse_weights(xs, s_new, k, kappa)
        if np.array_equal(s_new, s):
            break
        s = s_new
    return centroids, weights, s, _objective(xs, s, centroids, weights, lam)


def fit(x: np.ndarray, k: int, lam: float, *, seed: int = 0, max_iter: int = 100,
        kappa: float | None = None, vol_col: int = 0, n_init: int = N_INIT) -> JumpFit:
    """Deterministic for a given (x, k, lam, seed, n_init); STOCHASTIC across seeds.

    `n_init` restarts from independent k-means++ seedings and keeps the lowest objective, as
    every k-means-family fitter does (`jumpmodels` defaults to `n_init_jm=10`). A single start
    is not a cheaper version of this -- it is a worse fit, and it inflates exactly the
    initialisation sensitivity FR-022 exists to measure. Restarts do not remove that
    sensitivity, they only stop it being self-inflicted, which is why the seed ensemble stays
    mandatory either way.
    """
    xs, mu, sd = standardise(x)
    d = xs.shape[1]
    kappa = default_kappa(d) if kappa is None else kappa
    rng = np.random.default_rng(seed)

    best = None
    for _ in range(max(1, n_init)):
        cand = _one_start(xs, k, lam, kappa, rng, max_iter)
        if best is None or cand[3] < best[3]:
            best = cand
    centroids, weights, _s, _obj = best

    # Volatility ordering (FR-006): rank raw states by their realised-vol centroid.
    #
    # `vol_col` is the column index of `rvol_realised`. It is passed explicitly rather than
    # assumed, because the sparse weighting can drive that feature's WEIGHT to ~0 while the
    # ordering still depends on its CENTROID -- an implicit coupling that would silently
    # invert "high vol" if the feature order ever changed. `refit.py` asserts the column name.
    order = tuple(int(i) for i in np.argsort(centroids[:, vol_col]))
    return JumpFit(centroids=centroids, weights=weights, scaler_mean=mu, scaler_scale=sd,
                   lam=float(lam), k=int(k), seed=int(seed), state_order=order)


def fit_ensemble(x: np.ndarray, k: int, lam: float,
                 seeds: Sequence[int] = SEEDS, **kw) -> list[JumpFit]:
    """FR-022 -- >= 5 seeds. Report the IQM across these, never one fit and never the best."""
    if len(seeds) < 5:
        raise ValueError(f"FR-022 requires >= 5 seeds, got {len(seeds)}")
    return [fit(x, k, lam, seed=s, **kw) for s in seeds]


def seed_agreement(fits: list[JumpFit], x: np.ndarray) -> float:
    """Mean pairwise state-for-state agreement of the FILTERED sequences across seeds.

    Below SEED_AGREEMENT_FLOOR the regime structure is a k-means initialisation artifact and
    the feature stops (FR-023).
    """
    from .online import filtered_states
    xs = (x - fits[0].scaler_mean) / fits[0].scaler_scale
    seqs = [filtered_states(xs, f) for f in fits]
    pairs = [(np.asarray(a) == np.asarray(b)).mean()
             for i, a in enumerate(seqs) for b in seqs[i + 1:]]
    return float(np.mean(pairs)) if pairs else 1.0


def calibrate_lam(x: np.ndarray, k: int, target_switch_rate: float,
                  seed: int = 0, grid: np.ndarray | None = None,
                  min_state_share: float = MIN_STATE_SHARE) -> dict:
    """Called ONCE, on fold 0's training window; the result is frozen for the whole run.

    Principle VIII -- one parameter space at a time. A per-fold lam search would be a joint
    search over the jump model and the incumbent's gate, which is how brief 05 reached
    PBO 0.7128 over 48,015 evaluations.

    TWO THINGS THIS GETS RIGHT THAT A FIRST PASS DID NOT, both measured 2026-09-15:

    1. IT MEASURES THE FILTERED SWITCH RATE, because filtered states are what the gate
       actually consumes. The smoothed assignment used during fitting switches materially
       less (4 vs 14 per year at lam=658 on SOL), so calibrating on it would silently deliver
       a churnier gate than the target asked for.

    2. IT REFUSES DEGENERATE lam. Above lam ~1900 on SOL the smoothed assignment becomes
       CONSTANT and one state absorbs 91% of bars -- the model has stopped finding regimes and
       started finding nothing. The switch rate keeps falling monotonically into that limit, so
       an unguarded "closest to target" search walks straight into it. This is brief 06's
       recorded selection-by-best-calibration failure, where the criterion fell monotonically
       in the tuned parameter and drove the choice to a degenerate limit. Fixing the cause
       beats widening the band to accommodate it.

    Returns the chosen lam with the diagnostics that justify it, so the report can state
    whether the degeneracy guard bound rather than leaving it implicit.
    """
    from .online import filtered_states
    grid = np.geomspace(1e0, 1e5, 30) if grid is None else grid
    rows, best, best_err = [], None, np.inf
    for lam in grid:
        # n_init=2: this is a SCAN over lam, not the fit that produces the verdict. Ten
        # restarts per grid point cost 20 minutes on fold 0 and moved the selected lam by
        # nothing measurable.
        f = fit(x, k, float(lam), seed=seed, n_init=2)
        xs = (x - f.scaler_mean) / f.scaler_scale
        s = filtered_states(xs, f)
        shares = np.array([(s == c).mean() for c in range(k)])
        rate = float((np.diff(s) != 0).mean())
        degenerate = bool(shares.min() < min_state_share)
        rows.append({"lam": float(lam), "switch_rate": rate,
                     "min_state_share": float(shares.min()), "degenerate": degenerate})
        if degenerate:
            continue
        err = abs(rate - target_switch_rate)
        if err < best_err:
            best, best_err = float(lam), err
    if best is None:
        raise SystemExit(
            "every lam on the grid is degenerate (one state below "
            f"{min_state_share:.0%} occupancy) -- the model finds no usable regime structure")
    chosen = next(r for r in rows if r["lam"] == best)
    return {"lam": best, "achieved_switch_rate": chosen["switch_rate"],
            "target_switch_rate": float(target_switch_rate),
            "min_state_share": chosen["min_state_share"],
            # TWO-SIDED. A one-sided flag reported "guard did not bind" on fold 0 while the
            # achieved rate was 1.2 switches/yr against a target of 6 -- the guard had in fact
            # pushed the choice clean past the target, because every rate between them was
            # degenerate. Missing the target by 5x in either direction is the same fact.
            "guard_bound": bool(chosen["switch_rate"] > target_switch_rate * 1.5
                                or chosen["switch_rate"] < target_switch_rate / 1.5),
            "n_degenerate": int(sum(g["degenerate"] for g in rows)),
            "n_grid": int(len(rows)),
            "grid": rows}
