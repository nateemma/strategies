"""CMA-ES and the identical-budget random control (T033, T034).

The reference `cma` package is used deliberately rather than a hand-rolled implementation:
the feature's expected deliverable is a NEGATIVE, and a negative must not be attributable to
a home-made optimiser. cma 4.5.0, installed into freqtrade/.venv 2026-09-13.

BUDGET IS A COUNT OF EVALUATIONS, NOT WALL-CLOCK. That is the unit the deflation counts and
the unit the optimiser is being credited with using well, and "identical budget" is otherwise
ambiguous. Both methods draw from the SAME bounded space, so the control is a genuine null
rather than a differently-shaped search.
"""
from __future__ import annotations

from typing import Callable

import numpy as np

from research.labelfree.es import space as SP


def _z_bounds() -> tuple[list[float], list[float]]:
    """Search in z-space but keep it finite; logit(1e-4)..logit(1-1e-4) spans the box."""
    lim = 9.2
    return [-lim] * len(SP.DIMS), [lim] * len(SP.DIMS)


def run_cmaes(evaluate: Callable[[dict], float], budget: int, seed: int,
              sigma0: float = 2.0) -> list[dict]:
    """Ask/tell CMA-ES against `evaluate`, which MAXIMISES. Returns the history.

    `cma` minimises, so the objective is negated on the way in. Inadmissible candidates score
    -inf, which becomes +inf for cma: a hard rejection it handles by resampling pressure
    rather than by crashing.
    """
    import cma

    lo, hi = _z_bounds()
    es = cma.CMAEvolutionStrategy(
        [0.0] * len(SP.DIMS), sigma0,
        {"seed": seed, "bounds": [lo, hi], "verbose": -9,
         "maxfevals": budget})

    hist: list[dict] = []
    gen = 0
    while len(hist) < budget and not es.stop():
        zs = es.ask()
        losses = []
        for z in zs:
            if len(hist) >= budget:
                losses.append(0.0)
                continue
            params = SP.decode(z)
            v = evaluate(params, gen)
            hist.append({"params": params, "value": v, "generation": gen})
            losses.append(1e12 if not np.isfinite(v) else -v)
        es.tell(zs, losses)
        gen += 1
    return hist[:budget]


def run_random(evaluate: Callable[[dict], float], budget: int, seed: int) -> list[dict]:
    """Uniform draws from the SAME bounded space, at the SAME evaluation count."""
    rng = np.random.default_rng(seed)
    hist: list[dict] = []
    for _ in range(budget):
        params = {d.name: (int(rng.integers(d.lo, d.hi + 1)) if d.integer
                           else float(rng.uniform(d.lo, d.hi))) for d in SP.DIMS}
        hist.append({"params": params, "value": evaluate(params, 0), "generation": 0})
    return hist


METHODS = {"cmaes": run_cmaes, "random": run_random}
