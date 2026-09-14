"""Optimiser wiring (T030-T032)."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from research.labelfree.es import cmaes as CM            # noqa: E402
from research.labelfree.es import space as SP            # noqa: E402


def _sphere_at(target: dict):
    """A function with a KNOWN optimum, so the ask/tell loop and bounds mapping are testable
    independently of anything this feature measures."""
    def f(params, gen=0):
        return -sum(((params[d.name] - target[d.name]) / d.width) ** 2 for d in SP.DIMS)
    return f


def test_finds_known_optimum():
    target = {d.name: d.lo + 0.5 * d.width for d in SP.DIMS}
    hist = CM.run_cmaes(_sphere_at(target), budget=1200, seed=7)
    best = max(hist, key=lambda h: h["value"])
    assert best["value"] > -0.05, (
        f"CMA-ES did not approach the known optimum (best {best['value']:.4f}); "
        f"the ask/tell loop or the bounds mapping is wrong")


def test_cmaes_beats_random_on_a_smooth_function():
    """Sanity on the CONTROL too: on a smooth unimodal function the optimiser should win.

    If it does not, a later 'random matched CMA-ES' result would be uninformative — it would
    mean the harness is broken, not that the landscape is noise.
    """
    target = {d.name: d.lo + 0.5 * d.width for d in SP.DIMS}
    f = _sphere_at(target)
    b_cma = max(h["value"] for h in CM.run_cmaes(f, budget=800, seed=7))
    b_rnd = max(h["value"] for h in CM.run_random(f, budget=800, seed=7))
    assert b_cma > b_rnd


def test_identical_budgets():
    f = _sphere_at(dict(SP.INCUMBENT))
    for n in (50, 237):
        assert len(CM.run_cmaes(f, budget=n, seed=1)) == n
        assert len(CM.run_random(f, budget=n, seed=1)) == n


def test_seeded_reproducibility():
    f = _sphere_at(dict(SP.INCUMBENT))
    for fn in (CM.run_cmaes, CM.run_random):
        a = [h["params"] for h in fn(f, budget=60, seed=11)]
        b = [h["params"] for h in fn(f, budget=60, seed=11)]
        c = [h["params"] for h in fn(f, budget=60, seed=12)]
        assert a == b, f"{fn.__name__} is not reproducible under a fixed seed"
        assert a != c, f"{fn.__name__} ignores its seed"


def test_random_control_draws_from_the_whole_space():
    """A control clustered near the incumbent would be a straw man, not a null."""
    hist = CM.run_random(lambda p, g=0: 0.0, budget=500, seed=3)
    for d in SP.DIMS:
        vals = np.array([h["params"][d.name] for h in hist], dtype=float)
        assert vals.min() < d.lo + 0.1 * d.width
        assert vals.max() > d.hi - 0.1 * d.width


def test_inadmissible_candidates_do_not_crash_the_search():
    """floor/cap violations are common in a uniform draw and must be survivable."""
    def f(params, gen=0):
        try:
            SP.validate(params)
        except SP.Inadmissible:
            return float("-inf")
        return -SP.distance_from_incumbent(params)
    hist = CM.run_cmaes(f, budget=300, seed=5)
    assert len(hist) == 300
    assert any(np.isfinite(h["value"]) for h in hist)
