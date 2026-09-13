"""Tests for the pooled RL environment (feature 012, brief 04).

Each test pins a defect this study has actually had, not a hypothetical.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from research.labelfree import rl as RL          # noqa: E402


# --- T003: the dependency surface -------------------------------------------

def test_dependency_versions():
    """A version drift changes PPO's behaviour silently; fail loudly instead."""
    import gymnasium, stable_baselines3, sb3_contrib, torch

    def ge(v, want):
        return tuple(int(x) for x in v.split(".")[:2]) >= want

    assert ge(gymnasium.__version__, (1, 3))
    assert ge(stable_baselines3.__version__, (2, 9))
    assert ge(sb3_contrib.__version__, (2, 9))
    assert ge(torch.__version__.split("+")[0], (2, 13))


def test_liquidity_fraction_is_imported_not_redeclared():
    """A second copy would drift from the simulator the equivalence gate compares against,
    and the drift would be invisible."""
    from research.labelfree import sim
    assert RL.LIQUIDITY_FRACTION is sim.LIQUIDITY_FRACTION


def test_screen_bar_is_pinned_before_any_run():
    """analyze H1: 'distinguishable' without a threshold always passes, so the gate stops
    gating. The value must exist as a constant, not be chosen after seeing the screen."""
    assert 0.0 < RL.DISTINGUISHABLE_MIN_BAR_FRACTION < 1.0


def test_probe_threshold_is_pinned():
    """analyze M1: 'turnover rises measurably' passes on a 1% rise."""
    assert RL.PROBE_MIN_TURNOVER_RATIO >= 2.0


# --- Gate A (T007): the four cases a naive step gets wrong -------------------

import mlx.core as mx                                          # noqa: E402
from research.labelfree import sim                             # noqa: E402
from research.labelfree.env import step_economics, bar_returns_from_open  # noqa: E402
from research.labelfree.rl import equivalence as EQ            # noqa: E402


@pytest.mark.parametrize("kind", ["random", "opens_long", "ends_in_position",
                                  "cap_binds", "zero_volume"])
def test_logic_is_exact_against_the_simulator(kind):
    """T007. Given the simulator's OWN bar returns, the env's economics must match
    EXACTLY — boundary, cap, accumulation and lag. The library's log() difference is
    excluded here deliberately; Part 2 bounds that separately (research R1a)."""
    rng = np.random.default_rng(0)
    w, px, cost, vol = EQ.make_case(kind, 400, rng)
    mine = EQ.logic_only(w, px, cost, vol)
    theirs = EQ.vectorised(w, px, cost, vol)
    n = min(len(mine), len(theirs))
    assert np.array_equal(mine[:n], theirs[:n].astype(np.float32))


def test_precision_residual_stays_within_ceiling():
    """Part 2. Above the ceiling is a defect; at it is two correct log implementations."""
    rng = np.random.default_rng(1)
    w, px, cost, vol = EQ.make_case("random", 400, rng)
    mine, theirs = EQ.step_path(w, px, cost, vol), EQ.vectorised(w, px, cost, vol)
    n = min(len(mine), len(theirs))
    assert np.max(np.abs(mine[:n] - theirs[:n])) <= EQ.PRECISION_CEILING


def test_opening_boundary_is_applied_once():
    """w_{-1} = 0 must be applied at the start and NEVER re-applied. Re-zeroing the carried
    position each step is the plausible-looking bug that rejected the 2-bar-window design."""
    r, c, v = 0.01, 1e-4, 1e9
    first, _, _ = step_economics(1.0, 0.0, r, c, v)       # opening: requested = 1.0
    held, _, _ = step_economics(1.0, 1.0, r, c, v)        # holding: requested = 0.0
    assert first == pytest.approx(1.0 * r - c * 1.0)
    assert held == pytest.approx(1.0 * r)                 # no cost when nothing is executed


def test_cost_is_charged_on_executed_not_requested():
    """The liquidity cap is not a free brake."""
    r, c = 0.01, 1e-2
    capped, req, ex = step_economics(1.0, 0.0, r, c, quote_volume=1.0)
    assert req == pytest.approx(1.0)
    assert ex == pytest.approx(RL.LIQUIDITY_FRACTION * 1.0)
    assert capped == pytest.approx(1.0 * r - c * ex)


def test_execution_lag_is_two_bars():
    """R2: the action on the observation ending at bar i earns open[i+1] -> open[i+2]."""
    px = np.array([100.0, 110.0, 121.0, 133.1])
    got = bar_returns_from_open(px)
    assert len(got) == len(px) - RL.EXEC_LAG_BARS
    assert got[0] == pytest.approx(np.log(121.0) - np.log(110.0))
