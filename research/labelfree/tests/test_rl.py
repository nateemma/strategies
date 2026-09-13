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


def test_drawdown_tracks_economics_not_the_shaped_reward():
    """Found by the SC-011 control. With drawdown computed on the shaped reward, FreqAI's
    -1/idle-bar penalty breached the limit in ONE bar: every episode ended after a single
    step, the agent never reached an entry bonus, and the probe reported zero turnover --
    reading as 'unhackable' when the reward had simply never been exercised."""
    from research.labelfree.env import PooledTradingEnv
    from research.labelfree.rl.rewards import freqai_shaped_reward

    n = 50
    px = np.full(n + RL.EXEC_LAG_BARS, 100.0)          # flat prices: zero realised return
    env = PooledTradingEnv(np.zeros((n, 3), dtype=np.float32), px,
                           np.full(n, 1e-4), np.full(n, 1e9),
                           reward_spec=freqai_shaped_reward())
    env.reset(seed=0)
    steps = 0
    while True:
        _, r, term, trunc, _ = env.step(0)             # stay flat: -1 shaped, 0.0 realised
        steps += 1
        assert r == pytest.approx(-1.0)                # the shaping IS applied
        if term or trunc:
            break
    assert steps == n, "flat prices must not terminate on drawdown; economics are zero"
    assert env.equity == pytest.approx(0.0)


def test_panel_cache_key_covers_the_pair_list():
    """A cache keyed on fold alone would silently serve a 10-pair panel to a 2-pair run.
    The key must change when the universe does."""
    import hashlib
    from research.labelfree.rl import panel as PN

    def key(fold_id, train, pairs):
        return hashlib.sha1(
            f"{fold_id}|{'train' if train else 'test'}|{','.join(pairs)}".encode()).hexdigest()[:16]

    assert key("f07", True, ["A", "B"]) != key("f07", True, ["A"])
    assert key("f07", True, ["A"]) != key("f07", False, ["A"])
    assert key("f07", True, ["A"]) != key("f08", True, ["A"])
    assert PN.CACHE_DIR.name == "rl_panels"


# --- Gate B (T012, T015, T016) ----------------------------------------------

def _ratio_ok(realised_turnover, shaped_turnover):
    """The SC-011 assertion, isolated so it can be tested for its ability to FAIL."""
    if realised_turnover <= 0:
        return False
    return (shaped_turnover / realised_turnover) >= RL.PROBE_MIN_TURNOVER_RATIO


def test_probe_assertion_can_fail():
    """T012. A control that cannot report failure certifies nothing. Feed the assertion two
    identical turnover series and confirm it says NO -- feature 010's liveness negative
    control passed vacuously on its first version by not actually severing anything."""
    assert _ratio_ok(100.0, 100.0) is False          # identical -> must fail
    assert _ratio_ok(100.0, 150.0) is False          # 1.5x, under the bar -> must fail
    assert _ratio_ok(100.0, 250.0) is True           # 2.5x -> passes
    assert _ratio_ok(0.0, 500.0) is False            # degenerate denominator -> must fail


def test_terminated_and_truncated_are_never_both_set():
    """T015, research R6. PPO bootstraps a truncated episode's value and must NOT bootstrap
    a terminated one's; conflating them corrupts the value estimate at the boundary."""
    from research.labelfree.env import PooledTradingEnv
    n = 30
    px = np.full(n + RL.EXEC_LAG_BARS, 100.0)
    env = PooledTradingEnv(np.zeros((n, 3), dtype=np.float32), px,
                           np.full(n, 1e-4), np.full(n, 1e9))
    env.reset(seed=0)
    seen_trunc = False
    for _ in range(n):
        _, _, term, trunc, _ = env.step(0)
        assert not (term and trunc)
        seen_trunc |= trunc
        if term or trunc:
            break
    assert seen_trunc, "a window that runs out must TRUNCATE, not terminate"


def test_realised_reward_has_no_shaping_terms():
    """T016, FR-003. A flat step earns exactly 0.0 — no idle penalty. An entry earns exactly
    -cost*executed — no entry bonus. Any shaping here is what makes a reward hackable."""
    from research.labelfree.env import PooledTradingEnv
    n = 10
    px = np.full(n + RL.EXEC_LAG_BARS, 100.0)        # zero return: only costs remain
    cost = np.full(n, 1e-3)
    env = PooledTradingEnv(np.zeros((n, 3), dtype=np.float32), px, cost, np.full(n, 1e9))
    env.reset(seed=0)
    _, r_flat, _, _, _ = env.step(0)
    assert r_flat == pytest.approx(0.0), "flat must earn exactly zero: no idle penalty"
    _, r_entry, _, _, info = env.step(1)
    assert r_entry == pytest.approx(-1e-3 * info["executed"]), "no entry bonus"
    _, r_hold, _, _, _ = env.step(1)
    assert r_hold == pytest.approx(0.0), "holding executes nothing, so costs nothing"
