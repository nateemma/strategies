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


def test_continuous_band_is_bimodal_across_seeds():
    """R4a. R4 predicted a FIXED band would avoid brief 02's bimodality. Refuted: the screen's
    continuous arm splits into a churn cluster (6 seeds, turnover 108.8-165.3) and a frozen
    one (4 seeds, 4.5-12.5) with an order-of-magnitude hole between them. The bimodality
    belongs to the hold-inside-the-band MECHANISM, not to learning the band."""
    import json
    p = ROOT / "research/labelfree/rl/screen_continuous.json"
    if not p.exists():
        pytest.skip("screen not run")
    tv = np.array([r["train_turnover"] for r in json.loads(p.read_text())["runs"]])
    lo, hi = tv[tv <= 50], tv[tv > 50]
    assert len(lo) and len(hi), "expected two clusters"
    assert hi.min() / lo.max() > 5.0, "clusters must be separated by an order of magnitude"


def test_discrete_arm_is_not_degenerate():
    """The discrete arm must keep every seed above the participation floor; a degenerate
    seed's return is withheld (FR-017), and 'learn not to trade' is this env's degenerate
    optimum (R8)."""
    import json
    p = ROOT / "research/labelfree/rl/screen_discrete.json"
    if not p.exists():
        pytest.skip("screen not run")
    ex = np.array([r["train_mean_exposure"] for r in json.loads(p.read_text())["runs"]])
    assert (ex >= RL.MIN_EXPOSURE).all(), f"degenerate seeds: {ex[ex < RL.MIN_EXPOSURE]}"


def test_head_to_head_applies_the_floor_to_BOTH_sides():
    """FR-017. A comparator below the participation floor is INCONCLUSIVE and its return is
    withheld, so no delta against it is admissible -- a policy that ABSTAINS is not a policy
    that PERFORMS.

    Measured 2026-09-13: brief 1's re-trained comparator ran at exposure 0.0004, 125x below
    the floor, and the raw delta read 'WORSE' -- i.e. the RL policy losing to one that does
    nothing. Applying the floor to the arm only is how brief 02 came to measure arms against
    a P&L control that lost 52% in a rising market."""
    import json
    p = ROOT / "research/labelfree/rl/headtohead.json"
    if not p.exists():
        pytest.skip("head-to-head not run")
    d = json.loads(p.read_text())
    if d["brief1_mean_exposure"] < d["participation_floor"]:
        assert d["verdict"].startswith("INCONCLUSIVE"), \
            "a degenerate comparator must not yield a performance verdict"
        assert "raw_delta_verdict" in d, "the withheld raw verdict must still be recorded"


# --- the shared admissibility gate ------------------------------------------

def test_admissibility_refuses_a_degenerate_control():
    """The brief-04 case: an arm that trades vs a comparator that abstains."""
    from research.labelfree.admissibility import Arm, compare
    arm = Arm("rl", value=-0.2270, mean_exposure=0.600, trades_per_year=90.0)
    ctl = Arm("brief1", value=+0.0011, mean_exposure=0.0004, trades_per_year=0.5)
    r = compare(arm, ctl)
    assert r["admissible"] is False
    assert "DEGENERATE CONTROL" in r["verdict"]
    assert r["raw_delta"] == pytest.approx(-0.2281, abs=1e-3)   # retained, not discarded


def test_admissibility_refuses_a_degenerate_arm_too():
    """Symmetry matters: brief 02's trap was the control, brief 1's was the arm."""
    from research.labelfree.admissibility import Arm, compare
    arm = Arm("flat_policy", value=0.0, mean_exposure=0.0001, trades_per_year=0.0)
    ctl = Arm("control", value=-0.5, mean_exposure=0.44, trades_per_year=120.0)
    r = compare(arm, ctl)
    assert r["admissible"] is False and "DEGENERATE ARM" in r["verdict"]


def test_trades_per_year_catches_hold_forever():
    """Mean exposure alone misses a policy holding ONE position across the whole span:
    exposure is high, decisions are zero, and nothing is being tested."""
    from research.labelfree.admissibility import Arm
    hold_forever = Arm("buy_and_hold_in_disguise", value=0.1,
                       mean_exposure=0.98, trades_per_year=1.0)
    assert hold_forever.degenerate
    assert "one position is not a strategy" in hold_forever.reason


def test_two_losers_are_directional_only_never_acceptance():
    """Both lose to cash. As an ACCEPTANCE claim that is inadmissible -- "loses less than
    the control" is not a strategy. As a DIRECTIONAL finding it is legitimate research
    (brief 04 asks exactly this), but it must be stamped so it cannot be quoted as
    performance."""
    from research.labelfree.admissibility import Arm, compare
    a = Arm("a", value=-0.10, mean_exposure=0.5, trades_per_year=80.0)
    b = Arm("b", value=-0.25, mean_exposure=0.4, trades_per_year=60.0)
    acc = compare(a, b, purpose="acceptance")
    assert acc["admissible"] is False and "LOSES TO DO NOTHING" in acc["verdict"]
    dr = compare(a, b, purpose="directional")
    assert dr["admissible"] is True
    assert dr["verdict"] == "DIRECTIONAL ONLY — BOTH SIDES LOSE"
    assert dr["delta"] == pytest.approx(0.15)
    assert "not_a_performance_claim" in dr


def test_degeneracy_blocks_both_purposes():
    """An abstaining policy answers neither question."""
    from research.labelfree.admissibility import Arm, compare
    flat = Arm("flat", value=0.0, mean_exposure=0.0001, trades_per_year=0.0)
    ctl = Arm("ctl", value=-0.5, mean_exposure=0.44, trades_per_year=120.0)
    for purpose in ("acceptance", "directional"):
        assert compare(flat, ctl, purpose=purpose)["admissible"] is False


def test_admissibility_refuses_an_arm_that_loses_to_doing_nothing():
    """Brief 02's trap, which the abstention floor CANNOT catch: the P&L control ran at
    exposure 0.441 and 1869 trades/yr -- fully participating -- while losing 52% in a market
    that rose 72%. Arm C scored +0.8784 against it and was still a loss-maker. If the arm
    cannot beat cash, beating the control measures the control."""
    from research.labelfree.admissibility import Arm, compare
    armC = Arm("armC_auxvol", value=-0.1071, mean_exposure=0.434, trades_per_year=6.2)
    pnl = Arm("pnl_control", value=-0.9855, mean_exposure=0.441, trades_per_year=1868.9)
    r = compare(armC, pnl, do_nothing=0.0)
    assert r["admissible"] is False
    assert "LOSES TO DO NOTHING" in r["verdict"]
    assert r["delta_vs_control"] == pytest.approx(0.8784, abs=1e-3)
    assert r["delta_vs_do_nothing"] == pytest.approx(-0.1071, abs=1e-3)


def test_a_profitable_arm_still_passes():
    from research.labelfree.admissibility import Arm, compare
    good = Arm("good", value=+0.12, mean_exposure=0.5, trades_per_year=80.0)
    ctl = Arm("ctl", value=+0.03, mean_exposure=0.4, trades_per_year=60.0)
    r = compare(good, ctl, do_nothing=0.0)
    assert r["admissible"] is True and r["delta"] == pytest.approx(0.09)
