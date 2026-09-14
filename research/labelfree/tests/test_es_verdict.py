"""Protocol §5 gate logic (T049a-T049f, T051).

The gates decide whether an operating point is adopted, so their failure modes matter more
than their success path. Every test here asserts a REFUSAL.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from research.labelfree.es import verdict as VD          # noqa: E402


def _folds(vals):
    return [{"fold_id": f"f{i:02d}", "annualised_pp": v, "test_days": 90}
            for i, v in enumerate(vals)]


# --------------------------------------------------------------------------------- SC-001

def test_pooled_requires_the_floor_not_merely_a_positive_delta():
    """+2 pp/yr over the control is an improvement and still fails: the floor is 3.35."""
    g = VD.pooled(_folds([12.0] * 15), control_pp=10.0)
    assert g["delta_pp"] < VD.FLOOR_PP
    assert not g["passed"]


def test_pooled_passes_a_clear_win():
    g = VD.pooled(_folds([20.0] * 15), control_pp=10.0)
    assert g["passed"] and g["ci_excludes_zero"]


def test_pooled_fails_when_the_interval_straddles_zero():
    """A big mean delta driven by one fold must not pass on the point estimate alone."""
    vals = [-5.0] * 14 + [200.0]
    g = VD.pooled(_folds(vals), control_pp=0.0)
    assert not g["passed"] or g["ci_excludes_zero"]


# --------------------------------------------------------------------- SC-002, Principle II

def test_fold_concentrated_win_fails_persistence():
    """The exact failure Principle II exists to catch."""
    vals = [-1.0] * 12 + [90.0, 90.0, 90.0]
    assert not VD.per_fold_positivity(_folds(vals))["passed"]


def test_per_fold_requires_the_full_plan():
    """10/10 positive is not 10/15 — a partial fold set cannot pass."""
    assert not VD.per_fold_positivity(_folds([5.0] * 10))["passed"]
    assert VD.per_fold_positivity(_folds([5.0] * 15))["passed"]


def test_single_window_needs_all_three_and_fails_on_a_missing_one():
    assert VD.single_window({"W1": 1.0, "W2": 1.0, "W3": 1.0})["passed"]
    assert not VD.single_window({"W1": 1.0, "W2": -0.1, "W3": 1.0})["passed"]
    g = VD.single_window({"W1": 1.0, "W2": 1.0})
    assert not g["passed"] and g["missing"] == ["W3"]


# --------------------------------------------------------------------------------- SC-003

def test_baseline_bar_is_the_production_control_not_zero():
    """+3 pp/yr is profitable and still loses to the +7.86 production control."""
    g = VD.baselines(3.0, 0.06, 150.0)
    assert g["bar_pp"] == VD.BASELINE_BAR_PP
    assert not g["passed"]
    assert VD.baselines(12.0, 0.06, 150.0)["passed"]


def test_abstaining_candidate_cannot_clear_the_baselines():
    """Exposure below the participation floor: inadmissible however good the number."""
    g = VD.baselines(99.0, 0.0001, 0.5)
    assert not g["passed"]
    assert not g["admissibility"]["admissible"]


# --------------------------------------------------------------------------------- SC-008

def test_overfit_gap_catches_a_train_test_collapse():
    rows = [{"fold_id": "f01", "candidate_id": "a", "train_pp": 30.0, "test_pp": 1.0}]
    g = VD.overfit_gap(rows)
    assert not g["passed"] and g["n_overfit"] == 1


def test_overfit_gap_allows_a_small_gap():
    rows = [{"fold_id": "f01", "candidate_id": "a", "train_pp": 10.0, "test_pp": 8.0}]
    assert VD.overfit_gap(rows)["passed"]


# ------------------------------------------------------------------------- derived status

def _all_pass():
    return {"SC-001": VD.pooled(_folds([20.0] * 15), 10.0),
            "SC-002a": VD.per_fold_positivity(_folds([20.0] * 15)),
            "SC-002b": VD.single_window({"W1": 1.0, "W2": 1.0, "W3": 1.0}),
            "SC-003": VD.baselines(20.0, 0.06, 150.0),
            "SC-008": VD.overfit_gap([])}


def test_high_pbo_kills_even_when_every_gate_passes():
    """Clarification Q3: PBO > 0.2 kills the candidate; the measured value is the finding."""
    s = VD.status(_all_pass(), pbo=0.45, delta_vs_incumbent_pp=20.0, beats_random=True)
    assert s["status"] == "KILL"
    assert any("PBO" in r for r in s["reasons"])


def test_random_control_matching_the_search_kills():
    s = VD.status(_all_pass(), pbo=0.05, delta_vs_incumbent_pp=20.0, beats_random=False)
    assert s["status"] == "KILL"
    assert any("fit noise" in r for r in s["reasons"])


def test_move_inside_the_sweep_noise_band_is_not_resolved_never_an_improvement():
    s = VD.status(_all_pass(), pbo=0.05, delta_vs_incumbent_pp=4.0, beats_random=True)
    assert s["status"] == "NOT RESOLVED"
    assert any("sweep-noise band" in r for r in s["reasons"])
    assert not any("improvement" in r.lower() for r in s["reasons"])


def test_accept_requires_everything():
    s = VD.status(_all_pass(), pbo=0.05, delta_vs_incumbent_pp=20.0, beats_random=True)
    assert s["status"] == "ACCEPT"


def test_failing_sc001_is_not_resolved_never_no_effect():
    """SC-007 in wording as well as logic."""
    g = _all_pass()
    g["SC-001"] = VD.pooled(_folds([10.5] * 15), 10.0)
    s = VD.status(g, pbo=0.05, delta_vs_incumbent_pp=20.0, beats_random=True)
    assert s["status"] == "NOT RESOLVED"
    # The STATUS is what gets quoted, and it must never be "no effect". The reason text
    # deliberately contains that phrase as a disclaimer ("never 'no effect'"), so assert on
    # the status and on SC-007 being cited, not on a substring of the explanation.
    assert s["status"] != "NO EFFECT"
    assert any("SC-007" in r for r in s["reasons"])
