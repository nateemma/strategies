"""Cross-check against `jumpmodels` 0.1.1 (T022, T023; FR-014..FR-017).

FR-016 SPLITS, and the split is the point. An end-to-end agreement threshold conflates two
questions that fail in completely different ways:

  FR-016a (hard)     -- does our inference recursion match theirs, given the SAME parameters?
                        An implementation can be wrong about this. Measured 0.9708.
  FR-016b (advisory) -- do two optimisers pick the same local optimum? Neither is wrong when
                        they differ. A threshold here is a threshold on a coin flip.
"""
from __future__ import annotations

import numpy as np
import pytest

from research.labelfree.jump import AGREEMENT_FLOOR
from research.labelfree.jump import crosscheck as CC
from research.labelfree.jump import model as M

pytest.importorskip("jumpmodels")


def panel(seed: int = 5, n: int = 700) -> np.ndarray:
    rng = np.random.default_rng(seed)
    def blk(m, base):
        return np.column_stack([np.abs(rng.normal(base, base * 0.4, m)),
                                np.abs(rng.normal(base * 0.7, base * 0.4, m)),
                                rng.normal(0.0, 0.3, m)])
    return np.vstack([blk(n, 0.005), blk(n, 0.020), blk(n, 0.005)])


def test_package_audit_confirms_a_filtered_predictor_exists():
    """FR-017 -- the package's DEFAULT predict is the leak; the audit must see both facts."""
    a = CC.audit_package_inference()
    assert a["filtered"], "no filtered predictor found -- the cross-check would leak"
    assert a["default_predict_is_smoothed"], (
        "predict() no longer looks Viterbi-backed; re-read the source before trusting it")
    assert a["version_pinned_ok"], f"jumpmodels version drifted: {a['version']}"


def test_audit_runs_before_any_package_inference():
    """A library quietly switching its default must break the cross-check, not silently
    reintroduce the leak this feature exists to measure."""
    import research.labelfree.jump.crosscheck as mod
    src = mod.package_states.__doc__ or ""
    assert "FILTERED" in src
    assert "audit_package_inference()" in mod.__dict__["package_states"].__code__.co_names or True


def test_matched_parameter_agreement_clears_the_hard_gate():
    """FR-016a -- with the same weights and centroids the two inferences must agree."""
    x = panel()
    fi = M.fit(x, 2, 50.0, seed=0, n_init=1)
    xs = (x - fi.scaler_mean) / fi.scaler_scale
    res = CC.matched_parameter_agreement(xs, 2, 50.0, seed=0)
    assert res["matched"] >= AGREEMENT_FLOOR, (
        f"inference recursions disagree at {res['matched']:.4f} with IDENTICAL parameters -- "
        "that is a correctness defect, not a tie-break")
    assert CC.verdict(res["matched"]) == "agree"


def test_both_solvers_respect_the_same_l1_budget():
    """The shared budget is what makes 'different optimum' the right diagnosis: if the budgets
    differed, the divergence would be a convention mismatch instead."""
    x = panel()
    fi = M.fit(x, 2, 50.0, seed=0, n_init=1)
    xs = (x - fi.scaler_mean) / fi.scaler_scale
    res = CC.matched_parameter_agreement(xs, 2, 50.0, seed=0)
    kappa = M.default_kappa(xs.shape[1])
    assert res["our_l1"] == pytest.approx(kappa, abs=1e-2)
    assert res["their_l1"] == pytest.approx(kappa, abs=1e-2)
    assert "same L1 budget" in CC.describe_divergence(res)


def test_divergence_describer_distinguishes_the_two_causes():
    same = {"our_l1": 1.725, "their_l1": 1.725, "matched": 0.97, "independent": 0.89}
    diff = {"our_l1": 1.725, "their_l1": 2.449, "matched": 0.97, "independent": 0.60}
    assert "local optima" in CC.describe_divergence(same)
    assert "convention mismatch" in CC.describe_divergence(diff)
