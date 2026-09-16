"""Sparse jump model: determinism, label stability, frozen lam, seeds (T019, T019a).

FR-022/FR-023 exist because the backtests in this feature are deterministic -- gated arms
reuse the ungated fold weights and train nothing -- while the FIT that drives them is not.
Determinism in the expensive half of a pipeline reads as determinism in all of it.
"""
from __future__ import annotations

import numpy as np
import pytest

from research.labelfree.jump import model as M
from research.labelfree.jump.online import filtered_states


def panel(seed: int = 3, n: int = 900) -> np.ndarray:
    rng = np.random.default_rng(seed)
    def blk(m, base):
        return np.column_stack([np.abs(rng.normal(base, base * 0.3, m)),
                                np.abs(rng.normal(base * 0.7, base * 0.3, m)),
                                rng.normal(0.0, 0.3, m)])
    return np.vstack([blk(n, 0.005), blk(n, 0.020), blk(n, 0.005)])


def test_fit_deterministic_for_a_given_seed():
    x = panel()
    a = M.fit(x, 2, 50.0, seed=0, n_init=2)
    b = M.fit(x, 2, 50.0, seed=0, n_init=2)
    np.testing.assert_allclose(a.centroids, b.centroids)
    np.testing.assert_allclose(a.weights, b.weights)
    assert a.state_order == b.state_order


def test_state_order_tracks_volatility_not_fitter_labels():
    """FR-006 -- without the remap, 'block the high-vol state' silently inverts at a refit."""
    x = panel()
    f = M.fit(x, 2, 50.0, seed=0, n_init=2)
    xs = (x - f.scaler_mean) / f.scaler_scale
    s = filtered_states(xs, f)
    lo = xs[s == 0, 0].mean()
    hi = xs[s == 1, 0].mean()
    assert hi > lo, "the high-vol state does not have the higher realised-vol centroid"
    assert f.high_vol_state == f.k - 1


def test_remap_is_a_permutation():
    f = M.fit(panel(), 2, 50.0, seed=0, n_init=2)
    raw = np.array([0, 1, 0, 1, 1, 0])
    out = f.remap(raw)
    assert set(np.unique(out)) <= set(range(f.k))
    assert len(np.unique(out)) == len(np.unique(raw))


def test_lam_and_k_are_frozen_across_an_ensemble():
    x = panel()
    fits = M.fit_ensemble(x, 2, 50.0, n_init=2)
    assert len({f.lam for f in fits}) == 1
    assert len({f.k for f in fits}) == 1


def test_ensemble_requires_at_least_five_seeds():
    """FR-022 -- >= 5 seeds. A caller cannot quietly ask for fewer."""
    with pytest.raises(ValueError, match="FR-022"):
        M.fit_ensemble(panel(), 2, 50.0, seeds=(0, 1, 2))


def test_seed_agreement_is_a_rate_in_unit_interval():
    fits = M.fit_ensemble(panel(), 2, 50.0, n_init=2)
    r = M.seed_agreement(fits, panel())
    assert 0.0 <= r <= 1.0


def test_seed_agreement_detects_disagreement():
    """The floor must be able to FAIL, or it is not a gate.

    Built by MOVING a centroid, not by permuting labels. A first attempt reversed the centroid
    array and `state_order` together; `remap` correctly cancelled that and the sequences came
    out identical -- which demonstrates the label-invariance property but tests nothing about
    disagreement.
    """
    x = panel()
    good = M.fit(x, 2, 50.0, seed=0, n_init=2)
    shifted = good.centroids.copy()
    shifted[1] = shifted[0] + 0.05 * (shifted[1] - shifted[0])     # collapse the split
    bad = M.JumpFit(centroids=shifted, weights=good.weights,
                    scaler_mean=good.scaler_mean, scaler_scale=good.scaler_scale,
                    lam=good.lam, k=good.k, seed=99, state_order=good.state_order)
    r = M.seed_agreement([good, bad], x)
    assert r < 1.0, "seed_agreement cannot fall below 1.0, so the FR-023 floor can never fire"


def test_warm_start_prevents_single_state_collapse():
    """Regression for a measured defect (2026-09-15).

    Initialising at random and running the DP straight at a large lam collapsed SOL/USDT to
    [0.004, 0.999, 0.996, 0.004, 0.005] high-vol share across five seeds -- the model flipping
    between 'everything calm' and 'everything turbulent' on the seed alone.
    """
    x = panel()
    for seed in range(5):
        f = M.fit(x, 2, 400.0, seed=seed, n_init=2)
        xs = (x - f.scaler_mean) / f.scaler_scale
        share = float((filtered_states(xs, f) == 1).mean())
        assert 0.02 < share < 0.98, f"seed {seed} collapsed to one state (share {share:.3f})"


def test_default_kappa_sits_inside_the_admissible_range():
    for d in (3, 6, 12):
        assert 1.0 <= M.default_kappa(d) <= np.sqrt(d)


def test_calibrate_lam_refuses_degenerate_solutions():
    """Brief 06's recorded selection-by-best-calibration failure: the criterion falls
    monotonically in the tuned parameter and drives the choice to a degenerate limit."""
    x = panel()
    out = M.calibrate_lam(x, 2, target_switch_rate=1e-4,
                          grid=np.geomspace(1e0, 1e5, 10), min_state_share=0.10)
    assert out["min_state_share"] >= 0.10
    assert any(g["degenerate"] for g in out["grid"]), "grid never reached the degenerate region"
    assert not next(g for g in out["grid"] if g["lam"] == out["lam"])["degenerate"]
