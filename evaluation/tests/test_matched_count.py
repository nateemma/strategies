"""T039 — the matched-count selection test (023 research R13, FR-014)."""

from __future__ import annotations

import numpy as np
import pytest
from evaluation.matched_count import matched_count


def _pop(n_per_fold=200, folds=4, seed=0):
    rng = np.random.default_rng(seed)
    return [
        {
            "fold_id": f"f{f:02d}",
            "pair": f"P{i}",
            "open_date": f"d{f}-{i}",
            "profit_abs": float(rng.normal(0, 10)),
            "scale_pp_per_abs": 0.01,
        }
        for f in range(folds)
        for i in range(n_per_fold)
    ]


def test_top_k_selection_is_detected():
    pop = _pop()
    sel = []
    for f in range(4):
        fold = sorted(
            (t for t in pop if t["fold_id"] == f"f{f:02d}"), key=lambda t: -t["profit_abs"]
        )
        sel += fold[:20]
    r = matched_count(sel, pop, n_draws=4000)
    assert r["z"] > 3 and r["p_one_sided"] < 0.001
    assert r["overlap_fraction"] == 1.0 and r["n_selected"] == 80


def test_random_subset_sits_at_the_null():
    pop = _pop()
    zs = []
    for seed in range(10):
        rng = np.random.default_rng(100 + seed)
        sel = [pop[i] for i in rng.choice(len(pop), 80, replace=False)]
        zs.append(matched_count(sel, pop, n_draws=2000, seed=seed)["z"])
    assert sum(abs(z) < 2 for z in zs) >= 8


def test_selection_larger_than_population_raises():
    pop = _pop(n_per_fold=5, folds=1)
    sel = pop + [dict(pop[0], open_date="extra")]
    with pytest.raises(ValueError):
        matched_count(sel, pop)


def test_stratified_draw_respects_fold_counts():
    # all selected trades in one fold: the null must be drawn from that fold only
    pop = _pop(folds=2)
    for t in pop:
        if t["fold_id"] == "f01":
            t["profit_abs"] = 1000.0
    sel = [t for t in pop if t["fold_id"] == "f00"][:10]
    r = matched_count(sel, pop, n_draws=500)
    assert r["null_mean_pp"] < 50  # never drew from the +1000 fold
