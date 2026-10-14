"""032 follow-up 1: run counting and the run-matched threshold pick."""

from __future__ import annotations

import numpy as np

from research.candidate_training import run_match as R


def test_runs_counts_maximal_blocks():
    assert R.runs(np.array([], bool)) == 0
    assert R.runs(np.array([1, 1, 0, 1, 0, 0, 1, 1, 1], bool)) == 3
    assert R.runs(np.array([0, 0], bool)) == 0
    assert R.runs(np.array([1], bool)) == 1


def test_runs_at_is_strict_and_sums_pairs():
    eff = [np.array([0.6, 0.7, 0.0, 0.6]), np.array([0.5, 0.0, 0.9])]
    assert R.runs_at(eff, 0.6) == 2          # 0.7 on pair 1, 0.9 on pair 2
    assert R.runs_at(eff, 0.55) == 3
    assert R.runs_at(eff, 0.0) == 4


def test_pick_hits_the_target_and_prefers_higher_threshold_on_ties():
    eff = [np.array([0.905, 0.0, 0.805, 0.0, 0.705, 0.0, 0.605])]
    t, r = R.pick_for_runs(eff, 2)
    assert (t, r) == (0.8, 2)                # highest grid value giving 2 runs
    t1, r1 = R.pick_for_runs(eff, 4)
    assert (t1, r1) == (0.6, 4)
    assert all(round(x, 2) == x for x in R.GRID)
