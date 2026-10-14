"""034 T010: S1 pieces -- AUC, split stats, degenerate folds, the both-reference rule, no late read."""

from __future__ import annotations

import inspect

import numpy as np
import pandas as pd

from research.label_screen import separation as S


def test_auc_hand_cases():
    assert S.auc([1, 2, 3, 4], [False, False, True, True]) == 1.0
    assert S.auc([4, 3, 2, 1], [False, False, True, True]) == 0.0
    assert S.auc([1, 1, 1, 1], [False, True, False, True]) == 0.5
    assert S.auc([1, 2], [True, True]) is None


def test_split_stats_gap():
    r = S.split_stats(np.array([1.0, 3.0, -1.0, 0.0]), np.array([True, True, False, False]))
    assert r["agree"] == {"n": 2, "pp_per_trade": 2.0} and r["gap"] == 2.5


def test_degenerate_folds():
    df = pd.DataFrame({"fold": ["f00"] * 3 + ["f01"] * 2 + ["f02"] * 2,
                       "a": [True, False, True, True, True, False, False]})
    assert S.degenerate(df, "a") == ["f01", "f02"]


def test_rule_needs_both_references_and_few_degenerate_folds():
    refs = {"pre_registered": 0.072, "control_at_share": 0.05}
    assert S.s1_pass(0.08, refs, 0)
    assert not S.s1_pass(0.06, refs, 0)            # beats the share reference only
    assert not S.s1_pass(0.08, {**refs, "control_at_share": 0.09}, 0)
    assert not S.s1_pass(0.08, refs, 3)
    assert not S.s1_pass(None, refs, 0)
    assert not S.s1_pass(0.08, {**refs, "control_at_share": None}, 0)


def test_early_half_only():
    assert S.EARLY == tuple(f"f{i:02d}" for i in range(8))
    assert "late" not in inspect.signature(S.main).parameters
    assert "after" not in inspect.getsource(S.main)
