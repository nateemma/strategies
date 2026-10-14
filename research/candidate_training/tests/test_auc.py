"""032 T019: S1 compares both models on IDENTICAL eligible, labelled bars."""

from __future__ import annotations

import numpy as np
import pandas as pd

from research.candidate_training import auc as A


def _t(dates, pairs, p, gm, bw):
    return pd.DataFrame({"date": pd.to_datetime(dates), "pair": pairs, "p_buy": p,
                         "guard_metric": gm, "bb_width": bw})


def test_join_keeps_only_bars_in_both_tables_that_are_eligible_and_labelled():
    d = ["2023-01-01 00:00", "2023-01-01 00:15", "2023-01-01 00:30", "2023-01-01 00:45"]
    cur = _t(d, ["X"] * 4, [0.1, 0.2, 0.3, 0.4], [0.1, 0.5, 0.1, 0.1], [0.05, 0.05, 0.02, 0.05])
    new = _t(d[:2] + d[3:], ["X"] * 3, [0.9, 0.8, 0.7], [0.0] * 3, [0.0] * 3)
    lab = {"X": pd.DataFrame({"date": pd.to_datetime(d).tz_localize("UTC"),
                              "label": [1, 0, 1, np.nan]})}
    j = A.joined(cur, new, lab)
    # row 0: eligible, in both, labelled; row 1: guard 0.5 fails; row 2: absent from new and
    # bb 0.02; row 3: unlabelled (NaN)
    assert len(j) == 1
    assert j.iloc[0]["p_cur"] == 0.1 and j.iloc[0]["p_new"] == 0.9 and j.iloc[0]["label"] == 1


def test_eligibility_comes_from_the_current_table_not_the_new_one():
    d = ["2023-01-01 00:00"]
    cur = _t(d, ["X"], [0.1], [0.1], [0.05])
    new = _t(d, ["X"], [0.9], [0.9], [0.0])  # the new table's columns are ignored
    lab = {"X": pd.DataFrame({"date": pd.to_datetime(d).tz_localize("UTC"), "label": [1]})}
    assert len(A.joined(cur, new, lab)) == 1


def test_diff_ci_brackets_a_clear_improvement():
    rng = np.random.default_rng(1)
    n = 400
    y = rng.integers(0, 2, n)
    d = pd.DataFrame({"pair": rng.choice(["A", "B"], n),
                      "date": pd.date_range("2023-01-01", periods=n, freq="6h", tz="UTC"),
                      "label": y, "p_cur": rng.random(n), "p_new": y * 0.5 + rng.random(n) * 0.6})
    lo, hi = A.diff_ci(d, np.random.default_rng(0))
    assert 0 < lo < hi
