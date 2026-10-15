"""035 S1b: run starts, early-only guard, and the scorer on synthetic data."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from research.new_inputs import screen_s1b as S1B


def test_run_starts():
    m = np.array([0, 1, 1, 0, 1, 0, 1, 1, 1], bool)
    assert S1B.run_starts(m).nonzero()[0].tolist() == [1, 4, 6]


def test_sample_refuses_late_folds():
    with pytest.raises(RuntimeError, match="early half only"):
        S1B.sample(("f08",))


def test_score_finds_an_incremental_input_and_counts_sign_folds():
    rng = np.random.default_rng(1)
    n = 4000
    s = pd.DataFrame({"fold": np.repeat([f"f{i:02d}" for i in range(8)], n // 8),
                      "bb_width": rng.normal(size=n), "p_ctl": rng.normal(size=n)})
    s["x"] = rng.normal(size=n)
    s["outcome"] = 0.2 * s["x"] + s["bb_width"] + rng.normal(size=n)
    r = S1B.score(s, "x")
    assert r["partial"] > 0.1 and r["sign_folds"] == 8 and r["coverage"] == 1.0
