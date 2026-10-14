"""033 T015: run starts, the effective BUY probability, and the share-quantile threshold."""

from __future__ import annotations

import numpy as np
import pytest

from research.width_relative import pick as P


def test_run_starts_marks_only_rising_edges():
    e = np.array([1, 1, 0, 1, 0, 0, 1, 1, 1], bool)
    assert P.run_starts(e).tolist() == [True, False, False, True, False, False, True, False, False]
    assert P.run_starts(np.zeros(3, bool)).sum() == 0


def test_effective_is_p_buy_only_where_buy_is_argmax():
    probs = np.array([[0.1, 0.3, 0.6], [0.2, 0.5, 0.3], [np.nan, np.nan, np.nan], [0.5, 0.1, 0.4]])
    assert P.effective(probs).tolist() == [0.6, 0.0, 0.0, 0.0]


def test_threshold_for_share_hits_the_share_at_four_decimals():
    rng = np.random.default_rng(0)
    eff = np.r_[rng.uniform(0.5, 0.9, 800), np.zeros(200)]
    t, got = P.threshold_for_share(eff, 0.5)
    assert round(t, 4) == t and abs(got - 0.5) < 0.01
    with pytest.raises(ValueError):
        P.threshold_for_share(np.empty(0), 0.5)
