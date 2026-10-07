# pragma pylint: disable=C0103, C0114, C0115, C0116
# type: ignore
"""Tests for evaluation/pick_threshold.py.

Covers the pure threshold math (no freqtrade, no I/O), the `TrainWindow` leakage guard, and
`effective_buy_probs`'s argmax-gating. Does NOT exercise `predicted_buy_probs` /
`_load_fold_context` (real freqtrade data loading + an already-trained MLX model) --
those were smoke-tested manually against one fold (NNNC_FMR_f00) per the task's GPU-contention
constraint and are not re-run on every `pytest` invocation.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))

from pick_threshold import (
    TrainWindow,
    effective_buy_probs,
    rate_for_threshold,
    threshold_for_rate,
)


FOLD = {
    "fold_id": "f00",
    "train_start": "2021-01-01",
    "train_end": "2022-12-31",
    "test_start": "2023-01-01",
    "test_end": "2023-04-01",
    "train_timerange": "20210101-20230101",
    "test_timerange": "20230101-20230401",
}


# ---------------------------------------------------------------------------
# TrainWindow -- the leakage guard
# ---------------------------------------------------------------------------


def test_train_window_uses_train_timerange_not_test():
    window = TrainWindow.from_fold(FOLD)
    assert window.timerange == FOLD["train_timerange"]
    assert window.timerange != FOLD["test_timerange"]
    assert window.fold_id == FOLD["fold_id"]


def test_train_window_refuses_a_window_that_overruns_test_start():
    leaky = dict(FOLD, train_timerange="20210101-20230201")  # ends AFTER test_start 20230101
    with pytest.raises(AssertionError):
        TrainWindow.from_fold(leaky)


def test_train_window_accepts_train_end_exactly_at_test_start():
    # The real foldplan.py geometry: train_end == test_start, never intersecting.
    touching = dict(FOLD, train_timerange="20210101-20230101", test_timerange="20230101-20230401")
    window = TrainWindow.from_fold(touching)
    assert window.timerange == "20210101-20230101"


# ---------------------------------------------------------------------------
# effective_buy_probs -- argmax-gated BUY probability
# ---------------------------------------------------------------------------


def test_effective_buy_probs_gates_on_argmax():
    df = pd.DataFrame({
        "%predict_prob_sell": [0.1, 0.3, np.nan, 0.5],
        "%predict_prob_hold": [0.2, 0.5, 0.3, 0.3],
        "%predict_prob_buy":  [0.7, 0.2, 0.4, 0.2],
    })
    out = effective_buy_probs(df)
    # row 0: BUY is argmax (0.7)      -> kept at 0.7
    # row 1: HOLD is argmax (0.5)     -> 0.0
    # row 2: NaN                      -> dropped entirely (not 0.0)
    # row 3: SELL is argmax (0.5)     -> 0.0
    assert len(out) == 3
    np.testing.assert_allclose(sorted(out), [0.0, 0.0, 0.7])


def test_effective_buy_probs_empty_when_all_nan():
    df = pd.DataFrame({
        "%predict_prob_sell": [np.nan],
        "%predict_prob_hold": [np.nan],
        "%predict_prob_buy": [np.nan],
    })
    out = effective_buy_probs(df)
    assert len(out) == 0


# ---------------------------------------------------------------------------
# threshold_for_rate / rate_for_threshold -- the core selection
# ---------------------------------------------------------------------------


def _synthetic_effective_probs(n=20000, seed=0):
    """Uniform(0,1) stand-in for a pooled effective-BUY-probability array -- every bar is
    already "BUY is the argmax", isolating the rate<->threshold inversion from the gating."""
    rng = np.random.default_rng(seed)
    return rng.uniform(0.0, 1.0, size=n)


def test_threshold_for_rate_achieves_the_target_rate():
    probs = _synthetic_effective_probs()
    for target in (0.01, 0.05, 0.10, 0.30):
        threshold, achieved = threshold_for_rate(probs, target)
        assert abs(achieved - target) < 0.01
        # achieved must be the TRUE rate at the returned threshold, not just a claim
        assert rate_for_threshold(probs, threshold) == pytest.approx(achieved)


def test_threshold_for_rate_is_monotonic_in_target_rate():
    probs = _synthetic_effective_probs()
    t_low, _ = threshold_for_rate(probs, 0.05)
    t_high, _ = threshold_for_rate(probs, 0.40)
    assert t_high < t_low, "a higher target rate must give a lower (more permissive) threshold"


def test_threshold_for_rate_caps_at_the_argmax_buy_frequency():
    # Half the bars can never be BUY (pinned at the 0.0 sentinel) regardless of threshold.
    probs = np.concatenate([np.zeros(5000), _synthetic_effective_probs(5000, seed=1)])
    threshold, achieved = threshold_for_rate(probs, target_rate=0.90)  # unreachable: cap is 0.5
    assert threshold == 0.0
    assert achieved == pytest.approx(0.5, abs=0.02)


def test_threshold_for_rate_rejects_out_of_range_target():
    probs = _synthetic_effective_probs(100)
    with pytest.raises(ValueError):
        threshold_for_rate(probs, 1.5)
    with pytest.raises(ValueError):
        threshold_for_rate(probs, -0.1)


def test_threshold_for_rate_rejects_empty_input():
    with pytest.raises(ValueError):
        threshold_for_rate(np.empty(0), 0.1)


def test_rate_for_threshold_zero_bars_is_zero_not_an_error():
    assert rate_for_threshold(np.empty(0), 0.5) == 0.0
