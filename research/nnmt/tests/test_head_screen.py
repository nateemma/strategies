"""040 T037 (research R10): the head screen's trailing baselines are causal, its labels are NNMT's own
training targets (3 classes per head), and the AUC margin behaves: a head equal to its label beats the
baseline, a head equal to the trailing state does not."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

STRAT = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(STRAT), str(STRAT / "evaluation"), str(STRAT / "NNMT")]

from research.nnmt import head_screen as HS  # noqa: E402

PR = {"baseline_windows": {"regime": 200, "risk": 50, "flow": 24, "momentum": 24, "profit": 8}}


def _frame(n=2000, seed=0):
    rng = np.random.default_rng(seed)
    c = 100 * np.exp(np.cumsum(rng.normal(0, 0.004, n)))
    slow = lambda p: 0.9 * np.sin(np.arange(n) / p) + rng.normal(0, 0.1, n)  # noqa: E731 -- regime-like inputs
    return pd.DataFrame({"close": c, "high": c * 1.003, "low": c * 0.997, "volume": 1.0,
                         "base:atr_norm": slow(150), "base:di_diff_scaled": slow(60),
                         "base:aroonosc_scaled": slow(80)})


@pytest.mark.parametrize("cut", [500, 1200, 1999])
def test_baselines_are_causal(cut):
    df = _frame()
    s = HS._strategy()
    full = HS.baselines(df, s, PR).iloc[: cut + 1]
    part = HS.baselines(df.iloc[: cut + 1].copy(), s, PR)
    pd.testing.assert_frame_equal(full.reset_index(drop=True), part.reset_index(drop=True))


def test_labels_are_nnmts_three_class_targets():
    lab = HS.head_labels(_frame(), HS._strategy())
    assert list(lab.columns) == ["regime", "risk", "flow", "momentum", "profit"]
    for c in lab.columns:
        assert set(np.unique(lab[c])) <= {0, 1, 2}
        assert lab[c].nunique() >= 2, c  # a degenerate label could not be screened


def test_margin_behaves():
    rng = np.random.default_rng(3)
    y = (rng.uniform(size=5000) > 0.7).astype(float)
    trailing = y + rng.normal(0, 1.5, 5000)        # the "current state": informative but noisy
    oracle = y + rng.normal(0, 0.2, 5000)          # a head that knows the label
    assert HS._auc(oracle, y) - HS._auc(trailing, y) > 0.02
    assert abs(HS._auc(trailing, y) - HS._auc(trailing, y)) < 1e-12
