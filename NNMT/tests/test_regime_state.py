"""040 T058 (preregistration_regime_state.json): the regime-state mixin's label is the transition study's
future state in MarketRegime coding, unknown rows one-hot to zero (no loss), and no label reads a bar after
the training frame."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

STRAT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(STRAT), str(STRAT / "NNMT"), str(STRAT / "evaluation")]

import _regime_state as RS  # noqa: E402
from research.nnmt import transition as T  # noqa: E402

PR = {"definition": {"deadband": 0.5, "sigma_window": 400, "sigma_min_periods": 100}}
H = 48


def _close(n=5000, seed=0):
    return pd.Series(100 * np.exp(np.cumsum(np.random.default_rng(seed).normal(0, 0.003, n))))


def test_label_matches_the_transition_state_in_marketregime_coding():
    c = _close()
    y = RS.future_state(c, H, 0.5, 400, 100)
    ref = T.future_state(c, H, PR)["fstate"].to_numpy()
    ok = ~np.isnan(ref)
    assert np.array_equal(y[ok], (ref[ok] + 1).astype(int))     # -1/0/+1 -> 0/1/2
    assert (y[~ok] == RS.UNKNOWN).all()
    assert (y[-H:] == RS.UNKNOWN).all()                           # future after the frame: no label


def test_no_label_reads_past_the_end_of_the_frame():
    c = _close()
    cut = 3500
    full, part = RS.future_state(c, H, 0.5, 400, 100), RS.future_state(c.iloc[:cut], H, 0.5, 400, 100)
    k = cut - H
    assert np.array_equal(full[:k], part[:k]) and (part[k:] == RS.UNKNOWN).all()


def test_unknown_one_hot_is_a_zero_row():
    u = RS._MaskedOneHot()
    oh = u.one_hot_encode(np.array([0, 2, RS.UNKNOWN, 1]), 3)
    assert oh.shape == (4, 3)
    assert oh.tolist() == [[1, 0, 0], [0, 0, 1], [0, 0, 0], [0, 1, 0]]
