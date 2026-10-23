"""040 T045: the transition head's label is the R11 UP/DOWN label (transition.labels) in DOWN 0 / NONE 1 / UP 2
coding, unknown rows carry no loss, and no label reads past the training frame."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

STRAT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(STRAT), str(STRAT / "NNMT"), str(STRAT / "evaluation")]

import _regime_state as RS  # noqa: E402
import _transition as TH  # noqa: E402
from research.nnmt import transition as T  # noqa: E402

PR = {"definition": {"deadband": 0.5, "sigma_window": 400, "sigma_min_periods": 100}}
H = 48


def _close(n=5000, seed=0):
    return pd.Series(100 * np.exp(np.cumsum(np.random.default_rng(seed).normal(0, 0.003, n))))


def test_label_matches_the_r11_labels():
    c = _close()
    y = TH.transition_target(c, H, 0.5, 400, 100)
    ref = T.labels(c, H, PR)
    ok = ref["up"].notna().to_numpy()
    assert (y[~ok] == RS.UNKNOWN).all() and (y[-H:] == RS.UNKNOWN).all()
    exp = np.where(ref["up"] == 1, 2, np.where(ref["down"] == 1, 0, 1))
    assert np.array_equal(y[ok], exp[ok])
    assert {0, 1, 2} <= set(y[ok].tolist())


def test_no_label_reads_past_the_end_of_the_frame():
    c = _close(seed=1)
    cut = 3500
    full, part = TH.transition_target(c, H, 0.5, 400, 100), TH.transition_target(c.iloc[:cut], H, 0.5, 400, 100)
    k = cut - H
    assert np.array_equal(full[:k], part[:k]) and (part[k:] == RS.UNKNOWN).all()
