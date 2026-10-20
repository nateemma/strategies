"""040 T041 (research R11, SC-005): the regime-transition label is causal at both ends, fires on a real
bear->bull turn, is NaN where its future end is unknown, and is symmetric on a driftless walk."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

STRAT = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(STRAT), str(STRAT / "evaluation")]

from research.nnmt import transition as T  # noqa: E402

PR = {"definition": {"deadband": 0.5, "sigma_window": 400, "sigma_min_periods": 100}}
H = 48


def _walk(n=6000, seed=0, drift=0.0):
    r = np.random.default_rng(seed).normal(drift, 0.003, n)
    return pd.Series(100 * np.exp(np.cumsum(r)))


def test_up_fires_before_a_bear_to_bull_turn_completes():
    rng = np.random.default_rng(1)
    r = np.r_[rng.normal(-0.002, 0.002, 1500), rng.normal(0.004, 0.002, 600)]  # bear, then a bull ramp
    lab = T.labels(pd.Series(100 * np.exp(np.cumsum(r))), H, PR)
    turn = 1500
    up = lab["up"].to_numpy()
    assert np.nanmax(up[turn - H: turn + 2 * H]) == 1.0   # UP is flagged around/just before the turn
    assert np.nansum(up[400: turn - H - 50]) == 0          # not during the steady bear phase


@pytest.mark.parametrize("cut", [2000, 3500, 5999])
def test_state_is_causal(cut):
    c = _walk()
    d = PR["definition"]
    full = T.state(T.zscore(c, H, d["sigma_window"], d["sigma_min_periods"]), d["deadband"]).to_numpy()
    part = T.state(T.zscore(c.iloc[: cut + 1], H, d["sigma_window"], d["sigma_min_periods"]), d["deadband"]).to_numpy()
    assert T._equal(full[: cut + 1], part)


@pytest.mark.parametrize("cut", [2000, 4000])
def test_label_from_a_prefix_matches_up_to_cut_minus_h_and_is_nan_after(cut):
    c = _walk()
    full, part = T.labels(c, H, PR), T.labels(c.iloc[: cut + 1], H, PR)
    k = cut + 1 - H
    for col in ("up", "down"):
        assert T._equal(full[col].to_numpy()[:k], part[col].to_numpy()[:k])
        assert part[col].iloc[k:].isna().all()


def test_label_is_nan_within_h_of_the_end():
    lab = T.labels(_walk(), H, PR)
    assert lab["up"].iloc[-H:].isna().all() and lab["down"].iloc[-H:].isna().all()


def test_driftless_walk_is_roughly_symmetric():
    ups, downs = [], []
    for seed in range(8):
        lab = T.labels(_walk(seed=seed), H, PR)
        ups.append(lab["up"].mean())
        downs.append(lab["down"].mean())
    assert abs(np.mean(ups) - np.mean(downs)) < 0.25 * np.mean(ups + downs)
