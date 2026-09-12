"""Regression tests for the Phase 6 modules (T027, T032a, T032b, T036, T037, T033b).

Each test pins a defect that was actually found while building these, not a hypothetical.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from research.labelfree import stats as ST                       # noqa: E402
from research.labelfree.arms import screen_degenerate            # noqa: E402
from research.labelfree.baselines import (_rolling_std,          # noqa: E402
                                          ema_trend_exposure)


# --- T036 activity ---------------------------------------------------------

def test_always_on_is_one_trade():
    a = ST.activity(np.ones((1, ST.BARS_PER_YEAR)))
    assert a["trades_per_year"] == 1.0
    assert a["mean_hold_hours"] == pytest.approx(8760.0)


def test_flat_path_has_no_trades_and_undefined_hold():
    a = ST.activity(np.zeros((1, 1000)))
    assert a["trades_per_year"] == 0.0
    assert np.isnan(a["mean_hold_hours"])


def test_trade_is_a_contiguous_run_not_a_weight_change():
    """Exposure ramping 0.2->0.9 inside one position is ONE trade, not seven."""
    w = np.concatenate([np.zeros(50), np.linspace(0.2, 0.9, 100), np.zeros(50)])[None, :]
    assert ST.activity(w)["trades_per_year"] * (w.shape[1] / ST.BARS_PER_YEAR) == 1.0


# --- T032a baselines -------------------------------------------------------

def test_ema_rule_actually_trades():
    """Found 2026-09-12: band +/-0.1 on the RAW ratio never fires on 15m bars, so the
    baseline sat 100% flat on both folds. A baseline that never trades is not a baseline."""
    t = np.linspace(0, 1, 5000)
    price = (100 * np.exp(0.5 * t + 0.02 * np.sin(40 * t)))[None, :]
    w = ema_trend_exposure(price)
    assert 0.0 < w.mean() < 1.0


def test_ema_dead_zone_limits_churn():
    t = np.linspace(0, 1, 5000)
    price = (100 * np.exp(0.5 * t + 0.02 * np.sin(40 * t)))[None, :]
    w = ema_trend_exposure(price)
    assert np.abs(np.diff(w, axis=1)).sum() < 0.2 * w.shape[1]


def test_rolling_std_is_causal():
    """A spike at the end must not change any earlier value, or the baseline peeks."""
    x = np.zeros((1, 500))
    base = _rolling_std(x, 60).copy()
    x[0, -1] = 99.0
    assert np.allclose(_rolling_std(x, 60)[0, :-1], base[0, :-1])


# --- T033b annualisation ---------------------------------------------------

def test_annualised_pp_identity_over_one_year():
    assert ST.annualised_pp(np.log(1.10), 365) == pytest.approx(10.0, abs=1e-6)


def test_annualisation_exponentiates_once():
    """Scaling a percentage instead of the log is the error that once turned a working
    instrument into a reported 0.62x regression."""
    naive = (np.exp(np.log(1.10)) - 1) * 100 * (365 / 90)
    correct = ST.annualised_pp(np.log(1.10), 90)
    # 47.19 vs 40.56 -- the gap is larger than the floor the study resolves at, so the
    # two methods disagree about whether an effect is real.
    assert abs(correct - naive) > ST.FLOOR_PP_PER_YEAR


# --- T027 degeneracy screen ------------------------------------------------

def _run(lr, w):
    return {"status": "ok", "test_log_return": lr, "test_mean_exposure": w}


def test_degenerate_arm_is_excluded_and_its_return_withheld():
    """An arm that does nothing posts a superb return by avoiding the control's losses."""
    s = screen_degenerate({"lazy": [_run(9.9, 0.001)] * 5,
                           "real": [_run(-0.1, 0.40)] * 5})
    assert "lazy" in s["inconclusive"] and "lazy" not in s["eligible"]
    assert s["inconclusive"]["lazy"]["return"] == "WITHHELD (SC-015)"
    assert "real" in s["eligible"]


def test_one_flat_seed_does_not_void_an_arm_that_trades():
    runs = [_run(-0.1, 0.40)] * 4 + [_run(-0.1, 0.0)]
    assert "arm" in screen_degenerate({"arm": runs})["eligible"]


# --- T037 deflation interpretability --------------------------------------

def test_deflation_refuses_to_pass_on_seed_axis_alone():
    """DSR/CSCV need a time axis; 5 seeds cannot satisfy SC-004 however good they look."""
    out = ST.trial_log_deflation()
    assert out["interpretable"] is False
    assert out["sc004_verdict"].startswith("CANNOT BE EVALUATED")
