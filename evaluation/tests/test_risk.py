"""026 addendum 4: paired risk verdict (daily Sharpe, max drawdown; stationary block bootstrap)."""

from __future__ import annotations

import numpy as np
import pandas as pd
from evaluation import risk as R


def _trades(pp: np.ndarray, start="2024-01-01") -> list[dict]:
    days = pd.date_range(start, periods=len(pp), freq="D", tz="UTC")
    return [{"close_date": str(d), "profit_abs": float(v), "scale_pp_per_abs": 1.0}
            for d, v in zip(days, pp, strict=True) if v != 0]


def _noise(seed=0, n=400, mu=0.05):
    return np.random.default_rng(seed).normal(mu, 1.0, n)


def test_identical_series_are_unchanged_with_zero_delta():
    x = _noise()
    r = R.compare(_trades(x), _trades(x))
    assert r["verdict_sharpe"] == r["verdict_maxdd"] == "UNCHANGED"
    assert r["sharpe_delta"] == 0 and r["maxdd_delta"] == 0


def test_half_size_copy_halves_drawdown_but_keeps_sharpe():
    x = _noise()
    r = R.compare(_trades(0.5 * x), _trades(x))
    assert abs(r["sharpe_delta"]) < 1e-9 and r["verdict_sharpe"] == "UNCHANGED"
    assert r["verdict_maxdd"] == "IMPROVED" and r["maxdd_delta"] < 0


def test_higher_mean_same_noise_improves_sharpe_and_lower_worsens():
    x = _noise(mu=0.0)
    assert R.compare(_trades(x + 0.3), _trades(x))["verdict_sharpe"] == "IMPROVED"
    assert R.compare(_trades(x - 0.3), _trades(x))["verdict_sharpe"] == "WORSENED"


def test_reproducible_and_paired_on_dates():
    x, y = _noise(1), _noise(2)
    assert R.compare(_trades(x), _trades(y)) == R.compare(_trades(x), _trades(y))
    # a control that only trades on later days is zero-filled over the shared span
    r = R.compare(_trades(x), _trades(y[:50], start="2024-06-01"))
    assert r["days"] == 400


def test_max_drawdown_counts_from_the_starting_zero():
    assert R.max_drawdown(np.array([-1.0, -1.0, 3.0, -0.5])) == 2.0


def test_block_indices_are_mostly_consecutive():
    idx = R._stationary_indices(500, 50, np.random.default_rng(0))
    steps = (np.diff(idx, axis=1) % 500) == 1
    assert 0.85 < steps.mean() < 0.95  # 1 - 1/BLOCK_MEAN_DAYS


def test_missing_close_dates_are_unresolvable_not_an_error():
    t = [{"close_date": None, "profit_abs": 1.0, "scale_pp_per_abs": 1.0}]
    r = R.compare(t, t)
    assert r["verdict_sharpe"] == r["verdict_maxdd"] == "UNRESOLVABLE"
    assert "UNRESOLVABLE" in R.line(r)
