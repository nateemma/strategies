"""037 T006: scores -- AUC, MI, lift, stratified AUC, aggregation."""
from __future__ import annotations

import numpy as np
from sklearn.metrics import roc_auc_score

from research.indicator_screen import scores as S

rng = np.random.default_rng(0)


def test_auc_equals_sklearn_with_ties():
    x = rng.integers(0, 10, 2000).astype(float)
    y = (rng.random(2000) < 0.3).astype(float)
    assert abs(S.auc(x, y) - roc_auc_score(y, x)) < 1e-12


def test_auc_drops_nan_and_is_nan_with_one_class():
    x = np.array([1.0, np.nan, 3.0, 2.0])
    y = np.array([0, 1, 1, np.nan])
    assert S.auc(x, y) == 1.0
    assert np.isnan(S.auc([1, 2, 3], [1, 1, 1]))


def test_perfect_column_gives_auc_one_and_mi_one():
    y = (np.arange(5000) % 2).astype(float)            # exactly balanced: the median edge splits it
    x = y + rng.random(5000) * 0.1
    assert S.auc(x, y) == 1.0
    assert abs(S.mi_norm(x, y, S.quantile_edges(x, 20)) - 1.0) < 1e-9
    assert S.direction(S.auc(-x, y)) == -1 and S.strength(S.auc(-x, y)) == 0.5


def test_independent_column_has_mi_near_zero():
    x = rng.random(50000)
    y = (rng.random(50000) < 0.5).astype(float)
    assert S.mi_norm(x, y, S.quantile_edges(x, 20)) < 0.002


def test_lift_and_not_computable_cases():
    y = np.array([1, 0, 0, 0] * 50, dtype=float)       # base 0.25
    s = np.array([1, 0, 0, 0] * 50, dtype=float)        # fires exactly on positives
    l, rate, n = S.lift(s, y, min_fires=10, max_fire_rate=0.95)
    assert (l, rate, n) == (4.0, 0.25, 50)
    assert np.isnan(S.lift(s, y, min_fires=60, max_fire_rate=0.95)[0])
    assert np.isnan(S.lift(np.ones(200), y, min_fires=10, max_fire_rate=0.95)[0])


def test_stratified_auc_of_the_stratifying_column_is_half_and_single_cell_is_plain():
    n = 20000
    a = rng.random(n)
    b = rng.random(n)
    y = (rng.random(n) < 0.2 + 0.15 * np.floor(a * 5)).astype(float)   # a matters only by quintile
    ea, eb = S.quantile_edges(a, 5), S.quantile_edges(b, 5)
    assert abs(S.auc(a, y) - 0.5) > 0.2
    assert abs(S.stratified_auc(a, y, a, b, ea, eb) - 0.5) < 0.05
    x = rng.random(n)
    assert abs(S.stratified_auc(x, y, a, b, [], []) - S.auc(x, y)) < 1e-12


def test_aggregation_drops_small_windows_and_reports_halves():
    ws = [{"window": f"f{i:02d}", "half": "early" if i < 8 else "late", "n_rows": 500, "auc": 0.6}
          for i in range(15)]
    ws[3]["auc"] = 0.4
    ws.append({"window": "f99", "half": "late", "n_rows": 5, "auc": 0.0})
    ws.append({"window": "E0", "half": "e0", "n_rows": 500, "auc": 0.45})
    r = S.aggregate(ws, min_rows=100)
    assert r["median"] == 0.6 and r["median_direction"] == 1
    assert abs(r["same_direction_share"] - 14 / 15) < 1e-12
    assert r["halves"]["e0"] == {"median": 0.45, "n_windows": 1}
    assert r["halves"]["late"]["n_windows"] == 7


def test_auc_multi_equals_auc_per_target_with_nan_and_ties():
    x = rng.integers(0, 50, 5000).astype(float)
    x[::17] = np.nan
    y1 = (rng.random(5000) < 0.3).astype(float)
    y2 = (rng.random(5000) < 0.6).astype(float)
    y2[::5] = np.nan
    got = S.auc_multi(x, [y1, y2, np.zeros(5000)])
    assert abs(got[0] - S.auc(x, y1)) < 1e-12 and abs(got[1] - S.auc(x, y2)) < 1e-12
    assert np.isnan(got[2])
