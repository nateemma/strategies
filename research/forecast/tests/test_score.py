"""038 scorer (T006): causal threshold, long-only signals, one position per pair, costs, benchmark,
bootstrap, zero-forecast skill."""
from __future__ import annotations

import numpy as np
import pandas as pd

from research.forecast import common as C
from research.forecast import score as S


def test_threshold_is_causal():
    rng = np.random.default_rng(0)
    f = rng.normal(size=500)
    thr = S.causal_threshold(f, lookback=50, q=0.9)
    assert np.isnan(thr[:50]).all() and np.isfinite(thr[50:]).all()
    g = f.copy()
    g[300:] += 10.0  # perturb the future
    thr2 = S.causal_threshold(g, lookback=50, q=0.9)
    np.testing.assert_array_equal(thr[:301], thr2[:301])  # row 300 uses rows 250..299 only
    np.testing.assert_allclose(thr[100], np.quantile(f[50:100], 0.9))


def test_signal_needs_positive_forecast():
    assert not S.long_signals(np.zeros(400), lookback=50, q=0.9).any()
    assert not S.long_signals(-np.abs(np.random.default_rng(1).normal(size=400)) - 0.1, 50, 0.9).any()
    f = np.r_[np.zeros(60), 1.0, np.zeros(10)]
    sig = S.long_signals(f, lookback=50, q=0.9)
    assert sig[60] and sig.sum() == 1


def test_one_position_per_pair():
    rows = np.array([10, 11, 13, 14, 20])
    eligible = np.ones(100, dtype=bool)
    taken = S.simulate(rows, eligible, n=4)
    # 10 -> trade over open[11..15]; next allowed signal t >= 14
    assert taken.tolist() == [10, 14, 20]
    eligible[14] = False
    assert S.simulate(rows, eligible, n=4).tolist() == [10, 20]


def test_net_is_gross_minus_round_trip():
    assert C.cost_rt(5.0) == 2 * (0.0010 + 0.0005)
    tr = pd.DataFrame({"gross": [0.01, -0.002]})
    net = S.net(tr["gross"].to_numpy(), 5.0)
    np.testing.assert_allclose(net, [0.01 - 0.003, -0.002 - 0.003])


def test_pair_matched_random_benchmark():
    trades = pd.DataFrame({"pair": ["A", "A", "B"], "half": ["early"] * 3})
    rand = {("A", "early"): 0.001, ("B", "early"): -0.002}
    assert np.isclose(S.matched_benchmark(trades, rand), (0.001 + 0.001 - 0.002) / 3)


def test_bootstrap_reproducible_and_brackets_mean():
    rng = np.random.default_rng(2)
    d = pd.Series(pd.date_range("2024-01-01", periods=600, freq="6h", tz="UTC"))
    x = rng.normal(0.001, 0.01, 600)
    a = S.bootstrap_lb(x, d, n_boot=500, seed=0)
    assert a == S.bootstrap_lb(x, d, n_boot=500, seed=0)
    assert a < x.mean()


def test_zero_forecast_has_exactly_zero_skill():
    r = np.random.default_rng(3).normal(size=1000)
    assert S.r2_oos(np.zeros(1000), r) == 0.0
