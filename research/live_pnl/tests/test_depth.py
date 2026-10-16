"""036 T019: book walking, monotone curves and tiers."""

from __future__ import annotations

import numpy as np
import pytest

from research.live_pnl import depth as D


ASKS = [[101.0, 1.0], [102.0, 2.0]]   # 101 + 204 quote
BIDS = [[99.0, 1.0], [98.0, 2.0]]


def test_buy_walks_asks_against_mid():
    c, ext = D.walk_cost_bps(ASKS, 100.0, 101.0, "buy")
    assert not ext and c == pytest.approx(100.0)  # fills at 101 vs mid 100
    c2, _ = D.walk_cost_bps(ASKS, 100.0, 305.0, "buy")
    vwap = 305.0 / (1.0 + 2.0)
    assert c2 == pytest.approx((vwap / 100 - 1) * 1e4)


def test_sell_walks_bids():
    c, ext = D.walk_cost_bps(BIDS, 100.0, 99.0, "sell")
    assert not ext and c == pytest.approx(100.0)


def test_beyond_depth_is_extrapolated():
    c, ext = D.walk_cost_bps(ASKS, 100.0, 1000.0, "buy")
    assert ext and c > 150


def test_monotone_fit_and_inverse_round_trip():
    rng = np.random.default_rng(0)
    p = rng.uniform(0, 0.02, 2000)
    cost = 5 + 2000 * p + rng.normal(0, 1, 2000)
    curve = D.MonotoneCurve.fit(p, cost)
    assert np.all(np.diff(curve.cost) >= 0)
    for budget in (10, 20, 30):
        pm = curve.max_participation(budget)
        assert float(curve(pm)) == pytest.approx(budget, abs=0.5)
    assert curve.max_participation(0.0) == 0.0
    j = D.MonotoneCurve.from_json(curve.to_json())
    assert np.allclose(j(p[:10]), curve(p[:10]))


def test_tiers_are_terciles():
    t = D.tiers({f"P{i}": i for i in range(9)})
    assert [t[f"P{i}"] for i in range(9)] == [0, 0, 0, 1, 1, 1, 2, 2, 2]
