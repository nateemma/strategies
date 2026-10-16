"""036 T022: cost-model refusals, fitting rule and the cost lookup."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from research.live_pnl import cost_model as CM


def test_refusals():
    assert CM.refusals(False, True) and CM.refusals(True, False)
    assert CM.refusals(True, True) == []


def _rows(n_days=8, seed=0):
    rng = np.random.default_rng(seed)
    out = []
    for d in range(n_days):
        for pair, depth, qv in (("A", 1e5, 1e6), ("B", 5e4, 5e5), ("C", 2e4, 2e5)):
            for stake in (100, 300, 900, 2500):
                for side in ("buy", "sell"):
                    p = stake / (qv * rng.uniform(0.8, 1.2))
                    out.append({"pair": pair, "day": f"2026-11-{d + 1:02d}", "stake": stake,
                                "side": side, "cost": 3 + 3000 * p + rng.normal(0, 0.3),
                                "extrapolated": False, "p": p, "depth_quote": depth})
    return pd.DataFrame(out)


def test_fit_chooses_a_model_and_holds_out_the_last_quarter():
    m = CM.fit(_rows(), {"A": 1.0, "B": 0.95, "C": 0.5})
    assert m["holdout_days"] == ["2026-11-07", "2026-11-08"]
    assert m["choice"] in ("participation", "per_pair")
    assert "C" in m["excluded_pairs"]          # coverage floor
    assert set(m["tiers"]) == {"A", "B", "C"} and m["tiers"]["C"] == 0


def test_cost_lookup_and_exclusion():
    m = CM.fit(_rows(), {"A": 1.0, "B": 1.0, "C": 1.0})
    c = CM.cost_bps(m, "A", 900, 1e6, "buy")
    assert 3 < c < 10
    m["excluded_pairs"] = ["A"]
    with pytest.raises(KeyError):
        CM.cost_bps(m, "A", 900, 1e6, "buy")
