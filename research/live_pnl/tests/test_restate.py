"""036 T023: re-statement keeps trade identity and prices both legs."""

from __future__ import annotations

import pandas as pd
import pytest

from research.live_pnl import restate as R


def _trades():
    return pd.DataFrame({
        "pair": ["A", "B"], "stake_amount": [900.0, 300.0], "profit_abs": [20.0, -5.0],
        "open_date": pd.to_datetime(["2024-01-01 10:15", "2024-01-02 11:00"], utc=True),
        "close_date": pd.to_datetime(["2024-01-01 14:00", "2024-01-02 15:00"], utc=True)})


def test_zero_cost_is_gross():
    t = R.price_trades(_trades(), lambda *a: 0.0, lambda p, ts: 1e6)
    assert (t.net_abs == t.profit_abs).all()
    assert list(t[["pair", "open_date", "close_date"]].itertuples(index=False)) == \
        list(_trades()[["pair", "open_date", "close_date"]].itertuples(index=False))


def test_both_legs_priced_at_their_own_candles():
    seen = []

    def cost(pair, stake, qv, side):
        seen.append((pair, side, qv))
        return 10.0 if side == "buy" else 20.0

    t = R.price_trades(_trades(), cost, lambda p, ts: ts.hour * 1.0)
    assert t.cost_abs.tolist() == pytest.approx([900 * 30 / 1e4, 300 * 30 / 1e4])
    assert ("A", "buy", 10.0) in seen and ("A", "sell", 14.0) in seen
    s = R.summarise(t)
    assert s["net_abs"] == pytest.approx(15.0 - 2.7 - 0.9)
