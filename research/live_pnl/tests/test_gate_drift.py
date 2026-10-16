"""036 T015/T016: drift flagging and the recording gate."""

from __future__ import annotations

import pandas as pd

from research.live_pnl import drift as D
from research.live_pnl import gate as G


PAIRS = {f"P{i}": 5 for i in range(10)}


def test_gate_closed_before_28_days():
    assert not G.status(27.9, PAIRS)["open"]


def test_gate_closed_with_a_short_pair():
    assert not G.status(40, {**PAIRS, "P3": 4})["open"]


def test_gate_open_at_28_days_with_all_pairs():
    s = G.status(28, PAIRS)
    assert s["open"] and s["thin_pairs"] == []


def test_gate_opens_at_84_days_listing_thin_pairs():
    s = G.status(84, {**PAIRS, "P3": 2, "P7": 0})
    assert s["open"] and s["thin_pairs"] == ["P3", "P7"]


def _sig(rate, n=1000):
    k = int(rate * n)
    return pd.DataFrame({"candle": ["2026-10-20T10:00:00+00:00"] * n,
                         "p_buy": [0.9] * k + [0.1] * (n - k)})


def test_drift_in_band_not_flagged():
    w = D.weekly(_sig(0.015), 0.8, 0.014735)
    assert len(w) == 1 and not w[0]["flag"]


def test_drift_out_of_band_flagged():
    assert D.weekly(_sig(0.005), 0.8, 0.014735)[0]["flag"]
    assert D.weekly(_sig(0.040), 0.8, 0.014735)[0]["flag"]


def test_drift_uses_the_logged_threshold():
    sig = _sig(0.015)
    sig["p_buy"] = [0.785] * 15 + [0.1] * 985   # above the picked 0.7783, above applied 0.78
    sig["threshold"] = 0.79                      # applied threshold logged higher
    w = D.weekly(sig, 0.7783, 0.014735)
    assert w[0]["buy_rate"] == 0.0 and w[0]["flag"]
