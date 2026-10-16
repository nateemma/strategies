"""036 T025: bias flag on two consecutive out-of-tolerance checks."""

from __future__ import annotations

import pandas as pd

from research.live_pnl import model_check as MC


def _check(err):
    return {"date": "x", "n": 10, "median_signed_error_bps": err}


def test_tier_errors():
    rows = pd.DataFrame({"tier": [0, 0, 1], "predicted": [10, 12, 5], "walked": [4, 4, 5]})
    assert MC.tier_errors(rows) == {"0": 7.0, "1": 0.0}


def test_constant_offset_flags_on_the_second_check():
    s = MC.update({}, _check({"0": 8.0}), 5.0)
    assert not s["bias_flag"]
    s = MC.update(s, _check({"0": 7.5}), 5.0)
    assert s["bias_flag"]


def test_no_offset_never_flags():
    s = {}
    for _ in range(4):
        s = MC.update(s, _check({"0": 1.0, "1": -2.0}), 5.0)
    assert not s["bias_flag"]


def test_sign_flip_does_not_flag():
    s = MC.update({}, _check({"0": 8.0}), 5.0)
    assert not MC.update(s, _check({"0": -8.0}), 5.0)["bias_flag"]


def test_exit_uses_the_last_entry_stake():
    orders = pd.DataFrame({"pair": ["A", "A", "B"], "granted_stake": [300.0, 450.0, 900.0],
                           "order_time": ["2026-10-20T01:00", "2026-10-21T01:00", "2026-10-21T02:00"],
                           "rejected": [False, False, False]})
    assert MC.entry_stake(orders, "A", "2026-10-21T05:00") == 450.0
    assert MC.entry_stake(orders, "A", "2026-10-20T05:00") == 300.0
    assert MC.entry_stake(orders, "C", "2026-10-21T05:00") is None
