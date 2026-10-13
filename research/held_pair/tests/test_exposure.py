"""031 T026: exposure, crowd-out and the leverage control on a synthetic book with known answers."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest


STRAT = Path(__file__).resolve().parents[3]
if str(STRAT) not in sys.path:
    sys.path.insert(0, str(STRAT))

from research.held_pair import exposure as X  # noqa: E402


def _ms(s):
    return int(pd.Timestamp(s, tz="UTC").value // 10**6)


def _t(pair, o, c, profit, stake=900.0, fills=None, fold="f00"):
    return {"fold_id": fold, "pair": pair, "open_date": o, "close_date": c,
            "profit_abs": profit, "scale_pp_per_abs": 0.01, "stake_amount": stake,
            "entry_fills": fills, "n_entries": len(fills) if fills else 1}


CTL = [_t("A", "2023-01-01 00:00", "2023-01-01 10:00", 9.0),       # 900 x 10 h
       _t("A", "2023-01-01 12:00", "2023-01-01 14:00", 4.0)]       # held in the arm
ARM = [_t("A", "2023-01-01 00:00", "2023-01-01 15:00", 30.0, stake=1800.0,
          fills=[(_ms("2023-01-01 00:00"), 900.0), (_ms("2023-01-01 05:00"), 900.0)]),
       _t("B", "2023-01-01 02:00", "2023-01-01 04:00", 1.0)]


def test_exposure_sweep():
    e = X.exposure(ARM)
    # A: 900 for 5 h, then 1800 for 10 h; B: 900 for 2 h overlapping A's first leg
    assert e["peak_stake"] == 1800.0
    assert e["stake_hours"] == pytest.approx(900 * 15 + 900 * 10 + 900 * 2)
    assert e["avg_stake"] == pytest.approx(e["stake_hours"] / 15.0)


def test_decomposition():
    d = X.decomposition(ARM, CTL)
    assert d["same_n"] == 1 and d["same_entry_pp"] == pytest.approx(0.30 - 0.09)
    assert d["forgone_n"] == 1 and d["forgone_pair_held_n"] == 1
    assert d["forgone_pp"] == pytest.approx(0.04)
    assert d["new_n"] == 1 and d["new_pp"] == pytest.approx(0.01)


def test_leverage_control():
    lev = X.leverage_control(ARM, CTL)
    rate = (0.09 + 0.04) / (900 * 10 + 900 * 2)  # control pp per stake-hour
    assert lev["leverage_pp"] == pytest.approx(900 * 10 * rate)  # the add: 900 for 10 h
