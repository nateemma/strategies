"""026 addendum 2: separate books combine per fold and report what one wallet could not hold."""

from __future__ import annotations

import pytest
from evaluation import books as B


def _p(run, rets, trades):
    return {
        "run_id": run, "family": "x", "status": "complete", "folds_missing": [],
        "per_fold": [{"fold_id": f, "n_trades": 1, "return_pct": r, "test_days": 90,
                      "status": "evaluated"} for f, r in rets.items()],
        "trades": trades, "n_trades_pooled": len(trades), "folds_complete": len(rets),
    }


def _t(pair, o, c, fold="f00"):
    return {"fold_id": fold, "pair": pair, "open_date": o, "close_date": c, "profit_abs": 1.0,
            "scale_pp_per_abs": 0.01}


def test_combine_sums_fold_returns_and_unions_trades():
    a = _p("a", {"f00": 1.0, "f01": None}, [_t("X", "2023-01-01 00:00", "2023-01-01 01:00")])
    b = _p("b", {"f00": -0.5, "f01": 2.0}, [_t("Y", "2023-01-01 00:00", "2023-01-01 01:00")])
    c = B.combine({"a": a, "b": b})
    assert [f["return_pct"] for f in c["per_fold"]] == [0.5, 2.0]
    assert c["n_trades_pooled"] == 2 and {t["book"] for t in c["trades"]} == {"a", "b"}


def test_stake_scale_scales_returns_and_trade_profit():
    a = _p("a", {"f00": 1.0}, [_t("X", "2023-01-01 00:00", "2023-01-01 01:00")])
    b = _p("b", {"f00": 3.0}, [])
    c = B.combine({"a": a, "b": b}, stake_scale=0.5)
    assert c["per_fold"][0]["return_pct"] == 2.0 and c["trades"][0]["profit_abs"] == 0.5


def test_combine_refuses_mismatched_folds():
    with pytest.raises(ValueError):
        B.combine({"a": _p("a", {"f00": 1.0}, []), "b": _p("b", {"f01": 1.0}, [])})


def test_capacity_counts_overflow_and_same_pair_across_books():
    tr = [
        {**_t("X", "2023-01-01 00:00", "2023-01-01 05:00"), "book": "a"},
        {**_t("Y", "2023-01-01 00:15", "2023-01-01 05:00"), "book": "a"},
        {**_t("X", "2023-01-01 01:00", "2023-01-01 02:00"), "book": "b"},  # X held by book a
        {**_t("X", "2023-01-01 06:00", "2023-01-01 07:00"), "book": "b"},  # free again
    ]
    cap = B.capacity(tr, max_open=2)
    assert cap["same_pair_other_book"] == 1
    assert cap["over_max_open"] == 1  # the 01:00 entry found 2 positions open
    assert cap["peak_open"] == 3
