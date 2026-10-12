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


def test_weights_scale_each_book_separately():
    a = _p("a", {"f00": 2.0}, [_t("X", "2023-01-01 00:00", "2023-01-01 01:00")])
    b = _p("b", {"f00": 4.0}, [_t("Y", "2023-01-01 00:00", "2023-01-01 01:00")])
    c = B.combine({"a": a, "b": b}, weights={"a": 0.25, "b": 0.75})
    assert c["per_fold"][0]["return_pct"] == 3.5
    assert sorted(t["profit_abs"] for t in c["trades"]) == [0.25, 0.75]


def test_risk_weights_use_only_the_given_folds_and_favour_the_calmer_book():
    calm = [{**_t("X", f"2023-01-0{d} 00:00", f"2023-01-0{d} 01:00"),
             "profit_abs": (1.0 if d % 2 else 0.5)} for d in range(1, 9)]
    wild = [{**_t("Y", f"2023-01-0{d} 00:00", f"2023-01-0{d} 01:00"),
             "profit_abs": (5.0 if d % 2 else -5.0)} for d in range(1, 9)]
    late = [{**_t("X", "2024-01-01 00:00", "2024-01-01 01:00", fold="f09"), "profit_abs": 999.0}]
    books = {"calm": _p("c", {"f00": 0}, calm + late), "wild": _p("w", {"f00": 0}, wild)}
    w = B.risk_weights(books, {"f00"})
    assert w["calm"]["weight"] > w["wild"]["weight"]
    assert abs(w["calm"]["weight"] + w["wild"]["weight"] - 1) < 1e-12


def test_committed_capital_frees_before_reentry():
    tr = [{**_t("X", "2023-01-01 00:00", "2023-01-01 01:00"), "weight": 0.5},
          {**_t("X", "2023-01-01 01:00", "2023-01-01 02:00"), "weight": 0.5}]
    assert B.committed(tr, 900.0) == 450.0


def test_mean_committed_is_time_weighted_and_weighted():
    tr = [{**_t("X", "2023-01-01 00:00", "2023-01-01 01:00"), "weight": 1.0},
          {**_t("Y", "2023-01-01 01:00", "2023-01-01 02:00"), "weight": 0.5}]
    assert B.mean_committed(tr, 900.0) == (900.0 * 1 + 450.0 * 1) / 2
