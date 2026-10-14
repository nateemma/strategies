"""033 T021: S0 helpers -- label bounds, stake weights and shortfalls, trade identity, exits."""

from __future__ import annotations

import pandas as pd
import pytest

from research.width_relative import instrument as I


def _rep(shares=(0.5, 0.5), sp=(0.01, -0.02), buy=10, sell=10, kept=100):
    return {"buckets": [{"pos_share": s} for s in shares], "spearman_buy": sp[0],
            "spearman_sell": sp[1], "kept": kept, "invalid": 0, "labelled_buy": buy,
            "labelled_sell": sell}


def test_label_checks_bounds():
    assert I.label_checks(_rep(), {"train": {"class_counts": [1, 2, 3]}})["pass"]
    assert not I.label_checks(_rep(shares=(0.5, 0.62)), None)["pass"]
    assert not I.label_checks(_rep(sp=(0.06, 0.0)), None)["pass"]
    assert not I.label_checks(_rep(sell=0), None)["pass"]
    assert not I.label_checks(None, None)["pass"]


def test_stake_weights_and_shortfall_exclude_shared_liquidity_caps():
    assert I.stake_weight("r:model:agree") == pytest.approx(4 / 3)
    assert I.stake_weight("r:model:disagree") == pytest.approx(2 / 3)
    k = lambda i: {"pair": f"P{i}", "open_date": f"2023-01-0{i} 00:00"}
    width = [k(1) | {"stake_amount": 900.0}, k(2) | {"stake_amount": 900.0},
             k(3) | {"stake_amount": 900.0}, k(4) | {"stake_amount": 300.0}]
    tr = [k(1) | {"enter_tag": "r:model:agree", "stake_amount": 1200.0},
          k(2) | {"enter_tag": "r:model:disagree", "stake_amount": 600.0},
          k(3) | {"enter_tag": "r:model:agree", "stake_amount": 1000.0},   # wallet short
          k(4) | {"enter_tag": "r:model:agree", "stake_amount": 300.0}]    # capped in both
    s = I.stake_checks(tr, width)
    assert s["uncapped_matched"] == 3 and s["wallet_short"] == 1 and s["flag"]
    assert s["agree_share"] == 0.75
    assert s["capital_ratio_vs_width"] == round(3100 / 3000, 4)
    assert s["agree_disagree_stake_ratio"] == round((2500 / 3) / 600, 3)


def test_trade_identity_ignores_stake_and_profit():
    a = [{"pair": "X", "open_date": "2023-01-01 00:00", "close_date": "2023-01-02 00:00",
          "stake_amount": 900, "profit_ratio": 0.0123456}]
    b = [dict(a[0], stake_amount=1200, profit_ratio=0.01234561)]
    assert I.trade_keys(a) == I.trade_keys(b)


def test_exit_checks_liveness():
    idx = pd.MultiIndex.from_tuples(
        [("X", pd.Timestamp("2023-01-01 00:15", tz="UTC")),
         ("X", pd.Timestamp("2023-01-01 00:30", tz="UTC"))], names=["pair", "date"])
    table = pd.DataFrame({"rvol": [3.0, 0.5], "guard_metric": [0.2, 0.2]}, index=idx)
    tr = [{"pair": "X", "open_date": "2023-01-01 00:00", "close_date": "2023-01-01 01:00",
           "exit_reason": "roi"}]
    assert I.exit_checks(tr, table)["liveness_fail"]
    tr[0]["exit_reason"] = "model_exit "
    r = I.exit_checks(tr, table)
    assert r["model_exits"] == 1 and not r["liveness_fail"]
