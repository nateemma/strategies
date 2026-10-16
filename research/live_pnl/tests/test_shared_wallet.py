"""036 US4: the order-level shared-wallet replay."""

from __future__ import annotations

import pytest

from research.live_pnl import shared_wallet as SW


def _tr(profit, *orders):
    return {"profit_abs": profit, "orders": [
        {"order_filled_timestamp": ts, "ft_is_entry": e, "amount": amt, "safe_price": px,
         "cost": amt * px} for ts, e, amt, px in orders]}


def test_no_contention_is_the_sum():
    books = {"cap_aro_ref": [_tr(10, (1, True, 9, 100), (5, False, 9, 101))],
             "mom_basket": [_tr(50, (2, True, 30, 100), (6, False, 30, 102))]}
    r = SW.replay(books, capacity=10000)
    assert r["pnl"]["cap_aro_ref"]["shared"] == 10 and r["pnl"]["mom_basket"]["shared"] == 50
    assert r["stats"]["cap_aro_ref"]["refused"] == r["stats"]["mom_basket"]["shrunk"] == 0


def test_basket_fully_invested_refuses_nnnc():
    books = {"mom_basket": [_tr(100, (1, True, 100, 100), (9, False, 100, 101))],
             "cap_aro_ref": [_tr(10, (2, True, 9, 100), (5, False, 9, 101))]}
    r = SW.replay(books, capacity=10000)
    assert r["stats"]["cap_aro_ref"]["refused"] == 1
    assert r["pnl"]["cap_aro_ref"]["shared"] == 0 and r["pnl"]["mom_basket"]["shared"] == 100


def test_shrink_scales_profit_and_trims_free_cash():
    books = {"mom_basket": [_tr(100, (1, True, 95, 100), (3, False, 50, 101), (9, False, 45, 101))],
             "cap_aro_ref": [_tr(10, (2, True, 9, 100), (5, False, 9, 101)),
                             _tr(20, (4, True, 9, 100), (6, False, 9, 101))]}
    r = SW.replay(books, capacity=10000)
    # t=2: 500 free of a 900 entry -> shrunk to 500; t=3 the basket sells 50/95 of its lots,
    # freeing 5000*... so the t=4 entry fills in full
    assert r["stats"]["cap_aro_ref"]["shrunk"] == 1 and r["stats"]["cap_aro_ref"]["refused"] == 0
    assert r["pnl"]["cap_aro_ref"]["shared"] == pytest.approx(10 * 500 / 900 + 20)


def test_exits_before_entries_at_the_same_time():
    books = {"mom_basket": [_tr(0, (1, True, 100, 100), (5, False, 100, 100))],
             "cap_aro_ref": [_tr(10, (5, True, 9, 100), (7, False, 9, 101))]}
    r = SW.replay(books, capacity=10000)
    assert r["stats"]["cap_aro_ref"]["refused"] == 0 and r["pnl"]["cap_aro_ref"]["shared"] == 10


def test_priority_breaks_same_time_entry_ties():
    books = {"cap_aro_ref": [_tr(10, (1, True, 9, 100), (5, False, 9, 101))],
             "mom_basket": [_tr(100, (1, True, 100, 100), (5, False, 100, 101))]}
    nn = SW.replay(books, capacity=10000, priority=("cap_aro_ref", "mom_basket"))
    bk = SW.replay(books, capacity=10000, priority=("mom_basket", "cap_aro_ref"))
    assert nn["stats"]["mom_basket"]["shrunk"] == 1 and nn["stats"]["cap_aro_ref"]["refused"] == 0
    assert bk["stats"]["cap_aro_ref"]["refused"] == 1


def test_realised_profit_returns_to_the_wallet():
    """A compounding book's later, larger orders must fit once its profit is back in the wallet
    (first replay returned only the basis and refused ~43% of the basket's own orders)."""
    books = {"mom_basket": [_tr(10000, (1, True, 100, 100), (2, False, 100, 200)),
                            _tr(0, (3, True, 150, 100), (4, False, 150, 100))],
             "cap_aro_ref": []}
    r = SW.replay(books, capacity=10000)
    assert r["stats"]["mom_basket"]["refused"] == 0 and r["stats"]["mom_basket"]["shrunk"] == 0


def test_small_recorded_orders_fill_when_cash_allows():
    books = {"mom_basket": [_tr(1, (1, True, 0.05, 100), (2, False, 0.05, 101))], "cap_aro_ref": []}
    r = SW.replay(books, capacity=10000)
    assert r["stats"]["mom_basket"]["refused"] == 0 and r["pnl"]["mom_basket"]["shared"] == 1
