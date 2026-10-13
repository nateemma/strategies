"""031 T017: position-add accounting in arm records -- additive, absent when off, and the
trades_hash definition unchanged."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd


STRAT = Path(__file__).resolve().parents[1]
if str(STRAT) not in sys.path:
    sys.path.insert(0, str(STRAT))

from regime import run_arm  # noqa: E402


def _o(entry, ts=1):
    return {"ft_is_entry": entry, "order_filled_timestamp": ts}


def _trades(with_add: bool):
    second = [_o(True), _o(True), _o(False)] if with_add else [_o(True), _o(False)]
    return pd.DataFrame({"pair": ["A/USDT", "B/USDT"], "orders": [[_o(True), _o(False)], second]})


def test_no_adds_adds_no_fields():
    assert run_arm.add_accounting(_trades(False)) == {}


def test_adds_are_counted():
    assert run_arm.add_accounting(_trades(True)) == {"n_adds": 1, "n_trades_with_adds": 1}


def test_unfilled_entry_orders_do_not_count():
    t = pd.DataFrame({"pair": ["A/USDT"], "orders": [[_o(True), _o(True, None), _o(False)]]})
    assert run_arm.add_accounting(t) == {}


def test_hp_counters_from_the_entry_condition_sink():
    ecc = {"guard_rvol": {"pass": 5, "total": 9}, "hp_adds": {"pass": 3, "total": 10},
           "hp_refused_no_signal": {"pass": 7, "total": 10}}
    assert run_arm.hp_counters(ecc) == {"adds": 3, "refused_no_signal": 7}
    assert run_arm.hp_counters({"guard_rvol": {"pass": 1, "total": 2}}) == {}
    assert run_arm.hp_counters(None) == {}


def test_trades_hash_columns_are_unchanged():
    import inspect

    src = inspect.getsource(run_arm.summarize)
    assert '("pair", "open_date", "close_date", "profit_ratio")' in src
