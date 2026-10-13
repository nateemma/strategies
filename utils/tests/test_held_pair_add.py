"""031 T007-T010: the held-pair add switch (HELD_PAIR_ADD) -- declaration, the AddDecision
checks, the trade clocks after an add, and the stop around an add. Stub strategies over
BaseStrategy; nothing loads data or a model."""

from __future__ import annotations

import inspect
import json
import sys
import types
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pandas as pd
import pytest


ROOT = Path(__file__).resolve().parents[2]
for p in (str(ROOT), str(ROOT / "utils")):
    if p not in sys.path:
        sys.path.insert(0, p)

from Framework.BaseStrategy import BaseStrategy  # noqa: E402
from utils.EntryFramework import EntryFrameworkError, EntryFrameworkMixin  # noqa: E402
from utils.StopInvariants import StopInvariantMixin  # noqa: E402


T0 = datetime(2026, 1, 1, 12, 0, tzinfo=UTC)
BAR = timedelta(minutes=15)


class _ModelParent(BaseStrategy):
    def get_entry_conditions(self, dataframe):
        return dataframe["m"] > 0


class Off(EntryFrameworkMixin, _ModelParent):
    ENTRY_MODE = "model"


class On(EntryFrameworkMixin, _ModelParent):
    ENTRY_MODE = "model"
    HELD_PAIR_ADD = {}


# ----------------------------------------------------------------------- T007 declaration


def test_off_by_default_touches_no_freqtrade_attribute():
    assert Off.HELD_PAIR_ADD_RESOLVED is None
    assert Off.position_adjustment_enable is False
    assert Off.max_entry_position_adjustment == -1


def test_empty_dict_turns_it_on_with_the_defaults():
    assert On.HELD_PAIR_ADD_RESOLVED == {"max_adds": 1, "min_age_hours": 1.0,
                                         "under_water_only": False}
    assert On.position_adjustment_enable is True
    assert On.max_entry_position_adjustment == 1


def test_subclass_merges_per_key_and_none_turns_it_off():
    class Child(On):
        HELD_PAIR_ADD = {"min_age_hours": 4}

    class Grand(Child):
        HELD_PAIR_ADD = {"under_water_only": True}

    class Back(Grand):
        HELD_PAIR_ADD = None

    assert Grand.HELD_PAIR_ADD_RESOLVED == {"max_adds": 1, "min_age_hours": 4.0,
                                            "under_water_only": True}
    assert Back.HELD_PAIR_ADD_RESOLVED is None


@pytest.mark.parametrize("decl", [{"bogus": 1}, {"max_adds": 0}, {"max_adds": 1.5},
                                  {"min_age_hours": -1}, {"under_water_only": 1}, [1]])
def test_invalid_declarations_raise_at_class_creation(decl):
    with pytest.raises(EntryFrameworkError):
        type("Bad", (EntryFrameworkMixin, _ModelParent), {"ENTRY_MODE": "model",
                                                          "HELD_PAIR_ADD": decl})


def test_simpletrigger_preset_path_resolves_it():
    from evaluation.fingerprint import load_strategy_class

    base = load_strategy_class("SimpleTrigger_RevOnlyVolreg2_P2_G24")
    cls = type("STOn", (base,), {"HELD_PAIR_ADD": {"max_adds": 2}})
    assert cls.HELD_PAIR_ADD_RESOLVED["max_adds"] == 2
    assert cls.position_adjustment_enable is True
    assert base.HELD_PAIR_ADD_RESOLVED is None


# --------------------------------------------------------------------- T008 AddDecision


def _frame(signal_date, enter=1, volume=1000.0, close=10.0):
    return pd.DataFrame({"date": [signal_date - BAR, signal_date],
                         "enter_long": [0, enter], "volume": [volume, volume],
                         "close": [close, close]})


def _trade(**kw):
    t = types.SimpleNamespace(pair="SOL/USDT", is_short=False, has_open_orders=False,
                              nr_of_successful_entries=1, date_last_filled_utc=T0,
                              open_date_utc=T0, open_rate=10.0, id=3)
    t.__dict__.update(kw)
    return t


def _strat(cls=On, frame=None, stake=900.0):
    s = object.__new__(cls)
    s.timeframe = "15m"
    s.config = {"max_open_trades": 10, "stake_amount": stake}
    s.wallets = types.SimpleNamespace(
        get_trade_stake_amount=lambda pair, mot, update=False: stake)
    s.dp = types.SimpleNamespace(
        get_analyzed_dataframe=lambda pair, timeframe: (frame, None))
    return s


NOW = T0 + timedelta(hours=2)  # add candle; signal candle = NOW - 15 min


def _call(s, trade=None, profit=-0.01, min_stake=10.0, max_stake=5000.0, now=NOW):
    return s.adjust_trade_position(trade or _trade(), now, 10.0, profit, min_stake, max_stake)


def test_off_returns_none_and_counts_nothing():
    s = _strat(Off, _frame(NOW - BAR))
    assert _call(s) is None
    assert "_hp_counts" not in s.__dict__


def test_accept_path_returns_the_fresh_entry_size():
    s = _strat(frame=_frame(NOW - BAR, volume=1e9))
    assert _call(s) == (900.0, "hp_add")
    assert s._hp_counts == {"adds": 1}


def test_stake_is_capped_by_liquidity_and_wallet():
    # quote volume 1000 * 10 = 10,000 -> fillable 1,000 / 10 = 1,000; wallet bound 400
    s = _strat(frame=_frame(NOW - BAR, volume=1000.0))
    assert _call(s, max_stake=400.0) == (400.0, "hp_add")
    s = _strat(frame=_frame(NOW - BAR, volume=500.0), stake=900.0)  # fillable 500
    assert _call(s) == (500.0, "hp_add")


@pytest.mark.parametrize("reason, trade_kw, frame_kw, call_kw, cls_decl", [
    ("not_long", {"is_short": True}, {}, {}, None),
    ("open_order", {"has_open_orders": True}, {}, {}, None),
    ("max_adds", {"nr_of_successful_entries": 2}, {}, {}, None),
    ("row_misaligned", {}, {"signal_date": NOW}, {}, None),
    ("no_signal", {}, {"enter": 0}, {}, None),
    ("too_young", {"date_last_filled_utc": NOW - timedelta(minutes=45)}, {}, {}, None),
    ("too_young", {"date_last_filled_utc": NOW}, {}, {}, {"min_age_hours": 0}),
    ("in_profit", {}, {}, {"profit": 0.01}, {"under_water_only": True}),
    ("thin", {}, {"volume": 50.0}, {}, None),
    ("below_min", {}, {}, {"min_stake": 2000.0}, None),
])
def test_each_refusal_is_counted(reason, trade_kw, frame_kw, call_kw, cls_decl):
    cls = On if cls_decl is None else type("X", (On,), {"HELD_PAIR_ADD": cls_decl})
    fk = {"signal_date": NOW - BAR, "volume": 1e6, **frame_kw}
    s = _strat(cls, _frame(fk.pop("signal_date"), **fk))
    assert _call(s, _trade(**trade_kw), **call_kw) is None
    assert s._hp_counts == {f"refused_{reason}": 1}
    assert reason in On.HP_REASONS


def test_counters_are_written_to_the_entry_condition_sink(tmp_path, monkeypatch):
    monkeypatch.setenv("FT_ENTRY_CONDITION_SINK", str(tmp_path))
    s = _strat(frame=_frame(NOW - BAR, volume=1e9))
    _call(s)
    _call(s, _trade(nr_of_successful_entries=2))
    s._hp_sink()
    out = json.loads((tmp_path / "On_hp_add.json").read_text())["counts"]
    assert out == {"hp_adds": {"pass": 1, "total": 2},
                   "hp_refused_max_adds": {"pass": 1, "total": 2}}


# ------------------------------------------------------------------------- T009 clocks


LAST_FILL = T0 + timedelta(hours=30)


def _ns(adjust: bool, **kw):
    s = types.SimpleNamespace(position_adjustment_enable=adjust, **kw)
    return s


def _added_trade():
    return _trade(open_date_utc=T0, date_last_filled_utc=LAST_FILL, open_rate=10.0,
                  nr_of_successful_entries=2)


@pytest.mark.parametrize("adjust, start", [(True, LAST_FILL), (False, T0)])
def test_trade_clock_start(adjust, start):
    assert BaseStrategy.trade_clock_start(_ns(adjust), _added_trade()) == start


def _grace_strat(adjust):
    return _ns(adjust, stoploss_grace_hours=24.0, stoploss_grace_level=-0.27,
               _resolve_normal_stop=lambda pair, trade: -0.05)


@pytest.mark.parametrize("adjust, expected", [(True, -0.27), (False, None)])
def test_grace_window_restarts_at_the_last_fill(adjust, expected):
    now = LAST_FILL + timedelta(hours=2)  # 32 h after open, 2 h after the add
    v = BaseStrategy.custom_stoploss(_grace_strat(adjust), "SOL/USDT", _added_trade(), now,
                                     10.0, 0.0, after_fill=True)
    if expected is None:  # off: grace long expired -> the entry-anchored normal stop
        assert v == pytest.approx(10.0 * 0.95 / 10.0 - 1.0)
    else:
        assert v == expected


def test_post_grace_stop_is_anchored_to_the_averaged_open_rate():
    # averaged open 9.0 after an add; 25 h after the add the grace has expired
    t = _added_trade()
    t.open_rate = 9.0
    now = LAST_FILL + timedelta(hours=25)
    v = BaseStrategy.custom_stoploss(_grace_strat(True), "SOL/USDT", t, now, 10.0, 0.0,
                                     after_fill=False)
    assert 10.0 * (1.0 + v) == pytest.approx(9.0 * 0.95)


@pytest.mark.parametrize("adjust, allowed", [(True, False), (False, True)])
def test_minimum_hold_counts_from_the_last_fill(adjust, allowed):
    s = _ns(adjust, dp=types.SimpleNamespace(runmode=types.SimpleNamespace(value="backtest")))
    now = LAST_FILL + timedelta(minutes=30)
    ok = BaseStrategy.confirm_trade_exit(s, "SOL/USDT", _added_trade(), "market", 1.0, 10.0,
                                         "GTC", "exit_signal", now)
    assert ok is allowed


def test_atr_take_profit_reads_the_add_signal_bar_and_re_resolves():
    from utils.ExitPolicy import ExitParams, atr_take_profit

    rows = pd.DataFrame({"date": [T0 - BAR, LAST_FILL - BAR], "atr_pct_roll": [0.004, 0.012]})
    s = _ns(True, timeframe="15m",
            dp=types.SimpleNamespace(get_analyzed_dataframe=lambda pair, tf: (rows, None)))
    p = ExitParams(take_profit_atr_mult=2.0)
    t = _added_trade()
    assert BaseStrategy._apply_atr_take_profit(s, p, t).take_profit == atr_take_profit(2.0, 0.012)
    s.position_adjustment_enable = False  # off: the entry's own signal bar, cached separately
    assert BaseStrategy._apply_atr_take_profit(s, p, t).take_profit == atr_take_profit(2.0, 0.004)


def test_every_age_site_uses_the_clock():
    """custom_exit's age, the hold window and the invariant checker read trade_clock_start (the
    behavioural tests above cover the stop, the minimum hold and the ATR take-profit)."""
    from NNNC._hold_window import HoldWindowMixin  # noqa: F401 -- import proves it loads

    src = inspect.getsource(BaseStrategy.custom_exit)
    assert "trade_clock_start" in src and "open_date_utc" not in src
    import NNNC._hold_window as hw

    assert "trade_clock_start" in inspect.getsource(hw)
    import utils.StopInvariants as si

    assert "trade_clock_start" in inspect.getsource(si)


# ------------------------------------------------------------------------- T010 stop


class _StopParent:
    position_adjustment_enable = True

    def trade_clock_start(self, trade):
        return BaseStrategy.trade_clock_start(self, trade)

    def custom_stoploss(self, pair, trade, current_time, current_rate, current_profit,
                        after_fill, **kwargs):
        return BaseStrategy.custom_stoploss(self, pair, trade, current_time, current_rate,
                                            current_profit, after_fill)

    def _resolve_normal_stop(self, pair, trade):
        return -0.04


class _Checked(StopInvariantMixin, _StopParent):
    stoploss = -0.05
    use_atr_adaptive_stoploss = True
    atr_stoploss_floor = -0.04
    atr_stoploss_cap = -0.02
    stoploss_grace_hours = 24.0
    stoploss_grace_level = -0.27

    def _stop_bound_inputs(self, trade):
        return True, -0.05, -0.04


def test_invariants_hold_through_entry_grace_add_grace_tightened():
    s = _Checked()
    t = _trade(open_date_utc=T0, date_last_filled_utc=T0, open_rate=10.0)
    seq = [(T0, 10.0, True), (T0 + timedelta(hours=10), 9.8, False),
           (T0 + timedelta(hours=25), 10.2, False)]
    for now, rate, af in seq:
        s.custom_stoploss("SOL/USDT", t, now, rate, 0.0, af)
    # the add fills 30 h in at 9.0: averaged open 9.5, clock restarts
    t.date_last_filled_utc, t.open_rate, t.nr_of_successful_entries = LAST_FILL, 9.5, 2
    assert s.custom_stoploss("SOL/USDT", t, LAST_FILL, 9.0, 0.0, True) == -0.27
    assert s.custom_stoploss("SOL/USDT", t, LAST_FILL + timedelta(hours=5), 9.4, 0.0,
                             False) == 1.0
    v = s.custom_stoploss("SOL/USDT", t, LAST_FILL + timedelta(hours=25), 9.8, 0.0, False)
    assert 9.8 * (1.0 + v) == pytest.approx(9.5 * 0.96)
