"""024 T009: FamilyExitMixin (contract C2, data-model FamilyExitSet / TradeFamilyTag).

The mixin turns a trade's entry tag (`"<family>:<trigger>[+ov]"`) into that family's exit
parameters, layered over the strategy's own `_exit_params()`. It is strategy-agnostic: it reads
`FAMILY_EXITS` and the tag, nothing else.
"""

from __future__ import annotations

import types

import pytest
from utils.ExitPolicy import ExitParams
from utils.FamilyExits import FamilyExitError, FamilyExitMixin, family_of


def _base_params() -> ExitParams:
    return ExitParams(
        atr_stoploss_multiplier=2.5,
        atr_stoploss_cap=-0.02,
        atr_stoploss_floor=-0.12,
        static_stoploss=-0.05,
        use_atr_adaptive_stoploss=True,
        use_volume_confirmation_stoploss=True,
        take_profit=0.022,
        max_hold_days=10,
        stoploss_grace_hours=6.5,
        stoploss_grace_level=-0.27,
    )


class _Base:
    SELL_OVERRIDES: dict = {}
    ATTR_OVERRIDES: dict = {}

    def _exit_params(self, trade=None):
        return _base_params()


def _strategy(family_exits, sell=None, attrs=None):
    return type(
        "S",
        (FamilyExitMixin, _Base),
        {"FAMILY_EXITS": family_exits, "SELL_OVERRIDES": sell or {}, "ATTR_OVERRIDES": attrs or {}},
    )


MOMENTUM = {"take_profit": 10.0, "unclog_12h_hours": 24.0}


def _trade(tag):
    return types.SimpleNamespace(enter_tag=tag, id=1)


def test_family_of_parses_the_prefix_and_ignores_the_overlap_suffix():
    assert family_of("momentum:breakout_gbb") == "momentum"
    assert family_of("reversion:gbb+ov") == "reversion"
    assert family_of("model_entry ") is None
    assert family_of(None) is None
    assert family_of("") is None
    assert family_of("sideways:x") is None  # unknown family


def test_family_set_is_applied_over_the_base_params():
    s = _strategy({"reversion": {}, "momentum": MOMENTUM, "unassigned": {}})()
    p = s._exit_params(_trade("momentum:breakout_gbb"))
    assert p.take_profit == 10.0 and p.unclog_12h_hours == 24.0
    assert p.atr_stoploss_floor == _base_params().atr_stoploss_floor  # untouched fields kept
    assert s._exit_params(_trade("reversion:gbb")) == _base_params()


def test_no_trade_unknown_or_missing_tag_falls_back_and_is_counted():
    s = _strategy({"momentum": MOMENTUM})()
    assert s._exit_params() == _base_params()  # no trade: not a fallback
    assert s.untagged_exits == 0
    assert s._exit_params(_trade("model_entry ")) == _base_params()
    assert s._exit_params(_trade(None)) == _base_params()
    assert s.untagged_exits == 2


def test_forbidden_keys_are_refused_at_class_creation():
    for bad in ({"stoploss_grace_hours": 1.0}, {"stoploss_grace_level": -0.1}, {"roi": 0.05}):
        with pytest.raises(FamilyExitError, match="shared"):
            _strategy({"momentum": bad})


def test_non_exitparams_key_and_unknown_family_are_refused():
    with pytest.raises(FamilyExitError, match="not an ExitParams field"):
        _strategy({"momentum": {"take_profitt": 1.0}})
    with pytest.raises(FamilyExitError, match="unknown family"):
        _strategy({"sideways": {}})


def test_exitparams_invariants_hold_after_replacement():
    with pytest.raises(FamilyExitError, match="invalid"):
        _strategy({"momentum": {"atr_stoploss_floor": -0.01, "atr_stoploss_cap": -0.05}})


def test_f3_key_in_both_a_family_set_and_sell_overrides_is_refused():
    with pytest.raises(FamilyExitError, match="ambiguous"):
        _strategy({"momentum": {"take_profit": 10.0}}, sell={"cexit_take_profit": 0.03})
    with pytest.raises(FamilyExitError, match="ambiguous"):
        _strategy({"momentum": {"atr_stoploss_floor": -0.1}}, attrs={"atr_stoploss_floor": -0.08})


def test_f5_hold_window_sanity(recwarn):
    with pytest.raises(FamilyExitError, match="max hold"):
        _strategy({"momentum": {"unclog_12h_hours": 24.0 * 11}})  # max_hold 10 days
    _strategy({"momentum": {"unclog_12h_hours": 30.0}})
    assert any("unclog_1d" in str(w.message) for w in recwarn.list)
    recwarn.clear()
    _strategy({"momentum": MOMENTUM})  # 24/24 is fine
    assert not [w for w in recwarn.list if "unclog_1d" in str(w.message)]


def test_empty_family_exits_is_a_no_op():
    s = _strategy({})()
    assert s._exit_params(_trade("momentum:breakout_gbb")) == _base_params()


def test_class_level_params_overlay_sell_params_values():
    # Parameter class-level .value is the declared default; sell_params is what freqtrade loads.
    from utils.FamilyExits import class_level_exit_params

    cls = type(
        "CL",
        (FamilyExitMixin, _Base),
        {"FAMILY_EXITS": {}, "sell_params": {"cexit_take_profit": 0.031, "cexit_max_days": 4}},
    )
    p = class_level_exit_params(cls)
    assert p.take_profit == 0.031 and p.max_hold_days == 4
    assert class_level_exit_params(cls, {"cexit_take_profit": 0.05}).take_profit == 0.05
