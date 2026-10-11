"""024 T014: entry rows carry `<family>:<trigger>`, and exits follow the STORED tag (FR-004)."""

from __future__ import annotations

import types

import numpy as np
import pandas as pd
import pytest
from evaluation.fingerprint import load_strategy_class
from Framework.BaseStrategy import BaseStrategy


def _frame():
    # gbb fires where guard_metric < 0.3 and bb_width > 0.035
    return pd.DataFrame(
        {"guard_metric": [0.1, 0.5, 0.2, 0.9], "bb_width": [0.05, 0.05, 0.01, 0.05]}
    )


def _bare(name):
    cls = load_strategy_class(name)
    return cls, object.__new__(cls)  # never instantiate a strategy in a unit test


def test_signal_rows_are_tagged_with_family_and_trigger(monkeypatch):
    cls, s = _bare("SimpleTrigger_FamGbb")

    def fake_base(self, df, metadata):  # BaseStrategy's tagging, without its guard chain
        mask = self.get_entry_conditions(df)
        df["enter_tag"] = ""
        df.loc[mask, "enter_tag"] = "model_entry "
        return df

    monkeypatch.setattr(BaseStrategy, "populate_entry_trend", fake_base)
    out = cls.populate_entry_trend(s, _frame(), {"pair": "SOL/USDT"})
    assert out["enter_tag"].tolist() == ["reversion:gbb", "", "", ""]


def _trade(tag):
    return types.SimpleNamespace(enter_tag=tag, id=7)


def test_shared_preset_ignores_the_family_tag():
    cls, s = _bare("SimpleTrigger_Gbb")  # ARM_D_OPEN: every family resolves to shared exits
    assert s._exit_params(_trade("momentum:breakout_gbb")) == s._exit_params()


def test_exits_follow_the_stored_tag_not_the_current_bar():
    # FR-004 / F8: FamGbb's own trigger is gbb (reversion), and its latest frame fires only gbb.
    # A trade tagged momentum at entry still gets the momentum set.
    cls, s = _bare("SimpleTrigger_FamGbb")
    s._st_tags = np.array(["reversion:gbb"])  # what the current bar would say
    p_mom = s._exit_params(_trade("momentum:breakout_gbb"))
    p_rev = s._exit_params(_trade("reversion:gbb"))
    assert p_mom.take_profit == 10.0 and p_mom.unclog_12h_hours == 24.0
    assert p_rev == s._exit_params()
    assert p_rev.take_profit != 10.0


def test_family_exits_resolved_from_the_preset():
    for name, momentum in (("SimpleTrigger_FamBgo", {"take_profit": 10.0, "unclog_12h_hours": 24.0}),
                           ("SimpleTrigger_Gbb", {})):
        cls = load_strategy_class(name)
        assert cls.FAMILY_EXITS["momentum"] == momentum, name
        assert cls.FAMILY_EXITS["reversion"] == {} and cls.FAMILY_EXITS["unassigned"] == {}


def test_declaring_family_exits_directly_is_refused():
    from SimpleTrigger.presets import PresetError
    from SimpleTrigger.SimpleTriggerStrategy import SimpleTriggerStrategy

    with pytest.raises(PresetError, match="FAMILY_EXIT_OVERRIDES"):
        type("Bad", (SimpleTriggerStrategy,), {"TRIGGER_NAME": "gbb", "FAMILY_EXITS": {}})
