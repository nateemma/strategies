"""026 addendum 2: MOMENTUM_PREEMPT -- a fillable momentum entry signal exits the reversion trade
holding that pair; momentum re-enters on its own (persisting) signal. The null check -- reversion
disabled reproduces the momentum-only arm exactly -- is a walk-forward record, not a unit test."""

from __future__ import annotations

from datetime import UTC, datetime
from types import SimpleNamespace

import pandas as pd
import pytest
from SimpleTrigger.presets import PresetError
from SimpleTrigger.SimpleTriggerStrategy import SimpleTriggerStrategy


def _arm(**attrs):
    attrs.setdefault("PRESET", "ARM_D_FAMILY")
    attrs.setdefault("TRIGGERS", ("gbb", "breakout_gbb"))
    return object.__new__(type("Pre", (SimpleTriggerStrategy,), attrs))


def test_preempt_and_recency_are_exclusive():
    with pytest.raises(PresetError):
        type("Both", (SimpleTriggerStrategy,), {"PRESET": "ARM_D_FAMILY",
             "TRIGGERS": ("gbb", "breakout_gbb"), "MOMENTUM_PREEMPT": True,
             "MOMENTUM_RECENCY_BARS": 2})


def test_preempt_needs_a_momentum_trigger():
    with pytest.raises(PresetError):
        type("NoMom", (SimpleTriggerStrategy,), {"PRESET": "ARM_D_FAMILY",
             "TRIGGERS": ("gbb", "breakout_gbb"), "MOMENTUM_PREEMPT": True,
             "DISABLED_TRIGGERS": ("breakout_gbb",)})


def _dp(last):
    frame = pd.DataFrame([last])
    return SimpleNamespace(get_analyzed_dataframe=lambda pair, timeframe: (frame, None),
                           runmode=SimpleNamespace(value="other"))


def test_preempt_fires_only_on_a_fillable_momentum_entry(monkeypatch):
    s = _arm(MOMENTUM_PREEMPT=True)
    monkeypatch.setattr(SimpleTriggerStrategy.__mro__[1], "custom_exit",
                        lambda *a, **k: None, raising=False)
    rev = SimpleNamespace(enter_tag="reversion:gbb", is_short=False)
    now = datetime(2023, 1, 1, 12, 15, tzinfo=UTC)
    mom = {"enter_long": 1, "enter_tag": "momentum:breakout_gbb", "volume": 100.0, "close": 50.0}
    s.dp = _dp(mom)
    assert s.custom_exit("X/USDT", rev, now, 1.0, 0.0) == "momentum_preempt"
    s.dp = _dp({**mom, "volume": 1.0})  # 50 USDT < MIN_QUOTE_VOLUME
    assert s.custom_exit("X/USDT", rev, now, 1.0, 0.0) != "momentum_preempt"
    s.dp = _dp({**mom, "enter_long": 0})  # tag without a final entry (guards refused it)
    assert s.custom_exit("X/USDT", rev, now, 1.0, 0.0) != "momentum_preempt"
    mo = SimpleNamespace(enter_tag="momentum:breakout_gbb", is_short=False)
    s.dp = _dp(mom)
    assert s.custom_exit("X/USDT", mo, now, 1.0, 0.0) != "momentum_preempt"  # never its own
