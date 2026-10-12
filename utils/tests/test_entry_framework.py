"""029 T010: the shared entry framework -- entry modes, tags, refusals, the MRO merge of family
exit overrides, and the regime gate on model entries. Stub strategies over BaseStrategy; the
model is a stub parent `get_entry_conditions`, triggers are patched, so no data or model loads."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from Framework.BaseStrategy import BaseStrategy  # noqa: E402
from Framework.TrainingSignals import LabelMethod  # noqa: E402
from utils.EntryFramework import (  # noqa: E402
    AUTO,
    EntryFrameworkError,
    EntryFrameworkMixin,
)


N = 8
MODEL = np.array([1, 1, 0, 1, 0, 0, 1, 0], dtype=bool)
TRIG = np.array([1, 0, 0, 1, 1, 0, 0, 0], dtype=bool)


class _ModelParent(BaseStrategy):
    """Stands in for BaseNNStrategy: the model's entry mask, as a Series."""

    def get_entry_conditions(self, dataframe):
        return dataframe["m"] > 0


def _df():
    return pd.DataFrame({"m": MODEL.astype(int),
                         "date": pd.date_range("2024-01-01", periods=N, freq="15min", tz="UTC")})


def _inst(cls):
    return object.__new__(cls)  # no freqtrade __init__: nothing loads


def _patch_triggers(monkeypatch, cls, families=("reversion",)):
    def fake(self, dataframe):
        return tuple(f"t{i}" for i in range(len(families))), [TRIG] * len(families), list(families)
    monkeypatch.setattr(cls, "_evaluate_triggers", fake)


# ------------------------------------------------------------------------------ model mode


class M(EntryFrameworkMixin, _ModelParent):
    ENTRY_MODE = "model"
    MODEL_ENTRY_FAMILY = "reversion"


class MNone(EntryFrameworkMixin, _ModelParent):
    ENTRY_MODE = "model"
    MODEL_ENTRY_FAMILY = None


def test_model_mode_returns_the_parent_mask_unchanged_and_tags_by_family():
    s, df = _inst(M), _df()
    mask = s.get_entry_conditions(df)
    pd.testing.assert_series_equal(mask, _ModelParent.get_entry_conditions(s, df))
    assert list(s._st_tags[MODEL]) == ["reversion:model"] * MODEL.sum()
    assert set(s._st_tags[~MODEL]) == {""}


def test_model_mode_without_a_family_leaves_tags_alone():
    s = _inst(MNone)
    s.get_entry_conditions(_df())
    assert s._st_tags is None


@pytest.mark.parametrize("training_type,family", [
    (LabelMethod.gbb_path, "reversion"), (LabelMethod.breakout_gbb, "momentum"),
    (LabelMethod.forward_mae, None),
])
def test_auto_family_comes_from_the_training_label(training_type, family):
    cls = type("A", (EntryFrameworkMixin, _ModelParent),
               {"ENTRY_MODE": "model", "MODEL_ENTRY_FAMILY": AUTO, "TRAINING_TYPE": training_type})
    assert cls._model_family() == family


def test_model_mode_has_no_family_exits_unless_declared():
    assert M.FAMILY_EXITS == {}


# -------------------------------------------------------------------- trigger_and_model mode


class TM(EntryFrameworkMixin, _ModelParent):
    ENTRY_MODE = "trigger_and_model"
    TRIGGER_NAME = "gbb"
    MODEL_ENTRY_FAMILY = "reversion"


def test_trigger_and_model_requires_both(monkeypatch):
    _patch_triggers(monkeypatch, TM)
    s = _inst(TM)
    mask = s.get_entry_conditions(_df())
    assert list(mask) == list(MODEL & TRIG)
    assert s._st_trigger_gate_refused == int((MODEL & ~TRIG).sum())
    assert list(s._st_tags[MODEL & TRIG]) == ["reversion:model"] * int((MODEL & TRIG).sum())


# ----------------------------------------------------------------------------- triggers mode


class T(EntryFrameworkMixin, BaseStrategy):
    ENTRY_MODE = "triggers"
    TRIGGERS = ("gbb", "breakout_gbb")


def test_triggers_mode_ors_triggers_and_tags_by_priority(monkeypatch):
    _patch_triggers(monkeypatch, T, families=("reversion", "momentum"))
    s = _inst(T)
    mask = s.get_entry_conditions(_df())
    assert list(mask) == list(TRIG)
    assert set(s._st_tags[TRIG]) == {"reversion:t0+ov"}  # both fired: reversion wins, +ov


def test_triggers_mode_refuses_without_a_trigger(capsys):
    cls = type("NoTrig", (EntryFrameworkMixin, BaseStrategy), {"ENTRY_MODE": "triggers"})
    assert not cls.get_entry_conditions(_inst(cls), _df()).any()
    assert "enters NO trades" in capsys.readouterr().out


# ---------------------------------------------------------------------------------- refusals


@pytest.mark.parametrize("attrs,match", [
    ({"ENTRY_MODE": "nonsense"}, "ENTRY_MODE"),
    ({"ENTRY_MODE": "model", "TRIGGER_NAME": "gbb"}, "ambiguous"),
    ({"ENTRY_MODE": "trigger_and_model"}, "needs a trigger"),
    ({"ENTRY_MODE": "model", "MOMENTUM_RECENCY_BARS": 4}, "ENTRY_MODE 'triggers'"),
    ({"ENTRY_MODE": "model", "MOMENTUM_PREEMPT": True}, "ENTRY_MODE 'triggers'"),
    ({"ENTRY_MODE": "model", "MODEL_ENTRY_FAMILY": "sideways"}, "MODEL_ENTRY_FAMILY"),
    ({"ENTRY_MODE": "model", "MODEL_ENTRY_FAMILY": None, "startup_candle_count": 4096,
      "REVERSION_REGIME_GATE": {"instrument": "volreg", "threshold": 1.0, "off_when": "lt"}},
     "INERT"),
    ({"ENTRY_MODE": "model", "MODEL_ENTRY_FAMILY": "reversion", "startup_candle_count": 64,
      "REVERSION_REGIME_GATE": {"instrument": "volreg", "threshold": 1.0, "off_when": "lt"}},
     "startup_candle_count"),
    ({"ENTRY_MODE": "model", "FAMILY_EXIT_OVERRIDES": {"sideways": {"take_profit": 0.03}}},
     "unknown family"),
])
def test_declarations_that_cannot_apply_are_refused_at_class_creation(attrs, match):
    with pytest.raises(EntryFrameworkError, match=match):
        type("Bad", (EntryFrameworkMixin, _ModelParent), dict(attrs))


# ------------------------------------------------------------------------- family-exit merge


def test_family_exit_overrides_merge_along_the_mro():
    class P(EntryFrameworkMixin, _ModelParent):
        ENTRY_MODE = "model"
        FAMILY_EXIT_OVERRIDES = {"reversion": {"take_profit": 0.03}}

    class C(P):
        FAMILY_EXIT_OVERRIDES = {"reversion": {"max_hold_days": 5},
                                 "momentum": {"take_profit": 10.0}}

    assert C.FAMILY_EXITS == {"reversion": {"take_profit": 0.03, "max_hold_days": 5},
                              "momentum": {"take_profit": 10.0}}
    assert P.FAMILY_EXITS == {"reversion": {"take_profit": 0.03}}


# ----------------------------------------------------------------- regime gate, model entries


class G(EntryFrameworkMixin, _ModelParent):
    ENTRY_MODE = "model"
    MODEL_ENTRY_FAMILY = "reversion"
    startup_candle_count = 4096
    REVERSION_REGIME_GATE = {"instrument": "volreg", "threshold": 1.0, "off_when": "lt"}


def test_regime_gate_drops_reversion_model_entries_in_the_off_state(monkeypatch):
    df = _df()
    state = pd.Series([2.0, 0.5, 2.0, 0.5, 2.0, 2.0, 0.5, 2.0], index=pd.DatetimeIndex(df["date"]))
    monkeypatch.setattr(G, "_regime_state", lambda self: state)
    s = _inst(G)
    mask = s.get_entry_conditions(df)
    off = state.to_numpy() < 1.0
    assert list(np.asarray(mask, bool)) == list(MODEL & ~off)
    assert s._st_regime_counts["regime_gate_refused"] == int((MODEL & off).sum())
