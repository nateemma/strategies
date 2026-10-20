"""039 T016 (research R6): NNPredict carries the NNNC entry framework, and the port does not move the
model key (so the framework arms reuse the trained weights). Identity of the trades is proven on the 15
folds (judge framework), not here. Since the 039 adoption (2026-10-21) the base declares the gbb gate
at the operating point; the 039 record classes pin what they ran."""

from __future__ import annotations

import sys
from pathlib import Path

STRAT = Path(__file__).resolve().parents[2]
for p in (STRAT, STRAT / "NNPredict"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from Framework import ModelKey as MK  # noqa: E402
from utils.EntryFramework import EntryFrameworkMixin  # noqa: E402
from NNPredict.NNPredictStrategy import NNPredictStrategy  # noqa: E402


def test_framework_is_on_the_family_base_with_only_the_adopted_gate():
    assert issubclass(NNPredictStrategy, EntryFrameworkMixin)
    assert NNPredictStrategy.ENTRY_MODE == "trigger_and_model"
    assert NNPredictStrategy._trigger_names() == ("gbb",)
    assert NNPredictStrategy.magnitude_floor_mode == "percentile"
    assert NNPredictStrategy.MODEL_ENTRY_FAMILY is None
    assert NNPredictStrategy.HELD_PAIR_ADD_RESOLVED is None
    assert NNPredictStrategy.FAMILY_EXITS == {}
    assert NNPredictStrategy.REVERSION_REGIME_GATE is None


def test_039_record_classes_pin_what_they_ran():
    from NNPredict.NNPredict_LSTM_RawInd_OP import NNPredict_LSTM_RawInd_OP
    from NNPredict.NNPredict_LSTM_RawInd_OP_GATE import NNPredict_LSTM_RawInd_OP_GATE
    from NNPredict.NNPredict_Ridge_Pre039 import NNPredict_Ridge_Pre039

    assert NNPredict_LSTM_RawInd_OP.ENTRY_MODE == "model" and not NNPredict_LSTM_RawInd_OP._trigger_names()
    assert NNPredict_LSTM_RawInd_OP_GATE.ENTRY_MODE == "trigger_and_model"
    assert NNPredict_Ridge_Pre039.ENTRY_MODE == "model"
    assert NNPredict_Ridge_Pre039.magnitude_floor_mode == "absolute"
    assert NNPredict_Ridge_Pre039.buy_params["entry_guard_threshold"] == -0.2


def test_model_tags_are_basestrategy_s():
    import numpy as np

    s = NNPredictStrategy.__new__(NNPredictStrategy)
    assert s._model_tags(np.array([True, False]), 2) is None  # None -> BaseStrategy's own entry tag


def test_framework_names_do_not_shape_the_weights():
    ignored = set(MK.merged(NNPredictStrategy, "MODEL_KEY_IGNORED"))
    assert {"ENTRY_MODE", "MODEL_ENTRY_FAMILY", "HELD_PAIR_ADD", "FAMILY_EXIT_OVERRIDES", "TRIGGER_NAME"} <= ignored
