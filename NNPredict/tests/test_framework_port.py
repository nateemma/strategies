"""039 T016 (research R6): NNPredict carries the NNNC entry framework with every feature off, and the
port does not move the model key (so the framework arms reuse the trained weights). Identity of the
trades is proven on the 15 folds (judge framework), not here."""

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


def test_framework_is_on_the_family_base_and_everything_is_off():
    assert issubclass(NNPredictStrategy, EntryFrameworkMixin)
    assert NNPredictStrategy.ENTRY_MODE == "model"
    assert NNPredictStrategy.MODEL_ENTRY_FAMILY is None
    assert NNPredictStrategy.HELD_PAIR_ADD_RESOLVED is None
    assert NNPredictStrategy.FAMILY_EXITS == {}
    assert NNPredictStrategy.REVERSION_REGIME_GATE is None
    assert not NNPredictStrategy._trigger_names()


def test_model_tags_are_basestrategy_s():
    import numpy as np

    s = NNPredictStrategy.__new__(NNPredictStrategy)
    assert s._model_tags(np.array([True, False]), 2) is None  # None -> BaseStrategy's own entry tag


def test_framework_names_do_not_shape_the_weights():
    ignored = set(MK.merged(NNPredictStrategy, "MODEL_KEY_IGNORED"))
    assert {"ENTRY_MODE", "MODEL_ENTRY_FAMILY", "HELD_PAIR_ADD", "FAMILY_EXIT_OVERRIDES", "TRIGGER_NAME"} <= ignored
