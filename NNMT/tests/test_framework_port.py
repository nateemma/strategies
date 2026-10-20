"""040 T016 (research R1): NNMT carries the NNNC entry framework with every feature off, and the port
does not move NNMT_MLX's model key (so mt_fw0 reuses mt's per-fold weights). Trade identity is proven
on the 15 folds (research.nnmt.judge identity), not here."""
from __future__ import annotations

import sys
from pathlib import Path

STRAT = Path(__file__).resolve().parents[2]
for p in (STRAT, STRAT / "NNMT"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from Framework import ModelKey as MK  # noqa: E402
from NNMT.NNMT_MLX import NNMT_MLX  # noqa: E402
from NNMT.NNMTStrategy import NNMTStrategy  # noqa: E402
from utils.EntryFramework import EntryFrameworkMixin  # noqa: E402

#: NNMT_MLX's model key (pinned f00 window, two-pair config) recorded BEFORE the port, after the 040
#: key hygiene (T014). The port must not change it.
PRE_PORT_KEY = "a975211fa4e1f219"
CONFIG = {"exchange": {"pair_whitelist": ["SOL/USDT", "LINK/USDT"]}, "timeframe": "15m"}


def test_framework_is_on_the_family_base_with_everything_off():
    assert issubclass(NNMTStrategy, EntryFrameworkMixin)
    assert NNMTStrategy.ENTRY_MODE == "model"
    assert NNMTStrategy.MODEL_ENTRY_FAMILY is None
    assert NNMTStrategy.HELD_PAIR_ADD_RESOLVED is None
    assert NNMTStrategy.FAMILY_EXITS == {}
    assert NNMTStrategy.REVERSION_REGIME_GATE is None
    assert not NNMTStrategy._trigger_names()


def test_model_tags_are_basestrategys():
    import numpy as np

    s = NNMTStrategy.__new__(NNMTStrategy)
    assert s._model_tags(np.array([True, False]), 2) is None


def test_the_port_does_not_move_the_model_key():
    pinned = type("NNMT_MLX_f", (NNMT_MLX,), {"MODEL_TRAIN_TIMERANGE": "20210101-20230101"})
    assert MK.key_hash(MK.resolve(pinned, CONFIG)) == PRE_PORT_KEY
