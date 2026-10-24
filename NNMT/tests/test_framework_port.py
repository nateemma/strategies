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


def test_framework_is_on_the_family_base_with_only_the_adopted_switches():
    """040 adoptions (2026-10-24): 24 h grace and the in-strategy tr_scale (entries tagged reversion so the
    scaler reads the mask); every other framework switch stays off. The 040 records pin both off."""
    from NNMT.NNMT_LR_RI_PreG24 import NNMT_LR_RI_PreG24
    from NNMT.NNMT_LR_RI_PreTR import NNMT_LR_RI_PreTR

    assert issubclass(NNMTStrategy, EntryFrameworkMixin)
    assert NNMTStrategy.ENTRY_MODE == "model"
    assert NNMTStrategy.MODEL_ENTRY_FAMILY == "reversion"
    assert (NNMTStrategy.TR_UP_CELL, NNMTStrategy.TR_DOWN_CELL, NNMTStrategy.TR_SOURCE) == \
        ("up_pair_96", "down_pair_672", "standin")
    for pin in (NNMT_LR_RI_PreG24, NNMT_LR_RI_PreTR):
        assert pin.MODEL_ENTRY_FAMILY is None and pin.TR_UP_CELL is None and pin.TR_DOWN_CELL is None
    assert NNMTStrategy.HELD_PAIR_ADD_RESOLVED is None
    assert NNMTStrategy.FAMILY_EXITS == {}
    assert NNMTStrategy.REVERSION_REGIME_GATE is None
    assert not NNMTStrategy._trigger_names()


def test_model_tags_are_basestrategys():
    import numpy as np

    s = NNMTStrategy.__new__(NNMTStrategy)
    assert s._model_tags(np.array([True, False]), 2).tolist() == ["reversion:model", ""]
    from NNMT.NNMT_LR_RI_PreTR import NNMT_LR_RI_PreTR

    r = NNMT_LR_RI_PreTR.__new__(NNMT_LR_RI_PreTR)
    assert r._model_tags(np.array([True, False]), 2) is None  # the pinned records keep BaseStrategy's tag


def test_the_port_does_not_move_the_model_key():
    pinned = type("NNMT_MLX_f", (NNMT_MLX,), {"MODEL_TRAIN_TIMERANGE": "20210101-20230101"})
    assert MK.key_hash(MK.resolve(pinned, CONFIG)) == PRE_PORT_KEY


def test_g24_is_the_default_and_the_040_records_pin_what_they_ran():
    """040 adoption (2026-10-24): the family default stop-loss grace is 24 h; the 040 record classes'
    _PreG24 twins pin the 2.6 h they ran, and their plans point at them."""
    import json

    from NNMT.NNMT_LR_RI import NNMT_LR_RI
    from NNMT.NNMT_LR_RI_PreG24 import NNMT_LR_RI_PreG24
    from NNMT.NNMT_MLX_PreG24 import NNMT_MLX_PreG24

    assert NNMTStrategy.sell_params["stoploss_grace_hours"] == 24.0
    assert NNMT_MLX.sell_params["stoploss_grace_hours"] == 24.0
    assert NNMT_LR_RI.sell_params["stoploss_grace_hours"] == 24.0
    assert NNMT_MLX_PreG24.sell_params["stoploss_grace_hours"] == 2.6
    assert NNMT_LR_RI_PreG24.sell_params["stoploss_grace_hours"] == 2.6
    assert {k: v for k, v in NNMT_LR_RI_PreG24.sell_params.items() if k != "stoploss_grace_hours"} == \
        {k: v for k, v in NNMT_LR_RI.sell_params.items() if k != "stoploss_grace_hours"}
    folds = STRAT / "evaluation" / "folds"
    for plan, base in (("plan_mt.json", "NNMT_MLX_PreG24"), ("plan_mt_lr_ri.json", "NNMT_LR_RI_PreG24"),
                       ("plan_mt_op.json", "NNMT_MLX_OP_PreG24"), ("plan_mt_lr_ri_rg.json", "NNMT_LR_RI_RG_PreG24")):
        assert json.loads((folds / plan).read_text())["base_strategy"] == base
