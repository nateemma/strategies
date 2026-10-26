"""040 follow-up (preregistration_own_trade_heads.json): HeadVetoMixin removes exactly the entries whose score is below
the fold's threshold (NaN vetoes), clears their tags, switches off with VETO_THRESHOLDS None / FT_VETO_OFF, keys
thresholds by train window and seed, and leaves the model key unchanged."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

STRAT = Path(__file__).resolve().parents[2]
for p in (STRAT, STRAT / "NNMT"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from Framework import ModelKey as MK  # noqa: E402
from NNMT._head_veto import HeadVetoMixin  # noqa: E402

CONFIG = {"exchange": {"pair_whitelist": ["SOL/USDT"]}, "timeframe": "15m"}


class _Base:
    def get_entry_conditions(self, dataframe):
        self._st_tags = np.array(["reversion:model"] * len(dataframe), dtype=object)
        return pd.Series(True, index=dataframe.index)


class _Arm(HeadVetoMixin, _Base):
    VETO_COLUMN = "atr_pct_roll"
    MODEL_TRAIN_TIMERANGE = "20210101-20230101"
    MODEL_SEED = 1
    VETO_THRESHOLDS = {"20210101-20230101": 0.5, "20210101-20230101_s2": 0.25}


def _df():
    return pd.DataFrame({"atr_pct_roll": [0.1, 0.3, 0.5, 0.9, np.nan]})


def test_veto_removes_below_threshold_and_nan():
    s = _Arm()
    m = s.get_entry_conditions(_df())
    assert m.tolist() == [False, False, True, True, False]
    assert list(s._st_tags) == ["", "", "reversion:model", "reversion:model", ""]


def test_seed_keyed_threshold():
    s = _Arm()
    s.MODEL_SEED = 2
    assert s.get_entry_conditions(_df()).tolist() == [False, True, True, True, False]


def test_off_switches(monkeypatch):
    s = _Arm()
    monkeypatch.setenv("FT_VETO_OFF", "1")
    assert s.get_entry_conditions(_df()).all()
    monkeypatch.delenv("FT_VETO_OFF")
    s.VETO_THRESHOLDS = None
    d = _df()
    assert s.get_entry_conditions(d).all() and "veto_score" in d


def test_missing_fold_threshold_raises():
    s = _Arm()
    s.MODEL_TRAIN_TIMERANGE = "20210101-20240101"
    with pytest.raises(KeyError):
        s.get_entry_conditions(_df())


@pytest.mark.parametrize("name", ["NNMT_LR_RI_G24_HDP", "NNMT_LR_RI_G24_ATRV"])
def test_arm_reuses_nnmt_lr_ri_weights(name):
    import importlib

    from NNMT.NNMT_LR_RI import NNMT_LR_RI

    cls = getattr(importlib.import_module(f"NNMT.{name}"), name)
    assert MK.key_hash(MK.resolve(cls, CONFIG)) == MK.key_hash(MK.resolve(NNMT_LR_RI, CONFIG))
    assert cls.TR_UP_CELL is None and cls.TR_DOWN_CELL is None  # tr_scale off, as mt_lr_ri_g24's records
