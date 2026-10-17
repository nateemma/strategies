"""ATR-barrier study: label without floor or peak filter; arm B adds the 7 inputs to ARO's."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

STRAT = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(STRAT), str(STRAT / "NNNC")]


def test_label_has_no_floor_and_no_peak_filter():
    from _atr_barrier_label import AtrBarrierLabelMixin as L

    assert L.LABEL_PARAMS["min_gain"] is None and L.LABEL_PARAMS_SELL["min_loss"] is None
    assert L.LABEL_PARAMS["atr_scale"] == 4.0 and not L.LABEL_PEAK_FILTER
    s = L.__new__(L)
    assert np.array_equal(s._filtered(None, pd.Series([0.0, 1.0, 0.7]), "buy").to_numpy(), [0.0, 1.0, 1.0])


def test_arm_b_inputs_extend_aro():
    import NNNC_CS_ARO as A
    import NNNC_CS_TBB as B

    b = B.NNNC_CS_TBB
    assert list(b.include_list) == list(A.NNNC_CS_ARO.include_list) + list(b.TB_INPUT_COLUMNS)
    assert set(b.TB_INPUT_COLUMNS) <= set(b.pre_normalized_columns)
    assert b.TRAIN_ROW_FILTER == (("volume", "gt", 0.0),)


def test_mcc():
    from research.tb_model.screen import mcc

    assert np.isclose(mcc(np.array([True, False]), np.array([True, False])), 1.0)
    assert mcc(np.array([True, True]), np.array([True, False])) == 0.0
