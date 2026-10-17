"""ARO volume-filter retrain: the class declares the filter as data and in its model key."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

STRAT = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(STRAT), str(STRAT / "NNNC")]

from Framework.RowFilter import row_filter_mask  # noqa: E402


def test_filter_is_volume_gt_zero_and_keyed():
    import NNNC_CS_AROV as M

    c = M.NNNC_CS_AROV
    assert c.TRAIN_ROW_FILTER == (("volume", "gt", 0.0),)
    assert "TRAIN_ROW_FILTER" in c.MODEL_KEY_ATTRIBUTES
    m = row_filter_mask(pd.DataFrame({"volume": [0.0, 1.0, np.nan]}), c.TRAIN_ROW_FILTER)
    assert m.tolist() == [False, True, False]
