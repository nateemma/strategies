"""Composite sizer: normalisation rows stop at each pair's tail start; the class reads its own tables."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

STRAT = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(STRAT), str(STRAT / "NNNC")]

from research.cmp_sizer import signals as SG  # noqa: E402


def test_composite_is_signed_mean_of_z_on_norm_rows():
    X = np.array([[1.0, 10.0], [3.0, 30.0], [5.0, np.nan]])
    rows = np.array([True, True, False])
    c = SG.composite(X, np.array([1.0, -1.0]), rows)
    assert np.allclose(c, [(-1 + 1) / 2, (1 - 1) / 2, (3 + 0) / 2])


def test_class_points_at_cmp_tables():
    import NNNC_CS_CMP as M
    assert M.NNNC_CS_CMP.HGB_SIGNALS.name == "cmp_sizer"
