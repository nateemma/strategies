"""composite.py: lags and the k-bar look-back stay inside one pair and never include bar t."""
from __future__ import annotations

import numpy as np

from research.indicator_screen import composite as C


def test_shift_and_recent_max_do_not_cross_pairs():
    x = np.arange(6.0)
    assert np.array_equal(C._shift(x, [3, 3]), [np.nan, 0, 1, np.nan, 3, 4], equal_nan=True)
    ext = np.array([1.0, 0, 0, 0, 1, 0])
    got = C._recent_max(ext, [3, 3], 2)
    assert np.array_equal(got, [np.nan, 1, 1, np.nan, 0, 1], equal_nan=True)   # bar t excluded
