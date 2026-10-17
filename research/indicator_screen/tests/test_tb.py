"""037 addendum (tb.py): the TB label, the width-book comparator, cmp: columns scored as signals."""
from __future__ import annotations

import numpy as np
import pandas as pd

from research.indicator_screen import tb
from research.indicator_screen import windows as W


def test_tb_label_is_gbb_free_and_last_horizon_is_nan():
    n = 200
    c = np.r_[np.full(100, 100.0), np.full(100, 103.0)]       # +3% jump at bar 100
    df = pd.DataFrame({"date": pd.date_range("2023-01-01", periods=n, freq="15min", tz="UTC"),
                       "open": c, "high": c, "low": c, "close": c, "volume": 1.0})
    y = tb.tb_label(df)
    assert np.isnan(y[-W.LABEL_HORIZON:]).all()
    assert y[99] == 1 and y[100 - W.LABEL_HORIZON - 1] == 0 and y[100 - W.LABEL_HORIZON] == 1


def test_width_book_uses_each_windows_cut():
    cuts = tb.width_cuts()
    f3 = W.by_id("f03")
    d = pd.date_range(f3.start, periods=4, freq="15min")
    frame = pd.DataFrame({"date": d, "base:guard_metric": [0.0, 0.0, 0.5, 0.0],
                          "base:bb_width": [cuts["f03"], cuts["f03"] - 1e-6, 1.0, 0.03]})
    assert np.array_equal(tb.width_book(frame, cuts), [1.0, 0.0, 0.0, 0.0])


def test_comparators_are_scored_as_signals():
    from research.indicator_screen import scores as S
    from research.indicator_screen.screen_single import _score_group

    recs = []
    x = np.r_[np.ones(50), np.zeros(150)]
    y = np.r_[np.ones(40), np.zeros(10), np.ones(20), np.zeros(130)]
    w = W.by_id("f03")
    _score_group(recs, "cmp:width_book", True, x, ["label:tb_buy"], [y], w, "all",
                 {"width_auc": {"label:tb_buy": 0.5}}, None, False, "tb", 20, 0.95)
    assert recs[0]["lift"] == S.lift(x, y, 20, 0.95)[0] > 1
