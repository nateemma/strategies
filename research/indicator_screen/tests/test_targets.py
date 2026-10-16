"""037 T018 / T028: targets -- full-range labels equal the per-fold labels; regime targets are
forward-only, baselines backward-only, cut points from the declared span."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from research.indicator_screen import windows as W


def test_full_range_labels_equal_the_per_fold_labels(loaded):
    """R4: labels are a function of the price path, so the full-range labels must equal the ones
    the label screen builds on fold f03's training window (except its last 48 bars, which have no
    future inside that window)."""
    from Framework.BaseStrategy import TradingAction
    from research.indicator_screen import targets as T
    from research.label_screen import audit as A

    H = loaded["H"]
    checked = 0
    for pair, strat, d in A._frames(H.ARO_CLASS, "f03"):
        full = H.load_ohlcv(loaded["cfg"], pair)
        buy, sell = T.labels_for(loaded["aro"], full, pair)
        k = np.asarray(strat.get_training_labels(d.copy()))[-len(d):]
        ref = pd.DataFrame({"date": d["date"].to_numpy(),
                            "b": (k == int(TradingAction.BUY)).astype(float),
                            "s": (k == int(TradingAction.SELL)).astype(float)}).iloc[:-W.LABEL_HORIZON]
        got = pd.DataFrame({"date": full["date"].to_numpy(), "b": buy, "s": sell})
        m = ref.merge(got, on="date", suffixes=("_ref", "_got"))
        assert len(m) == len(ref), pair
        assert np.array_equal(m["b_ref"], m["b_got"]), pair
        assert np.array_equal(m["s_ref"], m["s_got"]), pair
        checked += 1
    assert checked == 10
