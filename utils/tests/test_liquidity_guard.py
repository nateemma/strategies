"""LiquidityGuardMixin: off = parent mask; all-candle vs non-zero median baselines; tags cleared."""

import numpy as np
import pandas as pd
import pytest

from utils.LiquidityGuardMixin import LiquidityGuardMixin


class _Parent:
    startup_candle_count = 4

    def get_entry_conditions(self, dataframe):
        self._st_tags = np.array(["t"] * len(dataframe), dtype=object)
        return np.ones(len(dataframe), dtype=bool)


def _strat(guard):
    return type("S", (LiquidityGuardMixin, _Parent), {"LIQUIDITY_GUARD": guard})()


# a thin pair: most candles never trade
DF = pd.DataFrame({"volume": [0, 0, 0, 5, 0, 0, 2, 0, 3, 0], "close": [1.0] * 10})


def test_off_is_the_parent_mask():
    assert _strat(None).get_entry_conditions(DF).all()


def test_all_candle_median_is_zero_on_a_thin_pair():
    keep = _strat({"window": 4, "min_ratio": 1.0}).get_entry_conditions(DF)
    # first 3 rows lack a full window; every traded candle beats a zero median
    assert keep.tolist() == [False] * 3 + [True, False, False, True, False, True, False]


def test_nonzero_median_is_the_typical_traded_candle():
    s = _strat({"window": 4, "min_ratio": 1.0, "nonzero": True})
    keep = s.get_entry_conditions(DF)
    # row 3: only traded candle in window is itself (5 > 5 false); row 6: window {5, 2} median 3.5;
    # row 8: window {2, 3} median 2.5 -> 3 passes; untraded windows (row 5: {5}) refuse qv 0
    assert keep.tolist() == [False, False, False, False, False, False, False, False, True, False]
    assert (s._st_tags[~keep] == "").all() and s._st_tags[8] == "t"


def test_window_longer_than_startup_is_refused():
    with pytest.raises(ValueError, match="startup_candle_count"):
        _strat({"window": 8, "min_ratio": 1.0, "nonzero": True}).get_entry_conditions(DF)
