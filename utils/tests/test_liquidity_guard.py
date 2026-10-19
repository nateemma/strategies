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


class _Sparse(_Parent):
    """Signals on rows 1, 3, 4, 6, 8, 9; row 4 did not trade."""

    def get_entry_conditions(self, dataframe):
        m = np.zeros(len(dataframe), dtype=bool)
        m[[1, 3, 4, 6, 8, 9]] = True
        self._st_tags = np.where(m, "t", "").astype(object)
        return m


SR = pd.DataFrame({"volume": [9, 4, 9, 2, 0, 9, 6, 9, 3, 7], "close": [1.0] * 10, "date": pd.date_range("2024", periods=10, freq="15min")})


def _sr(guard):
    return type("SR", (LiquidityGuardMixin, _Sparse), {"LIQUIDITY_GUARD": guard})()


def test_signal_rank_uses_only_previous_traded_signals(monkeypatch, tmp_path):
    monkeypatch.setenv("FT_LIQ_GUARD_DUMP", str(tmp_path))
    s = _sr({"signals": 2, "quantile": 0.5})
    keep = s.get_entry_conditions(SR)
    # traded signal qv: r1 4, r3 2, r6 6, r8 3, r9 7. Thresholds = median of the previous two:
    # r1, r3 no history (kept); r4 {4,2}=3 vs 0 refused; r6 {4,2}=3 -> 6 kept; r8 {2,6}=4 -> 3 refused;
    # r9 {6,3}=4.5 -> 7 kept. Non-signal rows stay False.
    assert np.flatnonzero(keep).tolist() == [1, 3, 6, 9]
    assert (s._st_tags[[4, 8]] == "").all() and s._st_tags[9] == "t"
    d = pd.read_csv(next(tmp_path.glob("*.csv")))
    assert d["n"].tolist() == [0, 1, 2, 2, 2, 2] and d["keep"].tolist() == [True, True, False, True, False, True]


def test_signal_rank_observe_filters_nothing():
    keep = _sr({"signals": 2, "quantile": 0.5, "observe": True}).get_entry_conditions(SR)
    assert np.flatnonzero(keep).tolist() == [1, 3, 4, 6, 8, 9]


def test_signal_rank_with_no_traded_signal_keeps_everything():
    zero = SR.assign(volume=0)
    keep = _sr({"signals": 2, "quantile": 0.5}).get_entry_conditions(zero)
    assert np.flatnonzero(keep).tolist() == [1, 3, 4, 6, 8, 9]
