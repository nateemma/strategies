"""039 T009 (research R4): the plain-LSTM input sets -- causal, the right columns per set, finite, and
four distinct model keys (Ridge + three LSTM input sets)."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

STRAT = Path(__file__).resolve().parents[2]
for p in (STRAT, STRAT / "NNPredict"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import _inputs as IN  # noqa: E402


def _ohlcv(n=3500, seed=0):
    rng = np.random.default_rng(seed)
    close = 100 * np.exp(np.cumsum(rng.normal(0, 0.004, n)))
    high = close * (1 + rng.uniform(0, 0.003, n))
    low = close * (1 - rng.uniform(0, 0.003, n))
    open_ = np.r_[close[0], close[:-1]]
    vol = rng.uniform(1, 100, n) * (rng.uniform(0, 1, n) > 0.2)
    return pd.DataFrame({"open": open_, "high": high, "low": low, "close": close, "volume": vol})


class _Parent:
    include_list = ["adx_scaled", "bb_width", "current_gain", "recent_gain"]
    pre_normalized_columns = ["adx_scaled", "current_gain", "recent_gain"]

    def add_additional_indicators(self, df):
        return df


def _make(input_set):
    return type(f"S_{input_set}", (IN.InputSetMixin, _Parent), {"INPUT_SET": input_set})()


class _Scaling(_Parent):
    def scale_dataframe(self, df):  # stands in for the framework: every include_list column, normalised
        return df[self.include_list]


@pytest.mark.parametrize("input_set,expected", [
    ("raw", IN.R8 + IN.GAINS),
    ("raw_ret", IN.R8 + IN.LR + IN.GAINS),
    ("raw_ind", _Parent.include_list + IN.R8),
])
def test_model_sees_exactly_its_input_set(input_set, expected):
    s = type(f"T_{input_set}", (IN.InputSetMixin, _Scaling), {"INPUT_SET": input_set})()
    df = s.add_additional_indicators(_ohlcv())
    for c in _Parent.include_list:  # parent columns stay normalised for the framework's guards
        df[c] = 0.0
        assert c in s.include_list
    assert list(s.scale_dataframe(df).columns) == expected
    assert s.get_normalized_size(df) == len(expected)
    assert set(IN.R8) <= set(s.pre_normalized_columns)
    assert _Parent.include_list == ["adx_scaled", "bb_width", "current_gain", "recent_gain"]  # not mutated


def test_new_columns_are_causal_and_finite():
    df = _ohlcv()
    full = _make("raw_ret").add_additional_indicators(df.copy())
    cut = 2000
    part = _make("raw_ret").add_additional_indicators(df.iloc[: cut + 1].copy())
    for c in IN.R8 + IN.LR:
        np.testing.assert_allclose(full[c].to_numpy()[: cut + 1], part[c].to_numpy(), atol=1e-6, err_msg=c)
        v = full[c].to_numpy()
        assert np.isfinite(v).all() and np.abs(v).max() <= 1.0 + 1e-6


def test_unknown_input_set_is_refused():
    with pytest.raises(ValueError):
        type("Bad", (IN.InputSetMixin, _Parent), {"INPUT_SET": "nope"})


def test_four_distinct_model_keys(tmp_path, monkeypatch):
    from Framework import ModelKey as MK
    from Framework.BaseNNStrategy import BaseNNStrategy
    from NNPredict_LSTM_Raw import NNPredict_LSTM_Raw
    from NNPredict_LSTM_RawInd import NNPredict_LSTM_RawInd
    from NNPredict_LSTM_RawRet import NNPredict_LSTM_RawRet
    from NNPredict_Ridge import NNPredict_Ridge

    (tmp_path / "main_scaler.pkl").write_bytes(b"s")
    monkeypatch.setattr(BaseNNStrategy, "get_storage_location", lambda self: str(tmp_path) + "/")
    cfg = {"exchange": {"pair_whitelist": ["SOL/USDT"]}, "timeframe": "15m"}
    keys = {MK.key_hash(MK.resolve(c, cfg)) for c in
            (NNPredict_Ridge, NNPredict_LSTM_Raw, NNPredict_LSTM_RawInd, NNPredict_LSTM_RawRet)}
    assert len(keys) == 4
