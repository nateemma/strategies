"""037 T012b: the truncation adapter flags a centred column and passes a trailing one."""
from __future__ import annotations

import numpy as np
import pandas as pd

from research.indicator_screen import causality as C


def _ohlcv(n=1200, seed=0):
    rng = np.random.default_rng(seed)
    close = 100 * np.exp(np.cumsum(rng.normal(0, 0.01, n)))
    return pd.DataFrame({"date": pd.date_range("2024-01-01", periods=n, freq="15min", tz="UTC"),
                         "open": close, "high": close * 1.01, "low": close * 0.99, "close": close,
                         "volume": rng.random(n) * 1000})


def _harvest(ohlcv):
    c = ohlcv["close"]
    return pd.DataFrame({"date": ohlcv["date"], "trailing": c.rolling(20).mean(),
                         "centred": c.rolling(21, center=True).mean(),
                         "raw_buy:X": (c > c.shift(1)).astype(np.int8)})


def test_centred_fails_trailing_passes():
    res, meta = C.check_columns(_harvest, [("synthetic", _ohlcv())], cuts=8, warmup=200)
    assert res["trailing"]["verdict"] == "PASS" and res["trailing"]["comparisons"] == 8
    assert res["raw_buy:X"]["verdict"] == "PASS"
    assert res["centred"]["verdict"] == "FAIL"
    assert res["centred"]["first_mismatch"]["bars_before_cut"] <= 10
    assert meta["checked"] == ["synthetic"]


def test_nan_equals_nan_and_tolerance():
    a = np.array([np.nan, 1.0, 2.0])
    assert not C._mismatch(a, a.copy()).any()
    assert C._mismatch(a, np.array([np.nan, 1.0, 2.0001])).tolist() == [False, False, True]
