"""037 extras (post-lock additions): raw8 from its own code, ratio columns, 035 inputs joined by date."""
from __future__ import annotations

import numpy as np
import pandas as pd

from research.indicator_screen import extras as X


def _f(n=500):
    rng = np.random.default_rng(0)
    c = 100 * np.exp(np.cumsum(rng.normal(0, 0.01, n)))
    return pd.DataFrame({"date": pd.date_range("2024-01-01", periods=n, freq="15min", tz="UTC"),
                         "open": c, "high": c * 1.01, "low": c * 0.99, "close": c,
                         "volume": rng.random(n) * 100 + 1, "A:level": c * 1.02})


def test_raw8_is_the_clfstudy_function():
    from research.clfstudy.data import raw8

    f = _f()
    r = X.raw8_frame(f)
    assert list(r.columns) == [f"raw8:{c}" for c in ("ret1", "ret4", "log_hc", "log_lc", "log_oc", "range", "logvol_z", "vol_ratio")]
    assert np.allclose(r["raw8:ret1"], raw8(f[["date", "open", "high", "low", "close", "volume"]])["ret1"])


def test_ratio_is_value_over_close():
    f = _f()
    r = X.ratio_frame(f, ["A:level", "missing:col"])
    assert list(r.columns) == ["A:level/close"] and np.allclose(r["A:level/close"], 1.02)


def test_new_inputs_missing_file_gives_no_columns(tmp_path, monkeypatch):
    monkeypatch.setattr(X, "NEW_INPUTS", tmp_path)
    assert X.ni_frame("BTC/USDT", _f()["date"]).shape[1] == 0
    t = pd.DataFrame({"date": _f()["date"].iloc[::2], "ni_x": 1.0})
    t.to_parquet(tmp_path / "SOL_USDT.parquet")
    got = X.ni_frame("SOL/USDT", _f()["date"])
    assert list(got.columns) == ["ni:ni_x"] and got["ni:ni_x"].isna().sum() == 250
