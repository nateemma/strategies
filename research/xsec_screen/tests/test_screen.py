"""xsec screen: every signal at day t is computed from bars up to the end of day t only, and the IC is a
rank correlation across pairs."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from research.xsec_screen import screen as S


def _m(days=200, seed=0):
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2024-01-01", periods=days * 96, freq="15min", tz="UTC")
    c = 100 * np.exp(np.cumsum(rng.normal(0, 0.003, len(idx))))
    v = rng.uniform(0, 10, len(idx))
    v[::37] = 0.0
    return pd.DataFrame({"close": c, "volume": v}, index=idx)


@pytest.mark.parametrize("cut_day", [120, 150, 199])
def test_signals_are_causal(cut_day):
    m, b = _m(), _m(seed=1)
    full = S.signals(S.daily_features(m), np.log(S.daily_features(b)["close"]).diff())
    end = m.index[0].floor("D") + pd.Timedelta(days=cut_day + 1)  # bars strictly before the next day
    mt, bt = m[m.index < end], b[b.index < end]
    part = S.signals(S.daily_features(mt), np.log(S.daily_features(bt)["close"]).diff())
    pd.testing.assert_frame_equal(full.loc[part.index], part)


def test_ic_is_a_rank_correlation():
    days = pd.date_range("2024-01-01", periods=5, tz="UTC")
    sig = pd.DataFrame(np.tile(np.arange(8.0), (5, 1)), index=days)
    fwd = sig ** 3  # monotone -> rank IC exactly 1
    assert np.allclose(S.ic_series(sig, fwd, days).to_numpy(), 1.0)
    assert np.allclose(S.ic_series(sig, -fwd, days).to_numpy(), -1.0)
