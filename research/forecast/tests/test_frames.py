"""038 frames (T004), plus stat-tier causality (T021) and learned-tier purge (T024)."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from research.forecast import common as C
from research.forecast import frames as F


def _candles(n=300, seed=0, start="2023-01-01"):
    rng = np.random.default_rng(seed)
    close = 100 * np.exp(np.cumsum(rng.normal(0, 0.01, n)))
    date = pd.date_range(start, periods=n, freq="15min", tz="UTC")
    open_ = np.r_[close[0], close[:-1]] * np.exp(rng.normal(0, 0.001, n))
    vol = rng.uniform(1, 10, n)
    return pd.DataFrame({"date": date, "open": open_, "close": close, "volume": vol})


WIN = {"f00": (pd.Timestamp("2023-01-01", tz="UTC"), pd.Timestamp("2023-01-02", tz="UTC")),
       "f08": (pd.Timestamp("2023-01-02", tz="UTC"), pd.Timestamp("2023-01-03", tz="UTC"))}
FWD = pd.Timestamp("2023-01-03", tz="UTC")


def test_targets_trade_returns_and_tail_nan():
    c = _candles()
    d = F.build(c, c["date"], None, WIN, FWD)
    lc = np.log(c.close.to_numpy())
    lo = np.log(c.open.to_numpy())
    for n in C.HORIZONS:
        t = d[f"target_{n}"].to_numpy()
        np.testing.assert_allclose(t[: len(c) - n], lc[n:] - lc[:-n])
        assert np.isnan(t[len(c) - n:]).all()
        tr = d[f"trade_ret_{n}"].to_numpy()
        m = len(c) - 1 - n
        np.testing.assert_allclose(tr[:m], lo[1 + n: 1 + n + m] - lo[1: 1 + m])
        assert np.isnan(tr[m:]).all()
        assert not d[f"eligible_{n}"].to_numpy()[m:].any()


def test_eligible_needs_traded_real_entry_and_exit():
    c = _candles()
    c.loc[11, "volume"] = 0.0  # entry candle of t=10 has no trades
    raw = c["date"].drop(index=[30])  # candle 30 was gap-filled
    d = F.build(c, raw, None, WIN, FWD)
    assert not d.loc[10, "eligible_1"]          # entry 11 has zero volume
    assert not d.loc[9, "eligible_1"]           # exit 11 (t+1+n = 11) has zero volume
    assert not d.loc[29, "eligible_4"]          # entry 30 gap-filled
    assert not d.loc[25, "eligible_4"]          # exit 30 gap-filled
    assert d.loc[40, "eligible_4"]
    assert not d.loc[30, "has_data"] and d.loc[31, "has_data"]


def test_fold_and_half_edges():
    c = _candles()
    d = F.build(c, c["date"], None, WIN, FWD)
    by_date = d.set_index("date")
    assert by_date.loc[pd.Timestamp("2023-01-01 00:00", tz="UTC"), "fold"] == "f00"
    assert by_date.loc[pd.Timestamp("2023-01-01 23:45", tz="UTC"), "half"] == "early"
    assert by_date.loc[pd.Timestamp("2023-01-02 00:00", tz="UTC"), "fold"] == "f08"
    assert by_date.loc[pd.Timestamp("2023-01-02 00:00", tz="UTC"), "half"] == "late"
    assert by_date.loc[pd.Timestamp("2023-01-03 00:00", tz="UTC"), "fold"] == "fwd"


def test_data_ok_from_cache_then_has_data_after_cache():
    c = _candles()
    ok = pd.Series(1, index=c["date"].iloc[:200])
    ok.iloc[:50] = 0
    d = F.build(c, c["date"], ok, WIN, FWD)
    assert not d.data_ok.iloc[:50].any() and d.data_ok.iloc[50:200].all()
    # after the cache: has_data, but only from the forward window on
    assert d.data_ok.iloc[200:].tolist() == (d.date.iloc[200:] >= FWD).tolist()


def test_plan_folds_are_contiguous():
    assert F.check_folds() == []


def test_stat_tier_is_causal_and_strided():
    from research.forecast import stat_tier as S

    c = _candles(n=900)
    d = F.build(c, c["date"], None, WIN, FWD)
    rows = S.scored_rows(d, stride=16)
    assert len(rows) == len(S.scored_rows(d, stride=16))
    assert all(r % 16 == 0 for r in rows)
    t = int(rows[len(rows) // 2])
    base = S.forecast_rows(d, [t], "holt")
    d2 = d.copy()
    d2.loc[t + 1:, "lc"] += 0.5  # change the future only
    cols = [f"f_{n}" for n in C.HORIZONS]
    assert np.isfinite(base[cols].to_numpy()).all()
    np.testing.assert_allclose(base[cols].to_numpy(), S.forecast_rows(d2, [t], "holt")[cols].to_numpy())
    d3 = d.copy()
    d3.loc[t - 10:t, "lc"] += 0.05  # change the past: the forecast must move (liveness)
    assert not np.allclose(base[cols].to_numpy(), S.forecast_rows(d3, [t], "holt")[cols].to_numpy())


@pytest.mark.parametrize("n", C.HORIZONS)
def test_learned_purge_and_null_permutation(n):
    from research.forecast import learned_tier as L

    dates = pd.Series(pd.date_range("2022-12-01", periods=4000, freq="15min", tz="UTC"))
    start = pd.Timestamp("2023-01-01", tz="UTC")
    m = L.train_mask(dates, start, n)
    last = dates[m].max()
    assert (start - last) >= pd.Timedelta(minutes=15 * n)
    y = np.arange(m.sum(), dtype=float)
    p = L.null_targets(y, seed=1)
    assert sorted(p) == sorted(y) and not np.array_equal(p, y)
