"""026 T002: market-wide regime instruments are causal, bounded and pair-agnostic (contract C1, R5).

Causality is checked the 023 way: compute on the full data, then on data truncated at each cut;
every value at or before the cut must be identical. A planted centred rolling mean (which reads the
future) must FAIL the same check, or the check is vacuous.
"""

from __future__ import annotations

import inspect

import numpy as np
import pandas as pd
import pytest
from utils import RegimeState as RS


PAIRS = ["SOL", "LTC", "AAVE", "XRP", "ZEC", "BCH", "DOT"]
CENTRE = pd.Timestamp("2023-10-01", tz="UTC")
# +-120 days so every instrument is defined at every cut (2880-bar windows + 768-bar EMAs; L1)
LO, HI = CENTRE - pd.Timedelta(days=120), CENTRE + pd.Timedelta(days=120)


@pytest.fixture(scope="module")
def data():
    from research.clfstudy.data import load_pair

    frames = {}
    for p in PAIRS:
        d = load_pair(f"{p}/USDT")
        frames[p] = d[(d.date >= LO) & (d.date <= HI)].reset_index(drop=True)
    b = load_pair("BTC/USDT")
    return frames, b[(b.date >= LO) & (b.date <= HI)].reset_index(drop=True)


def _cuts(series: pd.Series, n=16):
    valid = series.dropna().index
    lo, hi = valid[len(valid) // 4], valid[-1]
    return pd.date_range(lo, hi, periods=n)


def _truncate(frames, btc, cut):
    return {p: f[f.date <= cut] for p, f in frames.items()}, btc[btc.date <= cut]


def _causal(fn, frames, btc) -> bool:
    full = fn(frames, btc)
    for cut in _cuts(full):
        part = fn(*_truncate(frames, btc, cut))
        a, b = full[full.index <= cut], part[part.index <= cut]
        if not a.index.equals(b.index):
            return False
        if not np.allclose(a.to_numpy(), b.to_numpy(), equal_nan=True, rtol=0, atol=1e-12):
            return False
    return True


@pytest.mark.parametrize("inst", ["breadth", "volreg", "persist", "btctrend"])
def test_instrument_is_causal_under_truncation(data, inst):
    frames, btc = data
    assert _causal(lambda f, b: RS.compute(inst, f, b), frames, btc)


def test_planted_centred_mean_fails_the_check(data):
    frames, btc = data

    def leaky(f, b):
        return b.set_index("date")["close"].rolling(97, center=True).mean()

    assert not _causal(leaky, frames, btc)


def test_breadth_bounds_and_min_pairs(data):
    frames, btc = data
    s = RS.compute("breadth", frames, btc)
    v = s.dropna()
    assert len(v) > 0 and v.between(0, 1).all()
    few = {p: frames[p] for p in PAIRS[:5]}  # 5 < min_pairs 6 -> all NaN
    assert RS.compute("breadth", few, btc).isna().all()


def _synthetic(kind: str, n=12000, seed=0):
    rng = np.random.default_rng(seed)
    if kind == "walk":
        r = rng.normal(0, 0.001, n)
    else:  # trending: positively autocorrelated returns
        e = rng.normal(0, 0.001, n)
        r = np.empty(n)
        r[0] = e[0]
        for i in range(1, n):
            r[i] = 0.5 * r[i - 1] + e[i]
    close = 100 * np.exp(np.cumsum(r))
    dates = pd.date_range("2023-01-01", periods=n, freq="15min", tz="UTC")
    return pd.DataFrame({"date": dates, "close": close})


def test_persist_is_near_one_on_a_random_walk_and_above_one_when_trending():
    walk = {f"P{i}": _synthetic("walk", seed=i) for i in range(6)}
    trend = {f"P{i}": _synthetic("trend", seed=i) for i in range(6)}
    btc = _synthetic("walk", seed=99)
    assert abs(RS.compute("persist", walk, btc).dropna().median() - 1) < 0.1
    assert RS.compute("persist", trend, btc).dropna().median() > 1.3


def test_outputs_are_utc_15m_date_indexed(data):
    frames, btc = data
    for inst in RS.INSTRUMENTS:
        s = RS.compute(inst, frames, btc)
        assert isinstance(s.index, pd.DatetimeIndex) and str(s.index.tz) == "UTC"
        assert (s.index.to_series().diff().dropna() >= pd.Timedelta(minutes=15)).all()


def test_no_pair_names_in_the_logic():
    src = inspect.getsource(RS)
    for p in ("SOL", "LTC", "AAVE", "XRP", "ZEC", "BCH", "DOT", "LINK", "NEAR", "AVAX", "USDT"):
        assert f'"{p}' not in src and f"'{p}" not in src, p
