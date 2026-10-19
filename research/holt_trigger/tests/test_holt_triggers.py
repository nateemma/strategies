"""holt_vote_up / holt_vote_fade: direction, grid, pair invariance, causality by truncation, cache."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from research.outcomes import holt_triggers as H
from research.outcomes import triggers as T


@pytest.fixture(autouse=True)
def isolated_cache(tmp_path, monkeypatch):
    monkeypatch.setattr(H, "CACHE_DIR", tmp_path / "cache")
    monkeypatch.setattr(H, "_CACHE", None)
    H._NEW.clear()


def _frame(close, start="2024-01-01"):
    return pd.DataFrame({"date": pd.date_range(start, periods=len(close), freq="15min", tz="UTC"),
                         "open": close, "high": close, "low": close, "close": close, "volume": 1.0})


def _rise_then_fall(n=900, turn=550, slope=0.0008):
    t = np.arange(n)
    up = 100 * np.exp(slope * t[:turn])
    return np.r_[up, up[-1] * np.exp(-slope * (t[turn:] - turn + 1))]


def test_up_fires_in_the_rise_and_fade_in_the_fall():
    df = _frame(_rise_then_fall())
    up, fade = H._up(df).to_numpy(), H._fade(df).to_numpy()
    assert up[300:540].any() and not up[700:].any()
    assert fade[700:].any() and not fade[300:540].any()


def test_only_grid_candles_with_full_history_vote():
    df = _frame(_rise_then_fall())
    ups, downs = H.votes(df)
    k = ((pd.DatetimeIndex(df["date"]) - H.EPOCH) // H.BAR).to_numpy()
    voted = (ups + downs) > 0
    assert (k[voted] % H.STRIDE == 0).all()
    assert not voted[: max(H.WINDOWS) - 1].any()


def test_causal_under_truncation(monkeypatch):
    monkeypatch.setenv("HOLT_TRIGGER_NOCACHE", "1")  # a shared cache would hide a forward read
    df = _frame(_rise_then_fall() * np.exp(np.random.default_rng(1).normal(0, 0.002, 900)))
    full = H._up(df).to_numpy() | H._fade(df).to_numpy()
    cut = 640
    part = H._up(df.iloc[: cut + 1]).to_numpy() | H._fade(df.iloc[: cut + 1]).to_numpy()
    np.testing.assert_array_equal(full[: cut + 1], part)


def test_pair_invariant_through_the_registry():
    T.ensure_builtins()
    df = _frame(_rise_then_fall())
    a = T.evaluate("holt_vote_up", df.assign(pair="SOL/USDT"))
    b = T.evaluate("holt_vote_up", df.assign(pair="ZEC/USDT"))
    pd.testing.assert_series_equal(a, b)
    assert T.get("holt_vote_up").family == "momentum" and T.get("holt_vote_fade").family == "reversion"


def test_cache_hits_return_the_fitted_values(monkeypatch):
    df = _frame(_rise_then_fall())
    first = H.votes(df)
    calls = []
    monkeypatch.setattr(H, "_holt", lambda x, n: calls.append(1) or 0.0)
    monkeypatch.setattr(H, "_CACHE", None)  # reload from the shard written by the first call
    again = H.votes(df)
    assert not calls
    for a, b in zip(first, again):
        np.testing.assert_array_equal(a, b)


def test_nocache_refits_and_writes_nothing(monkeypatch):
    monkeypatch.setenv("HOLT_TRIGGER_NOCACHE", "1")
    calls = []
    real = H._holt
    monkeypatch.setattr(H, "_holt", lambda x, n: calls.append(1) or real(x, n))
    df = _frame(_rise_then_fall())
    H.votes(df)
    n = len(calls)
    H.votes(df)
    assert len(calls) == 2 * n > 0
    assert not H.CACHE_DIR.exists()
