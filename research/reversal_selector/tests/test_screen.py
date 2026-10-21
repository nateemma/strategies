"""Reversal selector: rank scaling, causality of the trade rank, and the permutation p."""
import numpy as np
import pandas as pd

from research.reversal_selector.screen import perm_p, rank_pct, trade_ranks


def test_rank_pct_ends_and_ties():
    assert rank_pct(-0.1, [0.0, 0.1]) == 0.0
    assert rank_pct(0.2, [0.0, 0.1]) == 1.0
    assert rank_pct(0.0, [0.0, 0.1]) == 0.25  # tie averaged: rank 1.5 of 3


class _B:
    def __init__(self, c):
        self.close = c


def _bars(seed, n=400):
    idx = pd.date_range("2024-01-01", periods=n, freq="15min", tz="UTC")
    c = np.exp(np.cumsum(np.random.default_rng(seed).normal(0, 0.01, n)))
    return _B(c), idx


def test_rank_uses_only_bars_up_to_the_signal_bar():
    pairs = [f"P{i}" for i in range(6)]
    data = {p: _bars(i) for i, p in enumerate(pairs)}
    d = pd.DataFrame({"pair": ["P0"], "e": [300]})
    r0 = trade_ranks(d, pairs, lambda p: data[p], pd.Timedelta("1D"), 6)
    for p in pairs:  # scramble everything from the entry bar on
        data[p][0].close[300:] = np.random.default_rng(99).uniform(0.1, 10, 100)
    r1 = trade_ranks(d, pairs, lambda p: data[p], pd.Timedelta("1D"), 6)
    assert r0[0] == r1[0] and not np.isnan(r0[0])


def test_too_few_pairs_is_nan():
    data = {p: _bars(i) for i, p in enumerate(["A", "B"])}
    d = pd.DataFrame({"pair": ["A"], "e": [300]})
    assert np.isnan(trade_ranks(d, ["A", "B"], lambda p: data[p], pd.Timedelta("1D"), 6)[0])


def test_perm_p_small_for_a_real_split_and_large_for_noise():
    rng = np.random.default_rng(1)
    fold = np.repeat(["a", "b"], 200)
    agree = rng.random(400) < 0.5
    assert perm_p(np.where(agree, 1.0, 0.0) + rng.normal(0, 0.1, 400), agree, fold, n=500) < 0.01
    assert perm_p(rng.normal(0, 1, 400), agree, fold, n=500) > 0.01
