"""Donchian screen: channel states, causality, fill timing, band, bootstrap."""
import numpy as np
import pandas as pd

from research.donchian_screen.screen import block_bootstrap_p, ensemble, simulate


def _daily(close):
    idx = pd.date_range("2024-01-01", periods=len(close), freq="D", tz="UTC")
    c = pd.Series(close, index=idx, dtype=float)
    return pd.DataFrame({"open": c, "high": c, "low": c, "close": c})


def test_breakout_sets_and_lower_channel_resets():
    d = _daily([10, 10, 10, 11, 11, 11, 9, 9])  # L=2: entry channel = prior 2 days' high, exit = prior 1 day's low
    s = ensemble(d, [2]).to_numpy()
    assert list(s) == [0, 0, 1, 1, 1, 1, 0, 0]  # day 2 ties the channel high (>= is declared); day 6 breaks the low


def test_signal_at_t_ignores_later_days():
    rng = np.random.default_rng(0)
    c = np.exp(np.cumsum(rng.normal(0, 0.03, 300)))
    a = ensemble(_daily(c), [5, 10, 20])
    c2 = c.copy()
    c2[200:] = rng.uniform(0.1, 10, 100)
    b = ensemble(_daily(c2), [5, 10, 20])
    assert a.iloc[:200].equals(b.iloc[:200])


def test_target_fills_next_day_and_band_blocks_small_moves():
    idx = pd.date_range("2024-01-01", periods=4, freq="D", tz="UTC")
    tgt = pd.DataFrame({"A": [1.0, 1.0, 1.0, 1.0]}, index=idx)
    ret = pd.DataFrame({"A": [0.5, 0.1, 0.0, 0.0]}, index=idx)
    x = simulate(tgt, ret, cost=0.0)
    assert x["gross"].iloc[0] == 0.1  # day 0's target earns day 1's return, never day 0's
    tgt2 = pd.DataFrame({"A": [0.01, 0.01, 0.01, 0.01]}, index=idx)
    assert simulate(tgt2, ret, cost=0.0, band=0.02)["traded"].sum() == 0


def test_bootstrap_identical_books_is_not_significant():
    r = np.random.default_rng(1).normal(0.001, 0.02, 400)
    assert block_bootstrap_p(r, r, n=200) == 1.0
    assert block_bootstrap_p(r + 0.01, r, n=200) < 0.05
