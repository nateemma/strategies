"""Pair triggers: registered, reversion family, legs AND-ed, pair-invariant."""
from __future__ import annotations

import numpy as np
import pandas as pd

from research.outcomes import audit_triggers as A
from research.outcomes import triggers as T


def _df(n=400, seed=0):
    rng = np.random.default_rng(seed)
    c = 100 * np.exp(np.cumsum(rng.normal(0, 0.01, n)))
    return pd.DataFrame({"open": c, "high": c * 1.005, "low": c * 0.995, "close": c,
                         "volume": rng.uniform(1, 2, n)})


def test_registered_reversion_and_anded():
    T.ensure_builtins()
    df = _df()
    for a, b in A.PAIR_TRIGGERS:
        t = T.get(f"pair_{a}_{b}")
        assert t.family == "reversion"
        got = np.asarray(T.evaluate(t.name, df), bool)
        want = np.asarray(A.PAIR_LEGS[a](df), bool) & np.asarray(A.PAIR_LEGS[b](df), bool)
        assert np.array_equal(got, want)


def test_pair_column_is_ignored():
    T.ensure_builtins()
    df = _df()
    a = np.asarray(T.evaluate("pair_kst_zscore", df.assign(pair="SOL/USDT")), bool)
    b = np.asarray(T.evaluate("pair_kst_zscore", df.assign(pair="LTC/USDT")), bool)
    assert np.array_equal(a, b)
