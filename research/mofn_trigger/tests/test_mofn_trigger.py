"""mofn_4of8: registered, reversion family, fires iff >= MOFN_M member votes, pair-invariant."""
from __future__ import annotations

import numpy as np
import pandas as pd

from research.outcomes import audit_triggers as A
from research.outcomes import triggers as T


def _df(n=600, seed=0):
    rng = np.random.default_rng(seed)
    c = 100 * np.exp(np.cumsum(rng.normal(0, 0.01, n)))
    return pd.DataFrame({"date": pd.date_range("2024-01-01", periods=n, freq="15min", tz="UTC"),
                         "open": c, "high": c * 1.005, "low": c * 0.995, "close": c,
                         "volume": rng.uniform(1, 2, n)})


def test_vote_threshold_and_pair_invariance():
    T.ensure_builtins()
    t = T.get("mofn_4of8")
    assert t.family == "reversion"
    df = _df()
    votes = A.mofn_votes(df)
    assert list(votes.columns) == [f"{c}:{v}" if v else c for c, v in A.MOFN_MEMBERS]
    got = np.asarray(T.evaluate(t.name, df.assign(pair="SOL/USDT")), bool)
    assert np.array_equal(got, votes.sum(axis=1).to_numpy() >= A.MOFN_M)
    assert np.array_equal(got, np.asarray(T.evaluate(t.name, df.assign(pair="LTC/USDT")), bool))
