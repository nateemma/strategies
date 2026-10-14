"""034 T003: the declared candidates -- keys accepted and live on both sides; TB's sell mirror."""

from __future__ import annotations

import inspect
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


STRAT = Path(__file__).resolve().parents[3]
if str(STRAT) not in sys.path:
    sys.path.insert(0, str(STRAT))

from Framework import TrainingSignals as T  # noqa: E402
from research.label_screen import candidates as C  # noqa: E402


BASE = {"horizon": 48, "min_gain": 0.02, "min_loss": 0.02}


def _frame(n=3000, seed=0):
    rng = np.random.default_rng(seed)
    close = 100 * np.exp(np.cumsum(rng.normal(0, 0.006, n)))
    return pd.DataFrame({"open": close, "high": close * 1.002, "low": close * 0.998,
                         "close": close, "volume": rng.uniform(1, 2, n),
                         "guard_metric": rng.uniform(-1, 1, n), "bb_width": rng.uniform(0, 0.1, n),
                         "adx_scaled": rng.uniform(-1, 1, n)})


def test_tags_unique_and_table_shape():
    tags = [c.tag for c in C.CANDIDATES]
    assert len(tags) == len(set(tags)) == 9
    assert [c.tag for c in C.TRAINED] == ["FMAE", "TB", "MMR", "FEXP", "QF", "LMF", "BOG"]
    assert C.BY_TAG["BOG"].roles == ("sz",) and C.BY_TAG["CONTROL"].roles == ()
    assert {c.method for c in C.CANDIDATES} == {27, 28, 0, 1, 3, 12, 2, 4, 24}


@pytest.mark.parametrize("c", [c for c in C.CANDIDATES if c.label_params], ids=lambda c: c.tag)
def test_declared_buy_keys_are_in_the_signature_and_survive_strict_filtering(c):
    fn = T.METHODS[T.LabelMethod(c.method)]
    assert set(c.label_params) <= set(inspect.signature(fn).parameters)
    kept = T._filter_label_kwargs(fn, BASE | c.label_params, strict_keys=c.label_params.keys())
    assert all(kept[k] == v for k, v in c.label_params.items())


def test_a_bogus_buy_key_raises():
    with pytest.raises(Exception):
        T.get_train_buy_signals(_frame(300), method=0, params=BASE | {"bogus": 1},
                                strict_keys={"bogus"})


def _default(fn, k):
    return inspect.signature(fn).parameters[k].default


@pytest.mark.parametrize("c", [c for c in C.CANDIDATES if C.sell_params(c)], ids=lambda c: c.tag)
def test_declared_sell_keys_reach_the_sell_labeler(c):
    """Explicit sell branches filter to a hand-written allowed set and drop other keys silently, so
    each non-default declared sell key must CHANGE the sell labels (removed vs present)."""
    df = _frame()
    sp = C.sell_params(c)
    full = T.get_train_sell_signals(df, method=c.method, params=BASE | sp)
    for k in sp:
        without = T.get_train_sell_signals(df, method=c.method,
                                           params=BASE | {x: v for x, v in sp.items() if x != k})
        if k == "sl" and c.tag == "TB":  # 0.02 == the sell labeler's default and min_loss
            assert (full == without).all()
            continue
        assert (full != without).any(), f"{c.tag} sell key {k} has no effect"


def test_triple_barrier_sell_is_the_mirror_of_the_buy_rule():
    tb = C.BY_TAG["TB"]
    # falls 2.1% then rises 5.5%: the buy rule (+2% before -5%) fires, and so does the sell mirror
    # (-2% before +5%), because each side's stop is the wider 5%
    close = np.r_[100.0, 99.5, 97.9, 99.0, 101, 103, 105.5, np.full(60, 105.5)]
    df = pd.DataFrame({"open": close, "high": close, "low": close, "close": close,
                       "volume": 1.0})
    buy = T.get_train_buy_signals(df, method=1, params=BASE | tb.label_params,
                                  strict_keys=tb.label_params.keys())
    sell = T.get_train_sell_signals(df, method=1, params=BASE | C.sell_params(tb))
    assert sell.iloc[0] == 1 and buy.iloc[0] == 1
    # reusing the buy values on the sell side would require -5% first: no sell here
    assert T.get_train_sell_signals(df, method=1, params=BASE | tb.label_params).iloc[0] == 0


def test_table_plans_cover_the_trained_candidates():
    tp = C.table_plans()
    assert set(tp) == {f"LS_{c.tag}" for c in C.TRAINED} and tp["LS_FMAE"] == "ls_fmae_es"
