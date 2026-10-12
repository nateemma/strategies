"""029 T011: the label families of A′ -- a model's entries take the family its label's positives
imply. Each family is checked against the labeler's own behaviour on synthetic frames, and against
the same-named trigger's declared family."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from Framework import TrainingSignals as T
from Framework.TrainingSignals import LabelMethod as L
from utils.LabelFamilies import LABEL_FAMILY, label_family


N = 120


def _frame(guard: float, bbw: float = 0.06, trend: float = 0.003, new_high: bool = False):
    """A rising path (every bar's forward gain clears 2%) with a fixed band position."""
    close = 100.0 * np.cumprod(np.full(N, 1.0 + trend))
    high = close * 1.001
    if not new_high:
        high = np.full(N, close.max() * 2)  # nothing ever breaks the prior high
    return pd.DataFrame({
        "open": close, "close": close, "high": high, "low": close * 0.999,
        "volume": np.full(N, 1000.0), "guard_metric": np.full(N, guard),
        "bb_width": np.full(N, bbw), "rvol": np.full(N, 3.0),
    })


def _positives(fn, df, **kw) -> int:
    return int((np.asarray(fn(df, min_gain=0.02, horizon=24, **kw)) > 0.5)[:N - 30].sum())


def test_gbb_labels_fire_only_at_a_weak_band_position():
    for fn in (T.labels_gbb, T.labels_gbb_path):
        assert _positives(fn, _frame(guard=0.0)) > 0
        assert _positives(fn, _frame(guard=0.9)) == 0


def test_breakout_gbb_fires_only_at_a_strong_band_position():
    assert _positives(T.labels_breakout_gbb, _frame(guard=0.9)) > 0
    assert _positives(T.labels_breakout_gbb, _frame(guard=0.0)) == 0


def test_breakout_fires_only_on_a_new_high():
    assert _positives(T.labels_breakout, _frame(guard=0.5, new_high=True)) > 0
    assert _positives(T.labels_breakout, _frame(guard=0.5, new_high=False)) == 0


def test_families_match_the_same_named_triggers():
    from research.outcomes import triggers

    triggers.ensure_builtins()
    for name in ("gbb", "breakout_gbb", "breakout", "breakout_bb", "breakout_vol",
                 "breakout_squeeze", "geometry"):
        assert label_family(getattr(L, name)) == triggers.get(name).family, name


def test_arm_d_label_is_reversion_and_outcome_only_labels_have_none():
    assert label_family(L.gbb_path) == "reversion"
    assert label_family(int(L.gbb_path)) == "reversion"
    for m in (L.forward_mae, L.triple_barrier, L.quantile_future, L.future_sharpe):
        assert label_family(m) is None
    assert label_family(None) is None and label_family("nonsense") is None


@pytest.mark.parametrize("method", sorted(LABEL_FAMILY, key=int))
def test_every_row_has_a_reason(method):
    family, reason = LABEL_FAMILY[method]
    assert family in ("reversion", "momentum") and reason.strip()
