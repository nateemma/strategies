"""Entry and exit signal chains, recomputed from the side-car (T012, T019a).

These chains decide WHICH TRADES EXIST, so an error here is invisible in the exit logic and
fatal to everything downstream. `model_exit` ungated fired ~4x too often against a real
backtest (143 vs 36 on f07), which is what these tests exist to prevent recurring.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from research.labelfree.es import panel as PA           # noqa: E402
from research.labelfree.es import space as SP           # noqa: E402


def _d(n=10, **over):
    """A synthetic per-pair panel that passes every guard by default."""
    d = {"p_sell": np.zeros(n), "p_hold": np.zeros(n), "p_buy": np.ones(n) * 0.9,
         "volume": np.ones(n) * 100.0, "rvol": np.ones(n) * 3.0,
         "atr_pct_roll": np.ones(n) * 0.01, "guard_metric": np.zeros(n),
         "close_norm": np.ones(n) * -0.5, "adx": np.ones(n) * 50.0,
         "bb_width": np.ones(n) * 0.1}
    d.update(over)
    return d


def test_model_signals_follow_argmax_with_threshold():
    """BaseNNStrategy:779-783 — argmax only if max(p) > threshold, else HOLD."""
    d = _d(3, p_buy=np.array([0.9, 0.5, 0.4]),
           p_hold=np.array([0.05, 0.3, 0.5]), p_sell=np.array([0.05, 0.2, 0.1]))
    buy, sell = PA.model_signals(d, threshold=0.69)
    assert list(buy) == [True, False, False], "0.5 is the argmax but is below threshold -> HOLD"
    assert not sell.any()

    d2 = _d(2, p_sell=np.array([0.9, 0.4]), p_hold=np.array([0.05, 0.5]),
            p_buy=np.array([0.05, 0.1]))
    _b, s = PA.model_signals(d2, threshold=0.69)
    assert list(s) == [True, False]


def test_entry_guard_metric_is_an_UPPER_bound():
    """`guard_metric < entry_guard_threshold`. Raising the threshold LOOSENS the guard.

    Getting this backwards would make the search push the guard the wrong way, and it would
    still look like it was working.
    """
    d = _d(4, guard_metric=np.array([0.1, 0.4, 0.6, 0.9]))
    tight = PA.entries(d, dict(SP.INCUMBENT, entry_guard_threshold=0.2))
    loose = PA.entries(d, dict(SP.INCUMBENT, entry_guard_threshold=0.95))
    assert tight.sum() < loose.sum()
    assert list(tight) == [True, False, False, False]


def test_entry_rvol_is_a_LOWER_bound():
    d = _d(4, rvol=np.array([0.5, 1.5, 2.5, 3.5]))
    strict = PA.entries(d, dict(SP.INCUMBENT, entry_rvol_threshold=3.0))
    open_ = PA.entries(d, dict(SP.INCUMBENT, entry_rvol_threshold=0.25))
    assert strict.sum() < open_.sum()
    assert list(strict) == [False, False, False, True]


def test_exit_signal_is_gated_not_raw_predict_sell():
    """The bug this catches: model_exit fired 143 times against a real backtest's 36."""
    d = _d(3, p_sell=np.ones(3) * 0.9, p_buy=np.zeros(3), p_hold=np.zeros(3),
           rvol=np.array([0.5, 0.5, 3.0]), guard_metric=np.array([0.5, 0.5, 0.5]))
    raw = PA.model_signals(d)[1]
    gated = PA.exits(d)
    assert raw.sum() == 3, "all three bars carry a sell signal"
    assert gated.sum() == 1, "only the bar clearing exit_rvol_threshold survives the gate"


def test_exit_guard_metric_is_a_LOWER_bound_unlike_the_entry_one():
    """Same column, opposite direction on the two sides (BaseStrategy 1009-1015 vs 945-947)."""
    d = _d(3, p_sell=np.ones(3) * 0.9, p_buy=np.zeros(3), p_hold=np.zeros(3),
           guard_metric=np.array([-0.5, 0.0, 0.5]))
    assert list(PA.exits(d)) == [False, True, True]


def test_nan_guard_columns_do_not_admit_a_trade():
    """NaN comparisons are False, but nan_to_num on a bool cast could flip that."""
    d = _d(3, adx=np.array([np.nan, 50.0, np.nan]))
    assert list(PA.entries(d, dict(SP.INCUMBENT))) == [False, True, False]
