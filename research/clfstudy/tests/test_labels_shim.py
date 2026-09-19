"""Tests for research/clfstudy/labels.py (T006).

Fidelity is the whole point of this module: every attribute the shim exposes must equal
the `NNNC_MLX` class attribute, and every bound method must resolve to the exact function
`NNNC_MLX` itself would use (never a reimplementation).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from NNNC.NNNC_MLX import NNNC_MLX
from research.clfstudy import data
from research.clfstudy.labels import (
    _SHIM_ATTRS, _SHIM_METHODS, _LabelShim, label_key, ternary_labels,
)


def test_shim_attrs_match_nnnc_mlx():
    shim = _LabelShim()
    for name in _SHIM_ATTRS:
        assert getattr(shim, name) == getattr(NNNC_MLX, name), name


def test_shim_methods_resolve_to_nnnc_mlx_mro():
    shim = _LabelShim()
    for name in _SHIM_METHODS:
        bound = getattr(shim, name)
        assert bound.__func__ is getattr(NNNC_MLX, name), name


def _synthetic_ohlcv(seed: int = 2, n: int = 600) -> pd.DataFrame:
    """600-row synthetic OHLCV: quiet random walk, then a sharp dip (oversold +
    volatility expansion) followed by a forced rally, so the gbb-family label's
    guard_metric/bb_width trigger and the +2% forward-gain test can both fire."""
    rng = np.random.default_rng(seed)
    close = np.full(n, 100.0)
    for i in range(1, 280):
        close[i] = close[i - 1] + rng.normal(0, 0.02)

    fall_start, fall_len = 280, 10
    fall = np.linspace(0, -0.05, fall_len)
    close[fall_start : fall_start + fall_len] = close[fall_start - 1] * (1 + fall)

    rally_start, rally_len = fall_start + fall_len, 20
    rally = np.linspace(0, 0.12, rally_len)
    close[rally_start : rally_start + rally_len] = close[rally_start - 1] * (1 + rally)

    for i in range(rally_start + rally_len, n):
        close[i] = close[i - 1] + rng.normal(0, 0.02)

    high = close * (1 + np.abs(rng.normal(0, 0.002, n)))
    low = close * (1 - np.abs(rng.normal(0, 0.002, n)))
    volatile = slice(fall_start, rally_start + rally_len)
    high[volatile] = close[volatile] * 1.01
    low[volatile] = close[volatile] * 0.99
    openp = close + rng.normal(0, 0.005, n)
    volume = np.abs(rng.normal(100, 20, n))

    dates = pd.date_range("2022-01-01", periods=n, freq="15min", tz="UTC")
    return pd.DataFrame(
        {"date": dates, "open": openp, "high": high, "low": low, "close": close, "volume": volume}
    )


def test_ternary_labels_on_synthetic_frame():
    df = data.populate(_synthetic_ohlcv())
    labels = ternary_labels(df)

    assert len(labels) == len(df)
    assert set(np.unique(labels).tolist()) <= {0, 1, 2}

    counts = np.bincount(labels, minlength=3)
    assert counts[1] == counts.max()  # HOLD is the majority
    # The forced dip+rally clears the gbb trigger and the +2% forward-gain test;
    # if this ever regresses to 0 BUYs, the fixture (not the assertion) needs
    # revisiting per T006's documented fallback.
    assert counts[2] > 0


def test_ternary_labels_does_not_mutate_input():
    df = data.populate(_synthetic_ohlcv())
    before = df.copy(deep=True)
    ternary_labels(df)
    pd.testing.assert_frame_equal(df, before)


def test_ternary_labels_label_none_matches_gbb_default():
    df = data.populate(_synthetic_ohlcv())
    np.testing.assert_array_equal(ternary_labels(df), ternary_labels(df, label=None))
    np.testing.assert_array_equal(
        ternary_labels(df), ternary_labels(df, label={"name": "gbb"})
    )


# ---------------------------------------------------------------------------
# triple_barrier label (research: let an experiment select the label)
# ---------------------------------------------------------------------------


def test_ternary_labels_triple_barrier_on_synthetic_frame():
    df = data.populate(_synthetic_ohlcv())
    label = {"name": "triple_barrier", "horizon": 24, "up": 0.01, "down": 0.01}
    labels = ternary_labels(df, label=label)

    assert len(labels) == len(df)
    assert set(np.unique(labels).tolist()) <= {0, 1, 2}

    counts = np.bincount(labels, minlength=3)
    # the synthetic fixture's forced dip and forced rally should each clear a 1%
    # triple-barrier threshold well within a 24-bar horizon.
    assert counts[0] > 0, "expected at least one SELL label"
    assert counts[2] > 0, "expected at least one BUY label"


def test_ternary_labels_triple_barrier_does_not_mutate_input():
    df = data.populate(_synthetic_ohlcv())
    before = df.copy(deep=True)
    label = {"name": "triple_barrier", "horizon": 24, "up": 0.01, "down": 0.01}
    ternary_labels(df, label=label)
    pd.testing.assert_frame_equal(df, before)


def test_ternary_labels_triple_barrier_peak_filter_true_uses_production_filter():
    # peak_filter=True re-applies filter_peaks_by_future_performance (bound unchanged
    # from NNNC_MLX), which can only ever REMOVE peaks -- never add new ones -- so the
    # default (peak_filter=False) count is always >= the peak_filter=True count.
    df = data.populate(_synthetic_ohlcv())
    base_label = {"name": "triple_barrier", "horizon": 24, "up": 0.01, "down": 0.01}
    filtered_label = dict(base_label, peak_filter=True)

    unfiltered = ternary_labels(df, label=base_label)
    filtered = ternary_labels(df, label=filtered_label)

    assert np.count_nonzero(unfiltered == 2) >= np.count_nonzero(filtered == 2)
    assert np.count_nonzero(unfiltered == 0) >= np.count_nonzero(filtered == 0)


def test_ternary_labels_triple_barrier_unknown_name_rejected():
    df = data.populate(_synthetic_ohlcv())
    with pytest.raises(ValueError):
        ternary_labels(df, label={"name": "not_a_real_label"})


# ---------------------------------------------------------------------------
# label_key
# ---------------------------------------------------------------------------


def test_label_key_gbb():
    assert label_key(None) == "gbb"
    assert label_key({"name": "gbb"}) == "gbb"


def test_label_key_triple_barrier_defaults():
    label = {"name": "triple_barrier", "horizon": 48, "up": 0.02, "down": 0.02}
    assert label_key(label) == "tb_h48_u0.02_d0.02_cr1_pf0"


def test_label_key_triple_barrier_explicit_flags():
    label = {
        "name": "triple_barrier", "horizon": 24, "up": 0.01, "down": 0.015,
        "conflict_resolution": False, "peak_filter": True,
    }
    assert label_key(label) == "tb_h24_u0.01_d0.015_cr0_pf1"


def test_label_key_differs_by_params():
    a = label_key({"name": "triple_barrier", "horizon": 24, "up": 0.01, "down": 0.01})
    b = label_key({"name": "triple_barrier", "horizon": 48, "up": 0.01, "down": 0.01})
    assert a != b
