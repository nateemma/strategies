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
# triple_barrier label — `atr_scale` (ATR-scaled barriers)
# ---------------------------------------------------------------------------


def _synthetic_ohlcv_high_vol(seed: int = 2, n: int = 600) -> pd.DataFrame:
    """Same shape as `_synthetic_ohlcv`, but with a wider high/low range (+-4% instead
    of +-1%) during the dip/rally window, so ATR/close there clears the default
    (fixed) 3%/2% triple-barrier width at a modest `atr_scale` -- otherwise the fixed-
    barrier and ATR-scaled barrier labels are identical on this fixture at
    `atr_scale=2.0` (the synthetic volatility is too low relative to the defaults)."""
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
    high[volatile] = close[volatile] * 1.04
    low[volatile] = close[volatile] * 0.96
    openp = close + rng.normal(0, 0.005, n)
    volume = np.abs(rng.normal(100, 20, n))

    dates = pd.date_range("2022-01-01", periods=n, freq="15min", tz="UTC")
    return pd.DataFrame(
        {"date": dates, "open": openp, "high": high, "low": low, "close": close, "volume": volume}
    )


def test_ternary_labels_triple_barrier_atr_scale_in_range():
    df = data.populate(_synthetic_ohlcv_high_vol())
    label = {"name": "triple_barrier", "horizon": 24, "up": 0.005, "down": 0.005, "atr_scale": 2.0}
    labels = ternary_labels(df, label=label)

    assert len(labels) == len(df)
    assert set(np.unique(labels).tolist()) <= {0, 1, 2}


def test_ternary_labels_triple_barrier_atr_scale_differs_from_fixed():
    df = data.populate(_synthetic_ohlcv_high_vol())
    fixed = {"name": "triple_barrier", "horizon": 24, "up": 0.005, "down": 0.005}
    scaled = dict(fixed, atr_scale=2.0)

    fixed_labels = ternary_labels(df, label=fixed)
    scaled_labels = ternary_labels(df, label=scaled)

    assert not np.array_equal(fixed_labels, scaled_labels)


def test_ternary_labels_triple_barrier_atr_scale_omitted_equals_none():
    # atr_scale omitted entirely produces the same labels as atr_scale=None explicit
    # (both resolve to `label.get("atr_scale")` -> `None` -> fixed barriers).
    df = data.populate(_synthetic_ohlcv())
    omitted = {"name": "triple_barrier", "horizon": 24, "up": 0.01, "down": 0.01}
    explicit_none = dict(omitted, atr_scale=None)
    np.testing.assert_array_equal(
        ternary_labels(df, label=omitted),
        ternary_labels(df, label=explicit_none),
    )


# ---------------------------------------------------------------------------
# triple_barrier label — `up`/`down` set the ACTUAL barrier (not a floor under
# `labels_triple_barrier`'s pt=0.03/sl=0.02 defaults)
# ---------------------------------------------------------------------------


def _rise_df(n: int = 100, t: int = 29, rise_pct: float = 0.025,
             rise_bars: int = 10) -> pd.DataFrame:
    """Deterministic path: flat at 100 through bar `t`, then a straight-line rise
    over `rise_bars` bars reaching exactly `+rise_pct` at `t + rise_bars`, then flat
    forever after (never retraces). `high == low == close` so the labeler's
    close-based touch test is unambiguous."""
    close = np.full(n, 100.0)
    for i in range(t + 1, t + 1 + rise_bars):
        frac = (i - t) / rise_bars
        close[i] = 100.0 * (1 + rise_pct * frac)
    close[t + 1 + rise_bars:] = close[t + rise_bars]
    dates = pd.date_range("2022-01-01", periods=n, freq="15min", tz="UTC")
    return pd.DataFrame(
        {"date": dates, "open": close, "high": close, "low": close,
         "close": close, "volume": np.ones(n)}
    )


def _fall_df(n: int = 100, t: int = 29, fall_pct: float = 0.025,
             fall_bars: int = 10) -> pd.DataFrame:
    """Mirror of `_rise_df`: a straight-line fall reaching exactly `-fall_pct`,
    then flat forever after (never retraces)."""
    close = np.full(n, 100.0)
    for i in range(t + 1, t + 1 + fall_bars):
        frac = (i - t) / fall_bars
        close[i] = 100.0 * (1 - fall_pct * frac)
    close[t + 1 + fall_bars:] = close[t + fall_bars]
    dates = pd.date_range("2022-01-01", periods=n, freq="15min", tz="UTC")
    return pd.DataFrame(
        {"date": dates, "open": close, "high": close, "low": close,
         "close": close, "volume": np.ones(n)}
    )


def test_triple_barrier_up_down_set_the_actual_buy_barrier():
    # From bar t, close rises to exactly +2.5% within 48 bars and never falls.
    # up=down=0.02 clears a 2% profit-take before any (never-happening) 2% stop
    # -> BUY. up=0.03 is never reached (rise tops out at 2.5%) -> HOLD. Before the
    # fix, `pt = max(pt, min_gain)` meant `up=0.02` couldn't lower `pt` below the
    # default 0.03, so BOTH cases produced HOLD.
    t = 29
    df = _rise_df(t=t)

    buy_labels = ternary_labels(
        df, label={"name": "triple_barrier", "horizon": 48, "up": 0.02, "down": 0.02}
    )
    hold_labels = ternary_labels(
        df, label={"name": "triple_barrier", "horizon": 48, "up": 0.03, "down": 0.02}
    )
    assert buy_labels[t] == 2
    assert hold_labels[t] == 1


def test_triple_barrier_up_down_set_the_actual_sell_barrier():
    # Mirror: close drops to exactly -2.5% within 48 bars and never rises.
    # down=0.02 clears a 2% stop-loss before any (never-happening) 2% profit-take
    # -> SELL. down=0.03 is never reached -> HOLD. Before the fix,
    # `sl = max(sl, min_loss)` meant `down=0.02` couldn't lower `sl` below the
    # default 0.02 -- but `pt` (the mirror barrier) stayed pinned at the default
    # 0.03 regardless of `up`, since `labels_triple_barrier_sell` ignores `min_gain`
    # entirely; this test only exercises the `down`/`sl` side.
    t = 29
    df = _fall_df(t=t)

    sell_labels = ternary_labels(
        df, label={"name": "triple_barrier", "horizon": 48, "up": 0.02, "down": 0.02}
    )
    hold_labels = ternary_labels(
        df, label={"name": "triple_barrier", "horizon": 48, "up": 0.02, "down": 0.03}
    )
    assert sell_labels[t] == 0
    assert hold_labels[t] == 1


def test_triple_barrier_width_below_default_floor():
    # up=down=0.01 is BELOW `labels_triple_barrier`'s pt=0.03/sl=0.02 defaults --
    # before the fix this was structurally impossible to trade on (the max() floor
    # always won), so a +1.5% rise (well below the 3% default, but above the
    # requested 1%) was labeled HOLD. After the fix it must be BUY.
    t = 29
    df = _rise_df(t=t, rise_pct=0.015)

    labels = ternary_labels(
        df, label={"name": "triple_barrier", "horizon": 48, "up": 0.01, "down": 0.01}
    )
    assert labels[t] == 2


# ---------------------------------------------------------------------------
# label_key
# ---------------------------------------------------------------------------


def test_label_key_gbb():
    assert label_key(None) == "gbb"
    assert label_key({"name": "gbb"}) == "gbb"


def test_label_key_triple_barrier_defaults():
    label = {"name": "triple_barrier", "horizon": 48, "up": 0.02, "down": 0.02}
    assert label_key(label) == "tb_h48_u0.02_d0.02_cr1_pf0"


def test_label_key_triple_barrier_atr_scale():
    label = {
        "name": "triple_barrier", "horizon": 48, "up": 0.005, "down": 0.005, "atr_scale": 2.0,
    }
    assert label_key(label) == "tb_h48_u0.005_d0.005_cr1_pf0_atr2"


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


# ---------------------------------------------------------------------------
# generic PRICE-ONLY labels (research: let an experiment select ANY price-only labeler,
# not just gbb/triple_barrier)
# ---------------------------------------------------------------------------


def test_ternary_labels_forward_mae_on_synthetic_frame():
    df = data.populate(_synthetic_ohlcv())
    label = {"name": "forward_mae", "horizon": 12, "min_gain": 0.015, "max_drawdown": 0.05}
    labels = ternary_labels(df, label=label)

    assert len(labels) == len(df)
    assert set(np.unique(labels).tolist()) <= {0, 1, 2}
    counts = np.bincount(labels, minlength=3)
    assert counts[0] > 0, "expected at least one SELL label"
    assert counts[2] > 0, "expected at least one BUY label"


def test_ternary_labels_quantile_future_on_synthetic_frame():
    df = data.populate(_synthetic_ohlcv())
    label = {"name": "quantile_future", "horizon": 12, "top_quantile": 0.8}
    labels = ternary_labels(df, label=label)

    assert len(labels) == len(df)
    assert set(np.unique(labels).tolist()) <= {0, 1, 2}
    counts = np.bincount(labels, minlength=3)
    assert counts[0] > 0, "expected at least one SELL label"
    assert counts[2] > 0, "expected at least one BUY label"


def test_ternary_labels_generic_label_unknown_name_rejected():
    df = data.populate(_synthetic_ohlcv())
    with pytest.raises(ValueError):
        ternary_labels(df, label={"name": "not_a_real_generic_label", "horizon": 12})


def test_label_key_forward_mae():
    label = {"name": "forward_mae", "horizon": 12, "min_gain": 0.015, "max_drawdown": 0.05}
    assert label_key(label) == "fmae_h12_g0.015_dd0.05"


def test_label_key_quantile_future():
    label = {"name": "quantile_future", "horizon": 12, "top_quantile": 0.8}
    assert label_key(label) == "qfut_h12_q0.8"


def test_generic_label_rejects_a_key_the_labeler_cannot_accept():
    """A config key the labeler does not take is an error, not a silent no-op.

    Three defects in this study (rvol_threshold, the pt/sl barrier floors, and a label that
    never reached the isolated child) all presented as a silently ignored parameter, so the
    generic path refuses any caller-supplied key the labeler cannot honour.
    """
    import pytest
    from research.clfstudy import labels as labels_mod
    df = labels_mod_test_frame() if "labels_mod_test_frame" in globals() else None
    if df is None:
        from research.clfstudy import data as data_mod
        df = data_mod.populate(_synthetic_ohlcv())
    with pytest.raises(ValueError, match="does not accept label key"):
        labels_mod.ternary_labels(
            df, label={"name": "forward_mae", "horizon": 12, "top_quantile": 0.8}
        )
    # a key the labeler DOES accept still works
    out = labels_mod.ternary_labels(
        df, label={"name": "forward_mae", "horizon": 12, "min_gain": 0.015, "max_drawdown": 0.05}
    )
    assert set(np.unique(out)) <= {0, 1, 2}
