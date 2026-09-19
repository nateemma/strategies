# pragma pylint: disable=C0103, C0114, C0115, C0116, C0301, C0303, C0411, C0413
# pylint: disable=import-error
# flake8: noqa: F401, E402

"""Tests for research/clfstudy/labeltriage.py.

Fast and synthetic throughout: no `data.load_all` (would read real feathers), no
training. Covers the generic shim's label composition, the signature-filtering helper,
the persistence-MCC calculation, and the error-row-not-abort contract.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from Framework import TrainingSignals
from research.clfstudy import splits as S
from research.clfstudy.labeltriage import (
    LabelMethod,
    _filter_kwargs,
    _one_row,
    _safe_mcc,
    _trailing_features,
    ternary_labels_generic,
)


def _synthetic_ohlcv(seed: int = 3, n: int = 300) -> pd.DataFrame:
    """Plain OHLCV (no indicators -- the price-only labelers under test need none):
    a quiet random walk, then a forced dip and a forced rally, so both a SELL and a
    BUY threshold can clear within a short horizon."""
    rng = np.random.default_rng(seed)
    close = np.full(n, 100.0)
    for i in range(1, 140):
        close[i] = close[i - 1] + rng.normal(0, 0.02)

    fall_start, fall_len = 140, 8
    fall = np.linspace(0, -0.06, fall_len)
    close[fall_start : fall_start + fall_len] = close[fall_start - 1] * (1 + fall)

    rally_start, rally_len = fall_start + fall_len, 16
    rally = np.linspace(0, 0.14, rally_len)
    close[rally_start : rally_start + rally_len] = close[rally_start - 1] * (1 + rally)

    for i in range(rally_start + rally_len, n):
        close[i] = close[i - 1] + rng.normal(0, 0.02)

    high = close * (1 + np.abs(rng.normal(0, 0.002, n)))
    low = close * (1 - np.abs(rng.normal(0, 0.002, n)))
    openp = close + rng.normal(0, 0.005, n)
    volume = np.abs(rng.normal(100, 20, n))
    dates = pd.date_range("2022-01-01", periods=n, freq="15min", tz="UTC")
    return pd.DataFrame(
        {"date": dates, "open": openp, "high": high, "low": low, "close": close, "volume": volume}
    )


# ---------------------------------------------------------------------------
# 1. generic shim -> {0,1,2} for at least two methods
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "method", [LabelMethod.triple_barrier, LabelMethod.quantile_future]
)
def test_ternary_labels_generic_produces_ternary_classes(method):
    df = _synthetic_ohlcv()
    labels = ternary_labels_generic(df, method, horizon=24)

    assert len(labels) == len(df)
    assert labels.dtype == np.int8
    assert set(np.unique(labels).tolist()) <= {0, 1, 2}
    counts = np.bincount(labels, minlength=3)
    assert counts[1] == counts.max()  # HOLD is the majority


def test_ternary_labels_generic_triple_barrier_gets_both_sides():
    # the forced dip and forced rally should each clear a 2% triple-barrier threshold
    # well within a 24-bar horizon (mirrors test_labels_shim.py's fixture logic).
    df = _synthetic_ohlcv()
    labels = ternary_labels_generic(
        df, LabelMethod.triple_barrier, horizon=24, min_gain=0.02, min_loss=0.02
    )
    counts = np.bincount(labels, minlength=3)
    assert counts[0] > 0, "expected at least one SELL label"
    assert counts[2] > 0, "expected at least one BUY label"


# ---------------------------------------------------------------------------
# 2. persistence MCC
# ---------------------------------------------------------------------------


def test_persistence_mcc_near_one_for_a_perfectly_repeating_label():
    # two long, near-constant runs: only a single lag-1 transition among ~4000 pairs.
    y = np.concatenate([np.zeros(2000, dtype=int), np.full(2000, 2, dtype=int)])
    mcc = _safe_mcc(y[:-1], y[1:])
    assert mcc == pytest.approx(1.0, abs=0.01)


def test_persistence_mcc_near_zero_for_iid_random_label():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 3, size=5000)
    mcc = _safe_mcc(y[:-1], y[1:])
    assert abs(mcc) < 0.05


# ---------------------------------------------------------------------------
# 3. signature-filtering helper
# ---------------------------------------------------------------------------


def test_filter_kwargs_passes_only_accepted_arguments():
    def fn(df, horizon=1, min_gain=2):
        return None

    filtered = _filter_kwargs(fn, {"df": "x", "horizon": 24, "min_gain": 0.02, "rvol_threshold": 5})
    assert filtered == {"horizon": 24, "min_gain": 0.02}


def test_filter_kwargs_matches_real_labeler_signature():
    # labels_geometry does NOT accept rvol_threshold or min_ratio; only min_gain/min_loss/
    # prominence/horizon should survive.
    fn = TrainingSignals.METHODS[LabelMethod.geometry]
    filtered = _filter_kwargs(
        fn, {"horizon": 24, "min_gain": 0.02, "min_loss": 0.02, "rvol_threshold": 0.3, "min_ratio": 2.0}
    )
    assert filtered == {"horizon": 24, "min_gain": 0.02, "min_loss": 0.02}


# ---------------------------------------------------------------------------
# 4. a raising labeler is recorded as an error row, scan continues
# ---------------------------------------------------------------------------


def _tiny_split_set(pair: str, dates: pd.Series) -> S.SplitSet:
    w = S.Window(
        window_id="T",
        train_start=dates.iloc[0],
        val_start=dates.iloc[200],
        train_end=dates.iloc[240],
        test_start=dates.iloc[250],
        test_end=dates.iloc[-1],
        distant=False,
    )
    return S.SplitSet(
        split_set="test", embargo_bars=10, val_fraction=0.1, min_history_days=1,
        timeframe="15m", pairs=[pair], windows=[w],
    )


def test_one_row_records_error_and_does_not_abort(monkeypatch):
    pair = "FOO/USDT"
    df = _synthetic_ohlcv()
    frames = {pair: df}
    vol_by_pair = {}
    mom_by_pair = {}
    for p, frame in frames.items():
        vol_by_pair[p], mom_by_pair[p] = _trailing_features(frame)
    split_set = _tiny_split_set(pair, df["date"])

    def boom(df, **kwargs):
        raise TypeError("synthetic labeler failure")

    monkeypatch.setitem(TrainingSignals.METHODS, LabelMethod.forward_mae, boom)

    error_row = _one_row(
        LabelMethod.forward_mae, 24, frames, vol_by_pair, mom_by_pair, split_set,
        min_gain=0.02, min_loss=0.02,
    )
    assert error_row["error"] is not None
    assert "synthetic labeler failure" in error_row["error"]
    assert "balance_mean" not in error_row

    # the scan continues: a different (unpatched) method still produces a full row.
    ok_row = _one_row(
        LabelMethod.quantile_future, 24, frames, vol_by_pair, mom_by_pair, split_set,
        min_gain=0.02, min_loss=0.02,
    )
    assert ok_row["error"] is None
    assert "balance_mean" in ok_row
    assert len(ok_row["balance_mean"]) == 3


# ---------------------------------------------------------------------------
# trailing features (used by vol_auroc / dir_auroc) are causal
# ---------------------------------------------------------------------------


def test_trailing_features_are_causal_and_correct_length():
    df = _synthetic_ohlcv()
    vol, mom = _trailing_features(df)
    assert len(vol) == len(df)
    assert len(mom) == len(df)
    assert np.isnan(vol[:48]).all()
    assert not np.isnan(vol[48:]).any()
    assert np.isnan(mom[:16]).all()
    assert not np.isnan(mom[16:]).any()
