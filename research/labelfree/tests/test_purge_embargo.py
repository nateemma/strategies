"""Leakage gate — blocking for every arm (T006, SC-012).

Brief 02's arms all carry a forward-looking term: Arm A's warm-start classifier is
label-trained, Arm B has a cross-entropy term, Arm C predicts forward volatility. None is
exempt, so this gate blocks all of them.

Assertions are on VALUES, not shapes. Four of feature 009's defects were wrong-axis slices
that produced plausible shapes and wrong numbers with no error.
"""
from __future__ import annotations

import pathlib
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[3]))

from research.labelfree import labels as L                              # noqa: E402

H = L.DEFAULT_HORIZON


@pytest.fixture(scope="module")
def frame():
    """OHLCV plus the columns the gbb_path labelers gate on.

    `labels_gbb_path` requires guard_metric, bb_width and rvol. Passing raw OHLCV yields an
    all-HOLD target — silently, before the guard was added. This fixture supplies them so
    the leakage assertions run against a target that actually varies.
    """
    rng = np.random.default_rng(0)
    n = 2000
    px = 100 * np.exp(np.cumsum(rng.normal(0, 0.004, n)))
    return pd.DataFrame({
        "open": px, "close": px,
        "high": px * (1 + abs(rng.normal(0, 0.003, n))),
        "low": px * (1 - abs(rng.normal(0, 0.003, n))),
        "volume": rng.uniform(1e3, 1e5, n),
        "date": pd.date_range("2024-01-01", periods=n, freq="15min", tz="UTC"),
        "guard_metric": rng.uniform(-1.0, 1.0, n),
        "bb_width": rng.uniform(0.005, 0.15, n),
        "rvol": rng.uniform(0.0, 4.0, n),
    })


def test_raw_ohlcv_is_rejected_rather_than_silently_all_hold(frame):
    """The gbb_path labelers gate on indicator columns; without them every bar is HOLD."""
    bare = frame[["open", "high", "low", "close", "volume", "date"]]
    with pytest.raises(ValueError, match="guard_metric"):
        L.directional_label(bare)


def test_purge_length_is_the_label_horizon_not_the_peak_window(frame):
    """Purging PEAK_WINDOW (6) instead of HORIZON (48) leaves 42 bars of leakage."""
    assert H == 48, "label horizon changed; the purge length must follow it"
    m = L.purge_embargo_mask(2000, [1000], H)
    assert not m[1000 - H:1000 + H + 1].any(), "boundary neighbourhood not fully purged"
    assert m[:1000 - H - 1].all() and m[1000 + H + 1:].all(), "purge spread too far"


def test_no_target_within_one_horizon_of_a_boundary_survives(frame):
    """The core leakage assertion, on values rather than counts."""
    t = L.directional_label(frame, horizon=H)
    mask, rec = L.apply_leakage_control(t, boundaries=[800, 1500])
    for b in (800, 1500):
        window = mask[max(0, b - H): b + H + 1]
        assert not window.any(), f"boundary {b}: {int(window.sum())} bars survived the purge"
    assert rec["purge_bars"] == H and rec["embargo_bars"] == H


def test_record_reports_applied_values_not_requested(frame):
    """A purge that silently fell to zero must be detectable (FR-014)."""
    t = L.directional_label(frame, horizon=H)
    _, rec = L.apply_leakage_control(t, boundaries=[1000], horizon=0)
    assert rec["purge_bars"] == 0, "record must show what was APPLIED, so zero is visible"


def test_label_cannot_see_past_the_end_of_the_series(frame):
    """The last `horizon` bars have no future to look at."""
    t = L.directional_label(frame, horizon=H)
    assert not t.valid_mask[-H:].any()
    assert t.valid_mask[:-H].all()


def test_forward_vol_is_non_negative_and_carries_no_direction(frame):
    """Arm C's target must be a magnitude (FR-011)."""
    t = L.forward_realised_vol(frame, horizon=H)
    v = t.values[t.valid_mask]
    assert (v >= 0).all(), "forward realised vol went negative — it is not a magnitude"
    assert t.threshold is None
    # A sign-carrying target would correlate with forward return; a magnitude must not.
    close = frame["close"].to_numpy()
    fwd = np.full(len(close), np.nan)
    fwd[:-H] = close[H:] / close[:-H] - 1.0
    ok = t.valid_mask & ~np.isnan(fwd)
    c = np.corrcoef(v[: ok.sum()], fwd[ok])[0, 1]
    assert abs(c) < 0.25, f"forward vol correlates {c:+.3f} with forward return — not direction-free"


def test_both_label_thresholds_differ_as_labels(frame):
    """0.007 and 0.02 must produce different labels, even though the repo measured them
    as identical in P&L (332 trades, 10.45% both, commit 7ddcfb2)."""
    a = L.directional_label(frame, horizon=H, min_gain=L.PRODUCTION_THRESHOLD)
    b = L.directional_label(frame, horizon=H, min_gain=L.ALTERNATIVE_THRESHOLD)
    assert not np.array_equal(a.values, b.values), "the two thresholds produced identical labels"
    assert b.values.sum() <= a.values.sum(), "the stricter threshold produced MORE signals"
