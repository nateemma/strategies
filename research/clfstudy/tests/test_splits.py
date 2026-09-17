"""Tests for research/clfstudy/splits.py (T007/T008).

Covers: window disjointness/ordering, the embargo geometry, val-slice geometry,
`assign_roles` never double-assigning a row, `included_pairs` history cutoff, `validate`
raising on a hand-broken window, and save/load round-tripping.
"""
from __future__ import annotations

import dataclasses

import numpy as np
import pandas as pd
import pytest

from research.clfstudy import EMBARGO_BARS, HORIZON, MIN_HISTORY_DAYS, VAL_FRACTION
from research.clfstudy import splits as S


def test_build_v1_windows_disjoint_and_ordered():
    ss = S.build("v1")
    S.validate(ss)  # must not raise

    ordered = sorted(ss.windows, key=lambda w: w.test_start)
    assert [w.window_id for w in ordered] == ["A", "B", "C"]
    for prev, cur in zip(ordered, ordered[1:]):
        assert prev.test_end < cur.test_start


def test_train_end_plus_embargo_le_test_start():
    ss = S.build("v1")
    embargo = pd.Timedelta(minutes=15) * EMBARGO_BARS
    for w in ss.windows:
        assert w.train_end + embargo <= w.test_start


def test_val_slice_is_last_val_fraction_of_train_span_and_embargo_separated():
    ss = S.build("v1")
    embargo = pd.Timedelta(minutes=15) * EMBARGO_BARS
    for w in ss.windows:
        expected_val_start = w.train_end - VAL_FRACTION * (w.train_end - w.train_start)
        assert w.val_start == expected_val_start
        # val is separated from the training-only rows by a full embargo
        assert (w.val_start - embargo) > w.train_start
        assert w.val_start < w.train_end


def test_at_least_one_distant_window():
    ss = S.build("v1")
    assert any(w.distant for w in ss.windows)
    assert ss.windows[0].window_id == "A" and ss.windows[0].distant


def test_assign_roles_never_gives_two_roles_and_covers_embargo_gaps():
    ss = S.build("v1")
    w = ss.windows[0]
    embargo = pd.Timedelta(minutes=15) * EMBARGO_BARS

    dates = pd.date_range(
        w.train_start - pd.Timedelta(days=1),
        w.test_end + pd.Timedelta(days=1),
        freq="15min",
        tz="UTC",
    )
    roles = S.assign_roles(dates, w)

    assert set(np.unique(roles)) <= {"train", "embargo", "val", "test", "none"}
    assert len(roles) == len(dates)

    # every row gets exactly one role (roles is a 1-D array of scalars => trivially true,
    # but assert the role covers the full timeline with no unintended holes)
    counts = {r: int((roles == r).sum()) for r in ("train", "embargo", "val", "test", "none")}
    assert sum(counts.values()) == len(dates)
    assert counts["train"] > 0 and counts["val"] > 0 and counts["test"] > 0 and counts["embargo"] > 0

    # spot-check the two embargo gaps are present
    pre_val_embargo_start = w.val_start - embargo
    idx_in_pre_val_embargo = (dates >= pre_val_embargo_start) & (dates < w.val_start)
    assert np.all(roles[idx_in_pre_val_embargo] == "embargo")

    idx_in_pre_test_embargo = (dates >= w.train_end) & (dates < w.test_start)
    assert np.all(roles[idx_in_pre_test_embargo] == "embargo")

    idx_in_test = (dates >= w.test_start) & (dates <= w.test_end)
    assert np.all(roles[idx_in_test] == "test")

    idx_none = dates < w.train_start
    assert np.all(roles[idx_none] == "none")


def test_included_pairs_history_cutoff():
    ss = S.build("v1")
    w = ss.windows[0]

    short_history_date = w.val_start - pd.Timedelta(days=300)
    long_history_date = w.val_start - pd.Timedelta(days=400)
    assert MIN_HISTORY_DAYS == 365  # sanity: this test assumes the pinned constant

    first_dates = {"SHORT/USDT": short_history_date, "LONG/USDT": long_history_date}
    included = S.included_pairs(w, first_dates)

    assert "SHORT/USDT" not in included
    assert "LONG/USDT" in included


def test_validate_raises_on_hand_broken_window():
    ss = S.build("v1")
    broken_window = dataclasses.replace(ss.windows[0], train_end=ss.windows[0].test_start + pd.Timedelta(days=1))
    broken_ss = dataclasses.replace(ss, windows=[broken_window] + list(ss.windows[1:]))

    with pytest.raises(ValueError):
        S.validate(broken_ss)


def test_validate_raises_on_overlapping_windows():
    ss = S.build("v1")
    overlapping_b = dataclasses.replace(ss.windows[1], test_start=ss.windows[0].test_end - pd.Timedelta(days=1))
    broken_ss = dataclasses.replace(ss, windows=[ss.windows[0], overlapping_b, ss.windows[2]])

    with pytest.raises(ValueError):
        S.validate(broken_ss)


def test_save_load_round_trip(tmp_path):
    ss = S.build("v1")
    path = tmp_path / "v1.json"
    S.save(ss, path)
    loaded = S.load(path)

    assert loaded.split_set == ss.split_set
    assert loaded.embargo_bars == ss.embargo_bars
    assert loaded.val_fraction == ss.val_fraction
    assert loaded.min_history_days == ss.min_history_days
    assert loaded.timeframe == ss.timeframe
    assert loaded.pairs == ss.pairs
    assert len(loaded.windows) == len(ss.windows)
    for w_orig, w_loaded in zip(ss.windows, loaded.windows):
        assert w_orig == w_loaded
