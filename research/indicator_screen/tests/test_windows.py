"""037 T004: fold windows, E0, purge, training-row masks and the late lock."""
from __future__ import annotations

import subprocess

import numpy as np
import pandas as pd
import pytest

from research.indicator_screen import windows as W


def test_fold_spans_match_the_plan():
    f = W.folds()
    assert [w.id for w in f] == [f"f{i:02d}" for i in range(15)]
    assert f[0].start == pd.Timestamp("2023-01-01", tz="UTC") and f[0].end == pd.Timestamp("2023-04-01", tz="UTC")
    assert f[-1].end == pd.Timestamp("2026-09-07", tz="UTC")
    assert [w.half for w in f] == ["early"] * 8 + ["late"] * 7
    assert all(w.train_end == w.start for w in f)          # anchored, train ends at test start
    assert all(a.end <= b.start for a, b in zip(f, f[1:]))


def test_e0_has_no_train_span_and_ends_before_the_folds():
    e = W.e0()
    assert e.train_start is None and e.half == "e0"
    assert e.end <= W.folds()[0].start
    with pytest.raises(ValueError):
        W.train_rows_mask(pd.date_range("2021-01-01", periods=10, freq="15min", tz="UTC"), e, "label:buy")


def test_purge_by_target():
    assert W.purge("label:buy") == 48 and W.purge("B-btc:288") == 288 and W.purge("A2:672") == 672


def test_purge_excludes_the_last_n_bars_before_test_start():
    f = W.folds()[0]
    d = pd.date_range(pd.Timestamp("2021-01-01", tz="UTC"), f.start, freq="15min", tz="UTC", inclusive="left")
    m = W.train_rows_mask(d, f, "C-pair:288")
    assert not m[:W.WARMUP_BARS].any()
    last = d[m][-1]
    assert last == f.start - 289 * W.BAR                # 288 bars purged
    assert m[W.WARMUP_BARS:-288].all() and not m[-288:].any()


def _git(repo, *a):
    subprocess.run(["git", "-C", str(repo), *a], check=True, capture_output=True)


def test_late_lock_refuses_missing_uncommitted_and_modified(tmp_path):
    _git(tmp_path, "init", "-q")
    _git(tmp_path, "config", "user.email", "t@t")
    _git(tmp_path, "config", "user.name", "t")
    p = tmp_path / "preregistration.json"
    with pytest.raises(W.LateLockError, match="missing"):
        W.late_lock(p, tmp_path)
    p.write_text("{}")
    with pytest.raises(W.LateLockError, match="not committed"):
        W.late_lock(p, tmp_path)
    _git(tmp_path, "add", p.name)
    _git(tmp_path, "commit", "-qm", "lock")
    assert W.late_lock(p, tmp_path)
    p.write_text('{"x": 1}')
    with pytest.raises(W.LateLockError, match="uncommitted"):
        W.late_lock(p, tmp_path)


def test_window_mask_is_half_open():
    w = W.folds()[0]
    d = pd.DatetimeIndex([w.start - W.BAR, w.start, w.end - W.BAR, w.end])
    assert W.Window.mask(w, d).tolist() == [False, True, True, False]
