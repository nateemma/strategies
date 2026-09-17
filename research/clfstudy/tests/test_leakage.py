"""Leakage regression tests for splits.py / features.py / windows.py (T011, spec SC-003).

Builds a small synthetic 2-pair dataset with a toy `Window`, fits the scaler and class
weights on the resulting `train`-role rows only, and proves that arbitrarily corrupting
every `test`/`embargo`-role row (features *1000 and sign-flipped, label perturbed) leaves
the fitted scaler, the class weights and the train/val tensors byte-for-byte identical.

`seq_len=1` is used for this bit-identical check so the assertion isolates the *role
selection* boundary (the actual leakage risk: fitting/training on a test or embargo row)
from the orthogonal, intentional property that a later row's feature *context* legitimately
reaches back over real historical bars — including, for the first few `val` rows, bars in
the pre-val embargo zone. That reach-back is causal (it only uses the past) and is exactly
what the embargo purges against (label-horizon overlap, not feature history), so it is
correct behaviour, not a leakage bug; it is exercised separately (with seq_len>1) by
`test_test_window_context_is_causal_and_non_zero` below, which only inspects the test slice.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from research.clfstudy import splits as S
from research.clfstudy import features as F
from research.clfstudy import windows as W

N_ROWS_PER_PAIR = 3000
PAIRS = ["FOO/USDT", "BAR/USDT"]
COLS = ["f1", "f2", "f3", "f4", "f5", "f6"]
NEEDS_NORM_COLS = {"f1", "f2", "f3"}  # 3 of the 6 "need" normalisation, 3 pass through


def _make_frame(seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2024-01-01", periods=N_ROWS_PER_PAIR, freq="15min", tz="UTC")
    data = {c: rng.normal(size=N_ROWS_PER_PAIR) for c in COLS}
    data["date"] = dates
    data["label"] = rng.integers(0, 3, N_ROWS_PER_PAIR)
    return pd.DataFrame(data)


@pytest.fixture
def frames():
    return {pair: _make_frame(seed) for seed, pair in enumerate(PAIRS)}


@pytest.fixture
def toy_window(frames):
    dates = frames[PAIRS[0]]["date"]
    train_start = dates.iloc[0]
    test_start = dates.iloc[2400]
    test_end = dates.iloc[2600]
    return S._make_window("T", train_start, test_start, test_end, distant=False)


@pytest.fixture
def roles(frames, toy_window):
    return {pair: S.assign_roles(frame["date"], toy_window) for pair, frame in frames.items()}


@pytest.fixture(autouse=True)
def patch_needs_norm(monkeypatch):
    monkeypatch.setattr(F, "needs_norm", lambda cols: [c for c in cols if c in NEEDS_NORM_COLS])


def _fit_and_build(frames, roles, seq_len):
    train_parts = [
        frame.loc[roles[pair] == "train", COLS + ["label"]] for pair, frame in frames.items()
    ]
    train_df = pd.concat(train_parts, axis=0)

    scaler = F.fit_scaler(train_df, COLS)
    class_w = F.class_weights(train_df["label"].to_numpy())

    scaled_frames = {}
    for pair, frame in frames.items():
        scaled = F.apply_scaler(frame, scaler, COLS)
        scaled["date"] = frame["date"].to_numpy()
        scaled["label"] = frame["label"].to_numpy()
        scaled_frames[pair] = scaled

    tensors = W.make_tensors(scaled_frames, roles, COLS, seq_len=seq_len)
    return scaler, class_w, tensors


def _perturb(frames, roles):
    """Return a deep copy with every test/embargo-role row corrupted (features x -1000,
    label mapped to a value no legitimate label column would ever contain)."""
    out = {}
    for pair, frame in frames.items():
        f = frame.copy()
        mask = np.isin(roles[pair], ["test", "embargo"])
        f.loc[mask, COLS] = f.loc[mask, COLS] * -1000.0
        f.loc[mask, "label"] = -1000 - f.loc[mask, "label"]
        out[pair] = f
    return out


def test_scaler_and_class_weights_ignore_test_and_embargo_rows(frames, roles):
    scaler, class_w, tensors = _fit_and_build(frames, roles, seq_len=1)

    perturbed = _perturb(frames, roles)
    scaler2, class_w2, tensors2 = _fit_and_build(perturbed, roles, seq_len=1)

    assert np.array_equal(scaler.center_, scaler2.center_)
    assert np.array_equal(scaler.scale_, scaler2.scale_)
    assert class_w == class_w2

    train_X, train_y1h, train_yint = tensors["train"]
    train_X2, train_y1h2, train_yint2 = tensors2["train"]
    assert np.array_equal(train_X, train_X2)
    assert np.array_equal(train_y1h, train_y1h2)
    assert np.array_equal(train_yint, train_yint2)

    val_X, val_y1h, val_yint = tensors["val"]
    val_X2, val_y1h2, val_yint2 = tensors2["val"]
    assert np.array_equal(val_X, val_X2)
    assert np.array_equal(val_y1h, val_y1h2)
    assert np.array_equal(val_yint, val_yint2)

    # sanity: the corruption actually reached the test slice (otherwise this test would
    # pass vacuously)
    assert len(tensors["train"][2]) > 0 and len(tensors["val"][2]) > 0
    assert not np.array_equal(tensors["test"][1], tensors2["test"][1])


def test_test_window_context_is_causal_and_non_zero(frames, roles, toy_window):
    seq_len = 4
    _, _, tensors = _fit_and_build(frames, roles, seq_len=seq_len)
    test_X, test_yint, test_dates, test_pairs = tensors["test"]

    assert len(test_dates) > 0

    for pair, frame in frames.items():
        row_idx = frame.index[frame["date"] == toy_window.test_start][0]
        assert row_idx >= seq_len - 1  # plenty of real history precedes test_start

        context_dates = frame["date"].iloc[row_idx - (seq_len - 1): row_idx]
        assert (context_dates < toy_window.test_start).all()

        first_test_pos = np.where(test_pairs == pair)[0][0]
        first_test_window = test_X[first_test_pos]
        assert not np.all(first_test_window == 0.0)
