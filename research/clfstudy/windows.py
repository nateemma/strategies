# pragma pylint: disable=C0103, C0114, C0115, C0116, C0301, C0303, C0411, C0413
# pylint: disable=import-error
# flake8: noqa: F401, E402

"""
Per-pair windowing + role selection -> train/val/test tensors (research R5).

Windows are built on each pair's continuous, already-scaled frame BEFORE any role
selection, so a test row's context (the `seq_len` bars ending at that row) is drawn from
real preceding bars (train/embargo), never zero padding — the only zero-padded rows are the
first `seq_len - 1` of the whole frame (`DataframeUtils.df_to_tensor(..., method=3)`'s
leading pad), which never coincide with a role-selected row once the training span and
embargo are longer than `seq_len` (true for every window in split set v1).

Memory: at production sizes (~1.5M rows x seq_len=16 x nf=24, float32) the train tensor is
~1.5e6 * 16 * 24 * 4 bytes ~= 2.3 GB — fine for a single run.
"""

from __future__ import annotations

from typing import Dict, List

import numpy as np
import pandas as pd

from utils.DataframeUtils import DataframeUtils


def _tensor_for_pair(frame: pd.DataFrame, cols: List[str], seq_len: int) -> np.ndarray:
    du = DataframeUtils()
    raw = du.df_to_tensor(frame[cols], seq_len, method=3)
    X = np.asarray(raw, dtype=np.float32)

    # Guarantee (module docstring / research R4 #5): every row from seq_len-1 onward is
    # built entirely from real (non-padded) bars, so it must not be all-zero.
    if len(X) >= seq_len:
        real_rows = X[seq_len - 1:]
        all_zero = np.all(real_rows == 0.0, axis=(1, 2))
        if all_zero.any():
            bad = int(np.argmax(all_zero)) + (seq_len - 1)
            raise AssertionError(
                f"row {bad} has an all-zero context window past the padding region "
                f"(seq_len={seq_len}); this would indicate zero-padding leaking into "
                f"a real row"
            )
    return X


def make_tensors(
    frames: Dict[str, pd.DataFrame],
    roles: Dict[str, np.ndarray],
    cols: List[str],
    seq_len: int,
    stride: int = 1,
    label_col: str = "label",
) -> dict:
    """Build train/val/test tensors from per-pair, already-scaled, labelled frames.

    Args:
        frames: pair -> continuous DataFrame with `cols`, `label_col` and a `date` column.
        roles: pair -> array of role strings (from `splits.assign_roles`), same length/order
            as `frames[pair]`.
        cols: feature columns to build the tensor from (already scaled).
        seq_len: window length.
        stride: keep every `stride`-th row among the train-role rows (val/test unaffected).
        label_col: integer class-label column name.

    Returns:
        {"train": (X, y1h, y_int), "val": (X, y1h, y_int), "test": (X, y_int, dates, pair_ids)}
        — float32 numpy arrays (dates as datetime64[ns, UTC], pair_ids as an object array).
    """
    if stride < 1:
        raise ValueError(f"stride must be >= 1, got {stride}")

    du = DataframeUtils()

    train_X, train_y = [], []
    val_X, val_y = [], []
    test_X, test_y, test_dates, test_pairs = [], [], [], []

    for pair, frame in frames.items():
        role = np.asarray(roles[pair])
        if len(role) != len(frame):
            raise ValueError(f"{pair}: roles length {len(role)} != frame length {len(frame)}")

        X_all = _tensor_for_pair(frame, cols, seq_len)
        labels_all = frame[label_col].to_numpy()
        dates_all = pd.to_datetime(frame["date"], utc=True).to_numpy()

        train_idx = np.where(role == "train")[0][::stride]
        val_idx = np.where(role == "val")[0]
        test_idx = np.where(role == "test")[0]

        if len(train_idx):
            train_X.append(X_all[train_idx])
            train_y.append(labels_all[train_idx])
        if len(val_idx):
            val_X.append(X_all[val_idx])
            val_y.append(labels_all[val_idx])
        if len(test_idx):
            test_X.append(X_all[test_idx])
            test_y.append(labels_all[test_idx])
            test_dates.append(dates_all[test_idx])
            test_pairs.append(np.full(len(test_idx), pair, dtype=object))

    def _cat(parts, empty_shape):
        return np.concatenate(parts, axis=0) if parts else np.empty(empty_shape, dtype=np.float32)

    nf = len(cols)
    train_X_arr = _cat(train_X, (0, seq_len, nf))
    val_X_arr = _cat(val_X, (0, seq_len, nf))
    test_X_arr = _cat(test_X, (0, seq_len, nf))

    train_y_arr = np.concatenate(train_y, axis=0).astype(np.int64) if train_y else np.empty((0,), dtype=np.int64)
    val_y_arr = np.concatenate(val_y, axis=0).astype(np.int64) if val_y else np.empty((0,), dtype=np.int64)
    test_y_arr = np.concatenate(test_y, axis=0).astype(np.int64) if test_y else np.empty((0,), dtype=np.int64)

    test_dates_arr = (
        np.concatenate(test_dates, axis=0) if test_dates else np.empty((0,), dtype="datetime64[ns]")
    )
    test_pairs_arr = (
        np.concatenate(test_pairs, axis=0) if test_pairs else np.empty((0,), dtype=object)
    )

    train_y1h = np.asarray(du.one_hot_encode(train_y_arr, 3), dtype=np.float32)
    val_y1h = np.asarray(du.one_hot_encode(val_y_arr, 3), dtype=np.float32)

    return {
        "train": (train_X_arr, train_y1h, train_y_arr),
        "val": (val_X_arr, val_y1h, val_y_arr),
        "test": (test_X_arr, test_y_arr, test_dates_arr, test_pairs_arr),
    }
