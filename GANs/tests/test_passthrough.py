"""
Tests for ``GANs.passthrough.swap_passthrough_columns``.

The helper has three input shapes (2D ndarray, 3D ndarray, DataFrame)
and one invariant per shape: passthrough columns end up as exact
copies of values from the real pool, while non-passthrough columns
are left alone.

We also pin down behaviour that's easy to break:
  * Empty / None columns → no-op, no copy made.
  * Whole-sequence copy for 3D — a single passthrough column should
    have its full ``T``-length trace copied from one source row, not
    stitched together per-timestep from random sources (would destroy
    temporal structure).
  * Reproducibility with a seeded rng.
  * Resolved-column-indices helper drops missing names silently.
  * End-to-end through ``balance_multi_task`` — passthrough columns
    in the augmented output match real values, non-passthrough don't.
"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path
from typing import Dict, List
from unittest.mock import MagicMock

import numpy as np
import pandas as pd

STRATEGIES_ROOT = str(Path(__file__).parent.parent.parent)
if STRATEGIES_ROOT not in sys.path:
    sys.path.insert(0, STRATEGIES_ROOT)

# Stub freqtrade so the GANs module imports cleanly.
_ft_stub = MagicMock()
sys.modules.setdefault("freqtrade", _ft_stub)
sys.modules.setdefault("freqtrade.persistence", _ft_stub)
sys.modules.setdefault("freqtrade.strategy", _ft_stub)

from GANs.passthrough import (  # noqa: E402
    swap_passthrough_columns,
    resolve_column_indices,
)
from GANs.balance import balance_multi_task  # noqa: E402


def _one_hot(idx: np.ndarray, n: int) -> np.ndarray:
    return np.eye(n, dtype=np.float32)[idx.astype(int)]


class TestSwap2DArray(unittest.TestCase):
    """2D ndarray: per-row copy of selected columns."""

    def test_swap_replaces_named_columns_only(self) -> None:
        rng = np.random.default_rng(0)
        n_real, n_synth, F = 50, 30, 6
        real = rng.normal(size=(n_real, F)).astype(np.float32)
        synth = rng.normal(size=(n_synth, F)).astype(np.float32)
        synth_pre = synth.copy()

        out = swap_passthrough_columns(synth, real, columns=[0, 2], rng=rng)

        # Original synth not mutated.
        np.testing.assert_array_equal(synth, synth_pre)

        # Non-passthrough columns (1, 3, 4, 5) are untouched.
        for c in (1, 3, 4, 5):
            np.testing.assert_array_equal(out[:, c], synth[:, c])

        # Each row of out's passthrough columns must equal SOME real row's
        # passthrough columns (since we sampled with replacement).  Verify
        # by checking that every (out[i, 0], out[i, 2]) pair appears in
        # the real data as a single-row pair.
        real_pairs = {tuple(r) for r in real[:, [0, 2]].tolist()}
        for i in range(n_synth):
            self.assertIn(
                tuple(out[i, [0, 2]].tolist()),
                real_pairs,
                f"row {i}: passthrough values not found in any real row "
                f"— means the swap stitched columns from different sources",
            )

    def test_empty_columns_no_op(self) -> None:
        rng = np.random.default_rng(1)
        synth = rng.normal(size=(10, 4)).astype(np.float32)
        real = rng.normal(size=(20, 4)).astype(np.float32)

        out = swap_passthrough_columns(synth, real, columns=[], rng=rng)
        self.assertIs(out, synth)  # no-op short-circuits before copy


class TestSwap3DArray(unittest.TestCase):
    """3D ndarray: WHOLE-SEQUENCE copy preserves within-column temporal
    structure of passthrough features."""

    def test_whole_sequence_copy_for_3d(self) -> None:
        rng = np.random.default_rng(2)
        n_real, T, F = 40, 16, 5
        n_synth = 25

        # Make real data temporally smooth so a stitched-from-different-
        # rows copy would clearly differ from a whole-sequence copy:
        # within each row, feature 0 follows a deterministic ramp.
        real = rng.normal(size=(n_real, T, F)).astype(np.float32)
        for i in range(n_real):
            real[i, :, 0] = np.linspace(i, i + 0.15 * (T - 1), T)
        synth = rng.normal(size=(n_synth, T, F)).astype(np.float32)

        out = swap_passthrough_columns(synth, real, columns=[0], rng=rng)

        # Every output row's column 0 must match SOME real row's column 0
        # in full (T,) — not a stitch of multiple sources.
        for i in range(n_synth):
            seq = out[i, :, 0]
            # Find the real row whose column 0 equals seq exactly.
            matches = np.all(real[:, :, 0] == seq, axis=1)
            self.assertTrue(
                matches.any(),
                f"synth row {i}'s column-0 trace doesn't match any real "
                f"row — whole-sequence copy invariant broken",
            )

        # Non-passthrough columns (1..4) are untouched.
        for c in range(1, F):
            np.testing.assert_array_equal(out[:, :, c], synth[:, :, c])


class TestSwapDataFrame(unittest.TestCase):
    """DataFrame: addressable by column name."""

    def test_swap_by_name(self) -> None:
        rng = np.random.default_rng(3)
        cols = ["dow_sin", "dow_cos", "rsi", "ema"]
        real = pd.DataFrame(
            rng.normal(size=(40, 4)),
            columns=cols,
        )
        synth = pd.DataFrame(
            rng.normal(size=(25, 4)),
            columns=cols,
        )
        synth_pre = synth.copy()

        out = swap_passthrough_columns(
            synth, real,
            columns=["dow_sin", "dow_cos"],
            rng=rng,
        )

        # Original not mutated.
        pd.testing.assert_frame_equal(synth, synth_pre)
        # Non-passthrough columns identical to original synth.
        pd.testing.assert_series_equal(
            out["rsi"], synth["rsi"], check_names=False,
        )
        pd.testing.assert_series_equal(
            out["ema"], synth["ema"], check_names=False,
        )
        # Passthrough columns are now drawn from the real pool — every
        # (dow_sin, dow_cos) pair in `out` must appear in `real`.
        real_pairs = {
            (row.dow_sin, row.dow_cos) for row in real.itertuples()
        }
        for row in out.itertuples():
            self.assertIn(
                (row.dow_sin, row.dow_cos),
                real_pairs,
                "passthrough pair not found in real data — sampling "
                "is breaking inter-column dependencies",
            )

    def test_dataframe_int_columns_resolve_via_position(self) -> None:
        """Integer column refs on DataFrames map to positions."""
        rng = np.random.default_rng(4)
        cols = ["a", "b", "c"]
        real = pd.DataFrame(rng.normal(size=(30, 3)), columns=cols)
        synth = pd.DataFrame(rng.normal(size=(20, 3)), columns=cols)

        out = swap_passthrough_columns(synth, real, columns=[0, 2], rng=rng)
        # Column 'b' (index 1) should be untouched.
        pd.testing.assert_series_equal(
            out["b"], synth["b"], check_names=False,
        )


class TestSeededReproducibility(unittest.TestCase):
    def test_seeded_rng_reproducible(self) -> None:
        n = 100
        real = np.arange(n * 5, dtype=np.float32).reshape(n, 5)
        synth = np.zeros((50, 5), dtype=np.float32)

        out_a = swap_passthrough_columns(
            synth, real, columns=[0, 1],
            rng=np.random.default_rng(42),
        )
        out_b = swap_passthrough_columns(
            synth, real, columns=[0, 1],
            rng=np.random.default_rng(42),
        )
        np.testing.assert_array_equal(out_a, out_b)


class TestResolveColumnIndices(unittest.TestCase):
    def test_drops_missing_names_silently(self) -> None:
        feature_names = ["a", "b", "c", "d"]
        indices = resolve_column_indices(
            ["a", "missing", "c", "also_missing"], feature_names,
        )
        # "a" → 0, "c" → 2; missing entries silently dropped.
        self.assertEqual(indices, [0, 2])

    def test_empty_input_returns_empty(self) -> None:
        self.assertEqual(resolve_column_indices([], ["a", "b"]), [])

    def test_all_missing_returns_empty(self) -> None:
        self.assertEqual(
            resolve_column_indices(["x", "y"], ["a", "b"]),
            [],
        )


class TestErrorPaths(unittest.TestCase):
    def test_feature_count_mismatch_raises(self) -> None:
        synth = np.zeros((5, 4), dtype=np.float32)
        real = np.zeros((5, 6), dtype=np.float32)  # different feature count
        with self.assertRaises(ValueError):
            swap_passthrough_columns(synth, real, columns=[0])

    def test_3d_time_dim_mismatch_raises(self) -> None:
        synth = np.zeros((5, 16, 4), dtype=np.float32)
        real = np.zeros((5, 8, 4), dtype=np.float32)
        with self.assertRaises(ValueError):
            swap_passthrough_columns(synth, real, columns=[0])


# ---------------------------------------------------------------------------
# End-to-end through balance_multi_task
# ---------------------------------------------------------------------------


class _MockMTGenerator:
    """Returns batches whose passthrough columns are deliberately wrong
    (constant value).  The swap should overwrite them with real values."""

    def __init__(
        self,
        sample_shape,
        return_dataframe: bool = False,
        bad_value: float = -999.0,
        column_names: List[str] = None,
    ):
        self.sample_shape = sample_shape
        self.return_dataframe = return_dataframe
        self.bad_value = bad_value
        self.column_names = column_names

    def generate(self, n: int, *, task_labels: Dict[str, np.ndarray]):
        if self.return_dataframe:
            cols = self.column_names or [
                f"f{i}" for i in range(self.sample_shape[-1])
            ]
            data = pd.DataFrame(
                np.full((n, self.sample_shape[-1]), self.bad_value, dtype=np.float32),
                columns=cols,
            )
        else:
            data = np.full((n, *self.sample_shape[1:]), self.bad_value, dtype=np.float32)
        return data, task_labels


class TestBalanceMultiTaskPassthroughIntegration(unittest.TestCase):
    """End-to-end: balance_multi_task with passthrough_columns should
    overwrite the GAN's bad output with real values for those columns."""

    def test_3d_passthrough_overwrites_synthetic(self) -> None:
        rng = np.random.default_rng(7)
        n = 200
        T, F = 4, 5

        # Real data: each feature drawn from N(0, 1).  Passthrough
        # column 0 has a recognisable distribution we can verify.
        data = rng.normal(size=(n, T, F)).astype(np.float32)
        labels = {
            "trading": _one_hot(
                np.concatenate([np.zeros(60, int), np.ones(120, int), np.full(20, 2, int)]),
                3,
            ),
        }

        iface = _MockMTGenerator(sample_shape=(0, T, F), bad_value=-999.0)

        out_data, _ = balance_multi_task(
            iface, data, labels,
            target_ratios=0.5,
            batch_size=80, max_rounds=20,
            log=lambda *_: None, debug_log=lambda *_: None,
            passthrough_columns=[0, 2],
        )

        # The GAN returned -999.0 for every column, but passthrough
        # columns 0 and 2 in the *new* rows should now be real values
        # (not -999).  The original n=200 rows pass through unchanged.
        new_rows = out_data[n:]
        if new_rows.size:
            # Passthrough columns: every value should be in the real
            # data's range (no -999 leakage).
            self.assertFalse(
                (new_rows[:, :, 0] == -999.0).any(),
                "passthrough column 0 still contains the GAN's bad value",
            )
            self.assertFalse(
                (new_rows[:, :, 2] == -999.0).any(),
                "passthrough column 2 still contains the GAN's bad value",
            )
            # Non-passthrough columns: still the GAN's bad value.
            self.assertTrue(
                (new_rows[:, :, 1] == -999.0).all(),
                "non-passthrough column 1 was unexpectedly modified",
            )
            # Whole-sequence copy: each new row's column-0 trace must
            # equal some real row's column-0 trace exactly.
            for i in range(min(new_rows.shape[0], 10)):
                seq = new_rows[i, :, 0]
                matches = np.all(data[:, :, 0] == seq, axis=1)
                self.assertTrue(
                    matches.any(),
                    f"new row {i}: column-0 trace doesn't match any real row",
                )

    def test_dataframe_passthrough_by_name(self) -> None:
        rng = np.random.default_rng(8)
        n = 150
        cols = ["dow_sin", "dow_cos", "rsi", "ema"]
        data = pd.DataFrame(rng.normal(size=(n, 4)), columns=cols)
        labels = {
            "trading": _one_hot(
                np.concatenate([np.zeros(40, int), np.ones(90, int), np.full(20, 2, int)]),
                3,
            ),
        }

        iface = _MockMTGenerator(
            sample_shape=(0, 4), return_dataframe=True, bad_value=-999.0,
            column_names=cols,
        )

        out_data, _ = balance_multi_task(
            iface, data, labels,
            target_ratios=0.6,
            batch_size=60, max_rounds=20,
            log=lambda *_: None, debug_log=lambda *_: None,
            passthrough_columns=["dow_sin", "dow_cos"],
        )

        # New rows: passthrough columns should NOT be -999, others SHOULD be.
        new_rows = out_data.iloc[n:]
        self.assertGreater(len(new_rows), 0)
        self.assertFalse(
            (new_rows["dow_sin"] == -999.0).any(),
            "dow_sin still has bad GAN value",
        )
        self.assertFalse(
            (new_rows["dow_cos"] == -999.0).any(),
            "dow_cos still has bad GAN value",
        )
        self.assertTrue(
            (new_rows["rsi"] == -999.0).all(),
            "rsi was unexpectedly modified",
        )
        self.assertTrue(
            (new_rows["ema"] == -999.0).all(),
            "ema was unexpectedly modified",
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
