"""
Unit tests for ``GANs.mlx_ctab_helpers``.

These tests are pure-numpy and do NOT require MLX — the helpers module
is intentionally framework-free so it can be exercised on any machine.

Run from the freqtrade root:
    python -m pytest user_data/strategies/GANs/tests/test_mlx_ctab_helpers.py -v
"""

from __future__ import annotations

import os
import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
STRATEGIES_ROOT = str(Path(__file__).parent.parent.parent)
if STRATEGIES_ROOT not in sys.path:
    sys.path.insert(0, STRATEGIES_ROOT)

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

from GANs.mlx_ctab_helpers import (  # noqa: E402
    BestCheckpointTracker,
    DivergenceMonitor,
    EarlyStopping,
    ReduceLROnPlateau,
    evaluate_with_dataframes,
    fit_vgm_columns,
    inverse_transform_vgm,
    transform_vgm,
)


# ---------------------------------------------------------------------------
# Section 1 — VGM helpers
# ---------------------------------------------------------------------------


def _synthetic_df(n: int = 200, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    return pd.DataFrame(
        {
            "uniform": rng.uniform(-1, 1, n).astype(np.float32),
            "bimodal": np.concatenate([
                rng.normal(-2, 0.3, n // 2),
                rng.normal(2, 0.3, n - n // 2),
            ]).astype(np.float32),
            "constant": np.full(n, 5.0, dtype=np.float32),
        }
    )


class TestFitVGMColumns(unittest.TestCase):

    def test_returns_three_outputs(self):
        df = _synthetic_df()
        vgm, info, cinfo = fit_vgm_columns(df)
        self.assertEqual(set(vgm.keys()), set(df.columns))
        self.assertEqual(set(info.keys()), set(df.columns))
        self.assertEqual(len(cinfo), len(df.columns))

    def test_continuous_info_first_dim_is_one(self):
        df = _synthetic_df()
        _, _, cinfo = fit_vgm_columns(df)
        for v, _m in cinfo:
            self.assertEqual(v, 1)

    def test_constant_column_collapses_to_single_mode(self):
        df = _synthetic_df()
        _, info, _ = fit_vgm_columns(df)
        self.assertEqual(info["constant"]["n_modes"], 1)

    def test_integer_columns_skip_vgm(self):
        df = _synthetic_df()
        vgm, info, cinfo = fit_vgm_columns(df, integer_columns=["uniform"])
        self.assertIsNone(vgm["uniform"])
        self.assertEqual(info["uniform"]["n_modes"], 0)
        # Column-order stable: uniform is column 0
        self.assertEqual(cinfo[0], (1, 0))


class TestTransformInverseRoundTrip(unittest.TestCase):

    def test_encoding_shape_matches_continuous_info(self):
        df = _synthetic_df()
        vgm, info, cinfo = fit_vgm_columns(df)
        encoded = transform_vgm(df, vgm, info, list(df.columns))
        expected_dim = sum(v + m for v, m in cinfo)
        self.assertEqual(encoded.shape, (len(df), expected_dim))
        self.assertEqual(encoded.dtype, np.float32)

    def test_round_trip_preserves_values_approximately(self):
        df = _synthetic_df()
        vgm, info, _ = fit_vgm_columns(df)
        encoded = transform_vgm(df, vgm, info, list(df.columns))
        decoded = inverse_transform_vgm(encoded, vgm, info, list(df.columns))
        # Each column should round-trip within (4 * std) of its real value.
        # Bimodal can be slightly noisier — use a generous tolerance.
        for col in df.columns:
            real = df[col].values
            recovered = decoded[col]
            mae = float(np.mean(np.abs(real - recovered)))
            self.assertLess(
                mae,
                max(real.std() * 0.5, 0.5),
                f"Column {col} round-trip MAE too large: {mae}",
            )

    def test_decoded_values_clipped_to_training_range(self):
        df = _synthetic_df()
        vgm, info, _ = fit_vgm_columns(df)
        encoded = transform_vgm(df, vgm, info, list(df.columns))
        decoded = inverse_transform_vgm(encoded, vgm, info, list(df.columns))
        for col in df.columns:
            self.assertGreaterEqual(decoded[col].min(), info[col]["min"])
            self.assertLessEqual(decoded[col].max(), info[col]["max"])


# ---------------------------------------------------------------------------
# Section 2 — Quality evaluation
# ---------------------------------------------------------------------------


class TestEvaluateWithDataframes(unittest.TestCase):

    def setUp(self):
        self.df_real = _synthetic_df(n=300, seed=1)
        # Generate "fake" data with similar distribution but small offsets
        self.df_fake = self.df_real.copy() + np.random.RandomState(2).normal(
            0, 0.05, self.df_real.shape
        )
        _, self.column_info, _ = fit_vgm_columns(self.df_real)
        self.continuous_columns = list(self.df_real.columns)
        self.column_order = list(self.df_real.columns)

    def test_returns_expected_keys(self):
        m = evaluate_with_dataframes(
            self.df_real,
            self.df_fake,
            continuous_columns=self.continuous_columns,
            column_info=self.column_info,
            column_order=self.column_order,
        )
        for key in ("diversity", "correlation", "statistics", "validity", "overall_score"):
            self.assertIn(key, m)

    def test_overall_score_keys(self):
        m = evaluate_with_dataframes(
            self.df_real,
            self.df_fake,
            continuous_columns=self.continuous_columns,
            column_info=self.column_info,
            column_order=self.column_order,
        )
        for key in (
            "diversity_score",
            "correlation_score",
            "statistical_score",
            "validity_score",
            "overall_quality",
        ):
            self.assertIn(key, m["overall_score"])

    def test_overall_quality_within_unit_interval(self):
        m = evaluate_with_dataframes(
            self.df_real,
            self.df_fake,
            continuous_columns=self.continuous_columns,
            column_info=self.column_info,
            column_order=self.column_order,
        )
        q = m["overall_score"]["overall_quality"]
        self.assertGreaterEqual(q, 0.0)
        self.assertLessEqual(q, 1.0)

    def test_identical_data_scores_high(self):
        m = evaluate_with_dataframes(
            self.df_real,
            self.df_real.copy(),
            continuous_columns=self.continuous_columns,
            column_info=self.column_info,
            column_order=self.column_order,
        )
        # Identical real-vs-real should score high (>= 0.85)
        self.assertGreater(m["overall_score"]["overall_quality"], 0.85)


# ---------------------------------------------------------------------------
# Section 3 — Callbacks
# ---------------------------------------------------------------------------


class _FakeOptimizer:
    """Minimal stand-in for an MLX Adam optimizer — exposes a mutable lr."""

    def __init__(self, lr: float):
        self.learning_rate = lr


class TestEarlyStopping(unittest.TestCase):

    def test_max_mode_does_not_stop_while_improving(self):
        es = EarlyStopping(patience=3, min_delta=0.0, mode="max")
        for score in (0.1, 0.2, 0.3, 0.4):
            self.assertFalse(es.update(score))

    def test_max_mode_stops_after_patience(self):
        es = EarlyStopping(patience=3, min_delta=0.0, mode="max")
        es.update(1.0)
        # Three flat epochs in a row — third call returns True.
        self.assertFalse(es.update(0.5))
        self.assertFalse(es.update(0.5))
        self.assertTrue(es.update(0.5))

    def test_min_mode_inverts_improvement(self):
        es = EarlyStopping(patience=2, min_delta=0.0, mode="min")
        es.update(1.0)
        self.assertFalse(es.update(0.5))  # improvement
        self.assertFalse(es.update(0.6))  # worse — counter=1
        self.assertTrue(es.update(0.7))   # worse — counter=2 → stop


class TestReduceLROnPlateau(unittest.TestCase):

    def test_lr_halves_after_plateau(self):
        opt = _FakeOptimizer(lr=1e-3)
        sched = ReduceLROnPlateau(patience=2, factor=0.5, min_lr=1e-8, mode="max")
        sched.update(1.0, [opt])
        sched.update(0.5, [opt])  # counter=1
        reduced, new_lr = sched.update(0.5, [opt])  # counter=2 → reduce
        self.assertTrue(reduced)
        self.assertAlmostEqual(new_lr, 5e-4, places=8)
        self.assertAlmostEqual(opt.learning_rate, 5e-4, places=8)

    def test_lr_does_not_drop_below_min(self):
        opt = _FakeOptimizer(lr=1e-6)
        sched = ReduceLROnPlateau(patience=1, factor=0.5, min_lr=1e-6, mode="max")
        sched.update(1.0, [opt])
        reduced, _ = sched.update(0.5, [opt])
        self.assertFalse(reduced)
        self.assertEqual(opt.learning_rate, 1e-6)

    def test_force_reduce_halves_immediately(self):
        opt = _FakeOptimizer(lr=1e-3)
        sched = ReduceLROnPlateau(patience=10, factor=0.5, min_lr=1e-8, mode="max")
        new_lr = sched.force_reduce([opt])
        self.assertAlmostEqual(new_lr, 5e-4, places=8)


class TestDivergenceMonitor(unittest.TestCase):

    def test_nan_loss_increments_counter(self):
        m = DivergenceMonitor(patience=2)
        self.assertFalse(m.update(float("nan"), 1.0, 0.0))
        self.assertTrue(m.update(float("nan"), 1.0, 0.0))

    def test_huge_loss_triggers_divergence(self):
        m = DivergenceMonitor(patience=1, d_ceil=10.0, g_ceil=10.0)
        # First call: count goes from 0 to 1. patience=1 so update returns True.
        self.assertTrue(m.update(50.0, 0.0, 1.0))

    def test_recovery_decrements_counter(self):
        m = DivergenceMonitor(patience=3, d_ceil=10.0, g_ceil=10.0, degrade_factor=2.0)
        m.update(50.0, 0.0, 1.0)  # divergent → count=1
        # Healthy step should decrement count back to 0.
        self.assertFalse(m.update(0.5, 0.5, 1.0))
        self.assertEqual(m.count, 0)


class TestBestCheckpointTracker(unittest.TestCase):

    class _FakeMod:
        """Stand-in MLX module — captures save_weights/load_weights paths."""

        def __init__(self, marker: str):
            self.marker = marker
            self.saved_to = None
            self.loaded_from = None

        def save_weights(self, path):
            self.saved_to = path
            with open(path, "w") as f:
                f.write(self.marker)

        def load_weights(self, path):
            self.loaded_from = path
            with open(path) as f:
                self.marker = f.read()

    def test_max_mode_tracks_improvement(self):
        ck = BestCheckpointTracker(mode="max", min_delta=0.0)
        gen = self._FakeMod("v0")
        critic = self._FakeMod("v0")
        self.assertTrue(ck.update(0.5, 0, gen, critic))
        self.assertFalse(ck.update(0.4, 1, gen, critic))
        self.assertTrue(ck.update(0.6, 2, gen, critic))
        self.assertEqual(ck.best_epoch, 2)
        ck.cleanup()

    def test_restore_loads_best_weights(self):
        ck = BestCheckpointTracker(mode="max", min_delta=0.0)
        gen = self._FakeMod("good")
        critic = self._FakeMod("good")
        ck.update(0.9, 0, gen, critic)
        # Mutate then restore — should bring back the marker.
        gen.marker = "corrupted"
        critic.marker = "corrupted"
        self.assertTrue(ck.restore(gen, critic))
        self.assertEqual(gen.marker, "good")
        self.assertEqual(critic.marker, "good")
        ck.cleanup()


if __name__ == "__main__":
    unittest.main(verbosity=2)
