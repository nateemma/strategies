"""
Tests for ``GANs.diagnostics.summarize_real_vs_synthetic``.

The diagnostic is a *reporter* — it never raises on bad data, it just
characterises it.  These tests pin down the three signals the
diagnostic exists to surface:

  * Identical real and synthetic distributions → near-zero shift,
    std_ratio ≈ 1.0, "ok" flag.
  * Mode-collapsed synthetic (zero variance per feature) → very low
    std_ratio_min, "MODE_COLLAPSE" in flag.
  * Mean-shifted synthetic (distribution moved by several σ) → large
    mean_shift_max, "OFF_DIST" in flag.

We also verify shape coercion: 2D ndarray, 3D ndarray (sequential), and
DataFrame inputs all flow through the same code path without crashing.
"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path
from typing import List
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

from GANs.diagnostics import summarize_real_vs_synthetic  # noqa: E402


def _one_hot(class_indices: np.ndarray, num_classes: int) -> np.ndarray:
    return np.eye(num_classes, dtype=np.float32)[class_indices.astype(int)]


class _Recorder:
    """Captures log lines for assertion."""
    def __init__(self) -> None:
        self.lines: List[str] = []

    def __call__(self, *args) -> None:
        self.lines.append(" ".join(str(a) for a in args))

    def joined(self) -> str:
        return "\n".join(self.lines)


class TestDiagnosticsHappyPath(unittest.TestCase):
    """Identical real + synthetic — diagnostic should report ~ok."""

    def setUp(self) -> None:
        rng = np.random.default_rng(0)
        n = 600
        # 2D feature matrix.
        self.real = rng.normal(size=(n, 8)).astype(np.float32)
        self.real_labels = {
            "task": _one_hot(rng.integers(0, 3, size=n), 3),
        }
        # Synthetic = same distribution, fresh sample.
        self.synth = rng.normal(size=(n, 8)).astype(np.float32)
        self.synth_labels = {
            "task": _one_hot(rng.integers(0, 3, size=n), 3),
        }

    def test_reports_low_shift_and_unit_std_ratio(self) -> None:
        log = _Recorder()
        results = summarize_real_vs_synthetic(
            self.real, self.real_labels,
            self.synth, self.synth_labels,
            log=log,
        )
        self.assertIn("task", results)
        for c in range(3):
            r = results["task"][c]
            self.assertIsNotNone(r["mean_shift_max"])
            self.assertLess(r["mean_shift_max"], 0.5)
            self.assertGreater(r["std_ratio_mean"], 0.7)
            self.assertLess(r["std_ratio_mean"], 1.4)
            self.assertEqual(r["flag"], "ok")
        self.assertIn("real vs synthetic", log.joined())


class TestDiagnosticsModeCollapse(unittest.TestCase):
    """Synthetic samples have zero variance — should flag MODE_COLLAPSE."""

    def test_mode_collapse_flagged(self) -> None:
        rng = np.random.default_rng(1)
        n = 300
        real = rng.normal(size=(n, 5)).astype(np.float32)
        real_labels = {"task": _one_hot(np.full(n, 0), 2)}
        # Synthetic: every row is the same value (zero variance).
        synth = np.zeros((n, 5), dtype=np.float32)
        synth_labels = {"task": _one_hot(np.full(n, 0), 2)}

        log = _Recorder()
        results = summarize_real_vs_synthetic(
            real, real_labels, synth, synth_labels, log=log,
        )

        r = results["task"][0]
        self.assertLess(r["std_ratio_min"], 0.05)
        self.assertIn("MODE_COLLAPSE", r["flag"])


class TestDiagnosticsMeanShift(unittest.TestCase):
    """Synthetic samples shifted by ~5σ — should flag OFF_DIST."""

    def test_mean_shift_flagged(self) -> None:
        rng = np.random.default_rng(2)
        n = 400
        real = rng.normal(loc=0.0, scale=1.0, size=(n, 4)).astype(np.float32)
        real_labels = {"task": _one_hot(np.full(n, 1), 2)}
        # Synthetic: mean shifted to +5 (5σ off real distribution).
        synth = rng.normal(loc=5.0, scale=1.0, size=(n, 4)).astype(np.float32)
        synth_labels = {"task": _one_hot(np.full(n, 1), 2)}

        log = _Recorder()
        results = summarize_real_vs_synthetic(
            real, real_labels, synth, synth_labels, log=log,
        )

        r = results["task"][1]
        self.assertGreater(r["mean_shift_max"], 4.0)
        self.assertIn("OFF_DIST", r["flag"])


class TestDiagnosticsShapeCoercion(unittest.TestCase):
    """3D sequential and DataFrame inputs both flow through cleanly."""

    def test_3d_sequential_input(self) -> None:
        rng = np.random.default_rng(3)
        n, t, f = 200, 16, 6
        real = rng.normal(size=(n, t, f)).astype(np.float32)
        real_labels = {"task": _one_hot(rng.integers(0, 2, size=n), 2)}
        synth = rng.normal(size=(n, t, f)).astype(np.float32)
        synth_labels = {"task": _one_hot(rng.integers(0, 2, size=n), 2)}

        log = _Recorder()
        results = summarize_real_vs_synthetic(
            real, real_labels, synth, synth_labels, log=log,
        )

        # Both classes should appear with stats computed (after time-axis
        # unrolling, every class has 200 * 16 / 2 ≈ 1600 rows).
        for c in range(2):
            r = results["task"][c]
            self.assertGreater(r["n_real"], 100)
            self.assertGreater(r["n_synth"], 100)

    def test_dataframe_input(self) -> None:
        rng = np.random.default_rng(4)
        n = 200
        cols = [f"feat_{i}" for i in range(5)]
        real = pd.DataFrame(rng.normal(size=(n, 5)), columns=cols)
        real_labels = {"task": _one_hot(rng.integers(0, 2, size=n), 2)}
        synth = pd.DataFrame(rng.normal(size=(n, 5)), columns=cols)
        synth_labels = {"task": _one_hot(rng.integers(0, 2, size=n), 2)}

        log = _Recorder()
        results = summarize_real_vs_synthetic(
            real, real_labels, synth, synth_labels,
            feature_names=cols, log=log,
        )

        # Worst-feature drilldown should reference a real column name,
        # not f0..fF.
        joined = log.joined()
        self.assertTrue(
            any(name in joined for name in cols),
            f"Expected a real feature name in the diagnostic output; got:\n{joined}",
        )

    def test_list_of_synthetic_batches(self) -> None:
        """The balancer hands diagnostics ``synth_data`` as a list of
        per-round batches.  Verify the diagnostic accepts that shape."""
        rng = np.random.default_rng(5)
        real = rng.normal(size=(300, 4)).astype(np.float32)
        real_labels = {"task": _one_hot(rng.integers(0, 2, size=300), 2)}
        synth_batches = [
            rng.normal(size=(50, 4)).astype(np.float32) for _ in range(3)
        ]
        synth_labels_full = _one_hot(rng.integers(0, 2, size=150), 2)
        synth_labels = {"task": synth_labels_full}

        log = _Recorder()
        results = summarize_real_vs_synthetic(
            real, real_labels, synth_batches, synth_labels, log=log,
        )

        # 150 synth rows distributed across 2 classes — both should have
        # at least one row of synthetic data.
        for c in range(2):
            r = results["task"][c]
            self.assertGreater(r["n_synth"], 0)


class TestDiagnosticsEmptyInputs(unittest.TestCase):
    """No synthetic generated — diagnostic should bail with a message."""

    def test_no_synthetic_does_not_raise(self) -> None:
        rng = np.random.default_rng(6)
        real = rng.normal(size=(100, 3)).astype(np.float32)
        real_labels = {"task": _one_hot(np.full(100, 0), 2)}
        synth = np.empty((0, 3), dtype=np.float32)
        synth_labels = {"task": np.empty((0, 2), dtype=np.float32)}

        log = _Recorder()
        results = summarize_real_vs_synthetic(
            real, real_labels, synth, synth_labels, log=log,
        )
        self.assertEqual(results, {})
        self.assertIn("no synthetic samples", log.joined().lower())


# ---------------------------------------------------------------------------
# Joint-distribution (correlation) tests
# ---------------------------------------------------------------------------


class TestJointDiagnosticsHappyPath(unittest.TestCase):
    """Identical real and synthetic — |Δρ| should be near zero."""

    def test_identical_distribution_low_corr_diff(self) -> None:
        rng = np.random.default_rng(10)
        n = 5000
        # Build real samples with strong feature correlation: f0, f1
        # correlated, f2, f3 also correlated, but the two pairs
        # independent.
        z1 = rng.normal(size=n)
        z2 = rng.normal(size=n)
        real = np.stack([
            z1 + 0.1 * rng.normal(size=n),
            z1 + 0.1 * rng.normal(size=n),
            z2 + 0.1 * rng.normal(size=n),
            z2 + 0.1 * rng.normal(size=n),
        ], axis=1).astype(np.float32)
        real_labels = {"task": _one_hot(np.full(n, 0), 2)}
        # Synthetic from the same generative process.
        z1s = rng.normal(size=n)
        z2s = rng.normal(size=n)
        synth = np.stack([
            z1s + 0.1 * rng.normal(size=n),
            z1s + 0.1 * rng.normal(size=n),
            z2s + 0.1 * rng.normal(size=n),
            z2s + 0.1 * rng.normal(size=n),
        ], axis=1).astype(np.float32)
        synth_labels = {"task": _one_hot(np.full(n, 0), 2)}

        log = _Recorder()
        results = summarize_real_vs_synthetic(
            real, real_labels, synth, synth_labels,
            log=log, rng=np.random.default_rng(0),
        )
        r = results["task"][0]
        self.assertIsNotNone(r["corr_max_abs_diff"])
        self.assertLess(r["corr_max_abs_diff"], 0.1)
        self.assertLess(r["corr_mean_abs_diff"], 0.05)
        self.assertEqual(r["joint_flag"], "ok")


class TestJointDiagnosticsCorrelationBroken(unittest.TestCase):
    """Real has correlated features, synth has uncorrelated features —
    should flag JOINT_BROKEN even though marginals match."""

    def test_correlation_breakage_flagged(self) -> None:
        rng = np.random.default_rng(11)
        n = 4000
        # Real: f0 and f1 perfectly anti-correlated (think dow_sin × dow_cos
        # on a unit circle), f2 independent.
        theta = rng.uniform(0, 2 * np.pi, size=n)
        real = np.stack([
            np.sin(theta),
            np.cos(theta),
            rng.normal(size=n),
        ], axis=1).astype(np.float32)
        real_labels = {"task": _one_hot(np.full(n, 0), 2)}

        # Synth: f0 and f1 still have the same marginals (uniformly
        # distributed ~ N(0, ~0.5)) but are independent — joint
        # structure broken.
        synth = rng.normal(loc=0.0, scale=0.71, size=(n, 3)).astype(np.float32)
        # Match the marginals more closely so this test isolates the
        # correlation breakage.
        synth[:, 0] = np.sin(rng.uniform(0, 2 * np.pi, size=n))
        synth[:, 1] = np.cos(rng.uniform(0, 2 * np.pi, size=n))
        # f2 keeps its N(0, 1) marginal.
        synth_labels = {"task": _one_hot(np.full(n, 0), 2)}

        log = _Recorder()
        results = summarize_real_vs_synthetic(
            real, real_labels, synth, synth_labels,
            feature_names=["sin", "cos", "noise"],
            log=log, rng=np.random.default_rng(0),
        )

        r = results["task"][0]
        # Real ρ(sin, cos) ≈ 0 globally (the two are orthogonal on the
        # unit circle), but their PAIRWISE relationship (sin² + cos² = 1)
        # produces strong constraints that synthetic independent sampling
        # destroys.  We expect the corr structure to differ.
        self.assertIsNotNone(r["corr_max_abs_diff"])
        # The drilldown should mention "sin" and "cos" — the broken pair.
        joined = log.joined()
        self.assertIn("=== Joint-distribution diagnostic", joined)


class TestJointDiagnosticsExplicitCorrelationDiff(unittest.TestCase):
    """A cleaner construction: real has +0.9 corr on a pair, synth has
    -0.9 — should produce |Δρ| ~ 1.8 for that pair."""

    def test_high_corr_diff(self) -> None:
        rng = np.random.default_rng(12)
        n = 3000

        # Real: features 0 and 1 strongly positively correlated (ρ ≈ +0.9).
        z = rng.normal(size=n)
        e1 = rng.normal(size=n) * 0.5
        e2 = rng.normal(size=n) * 0.5
        real = np.stack([z + e1, z + e2, rng.normal(size=n)], axis=1).astype(np.float32)
        real_labels = {"task": _one_hot(np.full(n, 0), 2)}

        # Synth: features 0 and 1 strongly negatively correlated (ρ ≈ -0.9),
        # marginals matched (mean 0, std ≈ same).
        zs = rng.normal(size=n)
        es1 = rng.normal(size=n) * 0.5
        es2 = rng.normal(size=n) * 0.5
        synth = np.stack([zs + es1, -zs + es2, rng.normal(size=n)], axis=1).astype(np.float32)
        synth_labels = {"task": _one_hot(np.full(n, 0), 2)}

        log = _Recorder()
        results = summarize_real_vs_synthetic(
            real, real_labels, synth, synth_labels,
            feature_names=["a", "b", "indep"],
            log=log, rng=np.random.default_rng(0),
        )

        r = results["task"][0]
        # |Δρ| for the (a, b) pair should be huge (~1.8) — well past
        # the JOINT_BROKEN threshold.
        self.assertGreater(r["corr_max_abs_diff"], 1.0)
        self.assertIn("JOINT_BROKEN", r["joint_flag"])


class TestJointDiagnosticsSmallNSkipped(unittest.TestCase):
    """When synth N is below the minimum, joint stats should be None
    (correlation on a handful of samples is meaningless)."""

    def test_small_n_skips_corr(self) -> None:
        rng = np.random.default_rng(13)
        # Need enough real data for marginal stats but only a few synth.
        real = rng.normal(size=(2000, 4)).astype(np.float32)
        real_labels = {"task": _one_hot(np.full(2000, 0), 2)}
        synth = rng.normal(size=(10, 4)).astype(np.float32)  # < _MIN_SAMPLES_FOR_CORR
        synth_labels = {"task": _one_hot(np.full(10, 0), 2)}

        log = _Recorder()
        results = summarize_real_vs_synthetic(
            real, real_labels, synth, synth_labels, log=log,
        )
        r = results["task"][0]
        # Marginal stats still computed.
        self.assertIsNotNone(r["mean_shift_max"])
        # Correlation stats skipped (small-N).
        self.assertIsNone(r["corr_max_abs_diff"])
        self.assertIsNone(r["joint_flag"])


class TestJointDiagnosticsSubsamplingDeterminism(unittest.TestCase):
    """A seeded rng should make the joint diagnostic reproducible."""

    def test_seeded_rng_reproducible(self) -> None:
        rng = np.random.default_rng(14)
        real = rng.normal(size=(200_000, 3)).astype(np.float32)
        real_labels = {"task": _one_hot(np.full(200_000, 0), 2)}
        synth = rng.normal(size=(200_000, 3)).astype(np.float32)
        synth_labels = {"task": _one_hot(np.full(200_000, 0), 2)}

        results_a = summarize_real_vs_synthetic(
            real, real_labels, synth, synth_labels,
            log=_Recorder(), rng=np.random.default_rng(99),
        )
        results_b = summarize_real_vs_synthetic(
            real, real_labels, synth, synth_labels,
            log=_Recorder(), rng=np.random.default_rng(99),
        )

        self.assertAlmostEqual(
            results_a["task"][0]["corr_max_abs_diff"],
            results_b["task"][0]["corr_max_abs_diff"],
            places=10,
        )


# ---------------------------------------------------------------------------
# Temporal-fidelity tests
# ---------------------------------------------------------------------------


def _make_smooth_3d(n: int, T: int, F: int, rng: np.random.Generator) -> np.ndarray:
    """Generate temporally-smooth 3D sequences via cumulative-sum random
    walks per feature.  Lag-1 autocorrelation of a random walk is high
    (~0.99) so this gives us a "real-data-like" reference."""
    steps = rng.normal(scale=0.05, size=(n, T, F)).astype(np.float32)
    starts = rng.normal(size=(n, 1, F)).astype(np.float32)
    return starts + np.cumsum(steps, axis=1)


class TestTemporalDiagnostics2DSkipped(unittest.TestCase):
    """2D / DataFrame inputs have no temporal axis — diagnostic should
    skip silently without leaking the temporal section."""

    def test_2d_input_no_temporal_stats(self) -> None:
        rng = np.random.default_rng(20)
        real = rng.normal(size=(500, 4)).astype(np.float32)
        real_labels = {"task": _one_hot(np.full(500, 0), 2)}
        synth = rng.normal(size=(500, 4)).astype(np.float32)
        synth_labels = {"task": _one_hot(np.full(500, 0), 2)}

        log = _Recorder()
        results = summarize_real_vs_synthetic(
            real, real_labels, synth, synth_labels,
            log=log, rng=np.random.default_rng(0),
        )
        r = results["task"][0]
        # Temporal stats should be absent (None) for 2D input.
        self.assertIsNone(r["ac_max_abs_diff"])
        self.assertIsNone(r["ac_mean_abs_diff"])
        self.assertIsNone(r["temporal_flag"])
        # The temporal section header should not appear in the log.
        self.assertNotIn("Temporal-distribution diagnostic", log.joined())


class TestTemporalDiagnosticsHappyPath(unittest.TestCase):
    """Identical real and synthetic 3D structure → low |Δac|."""

    def test_smooth_real_smooth_synth_flagged_ok(self) -> None:
        rng = np.random.default_rng(21)
        n, T, F = 400, 16, 5
        real = _make_smooth_3d(n, T, F, rng)
        synth = _make_smooth_3d(n, T, F, rng)
        real_labels = {"task": _one_hot(np.full(n, 0), 2)}
        synth_labels = {"task": _one_hot(np.full(n, 0), 2)}

        log = _Recorder()
        results = summarize_real_vs_synthetic(
            real, real_labels, synth, synth_labels,
            log=log, rng=np.random.default_rng(0),
        )
        r = results["task"][0]
        self.assertIsNotNone(r["ac_max_abs_diff"])
        self.assertLess(r["ac_max_abs_diff"], 0.05)
        self.assertEqual(r["temporal_flag"], "ok")
        self.assertIn("Temporal-distribution diagnostic", log.joined())


class TestTemporalDiagnosticsBrokenAutocorrelation(unittest.TestCase):
    """Real is temporally smooth, synth is i.i.d. step-to-step noise →
    flag TEMPORAL_BROKEN.  This is the failure mode we'd most expect
    from a poorly-trained sequential GAN."""

    def test_iid_synth_flagged_temporal_broken(self) -> None:
        rng = np.random.default_rng(22)
        n, T, F = 400, 16, 5
        real = _make_smooth_3d(n, T, F, rng)
        # Synth: each timestep independent — autocorr ~ 0 across the
        # board, while real autocorr ~ 0.99.
        synth = rng.normal(size=(n, T, F)).astype(np.float32)
        real_labels = {"task": _one_hot(np.full(n, 0), 2)}
        synth_labels = {"task": _one_hot(np.full(n, 0), 2)}

        log = _Recorder()
        results = summarize_real_vs_synthetic(
            real, real_labels, synth, synth_labels,
            log=log, rng=np.random.default_rng(0),
        )
        r = results["task"][0]
        # Real autocorr should be very high (0.95+) and synth ~ 0 →
        # |Δac| should be huge.
        self.assertGreater(r["ac_max_abs_diff"], 0.7)
        self.assertGreater(r["ac_mean_abs_diff"], 0.5)
        self.assertIn("TEMPORAL_BROKEN", r["temporal_flag"])
        self.assertIn("autocorr_skew", r["temporal_flag"])


class TestTemporalDiagnosticsWorstFeatureDrilldown(unittest.TestCase):
    """The worst-feature drilldown should name the column whose
    autocorrelation diverges most."""

    def test_drilldown_names_worst_feature(self) -> None:
        rng = np.random.default_rng(23)
        n, T, F = 300, 16, 4
        real = _make_smooth_3d(n, T, F, rng)
        # Synth: smooth on every feature EXCEPT feature index 2 (the
        # "bb_width" stand-in), which is i.i.d. noise.
        synth = _make_smooth_3d(n, T, F, rng)
        synth[:, :, 2] = rng.normal(size=(n, T))
        real_labels = {"task": _one_hot(np.full(n, 0), 2)}
        synth_labels = {"task": _one_hot(np.full(n, 0), 2)}

        log = _Recorder()
        summarize_real_vs_synthetic(
            real, real_labels, synth, synth_labels,
            feature_names=["dow_sin", "dow_cos", "bb_width", "rsi"],
            log=log, rng=np.random.default_rng(0),
        )
        joined = log.joined()
        # The drilldown should mention bb_width with the worst
        # autocorrelation disagreement.
        self.assertIn("bb_width", joined)


class TestTemporalDiagnosticsListOfBatches(unittest.TestCase):
    """``synth_data`` arriving as a list of per-round 3D batches —
    the format ``balance_multi_task`` actually passes — should also
    produce temporal stats."""

    def test_list_of_3d_batches(self) -> None:
        rng = np.random.default_rng(24)
        n, T, F = 200, 16, 4
        real = _make_smooth_3d(n, T, F, rng)
        # 3 synth batches that, concatenated, total enough for stats.
        synth_batches = [
            _make_smooth_3d(50, T, F, rng) for _ in range(3)
        ]
        real_labels = {"task": _one_hot(np.full(n, 0), 2)}
        synth_labels = {"task": _one_hot(np.full(150, 0), 2)}

        log = _Recorder()
        results = summarize_real_vs_synthetic(
            real, real_labels, synth_batches, synth_labels,
            log=log, rng=np.random.default_rng(0),
        )
        r = results["task"][0]
        self.assertIsNotNone(r["ac_max_abs_diff"])
        self.assertEqual(r["temporal_flag"], "ok")


if __name__ == "__main__":
    unittest.main(verbosity=2)
