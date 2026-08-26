"""Unit tests for the scorecard's manifold metrics and utility probe.

These guard the two failure modes GAN_TODO §5 had to diagnose by hand:
  - MT_DDPM synth SATURATING the ±4σ clip (invisible to RMSE-style metrics)
  - post-hoc dispersion widening that preserved correlations EXACTLY but sat
    OFF the nonlinear manifold (the AE rejected ~98%)
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from GANs.quality.manifold import clip_band_fraction, nn_distance_ratio
from GANs.quality.utility_probe import delta_val_mcc


class TestManifoldMetrics(unittest.TestCase):
    def setUp(self):
        self.rng = np.random.default_rng(0)
        self.real = self.rng.normal(size=(400, 6))

    def test_clip_band_fraction_zero_on_unclipped(self):
        self.assertLess(clip_band_fraction(self.real, 4.0), 1e-9)

    def test_clip_band_fraction_high_on_saturated(self):
        sat = np.clip(self.rng.normal(scale=9, size=(400, 6)), -4, 4)
        self.assertGreater(clip_band_fraction(sat, 4.0), 0.3)

    def test_nn_ratio_near_one_for_same_distribution(self):
        r = nn_distance_ratio(self.real, self.rng.normal(size=(400, 6)))
        self.assertGreater(r, 0.5)
        self.assertLess(r, 2.0)

    def test_nn_ratio_large_off_manifold(self):
        off = self.rng.normal(size=(400, 6)) * 4 + 9
        self.assertGreater(nn_distance_ratio(self.real, off), 3.0)

    def test_nn_ratio_survives_shape_mismatch(self):
        self.assertTrue(np.isnan(nn_distance_ratio(self.real, self.rng.normal(size=(50, 3)))))

    def test_metrics_accept_3d(self):
        r3 = self.rng.normal(size=(200, 4, 6))
        s3 = self.rng.normal(size=(200, 4, 6))
        self.assertFalse(np.isnan(nn_distance_ratio(r3, s3)))
        self.assertLess(clip_band_fraction(r3, 4.0), 1e-9)


class TestUtilityProbe(unittest.TestCase):
    def setUp(self):
        self.rng = np.random.default_rng(0)
        self.x = self.rng.normal(size=(500, 6))
        self.y = (self.x[:, 0] > 0).astype(int)

    def test_same_distribution_synth_is_roughly_neutral(self):
        sx = self.rng.normal(size=(300, 6))
        sy = (sx[:, 0] > 0).astype(int)
        d = delta_val_mcc(self.x, self.y, sx, sy)["delta"]
        self.assertIsNotNone(d)
        self.assertGreater(d, -0.15)

    def test_random_label_synth_degrades(self):
        sx = self.rng.normal(size=(300, 6))
        sy = self.rng.integers(0, 2, 300)
        self.assertLess(delta_val_mcc(self.x, self.y, sx, sy)["delta"], -0.02)

    def test_broken_variant_yields_a_row_not_an_exception(self):
        """A failing GAN must still produce a scorecard row."""
        for synth, why in ((None, "no synthetic"),
                           (np.zeros((0, 6)), "empty"),
                           (np.full((10, 6), np.nan), "non-finite"),
                           (self.rng.normal(size=(10, 3)), "wrong dim")):
            out = delta_val_mcc(self.x, self.y, synth,
                                np.zeros(len(synth)) if synth is not None else None)
            self.assertIsNone(out["delta"], why)
            self.assertTrue(out["reason"], why)
            self.assertIsNotNone(out["mcc_real"], "baseline should still be computed")

    def test_val_split_never_sees_synth(self):
        """Memorised synth must not inflate the score."""
        d = delta_val_mcc(self.x, self.y, self.x.copy(), self.y.copy())["delta"]
        self.assertLess(d, 0.30)


if __name__ == "__main__":
    unittest.main()
