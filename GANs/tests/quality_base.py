"""
Shared quality-test mixin for all GAN types.

Usage — concrete subclass must inherit from BOTH GANQualityMixin and
unittest.TestCase (mixin first):

    @unittest.skipUnless(_RUN, _SKIP_MSG)
    class TestWGANQuality(GANQualityMixin, unittest.TestCase):
        GAN_TYPE       = GANType.WGAN
        N_CLASSES      = 3
        MINORITY_CLASSES = [1, 2]

        @classmethod
        def _make_dataset(cls, seed: int = 42):
            ...

        @classmethod
        def _setup_generated(cls, iface) -> None:
            ...

Design notes:
    - GANQualityMixin does NOT inherit unittest.TestCase, so test runners
      skip it directly (no tests collected on the base).
    - setUpClass trains the model via iface.fit(), then delegates to
      _setup_generated() for type-specific generation and metric computation.
    - Quality thresholds are class-level constants that subclasses may override.
"""

from __future__ import annotations

from typing import Any, ClassVar

import numpy as np


# ---------------------------------------------------------------------------
# Training constants shared by all concrete subclasses
# ---------------------------------------------------------------------------
QUALITY_EPOCHS   = 100
QUALITY_BATCH    = 32
QUALITY_N_CRITIC = 3


class GANQualityMixin:
    """Shared quality tests exercised through GANInterface.fit()/generate().

    Do NOT inherit unittest.TestCase here — that would cause the test runner
    to collect (and fail) abstract hooks on this class directly.
    """

    # ------------------------------------------------------------------ #
    # Subclasses MUST override these                                       #
    # ------------------------------------------------------------------ #
    GAN_TYPE: ClassVar[Any]          # GANType enum value
    N_CLASSES: ClassVar[int]         # Number of output classes
    MINORITY_CLASSES: ClassVar[list] # Class indices to check for presence

    # ------------------------------------------------------------------ #
    # Thresholds — conservative sanity-check bars, not quality bars       #
    # ------------------------------------------------------------------ #
    MEAN_RMSE_THRESHOLD:      ClassVar[float] = 0.5
    STD_RMSE_THRESHOLD:       ClassVar[float] = 0.5
    RANGE_COVERAGE_THRESHOLD: ClassVar[float] = 0.5

    # ------------------------------------------------------------------ #
    # Set by setUpClass                                                    #
    # ------------------------------------------------------------------ #
    real_data:       np.ndarray   # Training data (2D or 3D)
    real_labels:     Any          # Labels (array or dict)
    gen_x:           np.ndarray   # Generated samples, 2D (N, F)
    gen_y_primary:   np.ndarray   # Primary-task one-hot for generated (N, C)
    quality_metrics: dict         # From assess_*_quality

    # ------------------------------------------------------------------ #
    # Hooks — subclasses implement these                                   #
    # ------------------------------------------------------------------ #

    @classmethod
    def _make_dataset(cls, seed: int = 42):
        """Return (data, labels) for training.  Labels may be array or dict."""
        raise NotImplementedError

    @classmethod
    def _fit_kwargs(cls) -> dict:
        """Extra kwargs merged into the fit() call (overrides type defaults)."""
        return {
            "epochs":         QUALITY_EPOCHS,
            "batch_size":     QUALITY_BATCH,
            "n_critic":       QUALITY_N_CRITIC,
            "assess_quality": False,
            "verbose":        False,
        }

    @classmethod
    def _setup_generated(cls, iface: Any) -> None:
        """Generate samples; populate cls.gen_x, cls.gen_y_primary, cls.quality_metrics.

        Args:
            iface: The trained GANInterface instance.
        """
        raise NotImplementedError

    # ------------------------------------------------------------------ #
    # Shared setUp                                                         #
    # ------------------------------------------------------------------ #

    @classmethod
    def setUpClass(cls) -> None:
        import gc

        import tensorflow as tf

        tf.keras.backend.clear_session()
        gc.collect()

        from GANs.GANInterface import GANInterface

        cls.real_data, cls.real_labels = cls._make_dataset()
        iface = GANInterface(cls.GAN_TYPE, save_path=None, prefer_mlx=False)
        iface.fit(
            cls.real_data.copy(),
            cls._copy_labels(cls.real_labels),
            **cls._fit_kwargs(),
        )
        cls._setup_generated(iface)

    @classmethod
    def _copy_labels(cls, labels: Any) -> Any:
        if isinstance(labels, dict):
            return {k: v.copy() for k, v in labels.items()}
        return labels.copy()

    # ------------------------------------------------------------------ #
    # Shared tests                                                         #
    # ------------------------------------------------------------------ #

    def test_generated_samples_exist(self) -> None:
        self.assertGreater(
            len(self.gen_x), 0,
            msg="No generated samples found; fit/generate may have failed",
        )

    def test_label_fidelity_above_chance(self) -> None:
        """A classifier trained on real data should classify generated samples above chance."""
        from sklearn.linear_model import LogisticRegression

        real_x   = self._to_2d(self.real_data)
        real_idx = np.argmax(self._primary_labels(self.real_labels), axis=1)
        gen_x    = self._to_2d(self.gen_x)
        gen_idx  = np.argmax(self.gen_y_primary, axis=1)

        clf      = LogisticRegression(max_iter=500, C=0.1).fit(real_x, real_idx)
        accuracy = clf.score(gen_x, gen_idx)

        chance = 1.0 / self.N_CLASSES
        self.assertGreater(
            accuracy, chance,
            msg=f"Label fidelity {accuracy:.3f} ≤ chance {chance:.3f}: "
                "generated samples don't align with their declared classes",
        )

    def test_all_minority_classes_generated(self) -> None:
        """Generated set must contain samples from all declared minority classes."""
        gen_idx = np.argmax(self.gen_y_primary, axis=1)
        for c in self.MINORITY_CLASSES:
            self.assertIn(
                c, gen_idx,
                msg=f"No generated samples found for minority class {c}",
            )

    def test_mean_rmse_below_threshold(self) -> None:
        self.assertLess(
            self.quality_metrics.get("mean_rmse", 999),
            self.MEAN_RMSE_THRESHOLD,
            msg=f"mean_rmse={self.quality_metrics.get('mean_rmse'):.4f} — "
                "feature means are too far from real",
        )

    def test_std_rmse_below_threshold(self) -> None:
        self.assertLess(
            self.quality_metrics.get("std_rmse", 999),
            self.STD_RMSE_THRESHOLD,
            msg=f"std_rmse={self.quality_metrics.get('std_rmse'):.4f} — "
                "feature spreads are too far from real",
        )

    def test_range_coverage_above_threshold(self) -> None:
        overlap = self.quality_metrics.get("range_coverage", {}).get("overlap", 0.0)
        self.assertGreater(
            overlap,
            self.RANGE_COVERAGE_THRESHOLD,
            msg=f"range_coverage={overlap:.2%} — generated samples don't cover "
                "enough of the real value range",
        )

    # ------------------------------------------------------------------ #
    # Helpers                                                              #
    # ------------------------------------------------------------------ #

    @classmethod
    def _to_2d(cls, arr) -> np.ndarray:
        """Collapse 3D (N, seq_len, F) to 2D (N, F), or convert DataFrame to numpy."""
        import pandas as pd
        if isinstance(arr, pd.DataFrame):
            return arr.values.astype("float32")
        if arr.ndim == 3:
            return arr[:, 0, :]
        return arr

    @classmethod
    def _primary_labels(cls, labels: Any) -> np.ndarray:
        """Return the primary-task label array from either array or dict form."""
        if isinstance(labels, dict):
            return labels.get("trading", next(iter(labels.values())))
        return labels
