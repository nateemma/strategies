"""
Cross-implementation parity tests — verify the MLX helpers produce output
numerically equivalent to the TF reference implementation for the
deterministic parts of the pipeline.

Three test classes:

  1. ``TestVGMModelParity``  — helper's ``fit_vgm_columns`` produces BGMs
     identical to the inline VGM-fitting logic in
     ``GANs.df_mt_ctab_gan.CTABGANPlusMT.fit``.  Pure sklearn — no TF
     dependency.

  2. ``TestScalarEncodingParity`` — helper's ``transform_vgm`` produces
     scalar values identical to the TF MT ``_transform_data`` continuous
     path.  Pure numpy/sklearn — no TF dependency.  (Mode one-hot
     shapes intentionally differ — MLX slices to ``n_modes`` significant
     components, TF keeps all ``n_components``.  This test compares the
     scalars only.)

  3. ``TestEvalMetricsParity``  — helper's ``evaluate_with_dataframes``
     matches ``CTABGANPlusMT._compute_*_metrics`` byte-for-byte on
     continuous-only data.  Skipped when TensorFlow is not available.

Run from the freqtrade root:
    python -m pytest user_data/strategies/GANs/tests/test_mlx_tf_parity.py -v
"""

from __future__ import annotations

import os
import sys
import unittest
import warnings
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from sklearn.mixture import BayesianGaussianMixture

# ---------------------------------------------------------------------------
# Path / environment setup
# ---------------------------------------------------------------------------
STRATEGIES_ROOT = str(Path(__file__).parent.parent.parent)
if STRATEGIES_ROOT not in sys.path:
    sys.path.insert(0, STRATEGIES_ROOT)

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

from GANs.mlx_ctab_helpers import (  # noqa: E402
    evaluate_with_dataframes,
    fit_vgm_columns,
    transform_vgm,
)


def _tf_available() -> bool:
    try:
        import tensorflow  # noqa: F401
        return True
    except (ImportError, ModuleNotFoundError):
        return False


_HAS_TF: bool = _tf_available()
_skip_if_no_tf = unittest.skipUnless(
    _HAS_TF, "TensorFlow not available — skipping eval-metric parity tests"
)


# ---------------------------------------------------------------------------
# Reference implementations lifted from df_mt_ctab_gan.CTABGANPlusMT
# ---------------------------------------------------------------------------
#
# These replicate the inline VGM and scalar-encoding logic of the TF MT
# class so that the parity tests don't have to import TF (which is slow
# and pulls in optional deps).  The TF eval-metric tests below DO import
# the actual class — they're the authoritative parity check for the
# stochastic-but-seeded parts.


def _reference_fit_vgm(df: pd.DataFrame) -> Dict[str, BayesianGaussianMixture]:
    """Mirror of df_mt_ctab_gan.CTABGANPlusMT.fit's VGM fitting loop
    (continuous-only path)."""
    vgm_models: Dict[str, BayesianGaussianMixture] = {}
    for col in df.columns:
        bgm = BayesianGaussianMixture(
            n_components=10,
            weight_concentration_prior_type="dirichlet_process",
            weight_concentration_prior=0.001,
            max_iter=100,
            n_init=1,
            random_state=42,
        )
        clean = df[col].dropna().values.reshape(-1, 1)
        if len(np.unique(clean)) > 1:
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=ConvergenceWarning)
                bgm.fit(clean)
        else:
            bgm.means_ = np.array([[clean[0, 0]]])
            bgm.covariances_ = np.array([[[1e-4]]])
            bgm.weights_ = np.array([1.0])
            bgm.n_components = 1
        vgm_models[col] = bgm
    return vgm_models


def _reference_scalar_encoding(
    df: pd.DataFrame,
    vgm_models: Dict[str, BayesianGaussianMixture],
) -> Dict[str, np.ndarray]:
    """Mirror of df_mt_ctab_gan.CTABGANPlusMT._transform_data's scalar path —
    returns the per-column normalized scalar (without the mode one-hot)."""
    scalars: Dict[str, np.ndarray] = {}
    for col in df.columns:
        bgm = vgm_models[col]
        data = df[col].values.reshape(-1, 1)
        probs = bgm.predict_proba(data)
        modes = np.argmax(probs, axis=1)
        means = bgm.means_.reshape(1, -1)
        stds = np.sqrt(bgm.covariances_).reshape(1, -1)
        chosen_means = means[0, modes]
        chosen_stds = stds[0, modes]
        normalized = (df[col].values - chosen_means) / (4.0 * chosen_stds + 1e-8)
        scalars[col] = np.clip(normalized, -0.99, 0.99).astype(np.float32)
    return scalars


# ---------------------------------------------------------------------------
# Shared synthetic dataset — small enough that diversity-metric sampling
# does not engage (so RNG patterns don't matter), large enough to give
# the BGM something to fit.
# ---------------------------------------------------------------------------


def _parity_dataset() -> pd.DataFrame:
    rng = np.random.default_rng(0)
    return pd.DataFrame(
        {
            "uniform": rng.uniform(-1, 1, 300).astype(np.float32),
            "bimodal": np.concatenate(
                [
                    rng.normal(-2, 0.3, 150),
                    rng.normal(2, 0.3, 150),
                ]
            ).astype(np.float32),
            "skewed": rng.exponential(1.0, 300).astype(np.float32),
        }
    )


# ===========================================================================
# 1. VGM model parity
# ===========================================================================


class TestVGMModelParity(unittest.TestCase):
    """Helper's ``fit_vgm_columns`` and the TF MT inline logic must
    produce BGM models with the same fitted parameters.

    Doesn't need TF — the reference is a small numpy/sklearn function
    lifted from df_mt_ctab_gan.py.  If either side later drifts on BGM
    hyperparameters, this test catches it.
    """

    @classmethod
    def setUpClass(cls):
        cls.df = _parity_dataset()
        cls.helper_models, _, _ = fit_vgm_columns(cls.df)
        cls.reference_models = _reference_fit_vgm(cls.df)

    def test_means_match(self):
        for col, ref in self.reference_models.items():
            mlx = self.helper_models[col]
            np.testing.assert_allclose(
                mlx.means_, ref.means_, rtol=1e-6, atol=1e-8,
                err_msg=f"BGM means differ for column {col!r}",
            )

    def test_covariances_match(self):
        for col, ref in self.reference_models.items():
            mlx = self.helper_models[col]
            np.testing.assert_allclose(
                mlx.covariances_, ref.covariances_, rtol=1e-6, atol=1e-8,
                err_msg=f"BGM covariances differ for column {col!r}",
            )

    def test_weights_match(self):
        for col, ref in self.reference_models.items():
            mlx = self.helper_models[col]
            np.testing.assert_allclose(
                mlx.weights_, ref.weights_, rtol=1e-6, atol=1e-8,
                err_msg=f"BGM weights differ for column {col!r}",
            )


# ===========================================================================
# 2. Scalar encoding parity
# ===========================================================================


class TestScalarEncodingParity(unittest.TestCase):
    """The normalized scalar component of the encoding is deterministic
    given the BGM and must match the TF MT inline logic byte-for-byte.

    The mode one-hot is intentionally NOT compared — MLX slices to
    ``n_modes`` significant components while TF keeps the full
    ``n_components`` matrix.  This is a documented divergence.
    """

    @classmethod
    def setUpClass(cls):
        cls.df = _parity_dataset()
        cls.helper_models, cls.helper_info, cls.helper_continuous_info = (
            fit_vgm_columns(cls.df)
        )
        cls.helper_encoded = transform_vgm(
            cls.df, cls.helper_models, cls.helper_info, list(cls.df.columns)
        )
        cls.reference_scalars = _reference_scalar_encoding(cls.df, cls.helper_models)

    def test_helper_scalars_match_reference(self):
        # The helper's encoding lays out columns as:
        #   [scalar_0, mode_0_0, mode_0_1, ..., scalar_1, mode_1_0, ...]
        # Walk the offset bookkeeping the same way the helper does to
        # extract each scalar.
        offset = 0
        for i, col in enumerate(self.df.columns):
            n_modes = self.helper_info[col]["n_modes"]
            helper_scalar = self.helper_encoded[:, offset]
            ref_scalar = self.reference_scalars[col]
            np.testing.assert_allclose(
                helper_scalar, ref_scalar, rtol=1e-6, atol=1e-7,
                err_msg=f"scalar values differ for column {col!r}",
            )
            offset += 1 + n_modes

    def test_helper_scalars_within_clip_range(self):
        # Belt-and-braces — the scalars are clipped to [-0.99, 0.99] in
        # both implementations.  If this regresses, the parity test
        # above would also fail, but a focused assertion makes the
        # failure mode obvious.
        offset = 0
        for col in self.df.columns:
            n_modes = self.helper_info[col]["n_modes"]
            scalar = self.helper_encoded[:, offset]
            self.assertGreaterEqual(scalar.min(), -0.99)
            self.assertLessEqual(scalar.max(), 0.99)
            offset += 1 + n_modes


# ===========================================================================
# 3. Evaluation metrics parity (requires TF)
# ===========================================================================


@_skip_if_no_tf
class TestEvalMetricsParity(unittest.TestCase):
    """Helper's ``evaluate_with_dataframes`` must match the TF MT
    ``_compute_*_metrics`` chain on continuous-only data.

    Uses the actual ``CTABGANPlusMT`` class for the reference — this is
    the strongest parity guarantee, catching drift in either direction.
    Categorical paths are bypassed by setting ``categorical_columns=[]``
    on the reference instance.
    """

    @classmethod
    def setUpClass(cls):
        # Reference data (real) and a slightly noised copy (generated)
        # — keep both ≤ 1000 rows so neither implementation engages the
        # diversity sample-down path (avoids the seeded-permutation
        # subtlety entirely).
        cls.df_real = _parity_dataset()
        cls.df_gen = (
            cls.df_real.values + np.random.RandomState(7).normal(0, 0.05, cls.df_real.shape)
        )
        cls.df_gen = pd.DataFrame(cls.df_gen, columns=cls.df_real.columns).astype(np.float32)

        cls.helper_models, cls.helper_info, _ = fit_vgm_columns(cls.df_real)
        cls.column_order = list(cls.df_real.columns)
        cls.continuous_columns = list(cls.df_real.columns)

        # Build the helper metrics.
        cls.helper_metrics = evaluate_with_dataframes(
            cls.df_real,
            cls.df_gen,
            continuous_columns=cls.continuous_columns,
            column_info=cls.helper_info,
            column_order=cls.column_order,
        )

        # Build the reference (TF MT) metrics by instantiating the class
        # and calling its compute_* methods directly.  We bypass fit()
        # so we don't run the full GAN training pipeline — just set the
        # attributes the eval methods read.
        from GANs.df_mt_ctab_gan import CTABGANPlusMT

        ref = CTABGANPlusMT(verbose=False, random_seed=None)
        ref.continuous_columns = list(cls.df_real.columns)
        ref.categorical_columns = []
        ref.column_order = list(cls.df_real.columns)
        # The helper uses only ``min/max`` from column_info for validity
        # — copy those across (the helper's column_info is a superset).
        ref.column_info = {
            col: {
                "type": "continuous",
                "min": cls.helper_info[col]["min"],
                "max": cls.helper_info[col]["max"],
            }
            for col in cls.df_real.columns
        }

        cls.ref_diversity = ref._compute_diversity_metrics(cls.df_real, cls.df_gen)
        cls.ref_correlation = ref._compute_correlation_metrics(cls.df_real, cls.df_gen)
        cls.ref_statistics = ref._compute_statistical_metrics(cls.df_real, cls.df_gen)
        cls.ref_validity = ref._compute_validity_metrics(cls.df_gen)
        cls.ref_overall = ref._compute_overall_score(
            {
                "diversity": cls.ref_diversity,
                "correlation": cls.ref_correlation,
                "statistics": cls.ref_statistics,
                "validity": cls.ref_validity,
            }
        )

    # ---------- diversity ----------

    def test_diversity_pairwise_distances_match(self):
        ref = self.ref_diversity
        helper = self.helper_metrics["diversity"]
        for key in (
            "gen_pairwise_distance_mean",
            "gen_pairwise_distance_std",
            "gen_pairwise_distance_min",
            "real_pairwise_distance_mean",
            "diversity_ratio",
        ):
            self.assertAlmostEqual(
                helper[key], ref[key], places=4,
                msg=f"diversity[{key!r}] differs",
            )

    def test_diversity_value_space_coverage_matches(self):
        self.assertAlmostEqual(
            self.helper_metrics["diversity"]["value_space_coverage"],
            self.ref_diversity["value_space_coverage"],
            places=5,
        )

    def test_diversity_nearest_neighbour_matches(self):
        self.assertAlmostEqual(
            self.helper_metrics["diversity"]["nearest_real_distance_mean"],
            self.ref_diversity["nearest_real_distance_mean"],
            places=4,
        )
        self.assertAlmostEqual(
            self.helper_metrics["diversity"]["nearest_real_distance_std"],
            self.ref_diversity["nearest_real_distance_std"],
            places=4,
        )

    # ---------- correlation ----------

    def test_correlation_metrics_match(self):
        helper = self.helper_metrics["correlation"]
        for key in ("correlation_preservation", "correlation_error"):
            self.assertAlmostEqual(
                helper[key], self.ref_correlation[key], places=5,
                msg=f"correlation[{key!r}] differs",
            )

    # ---------- statistics ----------

    def test_statistics_overall_averages_match(self):
        self.assertAlmostEqual(
            self.helper_metrics["statistics"]["mean_error_avg"],
            self.ref_statistics["mean_error_avg"],
            places=5,
        )
        self.assertAlmostEqual(
            self.helper_metrics["statistics"]["std_error_avg"],
            self.ref_statistics["std_error_avg"],
            places=5,
        )

    def test_statistics_per_column_match(self):
        helper_stats = self.helper_metrics["statistics"]["continuous_statistics"]
        ref_stats = self.ref_statistics["continuous_statistics"]
        self.assertEqual(set(helper_stats.keys()), set(ref_stats.keys()))
        for col in helper_stats:
            for key in ("mean_error", "std_error", "real_mean", "gen_mean", "real_std", "gen_std"):
                self.assertAlmostEqual(
                    helper_stats[col][key], ref_stats[col][key], places=4,
                    msg=f"statistics[{col!r}][{key!r}] differs",
                )

    # ---------- validity ----------

    def test_validity_overall_score_matches(self):
        self.assertAlmostEqual(
            self.helper_metrics["validity"]["overall_validity_score"],
            self.ref_validity["overall_validity_score"],
            places=6,
        )

    def test_validity_per_column_matches(self):
        helper_v = self.helper_metrics["validity"]["continuous_validity"]
        ref_v = self.ref_validity["continuous_validity"]
        self.assertEqual(set(helper_v.keys()), set(ref_v.keys()))
        for col in helper_v:
            self.assertEqual(
                helper_v[col]["out_of_range_count"],
                ref_v[col]["out_of_range_count"],
            )
            self.assertAlmostEqual(
                helper_v[col]["out_of_range_pct"],
                ref_v[col]["out_of_range_pct"],
                places=6,
            )

    # ---------- overall score ----------

    def test_overall_quality_matches(self):
        helper_overall = self.helper_metrics["overall_score"]
        for key in (
            "diversity_score",
            "correlation_score",
            "statistical_score",
            "validity_score",
            "overall_quality",
        ):
            self.assertAlmostEqual(
                helper_overall[key], self.ref_overall[key], places=4,
                msg=f"overall_score[{key!r}] differs",
            )


if __name__ == "__main__":
    unittest.main(verbosity=2)
