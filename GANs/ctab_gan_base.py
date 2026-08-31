"""Shared base for the CTAB-GAN+ trainers.

Holds the methods that were byte-identical between the single-task
(``df_ctab_gan.py``) and multi-task (``df_mt_ctab_gan.py``) trainers, so they live in exactly one place. The
single/multi-specific logic (``__init__``, ``_create_models``, ``_train``,
``fit``, ``generate``, ``save``/``load`` on the base classes, the metric
sub-computations) stays on the subclasses and resolves via normal MRO.

These methods call ``self.*`` helpers that differ per subclass; because the
bodies are byte-identical and ``self`` resolves per-instance, hoisting is
behaviour-neutral.
"""

from __future__ import annotations

import os
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd
import tensorflow as tf


class CTABGANPlusBase:
    def _set_random_seeds(self, seed: int):
        """Set random seeds for reproducibility across all libraries."""
        # Set Python's built-in random seed
        import random

        random.seed(seed)

        # Set NumPy random seed
        np.random.seed(seed)

        # Set TensorFlow random seed
        tf.random.set_seed(seed)

        # Set TensorFlow to use deterministic operations (may impact performance)
        # Note: Some GPU operations may still be non-deterministic
        try:
            tf.config.experimental.enable_op_determinism()
        except (AttributeError, RuntimeError):
            # Deterministic ops not available in this TensorFlow version
            pass

        if self.verbose:
            print(f"    Random seed set to {seed} for reproducibility")

    def _transform_data(self, df: pd.DataFrame) -> np.ndarray:
        """Transform dataframe to numpy array.
        Continuous data is encoded using Variational Gaussian Mixture (VGM) which produces two outputs per feature:
        1. A scalar representing the normalized value within the chosen mode
        2. A one-hot categorical vector indicating which mode was chosen.
        Categorical data is one-hot encoded."""

        continuous_data_list = []
        for col in self.continuous_columns:
            # Reconstruct the representation using VGM:
            # We transform value -> normalized offset in mode + one-hot mode
            # If a model isn't fitted for some reason, fallback to basic normalize
            if col in self.vgm_models and self.vgm_models[col] is not None:
                bgm = self.vgm_models[col]
                data = df[col].values.reshape(-1, 1)

                # Predict probabilities of each mode
                probs = bgm.predict_proba(data)

                # Assign to the mode with highest probability
                modes = np.argmax(probs, axis=1)

                # Normalize data within the selected mode using standardization: (x - mean) / (4 * std)
                means = bgm.means_.reshape(1, -1)
                stds = np.sqrt(bgm.covariances_).reshape(1, -1)

                # Find mean and std for the chosen modes
                chosen_means = means[0, modes]
                chosen_stds = stds[0, modes]

                # Normalize value within the mode to approximately [-1, 1] range
                # We divide by 4 std deviations to cover ~99.9% of the mass
                # Add epsilon to prevent division by zero
                normalized_value = (df[col].values - chosen_means) / (
                    4 * chosen_stds + 1e-8
                )
                normalized_value = np.clip(normalized_value, -0.99, 0.99).reshape(-1, 1)

                # One-hot encode the mode
                num_modes = bgm.n_components
                one_hot_mode = np.eye(num_modes, dtype=np.float32)[modes]

                continuous_data_list.append(normalized_value)
                continuous_data_list.append(one_hot_mode)
            else:
                # Fallback min-max normalization
                info = self.column_info[col]
                val = df[col].values
                norm = 2 * (val - info["min"]) / (info["max"] - info["min"] + 1e-8) - 1
                continuous_data_list.append(norm.reshape(-1, 1))

        if continuous_data_list:
            continuous_data = np.concatenate(continuous_data_list, axis=1)
        else:
            continuous_data = np.zeros((len(df), 0), dtype=np.float32)

        categorical_data_list = []
        for col in self.categorical_columns:
            info = self.column_info[col]
            cat_to_idx = info["cat_to_idx"]
            num_cats = info["num_categories"]

            # Map values to indices
            indices = np.array([cat_to_idx.get(val, 0) for val in df[col]])

            # One-hot encode
            one_hot = np.eye(num_cats, dtype=np.float32)[indices]
            categorical_data_list.append(one_hot)

        if categorical_data_list:
            categorical_data = np.concatenate(categorical_data_list, axis=1)
            # Continuous features first, then categorical (matches _create_models concat order)
            if continuous_data.shape[1] > 0:
                return np.concatenate([continuous_data, categorical_data], axis=1)
            else:
                return categorical_data
        return continuous_data

    # ------------------------------------------------------------------
    # Correlation matching (off-diagonal feature-matching)
    # ------------------------------------------------------------------
    #
    # CTAB-TF's discriminator is BLIND to joint structure: it scores
    # class-conditionally column-shuffled real data (marginals preserved
    # exactly, correlations destroyed) as essentially real
    # (joint_sensitivity 0.045 vs MLX 0.67, 3 seeds -- see
    # docs/GAN_SCORECARD.md).  The generator is therefore never told that
    # correlation is missing and never starts learning it, which is why
    # every generator-side change tried so far came back inert.
    #
    # This term supplies the missing gradient directly rather than waiting
    # for the discriminator to learn to supply it: match the off-diagonal
    # correlation vector of fake to real, in the DECODED feature space that
    # the fidelity probe measures.  Note the existing info_loss matches
    # first moments (mean/std) in ENCODED space -- this is the second-order
    # analogue, and the space matters because the metric under test
    # (mean|corr|, corr-of-corr) is computed post-decode.

    def _build_corr_decode_tables(self) -> None:
        """Precompute the VGM mode means/stds for a differentiable decode.

        ``generate`` decodes a continuous column as
        ``scalar * 4 * std[argmax(modes)] + mean[argmax(modes)]``.  The
        correlation loss needs the same quantity but differentiable, so the
        hard argmax is replaced by the Gumbel-Softmax mode probabilities the
        generator already emits.  For real data the mode block is exactly
        one-hot, so the soft decode reproduces the hard decode exactly and
        both sides of the loss are measured on the same footing.
        """
        plan: List[dict] = []
        offset = 0
        for col in self.continuous_columns:
            info = self.column_info[col]
            n_modes = int(info.get("vgm_components", 0))
            entry: dict = {"scalar_idx": offset, "n_modes": n_modes}
            if n_modes > 0:
                bgm = self.vgm_models[col]
                entry["mode_start"] = offset + 1
                entry["means"] = bgm.means_.reshape(-1).astype(np.float32)
                entry["stds"] = (
                    np.sqrt(bgm.covariances_).reshape(-1).astype(np.float32)
                )
            else:
                # Fallback min-max columns: decode is affine in the scalar.
                span = 0.5 * (info["max"] - info["min"])
                entry["scale"] = np.float32(span)
                entry["shift"] = np.float32(span + info["min"])
            plan.append(entry)
            offset += 1 + n_modes
        self._corr_decode_plan = plan

    def _soft_decode(self, encoded):
        """Differentiably decode the continuous block to raw feature values."""
        cols = []
        for e in self._corr_decode_plan:
            scalar = encoded[:, e["scalar_idx"]]
            if e["n_modes"] > 0:
                start = e["mode_start"]
                probs = encoded[:, start : start + e["n_modes"]]
                means = tf.constant(e["means"], dtype=encoded.dtype)
                stds = tf.constant(e["stds"], dtype=encoded.dtype)
                # Expectation over modes rather than argmax -- exact when the
                # mode block is one-hot, which it is for real data and which
                # the annealing Gumbel temperature drives fake towards.
                m = tf.reduce_sum(probs * means, axis=1)
                s = tf.reduce_sum(probs * stds, axis=1)
                cols.append(scalar * 4.0 * s + m)
            else:
                cols.append(
                    scalar * tf.cast(e["scale"], encoded.dtype)
                    + tf.cast(e["shift"], encoded.dtype)
                )
        return tf.stack(cols, axis=1)

    @staticmethod
    def _corr_offdiag(x):
        """Upper-triangular off-diagonal entries of the correlation matrix."""
        xc = x - tf.reduce_mean(x, axis=0, keepdims=True)
        denom = tf.sqrt(tf.reduce_mean(tf.square(xc), axis=0, keepdims=True) + 1e-8)
        xn = xc / denom
        n = tf.cast(tf.shape(xn)[0], xn.dtype)
        corr = tf.matmul(xn, xn, transpose_a=True) / n
        d = xn.shape[1]
        mask = tf.constant(np.triu(np.ones((d, d), dtype=bool), k=1))
        return tf.boolean_mask(corr, mask)

    @staticmethod
    def _second_moment(x, mu, sd, include_diag=True):
        """Second-moment matrix of x, normalised by the REAL feature stats.

        BOTH sides are divided by the SAME (real) sd, so on the fake side the
        DIAGONAL is (sigma_fake / sigma_real)^2 -- the dispersion ratio -- and
        the off-diagonal is correlation scaled by those ratios.  Matching this
        matrix therefore pins BOTH moments in a single term.

        `_corr_offdiag` normalises each side by its OWN std, which is
        scale-invariant by construction: it cannot see dispersion at all, which
        is why raising its weight inflates sigma on WGAN-MLX (1.10 -> 1.71) and
        why CGAN's sigma penalty destroys correlation.  Each term pins one
        moment and leaves the other free.
        """
        z = (x - mu) / sd
        zc = z - tf.reduce_mean(z, axis=0, keepdims=True)
        n = tf.cast(tf.shape(zc)[0], zc.dtype)
        cov = tf.matmul(zc, zc, transpose_a=True) / n
        d = zc.shape[1]
        mask = tf.constant(
            np.triu(np.ones((d, d), dtype=bool), k=0 if include_diag else 1)
        )
        return tf.boolean_mask(cov, mask)

    def _correlation_loss(self, real_data, fake_data):
        """MSE between the off-diagonal correlation vectors of real and fake.

        Forced to float32 regardless of the mixed-precision policy: the
        correlation is a sum of ~1e3 products, which fp16 accumulates too
        coarsely to give a usable gradient.  The caller casts the scalar
        back to the loss dtype.
        """
        real_d = self._soft_decode(tf.cast(real_data, tf.float32))
        fake_d = self._soft_decode(tf.cast(fake_data, tf.float32))
        if getattr(self, "corr_loss_mode", "correlation") == "covariance":
            # Normalise both sides by the REAL batch stats so the comparison
            # is on one scale and the diagonal carries the dispersion ratio.
            mu = tf.reduce_mean(real_d, axis=0, keepdims=True)
            sd = tf.sqrt(
                tf.reduce_mean(tf.square(real_d - mu), axis=0, keepdims=True)
                + 1e-8
            )
            loss = tf.reduce_mean(
                tf.square(
                    self._second_moment(fake_d, mu, sd)
                    - self._second_moment(real_d, mu, sd)
                )
            )
        else:
            loss = tf.reduce_mean(
                tf.square(self._corr_offdiag(fake_d) - self._corr_offdiag(real_d))
            )
        # Same NaN/blow-up discipline as the existing info/downstream terms.
        loss = tf.minimum(loss, 10.0)
        return tf.where(tf.math.is_finite(loss), loss, tf.zeros_like(loss))

    @staticmethod
    def _clip_grads(grads, clip_norm):
        """Per-tensor norm clipping; ``None`` passes gradients through."""
        if clip_norm is None:
            return grads
        return [tf.clip_by_norm(g, clip_norm) if g is not None else None
                for g in grads]

    def _wasserstein_loss(self, y_true, y_pred):
        """Wasserstein loss for WGAN."""
        return tf.reduce_mean(y_pred)

    def _compute_validity_metrics(self, generated_data: pd.DataFrame) -> Dict[str, Any]:
        """Check validity of generated samples."""
        metrics = {}

        # Check continuous columns are in valid ranges
        continuous_valid = {}
        for col in self.continuous_columns:
            if col in generated_data.columns and col in self.column_info:
                info = self.column_info[col]
                col_min = info["min"]
                col_max = info["max"]

                out_of_range = (
                    (generated_data[col] < col_min) | (generated_data[col] > col_max)
                ).sum()
                total = len(generated_data)

                continuous_valid[col] = {
                    "out_of_range_count": int(out_of_range),
                    "out_of_range_pct": (
                        float(out_of_range / total) if total > 0 else 0.0
                    ),
                }
        metrics["continuous_validity"] = continuous_valid

        # Check categorical columns have valid values
        categorical_valid = {}
        for col in self.categorical_columns:
            if col in generated_data.columns and col in self.column_info:
                info = self.column_info[col]
                valid_values = set(info["unique_values"])
                gen_values = set(generated_data[col].unique())

                invalid_values = gen_values - valid_values
                categorical_valid[col] = {
                    "invalid_value_count": len(invalid_values),
                    "invalid_values": list(invalid_values) if invalid_values else [],
                }
        metrics["categorical_validity"] = categorical_valid

        # Overall validity score
        total_invalid = sum(
            s["out_of_range_count"] for s in continuous_valid.values()
        ) + sum(s["invalid_value_count"] for s in categorical_valid.values())
        total_samples = len(generated_data)
        metrics["overall_validity_score"] = (
            1.0 - (total_invalid / total_samples) if total_samples > 0 else 1.0
        )

        return metrics

    def _compute_overall_score(self, metrics: Dict[str, Any]) -> Dict[str, float]:
        """Compute overall quality score from all metrics."""
        score = {}

        # Diversity score (0-1, higher is better)
        diversity_ratio = metrics["diversity"].get("diversity_ratio", 0.0)
        # Ideal is 1.0, penalize if too low (mode collapse) or too high (unrealistic)
        diversity_score = 1.0 - abs(1.0 - diversity_ratio) * 0.5  # Penalize deviation
        diversity_score = max(0.0, min(1.0, diversity_score))
        score["diversity_score"] = diversity_score

        # Correlation preservation score (0-1, higher is better)
        # correlation_preservation is corr(real_corr_vec, gen_corr_vec) clamped
        # at 0. Whenever the generator has not yet learned joint structure it is
        # <= 0, so 40% of `overall` becomes a CONSTANT and stops ranking epochs,
        # leaving diversity -- which is MAXIMAL for an untrained generator
        # (ratio 1.0 at init) -- to decide. Self-reinforcing: the selected model
        # never learns correlation, so the term never unpins. Measured on
        # CTAB-TF seed 42: best epoch = 1 at 8, 40 AND 100 epochs, identical
        # eval_quality=0.5766 -- every epoch of training discarded.
        #
        # correlation_error (mean |real_corr - gen_corr|) falls monotonically as
        # the model learns, so it keeps ranking below the point where
        # corr-of-corr saturates. Across 3 seeds this removed the failure mode
        # (worst-case sigma_ratio 0.828 -> 1.101) and gave the best mean
        # placement (worst dmu 1.033 -> 0.823). Joint structure itself is
        # unchanged (dcorr ~1.06) -- this fixes SELECTION, not learning.
        corr_err = metrics["correlation"].get("correlation_error")
        if corr_err is not None:
            corr_preservation = 1.0 - float(corr_err)
        else:
            corr_preservation = metrics["correlation"].get("correlation_preservation", 0.0)
        score["correlation_score"] = max(0.0, min(1.0, corr_preservation))

        # Statistical similarity score (0-1, higher is better)
        mean_error = metrics["statistics"].get("mean_error_avg", 1.0)
        std_error = metrics["statistics"].get("std_error_avg", 1.0)
        cat_error = metrics["statistics"].get("categorical_error_avg", 1.0)
        # Convert errors to scores (lower error = higher score)
        stat_score = 1.0 / (1.0 + mean_error + std_error + cat_error)
        score["statistical_score"] = max(0.0, min(1.0, stat_score))

        # Validity score (0-1, higher is better)
        validity_score = metrics["validity"].get("overall_validity_score", 0.0)
        score["validity_score"] = validity_score

        # Weighted overall score (emphasize diversity and correlation)
        # Weights: diversity=0.4, correlation=0.4, statistics=0.15, validity=0.05
        score["overall_quality"] = (
            diversity_score * 0.4
            + corr_preservation * 0.4
            + stat_score * 0.15
            + validity_score * 0.05
        )

        return score

    def evaluate_with_dataframes(
        self, real_data: pd.DataFrame, generated_data: pd.DataFrame
    ) -> Dict[str, Any]:
        """
        Evaluate generated samples against real data.

        This method allows evaluation with pre-generated dataframes, useful for
        evaluating in different data spaces (e.g., GAN space vs training space).

        Args:
            real_data: Real dataframe to compare against
            generated_data: Generated dataframe to evaluate

        Returns:
            Dictionary of evaluation metrics
        """
        # Ensure same columns and order
        real_data = real_data[self.column_order].copy()
        generated_data = generated_data[self.column_order].copy()

        metrics = {}

        # 1. DIVERSITY METRICS (Critical for avoiding overfitting)
        metrics["diversity"] = self._compute_diversity_metrics(
            real_data, generated_data
        )

        # 2. CORRELATION PRESERVATION (Critical for maintaining feature relationships)
        metrics["correlation"] = self._compute_correlation_metrics(
            real_data, generated_data
        )

        # 3. STATISTICAL SIMILARITY (Quality check)
        metrics["statistics"] = self._compute_statistical_metrics(
            real_data, generated_data
        )

        # 4. VALIDITY CHECKS
        metrics["validity"] = self._compute_validity_metrics(generated_data)

        # 5. OVERALL SCORE (weighted combination)
        metrics["overall_score"] = self._compute_overall_score(metrics)

        return metrics


class CTABGANPlusEnhancedMixin:
    def _train_auxiliary_step(
        self, real_data: tf.Tensor, real_labels: tf.Tensor
    ) -> None:
        """Train auxiliary classifier on real batch (one step)."""
        if self.auxiliary is None:
            return
        self.auxiliary.train_on_batch(real_data, real_labels)

    def save(
        self,
        filepath: str,
        min_buy_gain_threshold: Optional[float] = None,
        min_sell_loss_threshold: Optional[float] = None,
        training_type: Optional[int] = None,
    ):
        """Save generator, discriminator, and auxiliary (if present)."""
        super().save(
            filepath,
            min_buy_gain_threshold=min_buy_gain_threshold,
            min_sell_loss_threshold=min_sell_loss_threshold,
            training_type=training_type,
        )
        if self.auxiliary is not None:
            self.auxiliary.save(os.path.join(filepath, "auxiliary.keras"))

    def load(self, filepath: str) -> Dict[str, Optional[float]]:
        """Load generator, discriminator, and auxiliary if saved."""
        result = super().load(filepath)
        aux_path = os.path.join(filepath, "auxiliary.keras")
        if os.path.exists(aux_path):
            self.auxiliary = tf.keras.models.load_model(aux_path)
        return result
