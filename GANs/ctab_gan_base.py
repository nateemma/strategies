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
