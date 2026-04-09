# pragma pylint: disable=C0103, C0114, C0115, C0116, C0301, C0302, C0303, C0325, C0411, C0413
# pragma pylint: disable=W0105, W1203, W1309, W1514, W0613, W0621,
# type: ignore
# pylint: disable=import-error

"""
DebugEvaluateMTCtabGan - evaluates a trained Multi-Task CTAB-GAN+ model.
Collects dataframes and labels from all pairs, loads the saved model, and evaluates it.
Prints comprehensive metrics including diversity, correlation preservation, and overall quality.
"""

import sys
from pathlib import Path
from pandas import DataFrame
import numpy as np
import pandas as pd
import traceback
import os
from typing import Dict, Any

# Add parent directory to path to from Framework.BaseNNStrategy import BaseNNStrategy, HAS_MLX, StrategyConfig
from utils.Environment import Environment
from Framework.BaseStrategy import BaseStrategy, ScalerType, MarketRegime, TradingAction, FlowDirection, MomentumDirection, RiskLevel, GANType
from Framework.BaseNNStrategy import BaseNNStrategy, HAS_MLX, StrategyConfig
from utils.Environment import Environment
from Framework.BaseStrategy import BaseStrategy, ScalerType, MarketRegime, TradingAction, FlowDirection, MomentumDirection, RiskLevel, GANType
group_dir = str(Path(__file__).parent)
sys.path.append(group_dir)

from NNMTStrategy import NNMTStrategy
from utils.df_mt_ctab_gan import CTABGANPlusMT
from utils.DataframePopulator import DatasetType


class DebugEvaluateMTCtabGan(NNMTStrategy):
    """
    Strategy to evaluate a trained Multi-Task CTAB-GAN+ model.
    Collects data from all pairs and evaluates the model's performance.
    """

    pair_count = 0
    combined_df = None
    combined_labels = None
    processed_pairs = set()
    evaluation_done = False

    def populate_indicators(self, dataframe: DataFrame, metadata: Dict) -> DataFrame:
        """Populate indicators and collect data for evaluation."""
        # dataframe = super().populate_indicators(dataframe, metadata)

        whitelist = self.dp.current_whitelist()
        curr_pair = metadata.get("pair", "unknown")

        self.iteration_init()
        dataframe = self.check_precision_columns(dataframe)
        dataframe = self.dataframePopulator.add_indicators(
            dataframe, dataset_type=DatasetType.MINIMAL
        )
        dataframe = self.add_additional_indicators(dataframe)
        # seq_index removed as per USER request

        # Track unique pairs (avoid double counting)
        if curr_pair not in self.processed_pairs:
            self.processed_pairs.add(curr_pair)
            self.pair_count = len(self.processed_pairs)

            # Collect dataframe and labels for this pair
            if self.combined_df is None:
                self.combined_df = dataframe.reset_index(drop=True).copy()
                self.combined_labels = self.get_training_labels(dataframe)
            else:
                self.combined_df = pd.concat(
                    [self.combined_df, dataframe.reset_index(drop=True)],
                    ignore_index=True,
                )
                pair_labels = self.get_training_labels(dataframe)
                # Merge labels from all pairs
                for task in pair_labels.keys():
                    if task in self.combined_labels:
                        self.combined_labels[task] = np.concatenate(
                            [self.combined_labels[task], pair_labels[task]], axis=0
                        )
                    else:
                        self.combined_labels[task] = pair_labels[task]

            print(
                f"    Collected data from pair {curr_pair} ({self.pair_count}/{len(whitelist)} pairs)"
            )

        # When all pairs processed, run evaluation (only once)
        if (
            not self.evaluation_done
            and self.pair_count == len(whitelist)
            and self.combined_df is not None
            and len(self.combined_df) > 0
        ):
            self.evaluation_done = True
            self._run_evaluation()

        return dataframe

    def _run_evaluation(self) -> None:
        """Run the evaluation (called once when all pairs are processed)."""
        print("\n" + "=" * 80)
        print("MULTI-TASK CTAB-GAN+ MODEL EVALUATION")
        print("=" * 80)
        print(f"    Total samples collected: {len(self.combined_df)}")
        print(f"    Total unique pairs: {self.pair_count}")
        print()

        # Load the saved model
        save_location = os.path.join(self.get_storage_location(), "MTCTABGANs")
        if not os.path.exists(save_location):
            print(f"    ERROR: Multi-Task CTAB-GAN+ model not found at {save_location}")
            raise FileNotFoundError(
                f"Multi-Task CTAB-GAN+ model not found at {save_location}"
            )

        generator_path = os.path.join(save_location, "generator.keras")
        metadata_path = os.path.join(save_location, "metadata.pkl")
        if not os.path.exists(generator_path) or not os.path.exists(metadata_path):
            print(
                f"    ERROR: Multi-Task CTAB-GAN+ model files not found at {save_location}"
            )
            raise FileNotFoundError(
                f"Multi-Task CTAB-GAN+ model files not found at {save_location}"
            )

        print(f"    Loading Multi-Task CTAB-GAN+ model from {save_location}")
        try:
            model = CTABGANPlusMT()
            model.load(save_location)
            print("    ✅ Model loaded successfully")
        except Exception as e:
            print(f"    ERROR: Failed to load model: {e}")
            print(traceback.format_exc())
            raise

        # Load GAN scaler (REQUIRED for meaningful evaluation)
        print("    Loading GAN scaler...")
        scaler_dir = self.get_storage_location()
        # Import here to avoid circular dependencies
        from utils.Scalers import scaler_exists, load_scaler  # noqa: E402

        if not scaler_exists(scaler_dir, self.gan_scaler_a_name):
            error_msg = (
                f"ERROR: GAN scaler ({self.gan_scaler_a_name}) not found at {scaler_dir}. "
                "Evaluation requires the GAN scaler to match training data format. "
                "Please ensure CreateScalers has been run first."
            )
            print(f"    {error_msg}")
            raise FileNotFoundError(error_msg)

        self.gan_scaler_a = load_scaler(scaler_dir, self.gan_scaler_a_name)
        if self.gan_scaler_a is None:
            error_msg = f"ERROR: Failed to load GAN scaler ({self.gan_scaler_a_name})"
            print(f"    {error_msg}")
            raise ValueError(error_msg)

        print("    ✅ GAN scaler loaded successfully")
        print(f"      columns: {self.gan_scaler_a.feature_columns}")

        # seq_index removed as per USER request

        # Prepare data for evaluation (MUST use GAN-scaled format to match training)
        try:
            # Normalize first, then apply GAN scaling (same as training)
            normalized_df = self.rolling_dataframe_normalise(self.combined_df)
            eval_df = self.normalise_for_gan(normalized_df)
            print("    ✅ Data prepared using GAN-scaled format (matching training)")
        except Exception as e:
            error_msg = f"ERROR: Failed to prepare data with GAN scaler: {e}"
            print(f"    {error_msg}")
            print(traceback.format_exc())
            raise

        # Ensure columns match model
        model_cols = set(model.column_order)
        eval_cols = set(eval_df.columns)

        if not model_cols.issubset(eval_cols):
            missing = model_cols - eval_cols
            error_msg = f"ERROR: Data missing columns required by model: {missing}"
            print(f"    {error_msg}")
            raise ValueError(error_msg)

        # Use only columns that the model expects
        eval_df = eval_df[model.column_order].copy()

        # Convert labels to one-hot encoded format for GAN
        eval_labels_one_hot = {}
        for task, labels in self.combined_labels.items():
            arr = np.asarray(labels)
            if arr.ndim == 1:
                # Convert class indices to one-hot
                num_classes = int(arr.max()) + 1
                eval_labels_one_hot[task] = np.eye(num_classes, dtype=np.float32)[
                    arr.astype(int)
                ]
            elif arr.ndim == 2:
                eval_labels_one_hot[task] = arr.astype(np.float32)
            else:
                raise ValueError(f"Task '{task}' labels must be 1D or 2D array")

        # Shuffle and sample to get representative data from all pairs
        eval_sample_size = min(2000, len(eval_df))
        if len(eval_df) > eval_sample_size:
            # Sample indices
            eval_indices = np.random.choice(
                len(eval_df), eval_sample_size, replace=False
            )
            eval_df_sampled = eval_df.iloc[eval_indices].reset_index(drop=True)
            # Sample corresponding labels
            eval_labels_sampled = {
                task: labels[eval_indices]
                for task, labels in eval_labels_one_hot.items()
            }
            print(
                f"    Sampled {eval_sample_size} rows (shuffled) from {len(eval_df)} total rows"
            )
        else:
            eval_df_sampled = eval_df
            eval_labels_sampled = eval_labels_one_hot
            print(f"    Using all {len(eval_df)} rows for evaluation")

        # Evaluate the model in GAN space
        print(
            f"\n    Evaluating model in GAN space with {len(eval_df_sampled)} samples..."
        )
        print("    " + "=" * 76)
        print("    GAN SPACE EVALUATION (minmax normalized to [-1, 1])")
        print("    " + "=" * 76)
        try:
            # Generate data with task labels from real data
            generated_gan, _ = model.generate(
                num_samples=len(eval_df_sampled), task_labels=eval_labels_sampled
            )

            # Evaluate in GAN space
            metrics_gan = model.evaluate_with_dataframes(eval_df_sampled, generated_gan)
            print("    ✅ GAN space evaluation completed\n")

            # Print GAN space results
            self._print_metrics(metrics_gan, space_name="GAN Space")

        except Exception as e:
            print(f"    ERROR: GAN space evaluation failed: {e}")
            print(traceback.format_exc())
            raise

        # Evaluate the model in training space (denormalized)
        print(
            f"\n    Evaluating model in training space with {len(eval_df_sampled)} samples..."
        )
        print("    " + "=" * 76)
        print("    TRAINING SPACE EVALUATION (denormalized from GAN space)")
        print("    " + "=" * 76)
        print("    ⚠️  NOTE: Validity score in training space may be negative/invalid")
        print("    because validity checks use GAN space bounds. Focus on diversity,")
        print("    correlation, and statistical metrics instead.\n")
        try:
            # Denormalize both real and generated data
            eval_df_training = self.denormalise_from_gan(eval_df_sampled)
            generated_training = self.denormalise_from_gan(generated_gan)

            # Evaluate in training space
            metrics_training = model.evaluate_with_dataframes(
                eval_df_training, generated_training
            )
            print("    ✅ Training space evaluation completed\n")

            # Print training space results
            self._print_metrics(metrics_training, space_name="Training Space")

        except Exception as e:
            print(f"    ERROR: Training space evaluation failed: {e}")
            print(traceback.format_exc())
            raise

    def _print_metrics(self, metrics: Dict[str, Any], space_name: str = "") -> None:
        """Print evaluation metrics in a readable format."""
        overall = metrics.get("overall_score", {})
        diversity = metrics.get("diversity", {})
        correlation = metrics.get("correlation", {})
        statistics = metrics.get("statistics", {})
        validity = metrics.get("validity", {})

        if space_name:
            print(f"\n📊 OVERALL QUALITY SCORES ({space_name}):")
        else:
            print("📊 OVERALL QUALITY SCORES:")
        quality = overall.get("overall_quality", 0.0)
        div_score = overall.get("diversity_score", 0.0)
        corr_score = overall.get("correlation_score", 0.0)
        stat_score = overall.get("statistical_score", 0.0)
        valid_score = overall.get("validity_score", 0.0)
        print(f"  Overall Quality:     {quality:.4f} (0-1, higher is better)")
        print(f"  Diversity Score:     {div_score:.4f} (0-1, higher is better)")
        print(f"  Correlation Score:   {corr_score:.4f} (0-1, higher is better)")
        print(f"  Statistical Score:  {stat_score:.4f} (0-1, higher is better)")
        print(f"  Validity Score:     {valid_score:.4f} (0-1, higher is better)")

        print("\n🎲 DIVERSITY METRICS (Critical for avoiding overfitting):")
        div_ratio = diversity.get("diversity_ratio", 0.0)
        gen_dist = diversity.get("gen_pairwise_distance_mean", 0.0)
        real_dist = diversity.get("real_pairwise_distance_mean", 0.0)
        coverage = diversity.get("value_space_coverage", 0.0)
        nn_dist = diversity.get("nearest_real_distance_mean", 0.0)
        print(f"  Diversity Ratio:              {div_ratio:.4f} (target: ~1.0)")
        print(f"  Gen Pairwise Distance Mean:    {gen_dist:.4f}")
        print(f"  Real Pairwise Distance Mean:   {real_dist:.4f}")
        print(
            f"  Value Space Coverage:          {coverage:.4f} (0-1, higher is better)"
        )
        print(f"  Nearest Real Distance Mean:    {nn_dist:.4f}")

        print("\n🔗 CORRELATION PRESERVATION (Critical for feature relationships):")
        corr_pres = correlation.get("correlation_preservation", 0.0)
        corr_err = correlation.get("correlation_error", 0.0)
        print(
            f"  Correlation Preservation:      {corr_pres:.4f} (0-1, higher is better)"
        )
        print(f"  Correlation Error:            {corr_err:.4f} (lower is better)")

        print("\n📈 STATISTICAL SIMILARITY:")
        mean_err = statistics.get("mean_error_avg", 0.0)
        std_err = statistics.get("std_error_avg", 0.0)
        cat_err = statistics.get("categorical_error_avg", 0.0)
        print(f"  Mean Error (avg):              {mean_err:.4f} (lower is better)")
        print(f"  Std Error (avg):               {std_err:.4f} (lower is better)")
        print(f"  Categorical Error (avg):       {cat_err:.4f} (lower is better)")

        print("\n✅ VALIDITY CHECKS:")
        valid_score = validity.get("overall_validity_score", 0.0)
        print(
            f"  Overall Validity Score:        {valid_score:.4f} (0-1, higher is better)"
        )

        # Interpretation
        overall_quality = quality
        diversity_score = div_score

        print("\n💡 INTERPRETATION:")
        if overall_quality >= 0.8:
            print(
                "  ✅ Excellent: Model is generating high-quality, diverse samples with preserved correlations"
            )
        elif overall_quality >= 0.6:
            print(
                "  ⚠️  Good: Model is generating reasonable samples, but may need improvement"
            )
        elif overall_quality >= 0.4:
            print(
                "  ⚠️  Fair: Model needs improvement in diversity or correlation preservation"
            )
        else:
            print(
                "  ❌ Poor: Model is likely overfitting or not learning the data distribution well"
            )

        if diversity_score < 0.5:
            print("  ⚠️  WARNING: Low diversity - possible mode collapse detected!")
        if corr_score < 0.5:
            print(
                "  ⚠️  WARNING: Low correlation preservation - feature relationships may be lost!"
            )

        print("\n" + "=" * 80)

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """No-op for this debug strategy"""
        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """No-op for this debug strategy"""
        return dataframe
