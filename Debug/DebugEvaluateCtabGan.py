# pragma pylint: disable=C0103, C0114, C0115, C0116, C0301, C0302, C0303, C0325, C0411, C0413
# pragma pylint: disable=W0105, W1203, W1309, W1514, W0613, W0621,
# type: ignore
# pylint: disable=import-error

"""
DebugEvaluateCtabGan - evaluates a trained CTAB-GAN+ model.
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

from Framework.BaseNNStrategy import BaseNNStrategy, HAS_MLX, StrategyConfig
from utils.Environment import Environment
from Framework.BaseStrategy import BaseStrategy, ScalerType, MarketRegime, TradingAction, FlowDirection, MomentumDirection, RiskLevel, GANType
from Framework.BaseNNStrategy import BaseNNStrategy, HAS_MLX, StrategyConfig
from utils.Environment import Environment
from Framework.BaseStrategy import BaseStrategy, ScalerType, MarketRegime, TradingAction, FlowDirection, MomentumDirection, RiskLevel, GANType
from Framework.BaseNNStrategy import BaseNNStrategy, HAS_MLX, StrategyConfig
from utils.Environment import Environment
from Framework.BaseStrategy import BaseStrategy, ScalerType, MarketRegime, TradingAction, FlowDirection, MomentumDirection, RiskLevel, GANType
from Framework.BaseNNStrategy import BaseNNStrategy, HAS_MLX, StrategyConfig
from utils.Environment import Environment
from Framework.BaseStrategy import BaseStrategy, ScalerType, MarketRegime, TradingAction, FlowDirection, MomentumDirection, RiskLevel, GANType
from utils.DataframePopulator import DatasetType


class DebugEvaluateCtabGan(BaseNNStrategy):
    """
    Strategy to evaluate a trained CTAB-GAN+ model.
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
                self.combined_labels = np.concatenate(
                    [self.combined_labels, pair_labels], axis=0
                )

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
        print("CTAB-GAN+ MODEL EVALUATION")
        print("=" * 80)
        print(f"    Total samples collected: {len(self.combined_df)}")
        print(f"    Total unique pairs: {self.pair_count}")
        print()

        # seq_index removed as per USER request

        # Load the saved model
        save_location = os.path.join(self.get_storage_location(), "CTABGANs")
        if not os.path.exists(save_location):
            print(f"    ERROR: CTAB-GAN+ model not found at {save_location}")
            raise FileNotFoundError(f"CTAB-GAN+ model not found at {save_location}")

        generator_path = os.path.join(save_location, "generator.keras")
        metadata_path = os.path.join(save_location, "metadata.pkl")
        if not os.path.exists(generator_path) or not os.path.exists(metadata_path):
            print(f"    ERROR: CTAB-GAN+ model files not found at {save_location}")
            raise FileNotFoundError(
                f"CTAB-GAN+ model files not found at {save_location}"
            )

        print(f"    Loading CTAB-GAN+ model from {save_location}")
        try:
            model = CTABGANPlus()
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

        # Determine original (unflattened) column names from model's flattened column_order
        # Model columns have format like {original_col}_{time_tag} (e.g., "price_t-15", "price_t0")
        # Time tags are either "_t-{number}" or "_t0"
        print("    Determining original columns from model's flattened column names...")
        original_cols_from_model = set()
        for col in model.column_order:
            # Remove time tag - split on "_t-" or "_t0" pattern, not just "_t"
            # This avoids incorrectly splitting columns like "sar_trend_t-5" on the "t" in "trend"
            if "_t-" in col:
                # Column has format like "column_name_t-15"
                original_col = col.rsplit("_t-", 1)[0]
                original_cols_from_model.add(original_col)
            elif col.endswith("_t0"):
                # Column has format like "column_name_t0"
                original_col = col[:-3]  # Remove "_t0" suffix
                original_cols_from_model.add(original_col)

        if not original_cols_from_model:
            error_msg = (
                "ERROR: Could not determine original columns from model column names"
            )
            print(f"    {error_msg}")
            raise ValueError(error_msg)

        print(f"    Found {len(original_cols_from_model)} original columns from model")

        # Check that all required columns are present in eval_df
        # If columns are missing, it's a data preparation error - fail fast
        eval_df_cols = set(eval_df.columns)

        # Check for missing columns - fail fast if any are missing
        missing_orig = original_cols_from_model - eval_df_cols

        if missing_orig:
            missing_list = sorted(list(missing_orig))
            error_msg = (
                f"ERROR: Missing required columns in evaluation data: {missing_list}\n"
                f"    Model expects {len(original_cols_from_model)} original columns.\n"
                f"    Evaluation data has {len(eval_df_cols)} columns.\n"
                f"    Missing: {missing_list}\n"
                f"    This indicates a data preparation error.\n"
                f"    Check that populate_indicators() adds all required columns."
            )
            print(f"    {error_msg}")
            raise ValueError(error_msg)

        # Use all original columns for flattening
        eval_df_for_flatten = eval_df[list(original_cols_from_model)].copy()

        # Determine seq_len (needed for flattening)
        seq_len = None
        if hasattr(self, "seq_len"):
            strategy_seq_len = getattr(self, "seq_len", None)
            if strategy_seq_len is not None:
                seq_len = strategy_seq_len
                print(f"    Using seq_len={seq_len} from strategy configuration")

        if seq_len is None:
            # Infer seq_len from model column names
            # Count unique time tags
            time_tags = set()
            for col in model.column_order:
                if "_t" in col:
                    time_tag = col.split("_t")[-1]
                    time_tags.add(time_tag)
            if time_tags:
                # seq_len should be number of unique time tags
                seq_len = len(time_tags)
                print(
                    f"    Inferred seq_len={seq_len} from model column names ({len(time_tags)} unique time tags)"
                )
                # Store for later use
                self.seq_len = seq_len

        if seq_len is None:
            # Final fallback
            print("    WARNING: Could not determine seq_len, using default 16")
            seq_len = 16
            self.seq_len = seq_len

        # Use all original columns for flattening
        eval_df_for_flatten = eval_df[list(original_cols_from_model)].copy()

        # Flatten the evaluation data to match training format
        # The model was trained on flattened data with time windows
        print("    Flattening evaluation data to match training format...")
        eval_df_flat = self.window_and_flatten(eval_df_for_flatten, seq_len=seq_len)
        print(
            f"    Flattened: {len(eval_df_for_flatten)} rows -> {len(eval_df_flat)} rows"
        )
        print(
            f"    Flattened: {len(eval_df_for_flatten.columns)} cols -> {len(eval_df_flat.columns)} cols"
        )

        # Verify flattened columns match model exactly
        # If they don't match, it's a data preparation error - fail fast
        model_cols_set = set(model.column_order)
        flat_cols_set = set(eval_df_flat.columns)

        missing_cols = model_cols_set - flat_cols_set
        if missing_cols:
            # Try to identify which original columns are missing
            missing_original_from_flat = set()
            for col in missing_cols:
                if "_t" in col:
                    parts = col.split("_t")
                    if len(parts) > 1:
                        original_col = "_".join(parts[:-1])
                        missing_original_from_flat.add(original_col)

            missing_list = sorted(list(missing_cols))[:20]
            error_msg = (
                f"ERROR: Flattened data missing {len(missing_cols)} columns required by model.\n"
                f"    Model expects {len(model_cols_set)} flattened columns.\n"
                f"    Flattened data has {len(flat_cols_set)} columns.\n"
                f"    Missing original columns: {sorted(missing_original_from_flat)}\n"
                f"    Sample missing flattened columns: {missing_list}...\n"
                f"    This indicates a data preparation error.\n"
                f"    Check that seq_len matches and all required columns are present."
            )
            print(f"    {error_msg}")
            raise ValueError(error_msg)

        # Use all columns that the model expects (in flattened format)
        eval_df_flat = eval_df_flat[model.column_order].copy()

        # Shuffle and sample to get representative data from all pairs
        # Note: Use flattened data for sampling
        eval_sample_size = min(2000, len(eval_df_flat))
        if len(eval_df_flat) > eval_sample_size:
            eval_df_sampled = eval_df_flat.sample(
                n=eval_sample_size, random_state=42
            ).reset_index(drop=True)
            print(
                f"    Sampled {eval_sample_size} rows (shuffled) from {len(eval_df_flat)} total flattened rows"
            )
        else:
            eval_df_sampled = eval_df_flat
            print(f"    Using all {len(eval_df_flat)} flattened rows for evaluation")

        # Evaluate the model in GAN space
        print(
            f"\n    Evaluating model in GAN space with {len(eval_df_sampled)} samples..."
        )
        print("    " + "=" * 76)
        print("    GAN SPACE EVALUATION (minmax normalized to [-1, 1])")
        print("    " + "=" * 76)
        try:
            # Generate data for both evaluations
            generated_gan = model.generate(num_samples=len(eval_df_sampled))

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
        # Extract t0 columns from flattened data to get unflattened format
        # The t0 columns represent the "current" time step for each sample
        print(
            f"\n    Evaluating model in training space with {len(eval_df_sampled)} samples..."
        )
        print("    " + "=" * 76)
        print(
            "    TRAINING SPACE EVALUATION (denormalized, using t0 columns from flattened data)"
        )
        print("    " + "=" * 76)
        try:
            # Extract t0 columns (current time step) from flattened data
            # This gives us unflattened data that matches the GAN scaler format
            t0_cols = [col for col in eval_df_sampled.columns if col.endswith("_t0")]
            if not t0_cols:
                raise ValueError("No t0 columns found in flattened data")

            # Extract original column names (remove _t0 suffix)
            original_cols = [col[:-3] for col in t0_cols]  # Remove '_t0'

            # Create unflattened DataFrames with t0 columns only
            eval_df_unflat = eval_df_sampled[t0_cols].copy()
            eval_df_unflat.columns = original_cols

            generated_df_unflat = generated_gan[t0_cols].copy()
            generated_df_unflat.columns = original_cols

            # Now we can denormalize using the GAN scaler (expects unflattened format)
            eval_df_training = self.denormalise_from_gan(eval_df_unflat)
            generated_training = self.denormalise_from_gan(generated_df_unflat)

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
