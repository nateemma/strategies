# pragma pylint: disable=C0103, C0114, C0115, C0116, C0301, C0302, C0303, C0325, C0411, C0413
# pragma pylint: disable=W0105, W1203, W1309, W1514, W0613, W0621,
# type: ignore
# pylint: disable=import-error

"""
CreateCtabGanPlus - creates and saves CTAB-GAN+ models using data from all of the pairs in
the whitelist. CTAB-GAN+ is specifically designed for tabular data with mixed data types
(continuous and categorical).
"""

from __future__ import annotations

import sys
import traceback
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd
from pandas import DataFrame

group_dir = str(Path(__file__).parent)
sys.path.append(group_dir)

from CreateGANBase import CreateGANBase  # noqa: E402
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
from utils.df_ctab_gan import CTABGANPlusEnhanced  # noqa: E402
try:
    from utils.df_ctab_gan_mlx import CTABGANMLX
    from Framework.BaseNNStrategy import BaseNNStrategy, HAS_MLX, StrategyConfig
from utils.Environment import Environment
from Framework.BaseStrategy import BaseStrategy, ScalerType, MarketRegime, TradingAction, FlowDirection, MomentumDirection, RiskLevel, GANType
from Framework.BaseNNStrategy import BaseNNStrategy, HAS_MLX, StrategyConfig
from utils.Environment import Environment
from Framework.BaseStrategy import BaseStrategy, ScalerType, MarketRegime, TradingAction, FlowDirection, MomentumDirection, RiskLevel, GANType
except (ImportError, ModuleNotFoundError):
    HAS_CTAB_MLX = False
    HAS_MLX = False


class CreateCtabGanPlus(BaseNNStrategy):
    """
    Creates and saves CTAB-GAN+ models.

    NOTE: This class defines the MASTER threshold values that are stored in the GAN metadata.
    These values are the source of truth - any strategy that loads this GAN will use these
    thresholds, preventing mismatches between GAN training and model training.

    To change the thresholds, update them here and retrain the GAN. Strategies that use
    the GAN will automatically load and use these values from the GAN metadata.
    """

    # MASTER threshold values - these are stored in GAN metadata and used by all strategies
    # that load the GAN. Do not change these without retraining the GAN!
    MASTER_MIN_BUY_GAIN_THRESHOLD = 0.016
    MASTER_MIN_SELL_LOSS_THRESHOLD = 0.012
    MASTER_TRAINING_TYPE = 19
    # Keep local strategy defaults aligned with MASTER values
    MIN_BUY_GAIN_THRESHOLD = (
        MASTER_MIN_BUY_GAIN_THRESHOLD  # minimum gain for buy signals
    )
    MIN_SELL_LOSS_THRESHOLD = (
        MASTER_MIN_SELL_LOSS_THRESHOLD  # minimum loss for sell signals
    )
    TRAINING_TYPE = MASTER_TRAINING_TYPE

    DEFAULT_GAN_CONFIG: Dict[str, Any] = {
        "name": "CTAB-GAN+",
        "description": "CTAB-GAN+",
        "train_kwargs": {
            "epochs": 300,
            "batch_size": 2048,  # Power of 2 for optimal GPU utilization
            "latent_dim": 128,
            "generator_layers": [256, 256],
            "discriminator_layers": [256, 256],
            "learning_rate": 2e-4,
            "beta_1": 0.2,  # Reduced for better stability
            "beta_2": 0.999,
            "gp_weight": 10.0,
            "verbose": True,
            "integer_columns": [],
        },
        "augmentation_target_ratio": 1.0,  # Augment minority classes to % of majority class size
        "save_subdir": "CTABGANs",
        "multi_task": False,
        "categorical_columns": None,  # Will be auto-detected from one_hot_columns
    }

    aggregate_pairs = True  # use all pairs for training

    def __init__(self, gan_config: Optional[Dict[str, Any]] = None, **kwargs) -> None:
        super().__init__(gan_config=gan_config, **kwargs)

    def iteration_init(self):
        """Override to force MASTER threshold values for GAN creation.

        Create*GAN strategies are the source of truth for gain/loss/training_type parameters.
        This ensures that training labels are generated using the MASTER thresholds,
        which will then be stored in the GAN metadata. This prevents mismatches
        between the thresholds used for label generation and those stored in the GAN.
        """
        self._validate_master_thresholds()
        # Set MASTER values FIRST - these are the source of truth for GAN creation
        # This ensures training labels are generated with the same thresholds that will
        # be stored in the GAN metadata
        self.MIN_BUY_GAIN_THRESHOLD = self.MASTER_MIN_BUY_GAIN_THRESHOLD
        self.MIN_SELL_LOSS_THRESHOLD = self.MASTER_MIN_SELL_LOSS_THRESHOLD
        self.TRAINING_TYPE = self.MASTER_TRAINING_TYPE

        # Mark that we're a GAN creation strategy (prevents parent from overriding)
        self._is_gan_creation_strategy = True

        # Call parent - it will skip setting parameters since _is_gan_creation_strategy is True
        super().iteration_init()

    def _validate_master_thresholds(self) -> None:
        strategy_vals = {
            "MASTER_MIN_BUY_GAIN_THRESHOLD": BaseNNStrategy.MIN_BUY_GAIN_THRESHOLD,
            "MASTER_MIN_SELL_LOSS_THRESHOLD": BaseNNStrategy.MIN_SELL_LOSS_THRESHOLD,
            "MASTER_TRAINING_TYPE": BaseNNStrategy.TRAINING_TYPE,
        }
        local_vals = {
            "MASTER_MIN_BUY_GAIN_THRESHOLD": self.MASTER_MIN_BUY_GAIN_THRESHOLD,
            "MASTER_MIN_SELL_LOSS_THRESHOLD": self.MASTER_MIN_SELL_LOSS_THRESHOLD,
            "MASTER_TRAINING_TYPE": self.MASTER_TRAINING_TYPE,
        }
        mismatches = {
            key: (strategy_vals[key], local_vals[key])
            for key in local_vals
            if strategy_vals[key] != local_vals[key]
        }
        if mismatches:
            lines = [
                "CreateCtabGanPlus MASTER_* mismatch with NNStrategy:",
                *[
                    f"  {key}: NNStrategy={vals[0]} vs CreateCtabGanPlus={vals[1]}"
                    for key, vals in mismatches.items()
                ],
            ]
            raise ValueError("\n".join(lines))

    def _get_categorical_columns(self, dataframe: DataFrame) -> list[str]:
        """
        Identify categorical columns from the dataframe.

        Categorical columns are derived from one_hot_columns (e.g., flow_0, flow_2,
        regime_0, regime_2).
        """
        categorical_cols = []

        # Get one-hot encoded columns (e.g., flow_0, flow_2, regime_0, regime_2)
        for base_col in self.one_hot_columns:
            # Find all columns that start with base_col_ (e.g., flow_0, flow_2)
            matching_cols = [
                col for col in dataframe.columns if col.startswith(f"{base_col}_")
            ]
            categorical_cols.extend(matching_cols)

        # Remove duplicates and ensure columns exist in dataframe
        categorical_cols = list(
            set([col for col in categorical_cols if col in dataframe.columns])
        )

        return categorical_cols

    def run_gan_training(
        self,
        *,
        combined_df: DataFrame,
        train_data: np.ndarray,
        test_data: np.ndarray,
        train_labels: np.ndarray,
        test_labels: np.ndarray,
        config: Dict[str, Any],
    ) -> None:
        try:
            original_shape = np.shape(train_data)
            print(f"    Balancing training data with {config.get('name', 'CTAB-GAN+')}")

            if len(train_data) == 0:
                print("    No training data to balance")
                return

            # Get training kwargs
            train_kwargs = dict(config.get("train_kwargs", {}))
            save_path = train_kwargs.pop("save_path", self.get_gan_save_path(config))

            # Get column names: from GAN scaler normally, or from PCA passthrough+components
            if self.use_pca_reduction and self.pca_n_components is not None:
                passthrough = [
                    c for c in self.pca_passthrough_columns
                    if c in (self.pca_feature_columns or []) or c in (self.include_list or [])
                ]
                pca_cols = [f"pca_{i}" for i in range(self.pca_n_components)]
                train_df_columns = passthrough + pca_cols
            elif (
                hasattr(self, "gan_scaler_a")
                and self.gan_scaler_a is not None
                and hasattr(self.gan_scaler_a, "feature_columns")
            ):
                train_df_columns = self.gan_scaler_a.feature_columns
            else:
                raise ValueError(
                    "GAN scaler (gan_scaler_a) is not available. "
                    "CTAB-GAN+ requires the GAN-scaled dataframe with feature_columns."
                )

            if len(train_df_columns) != train_data.shape[1]:
                raise ValueError(
                    f"Column count mismatch: expected {len(train_df_columns)} columns, "
                    f"but train_data has {train_data.shape[1]} columns"
                )

            # Convert train_data back to DataFrame for CTAB-GAN+
            train_df = pd.DataFrame(train_data, columns=train_df_columns)

            # Get categorical columns (filter to only those that exist in train_df)
            categorical_columns = config.get("categorical_columns")
            if categorical_columns is None:
                # Auto-detect from the normalized dataframe columns
                categorical_columns = self._get_categorical_columns(train_df)
                print(f"    Auto-detected categorical columns: {categorical_columns}")
            else:
                # Filter to only include columns that exist in train_df
                categorical_columns = [
                    col for col in categorical_columns if col in train_df.columns
                ]
                print(
                    f"    Using provided categorical columns (filtered): {categorical_columns}"
                )

            # Process labels to ensure one-hot encoding
            labels_arr = np.asarray(train_labels)
            if labels_arr.ndim == 1:
                # Convert to one-hot if needed
                num_classes = int(labels_arr.max()) + 1
                train_labels_processed = np.eye(num_classes, dtype=np.float32)[
                    labels_arr.astype(int)
                ]
            else:
                train_labels_processed = labels_arr.astype(np.float32)

            train_idx = train_labels_processed.argmax(axis=1)
            classes, counts = np.unique(train_idx, return_counts=True)
            class_counts = dict(zip(classes.tolist(), counts.tolist()))
            print(
                f"    Train set size: {len(train_data)}  "
                f"Class counts: {class_counts}"
            )

            current_max = int(counts.max()) if counts.size > 0 else 0
            if current_max <= 0:
                print("    No majority class found, skipping balancing")
                return

            # Use augmentation_target_ratio
            augmentation_target_ratio = config.get("augmentation_target_ratio", 0.4)
            target = (
                int(current_max * augmentation_target_ratio)
                if current_max > 0
                else None
            )
            if target is None or target <= 0:
                print("    No target found, skipping balancing")
                return

            # Calculate needs for each class
            num_classes = train_labels_processed.shape[1]
            have_map = {
                c: int(train_labels_processed[:, c].sum()) for c in range(num_classes)
            }
            needs_map = {
                c: max(0, target - have_map.get(c, 0)) for c in range(num_classes)
            }
            print(
                f"    CTAB-GAN+ target per class: {target} "
                f"(ratio={augmentation_target_ratio})  Planned adds: {needs_map}"
            )
            if all(v <= 0 for v in needs_map.values()):
                print("    Already at or above target; skipping CTAB-GAN+")
                return

            # Print MASTER values that will be stored in GAN metadata (before training starts)
            print("    CTAB-GAN+ training starting...")
            print("    MASTER thresholds (will be stored in GAN metadata):")
            print(
                f"      MASTER_MIN_BUY_GAIN_THRESHOLD = {self.MASTER_MIN_BUY_GAIN_THRESHOLD:.4f}"
            )
            sell_thresh = self.MASTER_MIN_SELL_LOSS_THRESHOLD
            print(f"      MASTER_MIN_SELL_LOSS_THRESHOLD = {sell_thresh:.4f}")
            print(f"      MASTER_TRAINING_TYPE = {self.MASTER_TRAINING_TYPE}")

            # Ensure random seed is set in train_kwargs (defaults to 42 if not specified)
            if "random_seed" not in train_kwargs:
                train_kwargs["random_seed"] = 42

            # Initialize CTAB-GAN+ (Prefer MLX version for speed on Apple Silicon if no categorical columns)
            # Since the user has removed categorical columns, MLX is much faster here.
            if HAS_CTAB_MLX and HAS_MLX and (not categorical_columns or len(categorical_columns) == 0):
                print(f"    Using MLX-accelerated CTAB-GAN (epochs={train_kwargs.get('epochs', 100)})")
                ctab_gan = CTABGANMLX(
                    latent_dim=train_kwargs.get("latent_dim", 128),
                    epochs=train_kwargs.get("epochs", 300),
                    batch_size=train_kwargs.get("batch_size", 2048),
                    verbose=True
                )
            else:
                # Use standard Enhanced Keras version (supports Categorical + Auxiliary loss)
                print("    Using standard Enhanced Keras CTAB-GAN")
                ctab_gan = CTABGANPlusEnhanced(
                    use_auxiliary=True,
                    info_loss_weight=0.02,  # mean/std matching (keep low for stability)
                    downstream_loss_weight=0.02,  # CE(labels, A(fake))
                    generator_loss_weight=0.01,  # condition matching
                    **train_kwargs,
                )

            # Fit the model
            ctab_gan.fit(
                dataframe=train_df,
                labels=train_labels_processed,
                categorical_columns=categorical_columns,
                validation_split=0.1,
            )

            # Use MASTER threshold values - these are the source of truth stored in GAN metadata
            # All strategies that load this GAN will use these values, ensuring consistency
            min_buy_gain = self.MASTER_MIN_BUY_GAIN_THRESHOLD
            min_sell_loss = self.MASTER_MIN_SELL_LOSS_THRESHOLD
            training_type = self.MASTER_TRAINING_TYPE

            # Save the model with thresholds and training_type
            ctab_gan.save(
                save_path,
                min_buy_gain_threshold=min_buy_gain,
                min_sell_loss_threshold=min_sell_loss,
                training_type=training_type,
            )
            print(f"    CTAB-GAN+ model saved to {save_path}")
            if min_buy_gain is not None:
                min_sell_str = (
                    f"{min_sell_loss:.4f}" if min_sell_loss is not None else "None"
                )
                print(
                    f"      Stored thresholds: min_buy_gain={min_buy_gain:.4f}, "
                    f"min_sell_loss={min_sell_str}, training_type={training_type}"
                )

            # Evaluate the model after training (in both GAN space and training space)
            eval_sample_size = min(2000, len(train_df))
            print("\n    Evaluating CTAB-GAN+ model...")

            # Generate data once for both evaluations
            eval_indices = np.random.choice(
                len(train_df), eval_sample_size, replace=False
            )
            generated_gan = ctab_gan.generate(num_samples=eval_sample_size)

            # Evaluate in GAN space
            print("    GAN Space Evaluation (minmax normalized to [-1, 1]):")
            try:
                eval_df_gan = train_df.iloc[eval_indices]
                eval_metrics_gan = ctab_gan.evaluate_with_dataframes(
                    eval_df_gan, generated_gan
                )
                overall_gan = eval_metrics_gan.get("overall_score", {})
                print(
                    "      Overall Quality:     {:.4f}".format(
                        overall_gan.get("overall_quality", 0.0)
                    )
                )
                print(
                    "      Diversity Score:     {:.4f}".format(
                        overall_gan.get("diversity_score", 0.0)
                    )
                )
                print(
                    "      Correlation Score:   {:.4f}".format(
                        overall_gan.get("correlation_score", 0.0)
                    )
                )
                print(
                    "      Statistical Score:  {:.4f}".format(
                        overall_gan.get("statistical_score", 0.0)
                    )
                )
                print(
                    "      Validity Score:     {:.4f}".format(
                        overall_gan.get("validity_score", 0.0)
                    )
                )

                quality_gan = overall_gan.get("overall_quality", 0.0)
                if quality_gan < 0.6:
                    print(
                        f"      ⚠️  WARNING: Low overall quality ({quality_gan:.4f}) - model may need improvement"
                    )
                elif quality_gan >= 0.8:
                    print(f"      ✅ Excellent model quality ({quality_gan:.4f})")
            except Exception as eval_exc:
                print(f"      Evaluation failed: {eval_exc}")
                print(traceback.format_exc())

            # Evaluate in training space (denormalized)
            print("    Training Space Evaluation (denormalized from GAN space):")
            try:
                # Denormalize both real and generated data
                eval_df_training = self.denormalise_from_gan(
                    train_df.iloc[eval_indices]
                )
                generated_training = self.denormalise_from_gan(generated_gan)

                eval_metrics_training = ctab_gan.evaluate_with_dataframes(
                    eval_df_training, generated_training
                )
                overall_training = eval_metrics_training.get("overall_score", {})
                print(
                    "      Overall Quality:     {:.4f}".format(
                        overall_training.get("overall_quality", 0.0)
                    )
                )
                print(
                    "      Diversity Score:     {:.4f}".format(
                        overall_training.get("diversity_score", 0.0)
                    )
                )
                print(
                    "      Correlation Score:   {:.4f}".format(
                        overall_training.get("correlation_score", 0.0)
                    )
                )
                stat_score = overall_training.get("statistical_score", 0.0)
                print(f"      Statistical Score:  {stat_score:.4f}")
                print(
                    "      Validity Score:     {:.4f}".format(
                        overall_training.get("validity_score", 0.0)
                    )
                )

                quality_training = overall_training.get("overall_quality", 0.0)
                if quality_training < 0.6:
                    msg = f"      ⚠️  WARNING: Low overall quality ({quality_training:.4f})"
                    print(f"{msg} - model may need improvement")
                elif quality_training >= 0.8:
                    print(f"      ✅ Excellent model quality ({quality_training:.4f})")
            except Exception as eval_exc:
                print(f"      Evaluation failed: {eval_exc}")
                print(traceback.format_exc())

            # Generate augmented samples for each class that needs augmentation
            aug_data_list = [train_data]
            aug_labels_list = [train_labels_processed]

            for class_idx, need_count in needs_map.items():
                if need_count <= 0:
                    continue

                print(f"    Generating {need_count} samples for class {class_idx}")

                # Generate synthetic data
                generated_df = ctab_gan.generate(
                    num_samples=need_count,
                    class_label=int(class_idx),
                )

                # Convert back to numpy array (maintain column order from train_df)
                generated_array = generated_df[train_df_columns].values.astype(
                    np.float32
                )
                aug_data_list.append(generated_array)

                # Create labels for generated samples
                class_labels = np.zeros(
                    (need_count, train_labels_processed.shape[1]), dtype=np.float32
                )
                class_labels[:, class_idx] = 1.0
                aug_labels_list.append(class_labels)

            if aug_data_list:
                aug_x = np.concatenate(aug_data_list, axis=0)
                aug_y = np.concatenate(aug_labels_list, axis=0)

                aug_idx = aug_y.argmax(axis=1)
                aug_classes, aug_counts = np.unique(aug_idx, return_counts=True)
                aug_class_counts = dict(zip(aug_classes.tolist(), aug_counts.tolist()))

                print("    CTAB-GAN+ training complete")
                print(
                    f"    Augmented train size: {len(aug_x)}  Class counts: {aug_class_counts}"
                )
                print(
                    f"    CTAB-GAN+ effect: shape {original_shape} -> {np.shape(aug_x)}"
                )
            else:
                print("    CTAB-GAN+ training complete, but no augmentation needed")

        except Exception as exc:
            print("    CTAB-GAN+ encountered an error; returning original data")
            print(f"      Error: {exc}")
            print(traceback.format_exc())
