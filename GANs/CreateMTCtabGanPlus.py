# pragma pylint: disable=C0103, C0114, C0115, C0116, C0301, C0302, C0303, C0325, C0411, C0413
# pragma pylint: disable=W0105, W1203, W1309, W1514, W0613, W0621,
# type: ignore
# pylint: disable=import-error

"""
CreateMTCtabGanPlus - creates and saves Multi-Task CTAB-GAN+ models using data from all of the pairs in
the whitelist. CTAB-GAN+ is specifically designed for tabular data with mixed data types
(continuous and categorical) and supports multi-task learning.
"""

from __future__ import annotations

import sys
import traceback
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd
from pandas import DataFrame
import tensorflow as tf

group_dir = str(Path(__file__).parent)
sys.path.append(group_dir)

from CreateMTGANBase import CreateMTGANBase  # noqa: E402
from NNMTStrategy import NNMTStrategy  # noqa: E402
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

try:
    from utils.df_ctab_gan_mlx import CTABGANMLX
    from Framework.BaseNNStrategy import BaseNNStrategy, HAS_MLX, StrategyConfig
from utils.Environment import Environment
from Framework.BaseStrategy import BaseStrategy, ScalerType, MarketRegime, TradingAction, FlowDirection, MomentumDirection, RiskLevel, GANType
from Framework.BaseNNStrategy import BaseNNStrategy, HAS_MLX, StrategyConfig
from utils.Environment import Environment
from Framework.BaseStrategy import BaseStrategy, ScalerType, MarketRegime, TradingAction, FlowDirection, MomentumDirection, RiskLevel, GANType

    HAS_CTAB_MLX = True
except (ImportError, ModuleNotFoundError):
    HAS_CTAB_MLX = False
    HAS_MLX = False


def _concatenate_task_labels(task_labels_dict: Dict[str, np.ndarray]) -> np.ndarray:
    """Concatenate all task labels into a single condition vector."""
    sorted_keys = sorted(task_labels_dict.keys())
    return np.concatenate([task_labels_dict[k] for k in sorted_keys], axis=1)


class CreateMTCtabGanPlus(CreateMTGANBase, NNMTStrategy):
    """
    Creates and saves Multi-Task CTAB-GAN+ models.

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

    # DEFAULT_GAN_CONFIG: Dict[str, Any] = {
    #     "name": "Multi-Task CTAB-GAN+",
    #     "description": "Multi-Task CTAB-GAN+",
    #     "train_kwargs": {
    #         "epochs": 300,
    #         "batch_size": 2048,  # Power of 2 for optimal GPU utilization
    #         "latent_dim": 128,
    #         "generator_layers": [256, 256],
    #         "discriminator_layers": [256, 256],
    #         "learning_rate": 2e-4,
    #         "beta_1": 0.5,
    #         "beta_2": 0.999,
    #         "gp_weight": 10.0,
    #         "verbose": True,
    #         "integer_columns": [],
    #         "use_cnn": False,
    #         "use_auxiliary": False,
    #         "info_loss_weight": 1.0,
    #         "downstream_loss_weight": 0.0,
    #         "generator_loss_weight": 0.0,
    #     },
    #     "task_target_ratios": {
    #         "trading": 0.4,
    #         "regime": 0.4,
    #         "risk": 0.4,
    #         "momentum": 0.4,
    #         "flow": 0.4,
    #         "profit": 0.4,
    #     },
    #     "primary_task": "trading",
    #     "target_ratio": 0.4,
    #     "save_subdir": "MTCTABGANs",
    #     "multi_task": True,
    #     "categorical_columns": None,  # Will be auto-detected from one_hot_columns
    # }

    DEFAULT_GAN_CONFIG: Dict[str, Any] = {
        "name": "Multi-Task CTAB-GAN+",
        "description": "Multi-Task CTAB-GAN+",
        "train_kwargs": {
            "epochs": 300,
            "batch_size": 512,
            "latent_dim": 128,
            "generator_layers": [256, 256],
            "discriminator_layers": [256, 256],
            "learning_rate": 2e-4,
            "beta_1": 0.5,
            "beta_2": 0.999,
            "gp_weight": 10.0,
            "verbose": True,
            "integer_columns": [],
            # --- ENHANCEMENTS (CTAB-GAN+ Paper) ---
            # info_loss_weight ensures the GAN reproduces the marginal distributions (Stat Score)
            # generator_loss_weight ensures the GAN reproduces feature correlations (Corr Score)
            "use_cnn": False,
            "use_auxiliary": True,
            "info_loss_weight": 0.02,  # CRITICAL for high Stat Score and fidelity
            "downstream_loss_weight": 0.01,
            "generator_loss_weight": 0.01,  # Minor regularization for correlation matching
        },
        "task_target_ratios": {
            "trading": 0.4,
            "regime": 0.4,
            "risk": 0.4,
            "momentum": 0.4,
            "flow": 0.4,
            "profit": 0.4,
        },
        "primary_task": "trading",
        "target_ratio": 0.4,
        "save_subdir": "MTCTABGANs",
        "multi_task": True,
        "categorical_columns": [],
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
                "CreateMTCtabGanPlus MASTER_* mismatch with NNStrategy:",
                *[
                    f"  {key}: NNStrategy={vals[0]} vs CreateMTCtabGanPlus={vals[1]}"
                    for key, vals in mismatches.items()
                ],
            ]
            raise ValueError("\n".join(lines))

    def _get_categorical_columns(self, dataframe: DataFrame) -> list[str]:
        """
        Identify categorical columns from the dataframe.

        Categorical columns are derived from one_hot_columns (e.g., flow_0, flow_2,
        regime_0, regime_2) and other binary/categorical columns like sar_trend and trend_mode.
        """
        categorical_cols = []

        # Get existing categorical columns from config if available
        config_cats = self.gan_config.get("categorical_columns", [])
        if isinstance(config_cats, list):
            categorical_cols.extend([c for c in config_cats if c in dataframe.columns])

        # Get one-hot encoded columns (e.g., flow_0, flow_2, regime_0, regime_2)
        for base_col in getattr(self, "one_hot_columns", []):
            # Find all columns that start with base_col_ (e.g., flow_0, flow_2)
            matching_cols = [
                col for col in dataframe.columns if col.startswith(f"{base_col}_")
            ]
            categorical_cols.extend(matching_cols)

        # Remove duplicates and ensure columns exist in dataframe
        unique_categorical = list(dict.fromkeys(categorical_cols))
        return [col for col in unique_categorical if col in dataframe.columns]

    def run_gan_training(
        self,
        *,
        combined_df: DataFrame,
        train_data: np.ndarray,
        test_data: np.ndarray,
        train_labels: Dict[str, np.ndarray],
        test_labels: Dict[str, np.ndarray],
        config: Dict[str, Any],
    ) -> None:
        try:
            original_shape = np.shape(train_data)
            print(
                f"    Balancing training data with {config.get('name', 'Multi-Task CTAB-GAN+')}"
            )

            if len(train_data) == 0:
                print("    No training data to balance")
                return

            # Get training kwargs
            train_kwargs = dict(config.get("train_kwargs", {}))
            save_path = train_kwargs.pop("save_path", self.get_gan_save_path(config))

            # Get the actual column names from the GAN scaler
            # train_data comes from combined_minmax (GAN-scaled), which uses gan_scaler_a
            if (
                not hasattr(self, "gan_scaler_a")
                or self.gan_scaler_a is None
                or not hasattr(self.gan_scaler_a, "feature_columns")
            ):
                raise ValueError(
                    "GAN scaler (gan_scaler_a) is not available. "
                    "CTAB-GAN+ requires the GAN-scaled dataframe with feature_columns."
                )

            # Convert train_data back to DataFrame for CTAB-GAN+
            # CTAB-GAN+ is fundamentally tabular; if we receive 3D tensors, we flatten them
            # into a single wide row (B, S*F) to preserve sequence information in a tabular format.
            if train_data.ndim == 3:
                num_samples, seq_len, num_features = train_data.shape
                train_data_2d = train_data.reshape(num_samples, -1)

                # Generate wide column names (e.g., rsi_0, rsi_1, ...)
                feature_cols = self.gan_scaler_a.feature_columns
                train_df_columns = [
                    f"{c}_{s}" for s in range(seq_len) for c in feature_cols
                ]

                print(
                    f"    Flattened 3D tensors for CTAB-GAN+: {num_samples} samples, {len(train_df_columns)} features"
                )
                train_df = pd.DataFrame(train_data_2d, columns=train_df_columns)
            else:
                train_df_columns = self.gan_scaler_a.feature_columns
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

            # Validate task labels
            if not isinstance(train_labels, dict):
                raise ValueError(
                    "train_labels must be a dictionary for multi-task learning"
                )

            if len(train_labels) == 0:
                raise ValueError("train_labels dictionary cannot be empty")

            # Process task labels to ensure one-hot encoding
            train_labels_processed = {}
            for task, labels in train_labels.items():
                arr = np.asarray(labels)
                if arr.ndim == 1:
                    # Convert to one-hot if needed
                    num_classes = int(arr.max()) + 1
                    train_labels_processed[task] = np.eye(
                        num_classes, dtype=np.float32
                    )[arr.astype(int)]
                elif arr.ndim == 2:
                    train_labels_processed[task] = arr.astype(np.float32)
                else:
                    raise ValueError(f"Task '{task}' labels must be 1D or 2D array")

            # Determine balancing targets using task_target_ratios
            task_target_ratios = config.get("task_target_ratios")
            if task_target_ratios is None:
                # Fall back to primary_task behavior
                primary_task = config.get("primary_task", "trading")
                if primary_task not in train_labels_processed:
                    print(
                        f"    Primary task '{primary_task}' not found in labels. "
                        f"Available: {list(train_labels_processed.keys())}"
                    )
                    print("    Skipping CTAB-GAN+ balancing")
                    return

                primary_labels = train_labels_processed[primary_task]
                train_idx = primary_labels.argmax(axis=1)
                classes, counts = np.unique(train_idx, return_counts=True)
                class_counts = dict(zip(classes.tolist(), counts.tolist()))
                print(
                    f"    Train set size: {len(train_data)}  "
                    f"{primary_task} class counts: {class_counts}"
                )

                current_max = int(counts.max()) if counts.size > 0 else 0
                if current_max <= 0:
                    print("    No majority class found, skipping balancing")
                    return

                target_ratio = config.get("target_ratio", 0.4)
                target = int(current_max * target_ratio) if current_max > 0 else None
                if target is None or target <= 0:
                    print("    No target found, skipping balancing")
                    return

                # Calculate needs for primary task
                num_classes = primary_labels.shape[1]
                have_map = {
                    c: int(primary_labels[:, c].sum()) for c in range(num_classes)
                }
                needs_map = {
                    c: max(0, target - have_map.get(c, 0)) for c in range(num_classes)
                }
                print(
                    f"    CTAB-GAN+ target per class: {target} "
                    f"(ratio={target_ratio})  Planned adds: {needs_map}"
                )
                if all(v <= 0 for v in needs_map.values()):
                    print("    Already at or above target; skipping CTAB-GAN+")
                    return

                # Convert to task_target_ratios format for consistency
                task_target_ratios = {primary_task: target_ratio}
            else:
                # Validate task_target_ratios
                for task in task_target_ratios.keys():
                    if task is not None and task not in train_labels_processed:
                        print(
                            f"    Warning: task '{task}' in task_target_ratios not found in labels. "
                            f"Available: {list(train_labels_processed.keys())}"
                        )

            # Print MASTER values that will be stored in GAN metadata (before training starts)
            print("    Multi-Task CTAB-GAN+ training starting...")
            print(f"    MASTER thresholds (will be stored in GAN metadata):")
            print(
                f"      MASTER_MIN_BUY_GAIN_THRESHOLD = {self.MASTER_MIN_BUY_GAIN_THRESHOLD:.4f}"
            )
            print(
                f"      MASTER_MIN_SELL_LOSS_THRESHOLD = {self.MASTER_MIN_SELL_LOSS_THRESHOLD:.4f}"
            )
            print(f"      MASTER_TRAINING_TYPE = {self.MASTER_TRAINING_TYPE}")

            # Ensure random seed is set in train_kwargs (defaults to 42 if not specified)
            if "random_seed" not in train_kwargs:
                train_kwargs["random_seed"] = 42

            # Initialize and train CTAB-GAN+
            if (
                HAS_CTAB_MLX
                and HAS_MLX
                and (not categorical_columns or len(categorical_columns) == 0)
            ):
                print(
                    f"    Using MLX-accelerated Multi-Task CTAB-GAN (epochs={train_kwargs.get('epochs', 300)})"
                )
                ctab_gan = CTABGANMLX(
                    latent_dim=train_kwargs.get("latent_dim", 128),
                    epochs=train_kwargs.get("epochs", 300),
                    batch_size=train_kwargs.get("batch_size", 512),
                    verbose=True,
                )
                # Flatten labels for MLX version
                fit_labels = _concatenate_task_labels(train_labels_processed)
            else:
                # Use standard Enhanced Keras version (supports Categorical + Auxiliary loss)
                print("    Using standard Enhanced Keras Multi-Task CTAB-GAN")
                ctab_gan = CTABGANPlusMTEnhanced(**train_kwargs)
                fit_labels = train_labels_processed

            # Fit the model
            ctab_gan.fit(
                dataframe=train_df,
                labels=fit_labels,
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
            print(f"    Multi-Task CTAB-GAN+ model saved to {save_path}")
            if min_buy_gain is not None:
                min_sell_str = (
                    f"{min_sell_loss:.4f}" if min_sell_loss is not None else "None"
                )
                print(
                    f"      Stored thresholds: min_buy_gain={min_buy_gain:.4f}, min_sell_loss={min_sell_str}, training_type={training_type}"
                )

            # Evaluate the model after training (in both GAN space and training space)
            eval_sample_size = min(2000, len(train_df))
            print("\n    Evaluating Multi-Task CTAB-GAN+ model...")

            # Generate data once for both evaluations
            # Sample task labels from real data for evaluation
            eval_indices = np.random.choice(
                len(train_df), eval_sample_size, replace=False
            )
            eval_task_labels = {
                task: arr[eval_indices] for task, arr in train_labels_processed.items()
            }

            if HAS_CTAB_MLX and isinstance(ctab_gan, CTABGANMLX):
                eval_cond = _concatenate_task_labels(eval_task_labels)
                generated_gan = ctab_gan.generate(
                    num_samples=eval_sample_size, condition_vector=eval_cond
                )
            else:
                generated_gan, _ = ctab_gan.generate(
                    num_samples=eval_sample_size, task_labels=eval_task_labels
                )

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
                    print(
                        f"      ⚠️  WARNING: Low overall quality ({quality_training:.4f}) - model may need improvement"
                    )
                elif quality_training >= 0.8:
                    print(f"      ✅ Excellent model quality ({quality_training:.4f})")
            except Exception as eval_exc:
                print(f"      Evaluation failed: {eval_exc}")
                print(traceback.format_exc())

            # Generate augmented samples for each task/class combination that needs augmentation
            aug_data_list = [train_data]
            aug_labels_dict = {
                task: [labels] for task, labels in train_labels_processed.items()
            }

            # Determine generation needs from task_target_ratios
            generation_needs = {}  # {(task, class): need_count}

            for task, ratio_spec in task_target_ratios.items():
                if ratio_spec is None or task not in train_labels_processed:
                    continue

                task_labels = train_labels_processed[task]
                task_idx = np.argmax(task_labels, axis=1)
                unique, counts = np.unique(task_idx, return_counts=True)
                class_counts = {int(k): int(v) for k, v in zip(unique, counts)}
                current_max = int(np.max(counts)) if counts.size > 0 else 0

                # Determine ratios per class
                if isinstance(ratio_spec, dict):
                    class_ratios = ratio_spec
                else:
                    class_ratios = {c: ratio_spec for c in range(task_labels.shape[1])}

                for c in range(task_labels.shape[1]):
                    ratio = class_ratios.get(c, 0.0)
                    if ratio <= 0.0:
                        continue

                    target = int(current_max * ratio) if current_max > 0 else 0
                    have = class_counts.get(c, 0)
                    need = target - have

                    if need > 0:
                        generation_needs[(task, c)] = need

            if not generation_needs:
                print("    No classes need balancing")
                return

            # Generate samples for each task/class combination
            for (task, class_c), need_count in generation_needs.items():
                if need_count <= 0:
                    continue

                print(f"    Generating {need_count} samples for {task} class {class_c}")

                # Find real samples that match this task/class
                task_labels = train_labels_processed[task]
                task_idx = np.argmax(task_labels, axis=1)
                matching_mask = task_idx == class_c
                matching_samples = np.where(matching_mask)[0]

                if len(matching_samples) == 0:
                    # No real samples of this class, generate with default labels
                    task_labels_gen = {}
                    for other_task in sorted(train_labels_processed.keys()):
                        if other_task == task:
                            num_classes = train_labels_processed[other_task].shape[1]
                            one_hot = np.zeros(
                                (need_count, num_classes), dtype=np.float32
                            )
                            one_hot[:, class_c] = 1.0
                        else:
                            # Use most common class from real data
                            other_idx = np.argmax(
                                train_labels_processed[other_task], axis=1
                            )
                            most_common = int(np.bincount(other_idx).argmax())
                            num_classes = train_labels_processed[other_task].shape[1]
                            one_hot = np.zeros(
                                (need_count, num_classes), dtype=np.float32
                            )
                            one_hot[:, most_common] = 1.0
                        task_labels_gen[other_task] = one_hot
                else:
                    # Sample from real data labels that match this task/class
                    sampled_indices = np.random.choice(
                        matching_samples,
                        size=min(need_count, len(matching_samples)),
                        replace=True,
                    )
                    # If we need more than available, duplicate
                    if need_count > len(sampled_indices):
                        sampled_indices = np.concatenate(
                            [
                                sampled_indices,
                                np.random.choice(
                                    matching_samples,
                                    size=need_count - len(sampled_indices),
                                    replace=True,
                                ),
                            ]
                        )

                    # Copy labels from sampled real data
                    task_labels_gen = {
                        t: train_labels_processed[t][sampled_indices].copy()
                        for t in train_labels_processed.keys()
                    }
                    # Ensure the target task/class is correct
                    num_classes = train_labels_processed[task].shape[1]
                    task_labels_gen[task] = np.zeros(
                        (need_count, num_classes), dtype=np.float32
                    )
                    task_labels_gen[task][:, class_c] = 1.0

                # Generate synthetic data
                if HAS_CTAB_MLX and isinstance(ctab_gan, CTABGANMLX):
                    cond = _concatenate_task_labels(task_labels_gen)
                    generated_df = ctab_gan.generate(
                        num_samples=need_count, condition_vector=cond
                    )
                else:
                    generated_df, _ = ctab_gan.generate(
                        num_samples=need_count,
                        task_labels=task_labels_gen,
                    )

                # Convert back to numpy array (maintain column order from train_df)
                generated_array = generated_df[train_df_columns].values.astype(
                    np.float32
                )
                aug_data_list.append(generated_array)

                # Append generated labels
                for t in train_labels_processed.keys():
                    aug_labels_dict[t].append(task_labels_gen[t])

            if aug_data_list:
                aug_x = np.concatenate(aug_data_list, axis=0)
                aug_y = {
                    task: np.concatenate(aug_labels_dict[task], axis=0)
                    for task in train_labels_processed.keys()
                }

                # Display stats for first task
                display_task = list(train_labels_processed.keys())[0]
                aug_task_labels = aug_y[display_task]
                aug_idx = aug_task_labels.argmax(axis=1)
                aug_classes, aug_counts = np.unique(aug_idx, return_counts=True)
                aug_class_counts = dict(zip(aug_classes.tolist(), aug_counts.tolist()))

                print("    Multi-Task CTAB-GAN+ training complete")
                print(
                    f"    Augmented train size: {len(aug_x)}  "
                    f"{display_task} class counts: {aug_class_counts}"
                )
                print(
                    f"    CTAB-GAN+ effect: shape {original_shape} -> {np.shape(aug_x)}"
                )
            else:
                print(
                    "    Multi-Task CTAB-GAN+ training complete, but no augmentation needed"
                )

        except Exception as exc:
            print(
                "    Multi-Task CTAB-GAN+ encountered an error; returning original data"
            )
            print(f"      Error: {exc}")
            print(traceback.format_exc())
