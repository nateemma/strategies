# pragma pylint: disable=C0103, C0114, C0115, C0116, C0301, C0302, C0303, C0325, C0411, C0413
# pragma pylint: disable=W0105, W1203, W1309, W1514, W0613, W0621,
# type: ignore
# pylint: disable=import-error

"""
CreateCtabGanPlus - creates and saves CTAB-GAN+ models using data from all of the pairs in
the whitelist.  Uses GANInterface for backend-agnostic training (MLX preferred when no
categorical columns are present, Enhanced Keras otherwise).
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
from Framework.BaseStrategy import (
    BaseStrategy,
    ScalerType,
    MarketRegime,
    TradingAction,
    FlowDirection,
    MomentumDirection,
    RiskLevel,
    GANType,
)
from GANs.GANInterface import GANInterface  # noqa: E402


class CreateCtabGanPlus(CreateGANBase, BaseNNStrategy):
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
    MIN_BUY_GAIN_THRESHOLD = MASTER_MIN_BUY_GAIN_THRESHOLD
    MIN_SELL_LOSS_THRESHOLD = MASTER_MIN_SELL_LOSS_THRESHOLD
    TRAINING_TYPE = MASTER_TRAINING_TYPE

    DEFAULT_GAN_CONFIG: Dict[str, Any] = {
        "name": "CTAB-GAN+",
        "description": "CTAB-GAN+",
        "augmentation_target_ratio": 1.0,
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
        self.MIN_BUY_GAIN_THRESHOLD = self.MASTER_MIN_BUY_GAIN_THRESHOLD
        self.MIN_SELL_LOSS_THRESHOLD = self.MASTER_MIN_SELL_LOSS_THRESHOLD
        self.TRAINING_TYPE = self.MASTER_TRAINING_TYPE
        self._is_gan_creation_strategy = True
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

    def _get_categorical_columns(self, dataframe: DataFrame) -> list:
        """Identify categorical columns from the dataframe."""
        categorical_cols = []
        for base_col in self.one_hot_columns:
            matching_cols = [
                col for col in dataframe.columns if col.startswith(f"{base_col}_")
            ]
            categorical_cols.extend(matching_cols)
        return list(set([col for col in categorical_cols if col in dataframe.columns]))

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
            if len(train_data) == 0:
                print("    No training data to balance")
                return

            save_path = self.get_gan_save_path(config)

            # Rebuild column names for the GAN-scaled DataFrame
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

            train_df = pd.DataFrame(train_data, columns=train_df_columns)

            # Detect or filter categorical columns
            categorical_columns = config.get("categorical_columns")
            if categorical_columns is None:
                categorical_columns = self._get_categorical_columns(train_df)
                print(f"    Auto-detected categorical columns: {categorical_columns}")
            else:
                categorical_columns = [c for c in categorical_columns if c in train_df.columns]
                print(f"    Using provided categorical columns (filtered): {categorical_columns}")

            # Normalise labels to one-hot
            labels_arr = np.asarray(train_labels)
            if labels_arr.ndim == 1:
                num_classes = int(labels_arr.max()) + 1
                train_labels_processed = np.eye(num_classes, dtype=np.float32)[labels_arr.astype(int)]
            else:
                train_labels_processed = labels_arr.astype(np.float32)

            train_idx = train_labels_processed.argmax(axis=1)
            classes, counts = np.unique(train_idx, return_counts=True)
            class_counts = dict(zip(classes.tolist(), counts.tolist()))
            print(f"    Train set size: {len(train_data)}  Class counts: {class_counts}")

            # Compute augmentation targets
            augmentation_target_ratio = config.get("augmentation_target_ratio", 1.0)
            current_max = int(counts.max()) if counts.size > 0 else 0
            if current_max <= 0:
                print("    No majority class found, skipping CTAB-GAN+")
                return
            target = int(current_max * augmentation_target_ratio)
            num_classes = train_labels_processed.shape[1]
            have_map = {c: int(train_labels_processed[:, c].sum()) for c in range(num_classes)}
            needs_map = {c: max(0, target - have_map.get(c, 0)) for c in range(num_classes)}
            print(
                f"    CTAB-GAN+ target per class: {target} "
                f"(ratio={augmentation_target_ratio})  Planned adds: {needs_map}"
            )
            if all(v <= 0 for v in needs_map.values()):
                print("    Already at or above target; skipping CTAB-GAN+")
                return

            print("    CTAB-GAN+ training starting...")
            print("    MASTER thresholds (will be stored in GAN metadata):")
            print(
                f"      MASTER_MIN_BUY_GAIN_THRESHOLD = {self.MASTER_MIN_BUY_GAIN_THRESHOLD:.4f}"
            )
            print(
                f"      MASTER_MIN_SELL_LOSS_THRESHOLD = {self.MASTER_MIN_SELL_LOSS_THRESHOLD:.4f}"
            )
            print(f"      MASTER_TRAINING_TYPE = {self.MASTER_TRAINING_TYPE}")

            interface = GANInterface(GANType.CTAB_GAN, save_path=save_path)
            interface.fit(
                train_df,
                train_labels_processed,
                categorical_columns=categorical_columns,
            )
            interface.save(
                min_buy_gain_threshold=self.MASTER_MIN_BUY_GAIN_THRESHOLD,
                min_sell_loss_threshold=self.MASTER_MIN_SELL_LOSS_THRESHOLD,
                training_type=self.MASTER_TRAINING_TYPE,
            )
            print(f"    CTAB-GAN+ model saved to {save_path}")
            print(
                f"      Stored thresholds: min_buy_gain={self.MASTER_MIN_BUY_GAIN_THRESHOLD:.4f}, "
                f"min_sell_loss={self.MASTER_MIN_SELL_LOSS_THRESHOLD:.4f}, "
                f"training_type={self.MASTER_TRAINING_TYPE}"
            )

            # Evaluate the model
            eval_sample_size = min(2000, len(train_df))
            print("\n    Evaluating CTAB-GAN+ model...")
            eval_indices = np.random.choice(len(train_df), eval_sample_size, replace=False)
            generated_gan = interface.generate(n=eval_sample_size, class_label=0)

            print("    GAN Space Evaluation (minmax normalized to [-1, 1]):")
            try:
                eval_df_gan = train_df.iloc[eval_indices]
                eval_metrics_gan = interface._model.evaluate_with_dataframes(
                    eval_df_gan, generated_gan
                )
                overall_gan = eval_metrics_gan.get("overall_score", {})
                for metric, label in [
                    ("overall_quality",   "Overall Quality"),
                    ("diversity_score",   "Diversity Score"),
                    ("correlation_score", "Correlation Score"),
                    ("statistical_score", "Statistical Score"),
                    ("validity_score",    "Validity Score"),
                ]:
                    print(f"      {label:<22} {overall_gan.get(metric, 0.0):.4f}")
                quality = overall_gan.get("overall_quality", 0.0)
                if quality < 0.6:
                    print(f"      WARNING: Low overall quality ({quality:.4f})")
                elif quality >= 0.8:
                    print(f"      Excellent model quality ({quality:.4f})")
            except Exception as eval_exc:
                print(f"      Evaluation failed: {eval_exc}")
                print(traceback.format_exc())

            # Generate augmented samples for logging
            aug_data_list = [train_data]
            aug_labels_list = [train_labels_processed]
            for class_idx, need_count in needs_map.items():
                if need_count <= 0:
                    continue
                print(f"    Generating {need_count} samples for class {class_idx}")
                gen_df = interface.generate(n=need_count, class_label=int(class_idx))
                generated_array = gen_df[train_df_columns].values.astype(np.float32)
                aug_data_list.append(generated_array)
                class_labels = np.zeros(
                    (need_count, train_labels_processed.shape[1]), dtype=np.float32
                )
                class_labels[:, class_idx] = 1.0
                aug_labels_list.append(class_labels)

            if len(aug_data_list) > 1:
                aug_x = np.concatenate(aug_data_list, axis=0)
                aug_y = np.concatenate(aug_labels_list, axis=0)
                aug_idx = aug_y.argmax(axis=1)
                aug_classes, aug_counts = np.unique(aug_idx, return_counts=True)
                aug_class_counts = dict(zip(aug_classes.tolist(), aug_counts.tolist()))
                print("    CTAB-GAN+ training complete")
                print(
                    f"    Augmented train size: {len(aug_x)}  Class counts: {aug_class_counts}"
                )
            else:
                print("    CTAB-GAN+ training complete, no augmentation needed")

        except Exception as exc:
            print("    CTAB-GAN+ encountered an error; returning original data")
            print(f"      Error: {exc}")
            print(traceback.format_exc())
