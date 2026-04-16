# pragma pylint: disable=C0103, C0114, C0115, C0116, C0301, C0302, C0303, C0325, C0411, C0413
# pragma pylint: disable=W0105, W1203, W1309, W1514, W0613, W0621,
# type: ignore
# pylint: disable=import-error

"""
CreateMTCtabGanPlus - creates and saves Multi-Task CTAB-GAN+ models using data from all
of the pairs in the whitelist.  Uses GANInterface for backend-agnostic training.
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

from CreateMTGANBase import CreateMTGANBase  # noqa: E402

from NNMT.NNMTStrategy import NNMTStrategy  # noqa: E402
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
    MIN_BUY_GAIN_THRESHOLD = MASTER_MIN_BUY_GAIN_THRESHOLD
    MIN_SELL_LOSS_THRESHOLD = MASTER_MIN_SELL_LOSS_THRESHOLD
    TRAINING_TYPE = MASTER_TRAINING_TYPE

    DEFAULT_GAN_CONFIG: Dict[str, Any] = {
        "name": "Multi-Task CTAB-GAN+",
        "description": "Multi-Task CTAB-GAN+",
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
        """Override to force MASTER threshold values for GAN creation."""
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
                "CreateMTCtabGanPlus MASTER_* mismatch with NNStrategy:",
                *[
                    f"  {key}: NNStrategy={vals[0]} vs CreateMTCtabGanPlus={vals[1]}"
                    for key, vals in mismatches.items()
                ],
            ]
            raise ValueError("\n".join(lines))

    def _get_categorical_columns(self, dataframe: DataFrame) -> list:
        """Identify categorical columns from the dataframe."""
        categorical_cols = []
        config_cats = self.gan_config.get("categorical_columns", [])
        if isinstance(config_cats, list):
            categorical_cols.extend([c for c in config_cats if c in dataframe.columns])
        for base_col in getattr(self, "one_hot_columns", []):
            matching_cols = [
                col for col in dataframe.columns if col.startswith(f"{base_col}_")
            ]
            categorical_cols.extend(matching_cols)
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
            if len(train_data) == 0:
                print("    No training data to balance")
                return

            save_path = self.get_gan_save_path(config)

            # Get column names from the GAN scaler
            if (
                not hasattr(self, "gan_scaler_a")
                or self.gan_scaler_a is None
                or not hasattr(self.gan_scaler_a, "feature_columns")
            ):
                raise ValueError(
                    "GAN scaler (gan_scaler_a) is not available. "
                    "CTAB-GAN+ requires the GAN-scaled dataframe with feature_columns."
                )

            # Handle 3D tensors by flattening for tabular CTAB-GAN+
            if train_data.ndim == 3:
                num_samples, seq_len, num_features = train_data.shape
                train_data_2d = train_data.reshape(num_samples, -1)
                feature_cols = self.gan_scaler_a.feature_columns
                train_df_columns = [
                    f"{c}_{s}" for s in range(seq_len) for c in feature_cols
                ]
                print(
                    f"    Flattened 3D tensors for CTAB-GAN+: "
                    f"{num_samples} samples, {len(train_df_columns)} features"
                )
                train_df = pd.DataFrame(train_data_2d, columns=train_df_columns)
            else:
                train_df_columns = self.gan_scaler_a.feature_columns
                train_df = pd.DataFrame(train_data, columns=train_df_columns)

            # Detect or filter categorical columns
            categorical_columns = config.get("categorical_columns")
            if categorical_columns is None:
                categorical_columns = self._get_categorical_columns(train_df)
                print(f"    Auto-detected categorical columns: {categorical_columns}")
            else:
                categorical_columns = [
                    c for c in categorical_columns if c in train_df.columns
                ]
                print(
                    f"    Using provided categorical columns (filtered): {categorical_columns}"
                )

            # Validate and normalise task labels
            if not isinstance(train_labels, dict) or len(train_labels) == 0:
                raise ValueError(
                    "train_labels must be a non-empty dict for multi-task learning"
                )

            train_labels_processed: Dict[str, np.ndarray] = {}
            for task, labels in train_labels.items():
                arr = np.asarray(labels)
                if arr.ndim == 1:
                    num_classes = int(arr.max()) + 1
                    train_labels_processed[task] = np.eye(
                        num_classes, dtype=np.float32
                    )[arr.astype(int)]
                elif arr.ndim == 2:
                    train_labels_processed[task] = arr.astype(np.float32)
                else:
                    raise ValueError(f"Task '{task}' labels must be 1D or 2D array")

            # Log class distributions
            for task, lbls in train_labels_processed.items():
                task_idx = np.argmax(lbls, axis=1)
                unique, task_counts = np.unique(task_idx, return_counts=True)
                print(
                    f"    Task '{task}' class distribution: "
                    f"{dict(zip(unique.tolist(), task_counts.tolist()))}"
                )

            print("    Multi-Task CTAB-GAN+ training starting...")
            print("    MASTER thresholds (will be stored in GAN metadata):")
            print(
                f"      MASTER_MIN_BUY_GAIN_THRESHOLD = {self.MASTER_MIN_BUY_GAIN_THRESHOLD:.4f}"
            )
            print(
                f"      MASTER_MIN_SELL_LOSS_THRESHOLD = {self.MASTER_MIN_SELL_LOSS_THRESHOLD:.4f}"
            )
            print(f"      MASTER_TRAINING_TYPE = {self.MASTER_TRAINING_TYPE}")

            interface = GANInterface(GANType.MT_CTAB_GAN, save_path=save_path)
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
            print(f"    Multi-Task CTAB-GAN+ model saved to {save_path}")
            print(
                f"      Stored thresholds: min_buy_gain={self.MASTER_MIN_BUY_GAIN_THRESHOLD:.4f}, "
                f"min_sell_loss={self.MASTER_MIN_SELL_LOSS_THRESHOLD:.4f}, "
                f"training_type={self.MASTER_TRAINING_TYPE}"
            )

            # Evaluate the model
            eval_sample_size = min(2000, len(train_df))
            print("\n    Evaluating Multi-Task CTAB-GAN+ model...")
            eval_indices = np.random.choice(
                len(train_df), eval_sample_size, replace=False
            )
            eval_task_labels = {
                task: arr[eval_indices] for task, arr in train_labels_processed.items()
            }

            try:
                generated_gan, _ = interface.generate(
                    n=eval_sample_size, task_labels=eval_task_labels
                )
                print("    GAN Space Evaluation (minmax normalized to [-1, 1]):")
                eval_df_gan = train_df.iloc[eval_indices]
                eval_metrics = interface._model.evaluate_with_dataframes(
                    eval_df_gan, generated_gan
                )
                overall = eval_metrics.get("overall_score", {})
                for metric, label in [
                    ("overall_quality", "Overall Quality"),
                    ("diversity_score", "Diversity Score"),
                    ("correlation_score", "Correlation Score"),
                    ("statistical_score", "Statistical Score"),
                    ("validity_score", "Validity Score"),
                ]:
                    print(f"      {label:<22} {overall.get(metric, 0.0):.4f}")
                quality = overall.get("overall_quality", 0.0)
                if quality < 0.6:
                    print(f"      WARNING: Low overall quality ({quality:.4f})")
                elif quality >= 0.8:
                    print(f"      Excellent model quality ({quality:.4f})")
            except Exception as eval_exc:
                print(f"      Evaluation failed: {eval_exc}")
                print(traceback.format_exc())

            # Compute and log generation needs from task_target_ratios
            task_target_ratios = config.get("task_target_ratios", {})
            total_generated = 0
            for task, ratio_spec in task_target_ratios.items():
                if ratio_spec is None or task not in train_labels_processed:
                    continue
                task_lbls = train_labels_processed[task]
                task_idx = np.argmax(task_lbls, axis=1)
                unique, task_counts = np.unique(task_idx, return_counts=True)
                current_max = int(np.max(task_counts)) if task_counts.size > 0 else 0
                class_ratios = (
                    ratio_spec
                    if isinstance(ratio_spec, dict)
                    else {c: ratio_spec for c in range(task_lbls.shape[1])}
                )
                for c in range(task_lbls.shape[1]):
                    ratio = class_ratios.get(c, 0.0)
                    if ratio <= 0.0:
                        continue
                    have = int(np.sum(task_idx == c))
                    target = int(current_max * ratio)
                    need = max(0, target - have)
                    if need > 0:
                        print(f"    Would generate {need} samples for {task} class {c}")
                        total_generated += need

            if total_generated > 0:
                print(
                    f"    Multi-Task CTAB-GAN+ training complete "
                    f"(model trained; {total_generated} samples could be generated for augmentation)"
                )
            else:
                print("    Multi-Task CTAB-GAN+ training complete")

        except Exception as exc:
            print(
                "    Multi-Task CTAB-GAN+ encountered an error; returning original data"
            )
            print(f"      Error: {exc}")
            print(traceback.format_exc())
