# pragma pylint: disable=C0103, C0114, C0115, C0116, C0301, C0302, C0303, C0325, C0411, C0413
# pragma pylint: disable=W0105, W1203, W1309, W1514, W0613, W0621,
# type: ignore
# pylint: disable=import-error
# flake8: noqa: F401, E402, F541, W0718, W0719

"""
CreateMTWGAN - creates and saves Multi-Task WGAN-GP models using data from all of
the pairs in the whitelist. Concrete GAN parameters are supplied through
CreateGANBase for a unified API.
"""

from __future__ import annotations

import sys
import traceback
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
from pandas import DataFrame

group_dir = str(Path(__file__).parent)
sys.path.append(group_dir)

from CreateMTGANBase import CreateMTGANBase  # noqa: E402
from NNMTStrategy import NNMTStrategy  # noqa: E402
from utils.df_mt_wgan_gp import balance_with_mt_wgan_gp  # noqa: E402

try:
    from utils.df_mt_wgan_mlx import balance_with_mt_wgan_mlx

    HAS_MT_WGAN_MLX = True
except ImportError:
    HAS_MT_WGAN_MLX = False


class CreateMTWGAN(CreateMTGANBase, NNMTStrategy):
    """
    Creates and saves Multi-Task WGAN-GP models.

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
    MASTER_TRAINING_TYPE = 19  # Training type (label method) used for training labels

    DEFAULT_GAN_CONFIG: Dict[str, Any] = {
        "name": "Multi-Task WGAN-GP (MLX)" if HAS_MT_WGAN_MLX else "Multi-Task WGAN-GP",
        "description": (
            "Multi-Task WGAN-GP (MLX)" if HAS_MT_WGAN_MLX else "Multi-Task WGAN-GP"
        ),
        "balance_fn": (
            balance_with_mt_wgan_mlx if HAS_MT_WGAN_MLX else balance_with_mt_wgan_gp
        ),
        "train_kwargs": {
            "epochs": 100,
            "batch_size": 2048 if HAS_MT_WGAN_MLX else 256,
            "n_critic": 6,
            "verbose": True,
        },
        "task_target_ratios": {
            "trading": 0.2,
            "regime": 0.1,
            "risk": 0.1,
            "momentum": 0.1,
            "flow": 0.1,
            "profit": 0.1,
        },
        "primary_task": "trading",
        "target_ratio": 0.1,
        "save_subdir": "GANs",
        "multi_task": True,
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
        balance_fn = config.get("balance_fn")
        if balance_fn is None:
            print("    No balance function provided; skipping GAN training")
            return

        try:
            original_shape = np.shape(train_data)
            print(f"    Balancing training data with {config.get('name', 'GAN')}")

            if len(train_data) == 0:
                print("    No training data to balance")
                return

            train_kwargs = dict(config.get("train_kwargs", {}))
            requested_batch = int(train_kwargs.get("batch_size", len(train_data)))
            train_kwargs["batch_size"] = (
                min(requested_batch, len(train_data))
                if len(train_data) > 0
                else requested_batch
            )
            train_kwargs.setdefault("save_path", self.get_gan_save_path(config))

            # Pass MASTER threshold values and training_type to balance function - these are stored in GAN metadata
            train_kwargs["min_buy_gain_threshold"] = self.MASTER_MIN_BUY_GAIN_THRESHOLD
            train_kwargs["min_sell_loss_threshold"] = (
                self.MASTER_MIN_SELL_LOSS_THRESHOLD
            )
            train_kwargs["training_type"] = self.MASTER_TRAINING_TYPE

            # Print MASTER values that will be stored in GAN metadata (before training starts)
            print("    Multi-Task WGAN-GP training starting...")
            print(f"    MASTER thresholds (will be stored in GAN metadata):")
            print(
                f"      MASTER_MIN_BUY_GAIN_THRESHOLD = {self.MASTER_MIN_BUY_GAIN_THRESHOLD:.4f}"
            )
            print(
                f"      MASTER_MIN_SELL_LOSS_THRESHOLD = {self.MASTER_MIN_SELL_LOSS_THRESHOLD:.4f}"
            )
            print(f"      MASTER_TRAINING_TYPE = {self.MASTER_TRAINING_TYPE}")

            task_target_ratios = config.get("task_target_ratios")
            display_task: Optional[str] = None

            if task_target_ratios:
                train_kwargs["task_target_ratios"] = task_target_ratios
                display_task = next(iter(task_target_ratios.keys()), None)
                aug_x, aug_y = balance_fn(
                    train_data.astype("float32"),
                    train_labels,
                    **train_kwargs,
                )
            else:
                primary_task = config.get("primary_task", "trading")
                if primary_task not in train_labels:
                    print(
                        f"    Primary task '{primary_task}' not found in labels. "
                        f"Available: {list(train_labels.keys())}"
                    )
                    print("    Skipping WGAN-GP balancing")
                    return

                primary_labels = train_labels[primary_task]
                train_idx = primary_labels.argmax(axis=1)
                classes, counts = np.unique(train_idx, return_counts=True)
                class_counts = dict(zip(classes.tolist(), counts.tolist()))
                print(
                    f"    Train set size: {len(train_data)}  "
                    f"{primary_task} class counts: {class_counts}"
                )

                current_max = int(counts.max()) if counts.size > 0 else 0
                target_ratio = config.get("target_ratio", 0.1)
                target = int(current_max * target_ratio) if current_max > 0 else None
                if target is None or target <= 0:
                    print("    No majority class found, skipping balancing")
                    print(
                        f"    WGAN-GP effect: shape {original_shape} -> {np.shape(train_data)}; "
                        f"counts {class_counts} -> {class_counts}"
                    )
                    return

                train_kwargs["max_target"] = target
                train_kwargs["primary_task"] = primary_task
                display_task = primary_task
                # MASTER thresholds already set above
                aug_x, aug_y = balance_fn(
                    train_data.astype("float32"),
                    train_labels,
                    **train_kwargs,
                )

            if display_task and display_task in aug_y:
                aug_task_labels = aug_y[display_task]
                aug_idx = aug_task_labels.argmax(axis=1)
                aug_classes, aug_counts = np.unique(aug_idx, return_counts=True)
                aug_class_counts = dict(zip(aug_classes.tolist(), aug_counts.tolist()))
                print("    Multi-Task WGAN-GP training complete")
                print(
                    f"    Augmented train size: {len(aug_x)}  "
                    f"{display_task} class counts: {aug_class_counts}"
                )
                print(
                    f"    WGAN-GP effect: shape {original_shape} -> {np.shape(aug_x)}"
                )
            else:
                print("    Multi-Task WGAN-GP training complete")
                print(f"    Augmented train size: {len(aug_x)}")
                print(
                    f"    WGAN-GP effect: shape {original_shape} -> {np.shape(aug_x)}"
                )
        except Exception as exc:
            print(
                "    Multi-Task WGAN-GP encountered an error; returning original data"
            )
            print(f"      Error: {exc}")
            print(traceback.format_exc())
