# pragma pylint: disable=C0103, C0114, C0115, C0116, C0301, C0302, C0303, C0325, C0411, C0413
# pragma pylint: disable=W0105, W1203, W1309, W1514, W0613, W0621,
# type: ignore
# pylint: disable=import-error
# flake8: noqa: F401, E402, F541, W0718, W0719

"""
CreateMTWGAN - creates and saves Multi-Task WGAN-GP models using data from all of
the pairs in the whitelist.  Uses GANInterface for backend-agnostic training
(MLX preferred, TensorFlow fallback).
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
from NNMT.NNMTStrategy import NNMTStrategy  # noqa: E402
from GANs.GANInterface import GANInterface  # noqa: E402
from GANs.GANType import GANType  # noqa: E402


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
        "name": "Multi-Task WGAN-GP",
        "description": "Multi-Task WGAN-GP",
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
        try:
            if len(train_data) == 0:
                print("    No training data to balance")
                return

            # Log class distributions per task
            for task, lbls in train_labels.items():
                if hasattr(lbls, "argmax"):
                    task_idx = lbls.argmax(axis=1)
                    task_classes, task_counts = np.unique(task_idx, return_counts=True)
                    task_counts_map = dict(
                        zip(task_classes.tolist(), task_counts.tolist())
                    )
                    print(f"    Task '{task}' class distribution: {task_counts_map}")

            save_path = self.get_gan_save_path(config)
            print(
                f"    Training {config.get('name', 'Multi-Task WGAN-GP')} via GANInterface "
                f"(save_path={save_path})"
            )
            print("    MASTER thresholds (will be stored in GAN metadata):")
            print(
                f"      MASTER_MIN_BUY_GAIN_THRESHOLD = {self.MASTER_MIN_BUY_GAIN_THRESHOLD:.4f}"
            )
            print(
                f"      MASTER_MIN_SELL_LOSS_THRESHOLD = {self.MASTER_MIN_SELL_LOSS_THRESHOLD:.4f}"
            )
            print(f"      MASTER_TRAINING_TYPE = {self.MASTER_TRAINING_TYPE}")

            interface = GANInterface(GANType.MT_WGAN, save_path=save_path)
            interface.fit(
                train_data.astype("float32"),
                train_labels,
            )
            interface.save(
                min_buy_gain_threshold=self.MASTER_MIN_BUY_GAIN_THRESHOLD,
                min_sell_loss_threshold=self.MASTER_MIN_SELL_LOSS_THRESHOLD,
                training_type=self.MASTER_TRAINING_TYPE,
            )
            print(f"    Multi-Task WGAN-GP model saved to {save_path}")

        except Exception as exc:
            print("    Multi-Task WGAN-GP encountered an error during training")
            print(f"      Error: {exc}")
            print(traceback.format_exc())
