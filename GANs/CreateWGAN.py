# pragma pylint: disable=C0103, C0114, C0115, C0116, C0301, C0302, C0303, C0325, C0411, C0413
# pragma pylint: disable=W0105, W1203, W1309, W1514, W0613, W0621,
# type: ignore
# pylint: disable=import-error

"""
CreateWGAN - creates and saves WGAN-GP models using data from all of the pairs in
the whitelist.  Uses GANInterface for backend-agnostic training (MLX preferred,
TensorFlow fallback).
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


class CreateWGAN(CreateGANBase, BaseNNStrategy):

    MASTER_MIN_BUY_GAIN_THRESHOLD = 0.016
    MASTER_MIN_SELL_LOSS_THRESHOLD = 0.012
    MASTER_TRAINING_TYPE = 19  # Training type (label method) used for training labels

    DEFAULT_GAN_CONFIG: Dict[str, Any] = {
        "name": "WGAN-GP",
        "description": "WGAN-GP",
        "augmentation_target_ratio": 0.4,
        "noise_std": 0.02,
        "save_subdir": "GANs",
        "multi_task": False,
    }

    aggregate_pairs = True  # use all pairs for training

    def __init__(self, gan_config: Optional[Dict[str, Any]] = None, **kwargs) -> None:
        super().__init__(gan_config=gan_config, **kwargs)

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

            train_idx = train_labels.argmax(axis=1)
            classes, counts = np.unique(train_idx, return_counts=True)
            class_counts = dict(zip(classes.tolist(), counts.tolist()))
            print(
                f"    Train set size: {len(train_data)}  Class counts: {class_counts}"
            )

            save_path = self.get_gan_save_path(config)
            print(
                f"    Training {config.get('name', 'WGAN-GP')} via GANInterface "
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

            interface = GANInterface(GANType.WGAN, save_path=save_path)
            interface.fit(
                train_data.astype("float32"),
                train_labels.astype("float32"),
            )
            interface.save(
                min_buy_gain_threshold=self.MASTER_MIN_BUY_GAIN_THRESHOLD,
                min_sell_loss_threshold=self.MASTER_MIN_SELL_LOSS_THRESHOLD,
                training_type=self.MASTER_TRAINING_TYPE,
            )
            print(f"    WGAN-GP model saved to {save_path}")

        except Exception as exc:
            print("    WGAN-GP encountered an error during training")
            print(f"      Error: {exc}")
            print(traceback.format_exc())
