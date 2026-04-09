# pragma pylint: disable=C0103, C0114, C0115, C0116, C0301, C0302, C0303, C0325, C0411, C0413
# pragma pylint: disable=W0105, W1203, W1309, W1514, W0613, W0621,
# type: ignore
# pylint: disable=import-error

"""
CreateWGAN - creates and saves WGAN-GP models using data from all of the pairs in
the whitelist. Concrete GAN parameters are provided via configuration to the
CreateGANBase mixin.
"""

from __future__ import annotations

import sys
import traceback
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
from pandas import DataFrame
import tensorflow as tf

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

try:
    from utils.df_wgan_mlx import balance_with_wgan_mlx
    from Framework.BaseNNStrategy import BaseNNStrategy, HAS_MLX, StrategyConfig
from utils.Environment import Environment
from Framework.BaseStrategy import BaseStrategy, ScalerType, MarketRegime, TradingAction, FlowDirection, MomentumDirection, RiskLevel, GANType
from Framework.BaseNNStrategy import BaseNNStrategy, HAS_MLX, StrategyConfig
from utils.Environment import Environment
from Framework.BaseStrategy import BaseStrategy, ScalerType, MarketRegime, TradingAction, FlowDirection, MomentumDirection, RiskLevel, GANType

    HAS_WGAN_MLX = True
except (ImportError, ModuleNotFoundError):
    from Framework.BaseNNStrategy import BaseNNStrategy, HAS_MLX, StrategyConfig
from utils.Environment import Environment
from Framework.BaseStrategy import BaseStrategy, ScalerType, MarketRegime, TradingAction, FlowDirection, MomentumDirection, RiskLevel, GANType
from Framework.BaseNNStrategy import BaseNNStrategy, HAS_MLX, StrategyConfig
from utils.Environment import Environment
from Framework.BaseStrategy import BaseStrategy, ScalerType, MarketRegime, TradingAction, FlowDirection, MomentumDirection, RiskLevel, GANType

    HAS_WGAN_MLX = False


class CreateWGAN(BaseNNStrategy):

    MASTER_MIN_BUY_GAIN_THRESHOLD = 0.016
    MASTER_MIN_SELL_LOSS_THRESHOLD = 0.012
    MASTER_TRAINING_TYPE = 19  # Training type (label method) used for training labels

    DEFAULT_GAN_CONFIG: Dict[str, Any] = {
        "name": "WGAN-GP (MLX)" if (HAS_WGAN_MLX and HAS_MLX) else "WGAN-GP",
        "description": "WGAN-GP",
        "balance_fn": (
            balance_with_wgan_mlx
            if (HAS_WGAN_MLX and HAS_MLX)
            else balance_with_wgan_gp
        ),
        "train_kwargs": {
            "epochs": 100,
            "batch_size": 2048 if (HAS_WGAN_MLX and HAS_MLX) else 512,
            "n_critic": 5,
            "verbose": True,
        },
        "augmentation_target_ratio": 0.4,  # Augment minority classes to % of majority class size
        "noise_std": 0.02,  # Standard deviation of Gaussian noise to add to generated samples
        "save_subdir": "GANs",
        "multi_task": False,
    }

    aggregate_pairs = True  # only use 1 pair for training (debugging)

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

            train_idx = train_labels.argmax(axis=1)
            classes, counts = np.unique(train_idx, return_counts=True)
            class_counts = dict(zip(classes.tolist(), counts.tolist()))
            print(
                f"    Train set size: {len(train_data)}  Class counts: {class_counts}"
            )

            current_max = int(counts.max()) if counts.size > 0 else 0
            if current_max <= 0:
                print("    No majority class found, skipping balancing")
                print(
                    f"    WGAN-GP effect: shape {original_shape} -> {np.shape(train_data)}; "
                    f"counts {class_counts} -> {class_counts}"
                )
                return

            # Use augmentation_target_ratio (matches NNNC_WGAN approach)
            augmentation_target_ratio = config.get("augmentation_target_ratio", 0.4)
            target = (
                int(current_max * augmentation_target_ratio)
                if current_max > 0
                else None
            )

            have_map = {int(c): int(n) for c, n in class_counts.items()}
            needs_map = {
                c: max(0, target - have_map.get(c, 0))
                for c in range(train_labels.shape[1])
            }
            print(
                f"    WGAN target per class: {target} "
                f"(ratio={augmentation_target_ratio})  Planned adds: {needs_map}"
            )
            if all(v <= 0 for v in needs_map.values()):
                print("    Already at or above target; skipping WGAN-GP")
                print(
                    f"    WGAN-GP effect: shape {original_shape} -> {np.shape(train_data)}; "
                    f"counts {class_counts} -> {class_counts}"
                )
                return

            train_kwargs = dict(config.get("train_kwargs", {}))
            requested_batch = int(train_kwargs.get("batch_size", len(train_data)))
            train_kwargs["batch_size"] = (
                min(requested_batch, len(train_data))
                if len(train_data) > 0
                else requested_batch
            )
            train_kwargs["max_target"] = target
            train_kwargs["augmentation_target_ratio"] = augmentation_target_ratio
            train_kwargs["noise_std"] = config.get("noise_std", 0.02)
            train_kwargs.setdefault("save_path", self.get_gan_save_path(config))

            print("    WGAN-GP training starting...")
            # Set TensorFlow seed for deterministic operations (required when TF_DETERMINISTIC_OPS is enabled)
            tf.random.set_seed(42)

            aug_x, aug_y = balance_fn(
                train_data.astype("float32"),
                train_labels.astype("float32"),
                **train_kwargs,
                seq_len=1,  # WGAN always uses seq_len=1 for 2D data
            )

            aug_idx = aug_y.argmax(axis=1)
            aug_classes, aug_counts = np.unique(aug_idx, return_counts=True)
            aug_class_counts = dict(zip(aug_classes.tolist(), aug_counts.tolist()))
            print("    WGAN-GP training complete")
            print(
                f"    Augmented train size: {len(aug_x)}  Class counts: {aug_class_counts}"
            )
            print(
                f"    WGAN-GP effect: shape {original_shape} -> {np.shape(aug_x)}; "
                f"counts {class_counts} -> {aug_class_counts}"
            )
        except Exception as exc:
            print("    WGAN-GP encountered an error; returning original data")
            print(f"      Error: {exc}")
            print(traceback.format_exc())
