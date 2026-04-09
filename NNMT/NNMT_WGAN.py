# pragma pylint: disable=C0103, C0114, C0115, C0116, C0301, C0302, C0303, C0325, C0411, C0413
# pragma pylint: disable=W0105, W1203, W1309, W1514, W0613, W0621,
# type: ignore
# pylint: disable=import-error

"""
NNMT_WGAN - Subclass of NNMTStrategy using WGAN-GP for high-fidelity augmentation
"""

import sys
from pathlib import Path
from pandas import DataFrame
import numpy as np
import pandas as pd
import traceback
import os
from typing import Dict, Optional, Tuple

group_dir = str(Path(__file__).parent)
sys.path.append(group_dir)

from NNMTStrategy import NNMTStrategy
from utils.df_mt_wgan_gp import balance_with_mt_wgan_gp, load_mt_wgan_thresholds

try:
    from utils.df_mt_wgan_mlx import (
        balance_with_mt_wgan_mlx,
        load_mt_wgan_thresholds_mlx,
    )

    HAS_MT_WGAN_MLX = True
except ImportError:
    HAS_MT_WGAN_MLX = False
from utils.Scalers import load_scaler

# -----------


class NNMT_WGAN(NNMTStrategy):

    augment_training_data = (
        False  # we only want 'real' signals, since we are augmenting anyway
    )

    # High-fidelity defaults
    wgan_epochs = 100
    wgan_batch_size = 2048
    wgan_n_critic = (
        5  # Train critic n times per generator step (TTUR - helps stability)
    )
    wgan_target_ratio = (
        0.8  # upsample minorities to this fraction of majority (backward compatibility)
    )
    wgan_primary_task = (
        "trading"  # Task to use for determining class balance (backward compatibility)
    )
    # Flexible multi-task balancing: dict mapping task names to ratios (float or dict[int, float] for per-class)
    # If None, falls back to wgan_target_ratio and wgan_primary_task
    # Example: {"trading": 0.5, "profit": {0: 0.6, 1: 0.4, 2: 0.5}, "regime": None}
    wgan_task_target_ratios = {
        "trading": 0.8,
        "regime": 0.8,
        "risk": 0.8,
        "momentum": 0.8,
        "flow": 0.8,
        "profit": 0.8,
    }

    # -----------

    # -----------

    def enhance_training_data(
        self, train_df: DataFrame, train_labels: Dict[str, np.ndarray]
    ) -> Tuple[DataFrame, Dict[str, np.ndarray]]:
        """Optional hook to run GAN augmentation on 2D feature frames."""
        print("    Skipping 2D WGAN augmentation (using 3D sequential instead)")
        return train_df, train_labels

    def preprocess_training_data(
        self, dataframe: DataFrame, train_data, test_data, train_labels, test_labels
    ):
        # Clear augmented labels - will be set only if augmentation actually occurs
        self._augmented_labels = None
        try:
            original_shape = np.shape(train_data)
            print("    Balancing training data with MT WGAN-GP")
            if len(train_data) == 0:
                print("    No training data to balance")
                return train_data, test_data, train_labels, test_labels

            save_location = os.path.join(self.get_storage_location(), "GANs")

            # Load gan_scaler for minmax transformation
            # WGAN models work in minmax scaled space, but train_data is in normalized space

            # Transform train_data from normalized space to minmax space for WGAN
            # Reshape 3D tensor (batch, seq_len, num_features) to 2D (batch * seq_len, num_features)
            train_data_shape = train_data.shape
            num_features = train_data.shape[-1]
            train_data_2d = train_data.reshape(-1, num_features)
            gan_input = self._format_for_gan_scaler(train_data_2d)
            train_data_minmax_2d = self.normalise_for_gan(gan_input)
            if isinstance(train_data_minmax_2d, pd.DataFrame):
                train_data_minmax_2d = train_data_minmax_2d.to_numpy()
            train_data_minmax = train_data_minmax_2d.reshape(train_data_shape)

            # Use new task_target_ratios if provided, otherwise use backward-compatible approach
            if self.wgan_task_target_ratios is not None:
                # Flexible multi-task balancing
                if HAS_MT_WGAN_MLX:
                    aug_x, aug_y = balance_with_mt_wgan_mlx(
                        train_data_minmax.astype("float32"),
                        train_labels,
                        epochs=self.wgan_epochs,
                        batch_size=(
                            min(self.wgan_batch_size, len(train_data))
                            if len(train_data) > 0
                            else self.wgan_batch_size
                        ),
                        verbose=True,
                        save_path=save_location,
                        n_critic=self.wgan_n_critic,
                        task_target_ratios=self.wgan_task_target_ratios,
                        seq_len=getattr(
                            self, "wgan_seq_len", getattr(self, "seq_len", 1)
                        ),
                    )
                else:
                    aug_x, aug_y = balance_with_mt_wgan_gp(
                        train_data_minmax.astype("float32"),
                        train_labels,
                        epochs=self.wgan_epochs,
                        batch_size=(
                            min(self.wgan_batch_size, len(train_data))
                            if len(train_data) > 0
                            else self.wgan_batch_size
                        ),
                        verbose=True,
                        save_path=save_location,
                        n_critic=self.wgan_n_critic,
                        task_target_ratios=self.wgan_task_target_ratios,
                    )
                # For display, use first task in task_target_ratios
                display_task = (
                    list(self.wgan_task_target_ratios.keys())[0]
                    if self.wgan_task_target_ratios
                    else None
                )
            else:
                # Backward-compatible: validate primary_task
                if self.wgan_primary_task not in train_labels:
                    print(
                        f"    Primary task '{self.wgan_primary_task}' not found in labels. Available: {list(train_labels.keys())}"
                    )
                    print("    Skipping WGAN-GP balancing")
                    self._augmented_labels = None
                    return train_data, test_data, train_labels, test_labels

                # Determine per-class counts and target based on primary_task
                primary_labels = train_labels[self.wgan_primary_task]
                train_idx = primary_labels.argmax(axis=1)
                classes, counts = np.unique(train_idx, return_counts=True)
                print(
                    f"    Train set size: {len(train_data)}  {self.wgan_primary_task} class counts: {dict(zip(classes.tolist(), counts.tolist()))}"
                )
                original_counts_map = {
                    int(c): int(n) for c, n in zip(classes.tolist(), counts.tolist())
                }
                current_max = int(counts.max()) if counts.size > 0 else 0
                target = (
                    int(current_max * self.wgan_target_ratio)
                    if current_max > 0
                    else None
                )
                if target is None or target <= 0:
                    print("    No majority class found, skipping balancing")
                    print(
                        f"    WGAN-GP effect: shape {original_shape} -> {np.shape(train_data)}; counts {original_counts_map} -> {original_counts_map}"
                    )
                    self._augmented_labels = None
                    return train_data, test_data, train_labels, test_labels

                have_map = {
                    int(c): int(n) for c, n in zip(classes.tolist(), counts.tolist())
                }
                num_classes = primary_labels.shape[1]
                needs_map = {
                    c: max(0, target - have_map.get(c, 0)) for c in range(num_classes)
                }
                print(
                    f"    WGAN target per class: {target} (ratio={self.wgan_target_ratio})  Planned adds: {needs_map}"
                )
                if all(v <= 0 for v in needs_map.values()):
                    print("    Already at or above target; skipping WGAN-GP")
                    print(
                        f"    WGAN-GP effect: shape {original_shape} -> {np.shape(train_data)}; counts {original_counts_map} -> {original_counts_map}"
                    )
                    self._augmented_labels = None
                    return train_data, test_data, train_labels, test_labels

                if HAS_MT_WGAN_MLX:
                    aug_x, aug_y = balance_with_mt_wgan_mlx(
                        train_data_minmax.astype("float32"),
                        train_labels,
                        epochs=self.wgan_epochs,
                        batch_size=(
                            min(self.wgan_batch_size, len(train_data))
                            if len(train_data) > 0
                            else self.wgan_batch_size
                        ),
                        max_target=target,
                        verbose=True,
                        save_path=save_location,
                        n_critic=self.wgan_n_critic,
                        primary_task=self.wgan_primary_task,
                        seq_len=getattr(
                            self, "wgan_seq_len", getattr(self, "seq_len", 1)
                        ),
                    )
                else:
                    aug_x, aug_y = balance_with_mt_wgan_gp(
                        train_data_minmax.astype("float32"),
                        train_labels,
                        epochs=self.wgan_epochs,
                        batch_size=(
                            min(self.wgan_batch_size, len(train_data))
                            if len(train_data) > 0
                            else self.wgan_batch_size
                        ),
                        max_target=target,
                        verbose=True,
                        save_path=save_location,
                        n_critic=self.wgan_n_critic,
                        primary_task=self.wgan_primary_task,
                    )
                display_task = self.wgan_primary_task
                original_counts_map = {
                    int(c): int(n) for c, n in zip(classes.tolist(), counts.tolist())
                }

            # Print stats for display_task
            if display_task and display_task in aug_y:
                aug_task_labels = aug_y[display_task]
                aug_idx = aug_task_labels.argmax(axis=1)
                aug_classes, aug_counts = np.unique(aug_idx, return_counts=True)
                print(f"    WGAN-GP training complete")
                print(
                    f"    Augmented train size: {len(aug_x)}  {display_task} class counts: {dict(zip(aug_classes.tolist(), aug_counts.tolist()))}"
                )
                new_counts_map = {
                    int(c): int(n)
                    for c, n in zip(aug_classes.tolist(), aug_counts.tolist())
                }
                if "original_counts_map" in locals():
                    print(
                        f"    WGAN-GP effect: shape {original_shape} -> {np.shape(aug_x)}; counts {original_counts_map} -> {new_counts_map}"
                    )
                else:
                    print(
                        f"    WGAN-GP effect: shape {original_shape} -> {np.shape(aug_x)}"
                    )
            else:
                print(f"    WGAN-GP training complete")
                print(f"    Augmented train size: {len(aug_x)}")
                print(
                    f"    WGAN-GP effect: shape {original_shape} -> {np.shape(aug_x)}"
                )

            # Quality assessment is automatically performed by balance_with_mt_wgan_gp
            # (see assess_quality parameter, defaults to True)

            # Store augmented labels so class weights can be recalculated from augmented distribution
            self._augmented_labels = aug_y

            # Transform aug_x back from minmax scaled space to normalized space
            # WGAN generates data in minmax space, but classifier models expect normalized space
            # Reshape 3D tensor (batch, seq_len, num_features) to 2D (batch * seq_len, num_features)
            aug_x_shape = aug_x.shape
            num_features = aug_x.shape[-1]
            aug_x_2d = aug_x.reshape(-1, num_features)
            aug_input = self._format_for_gan_scaler(aug_x_2d)
            aug_x_2d = self.denormalise_from_gan(aug_input)
            if isinstance(aug_x_2d, pd.DataFrame):
                aug_x_2d = aug_x_2d.to_numpy()
            aug_x = aug_x_2d.reshape(aug_x_shape)

            return aug_x, test_data, aug_y, test_labels
        except Exception as e:
            print("    WGAN-GP encountered an error; returning original data")
            print(f"      Error: {e}")
            print(traceback.format_exc())
            print(
                f"    WGAN-GP effect: shape {original_shape} -> {np.shape(train_data)}; counts unchanged"
            )
            self._augmented_labels = None
            return train_data, test_data, train_labels, test_labels

    def _format_for_gan_scaler(self, array_2d):
        if isinstance(array_2d, pd.DataFrame):
            return array_2d
        if hasattr(self.gan_scaler_a, "feature_names_in_"):
            feature_names = list(self.gan_scaler_a.feature_names_in_)
            try:
                return pd.DataFrame(array_2d, columns=feature_names)
            except ValueError:
                pass
        return array_2d

    def _prepare_labels_for_wgan(
        self, labels: Dict[str, np.ndarray]
    ) -> Tuple[Dict[str, np.ndarray], Dict[str, Optional[Dict[int, int]]]]:
        processed: Dict[str, np.ndarray] = {}
        mappings: Dict[str, Optional[Dict[int, int]]] = {}

        for task, values in labels.items():
            arr = np.asarray(values)
            if arr.ndim == 2 and arr.shape[1] > 1:
                processed[task] = arr.astype("float32")
                mappings[task] = None
            else:
                flattened = arr.reshape(-1).astype(int)
                classes = np.sort(np.unique(flattened))
                class_to_index = {int(cls): idx for idx, cls in enumerate(classes)}
                index_to_class = {idx: int(cls) for idx, cls in enumerate(classes)}
                label_indices = np.array(
                    [class_to_index[int(cls)] for cls in flattened],
                    dtype=int,
                )
                one_hot = np.eye(len(classes), dtype="float32")[label_indices]
                processed[task] = one_hot
                mappings[task] = index_to_class

        return processed, mappings
