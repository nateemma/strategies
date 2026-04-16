# pragma pylint: disable=C0103, C0114, C0115, C0116, C0301, C0302, C0303, C0325, C0411, C0413
# pragma pylint: disable=W0105, W1203, W1309, W1514, W0613, W0621,
# type: ignore
# pylint: disable=import-error

"""
NNMT_WGAN - Subclass of NNMTStrategy using WGAN-GP for multi-task augmentation.

Uses GANInterface(GANType.MT_WGAN) to delegate all GAN-specific dispatch to the
interface, including the MLX / TensorFlow backend selection.
"""

import os
import sys
import traceback
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
from pandas import DataFrame

group_dir = str(Path(__file__).parent)
sys.path.append(group_dir)

from NNMTStrategy import NNMTStrategy  # noqa: E402
from GANs.GANType import GANType  # noqa: E402
from GANs.GANInterface import GANInterface  # noqa: E402


class NNMT_WGAN(NNMTStrategy):

    augment_training_data = (
        False  # only 'real' signals in 2-D mode; augmentation is done in 3-D
    )

    # WGAN hyper-parameters — override in subclasses as needed.
    wgan_epochs = 100
    wgan_batch_size = 2048
    wgan_n_critic = 5
    wgan_target_ratio = 0.8          # fallback when task_target_ratios is None
    wgan_primary_task = "trading"    # fallback when task_target_ratios is None
    wgan_task_target_ratios: Optional[Dict] = {
        "trading":  0.8,
        "regime":   0.8,
        "risk":     0.8,
        "momentum": 0.8,
        "flow":     0.8,
        "profit":   0.8,
    }

    # ---------------------------------------------------------------------- #
    # Hooks                                                                   #
    # ---------------------------------------------------------------------- #

    def enhance_training_data(
        self, train_df: DataFrame, train_labels: Dict[str, np.ndarray]
    ) -> Tuple[DataFrame, Dict[str, np.ndarray]]:
        """Skip 2-D augmentation — multi-task WGAN works on 3-D tensors."""
        print("    Skipping 2-D WGAN augmentation (using 3-D sequential instead)")
        return train_df, train_labels

    def preprocess_training_data(
        self,
        dataframe: DataFrame,
        train_data: np.ndarray,
        test_data: np.ndarray,
        train_labels: Dict[str, np.ndarray],
        test_labels: Dict[str, np.ndarray],
    ) -> Tuple[np.ndarray, np.ndarray, Dict[str, np.ndarray], Dict[str, np.ndarray]]:
        """Balance 3-D sequential training data via MT-WGAN using GANInterface."""
        self._augmented_labels = None
        original_shape = np.shape(train_data)

        if len(train_data) == 0:
            print("    No training data to balance")
            return train_data, test_data, train_labels, test_labels

        try:
            save_path = os.path.join(self.get_storage_location(), "GANs")

            # Transform 3-D tensor (batch, seq_len, features) → minmax space.
            train_data_shape = train_data.shape
            num_features = train_data.shape[-1]
            train_2d = train_data.reshape(-1, num_features)
            gan_input = self._format_for_gan_scaler(train_2d)
            train_minmax_2d = self.normalise_for_gan(gan_input)
            if isinstance(train_minmax_2d, pd.DataFrame):
                train_minmax_2d = train_minmax_2d.to_numpy()
            train_minmax = train_minmax_2d.reshape(train_data_shape)

            # Determine augmentation targets
            if self.wgan_task_target_ratios is not None:
                task_target_ratios = self.wgan_task_target_ratios
                display_task = next(iter(task_target_ratios), None)
            else:
                # Backward-compatible single-primary-task path.
                if self.wgan_primary_task not in train_labels:
                    print(
                        f"    Primary task '{self.wgan_primary_task}' not found in labels. "
                        f"Available: {list(train_labels.keys())}"
                    )
                    print("    Skipping WGAN-GP balancing")
                    return train_data, test_data, train_labels, test_labels

                primary = train_labels[self.wgan_primary_task]
                train_idx = primary.argmax(axis=1)
                _, counts = np.unique(train_idx, return_counts=True)
                current_max = int(counts.max()) if counts.size > 0 else 0
                target = int(current_max * self.wgan_target_ratio) if current_max > 0 else None
                if not target:
                    print("    No majority class found, skipping balancing")
                    return train_data, test_data, train_labels, test_labels

                task_target_ratios = {self.wgan_primary_task: self.wgan_target_ratio}
                display_task = self.wgan_primary_task

            print("    Balancing training data with MT WGAN-GP (via GANInterface)")
            interface = GANInterface(GANType.MT_WGAN, save_path=save_path)
            try:
                interface.load()
                print(f"    Loaded existing MT WGAN model from {save_path}.")
            except Exception as load_err:
                raise RuntimeError(
                    f"MT WGAN model not found at {save_path}. "
                    f"Run CreateMTWGAN first to train and save the model. Error: {load_err}"
                ) from load_err

            # Generate per-task per-class augmentation samples.
            # collect additions: aug_data (n, seq, F), aug_labels_dict
            aug_data_list: list = [train_minmax]
            aug_labels_dict: Dict[str, list] = {t: [v] for t, v in train_labels.items()}

            for task, ratio_spec in task_target_ratios.items():
                if ratio_spec is None or task not in train_labels:
                    continue

                task_lbls = train_labels[task]
                task_idx = np.argmax(task_lbls, axis=1)
                unique, task_counts = np.unique(task_idx, return_counts=True)
                current_max = int(np.max(task_counts)) if task_counts.size > 0 else 0

                class_ratios = (
                    ratio_spec if isinstance(ratio_spec, dict)
                    else {c: ratio_spec for c in range(task_lbls.shape[1])}
                )
                for c in range(task_lbls.shape[1]):
                    ratio = class_ratios.get(c, 0.0)
                    if ratio <= 0.0:
                        continue
                    have = int(np.sum(task_idx == c))
                    target_n = int(current_max * ratio)
                    need = max(0, target_n - have)
                    if need <= 0:
                        continue

                    # Build task_labels for this batch: target task gets one-hot class c,
                    # other tasks use most-common class from real data.
                    batch_task_labels: Dict[str, np.ndarray] = {}
                    for other_task, other_lbls in train_labels.items():
                        if other_task == task:
                            nc = task_lbls.shape[1]
                            oh = np.zeros((need, nc), dtype="float32")
                            oh[:, c] = 1.0
                            batch_task_labels[other_task] = oh
                        else:
                            other_idx = np.argmax(other_lbls, axis=1)
                            most_common = int(np.bincount(other_idx).argmax())
                            nc = other_lbls.shape[1]
                            oh = np.zeros((need, nc), dtype="float32")
                            oh[:, most_common] = 1.0
                            batch_task_labels[other_task] = oh

                    gen_data, gen_labels = interface.generate(
                        n=need, task_labels=batch_task_labels
                    )
                    # gen_data: (need, 1, F) — keep 3D to match train_minmax shape
                    aug_data_list.append(gen_data)
                    for t in train_labels:
                        aug_labels_dict[t].append(batch_task_labels[t])

            aug_x = np.concatenate(aug_data_list, axis=0)
            aug_y = {t: np.concatenate(aug_labels_dict[t], axis=0) for t in train_labels}

            # Log augmentation stats.
            if display_task and display_task in aug_y:
                aug_task_idx = aug_y[display_task].argmax(axis=1)
                aug_classes, aug_counts = np.unique(aug_task_idx, return_counts=True)
                aug_counts_map = dict(zip(aug_classes.tolist(), aug_counts.tolist()))
                print("    WGAN-GP training complete")
                print(
                    f"    Augmented train size: {len(aug_x)}  "
                    f"{display_task} class counts: {aug_counts_map}"
                )
            else:
                print("    WGAN-GP training complete")
                print(f"    Augmented train size: {len(aug_x)}")
            print(f"    WGAN-GP effect: shape {original_shape} -> {np.shape(aug_x)}")

            self._augmented_labels = aug_y

            # Transform aug_x back from minmax space to normalised space.
            aug_shape = aug_x.shape
            aug_2d = aug_x.reshape(-1, num_features)
            aug_input = self._format_for_gan_scaler(aug_2d)
            aug_2d_norm = self.denormalise_from_gan(aug_input)
            if isinstance(aug_2d_norm, pd.DataFrame):
                aug_2d_norm = aug_2d_norm.to_numpy()
            aug_x = aug_2d_norm.reshape(aug_shape)

            return aug_x, test_data, aug_y, test_labels

        except Exception as exc:
            print("    WGAN-GP encountered an error; returning original data")
            print(f"      Error: {exc}")
            print(traceback.format_exc())
            self._augmented_labels = None
            return train_data, test_data, train_labels, test_labels

    # ---------------------------------------------------------------------- #
    # Helpers                                                                 #
    # ---------------------------------------------------------------------- #

    def _format_for_gan_scaler(self, array_2d: np.ndarray):
        if isinstance(array_2d, pd.DataFrame):
            return array_2d
        if hasattr(self.gan_scaler_a, "feature_names_in_"):
            feature_names = list(self.gan_scaler_a.feature_names_in_)
            try:
                return pd.DataFrame(array_2d, columns=feature_names)
            except ValueError:
                pass
        return array_2d
