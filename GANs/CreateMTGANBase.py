# pragma pylint: disable=C0103, C0114, C0115, C0116, C0301, C0302, C0303, C0325, C0411, C0413
# pragma pylint: disable=W0105, W1203, W1309, W1514, W0613, W0621
# type: ignore

"""
CreateMTGANBase - specialized base class for creating Multi-Task GAN models
using aggregated sequential tensors (3D) instead of 2D dataframes.
Ensures that the temporal patterns of the sequences are preserved for training.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from pandas import DataFrame

group_dir = str(Path(__file__).parent)
sys.path.append(group_dir)

from CreateGANBase import CreateGANBase # noqa: E402

class CreateMTGANBase(CreateGANBase):
    """
    Subclass of CreateGANBase that overrides create_models to aggregate
    data into 3D tensors (Batch, Seq, Features) before training.
    """

    def create_models(
        self,
        dataframes: List[DataFrame],
        labels: List[Any],
        pair_names: Optional[List[str]] = None,
    ) -> None:
        config = self.gan_config
        descriptor = config.get("description") or config.get("name") or self.__class__.__name__
        print(f"    Creating {descriptor} models (Multi-Task Sequence Aggregation)")

        if not dataframes:
            print("    No dataframes supplied to create_models; skipping GAN creation")
            return

        if not labels or len(labels) != len(dataframes):
            print("    Label count mismatch in create_models; skipping GAN creation")
            return

        # 1. Sequential Transformation & Normalization per pair
        all_x_tensors = []
        all_y_dicts = []

        seq_len = getattr(self, "seq_len", 1)
        print(f"    Building sequential tensors for {len(dataframes)} pairs (seq_len={seq_len})")

        use_post_gan_scaling = bool(getattr(self, "use_post_gan_scaling", False))

        for i, df in enumerate(dataframes):
            if use_post_gan_scaling:
                # Post-GAN scaling pipeline: feed raw data to the GAN.
                # The GAN does its own internal z-score; a tensor scaler is
                # applied to the augmented tensor after generation.
                df_ready = df
            else:
                # Standard pipeline: pre-normalize before GAN sees the data.
                df_norm = self.scale_dataframe(df)
                df_ready = self.normalise_for_gan(df_norm)

            # Determine labels dict for this pair
            labels_obj = labels[i]
            if not isinstance(labels_obj, dict):
                # Fallback for single-task labels passed to MT base
                primary_task = config.get("primary_task", "trading")
                labels_dict = {primary_task: labels_obj}
            else:
                labels_dict = labels_obj

            # Build 3D tensors (B, S, F) using standard strategy logic
            # This ensures GAN input matches the architecture expected by the classifier
            x = self.dataframeUtils.df_to_tensor(
                df_ready, seq_len, method=getattr(self, "tensor_method", 1)
            )
            
            # Method 0 (TF) reduces sample count by (seq_len - 1), others preserve via padding
            offset = (seq_len - 1) if getattr(self, "tensor_method", 1) == 0 else 0
            y = {task: lbl[offset:] for task, lbl in labels_dict.items()}

            if x.size > 0:
                all_x_tensors.append(x)
                all_y_dicts.append(y)

        if not all_x_tensors:
            print("    No sequence data generated after filtering; skipping GAN training")
            return

        # 2. Global Aggregation across all pairs
        train_data = np.concatenate(all_x_tensors, axis=0)
        
        # Aggregate multi-task labels
        train_labels: Dict[str, np.ndarray] = {}
        # Iterate over tasks present in the first pair's built tensors
        tasks = all_y_dicts[0].keys()
        for task in tasks:
            task_arrays = [yd[task] for yd in all_y_dicts]
            concatenated = np.concatenate(task_arrays, axis=0)
            # Ensure proper one-hot encoding consistency
            train_labels[task] = self._ensure_one_hot(concatenated)

        # 3. Stats Logging
        print(f"    Aggregated 3D Tensor shape: {train_data.shape}")
        for task, lbls in train_labels.items():
            if lbls.ndim == 2:
                class_counts = np.sum(lbls, axis=0).astype(int).tolist()
                print(f"      Task '{task}' class distribution: {class_counts}")

        # 4. Global Shuffling (preserving multi-task/tensor alignment)
        if config.get("shuffle_before_gan", True):
            seed = config.get("train_shuffle_seed", 42)
            rng = np.random.RandomState(seed)
            indices = rng.permutation(len(train_data))
            train_data = train_data[indices]
            train_labels = {t: v[indices] for t, v in train_labels.items()}

        # 5. Invoke GAN Training implementation
        self.run_gan_training(
            combined_df=None, # Tensors are decoupled from raw DF stacking
            train_data=train_data,
            test_data=None,
            train_labels=train_labels,
            test_labels=None,
            config=config,
        )

        self.on_gan_training_complete()
        self._reset_state()
