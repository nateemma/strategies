# pragma pylint: disable=C0103, C0114, C0115, C0116, C0301, C0302, C0303, C0325, C0411, C0413
# pragma pylint: disable=W0105, W1203, W1309, W1514, W0613, W0621
# type: ignore
"""
CreateGANBase - shared functionality for building GAN training datasets across
single-task and multi-task strategy variants.

Concrete subclasses are expected to provide a `run_gan_training` implementation
that consumes a configuration dictionary and invokes the appropriate balancing
routine (e.g. WGAN-GP, MT-WGAN-GP, etc.).
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from pandas import DataFrame
from sklearn.utils import shuffle

group_dir = str(Path(__file__).parent)
sys.path.append(group_dir)

from utils.DataframePopulator import DatasetType  # noqa: E402


class CreateGANBase:
    """
    Mixin-style base class that encapsulates the common workflow for gathering,
    normalising, and shuffling data prior to training a GAN.
    """

    DEFAULT_GAN_CONFIG: Dict[str, Any] = {
        "name": "Generic GAN",
        "description": "Generic GAN",
        "balance_fn": None,
        "train_kwargs": {},
        "save_subdir": "GANs",
        "train_shuffle_seed": 42,
        "test_shuffle_seed": 42,
        "multi_task": False,
        "shuffle_before_gan": False,
    }

    def __init__(self, gan_config: Optional[Dict[str, Any]] = None, **kwargs) -> None:
        super().__init__(**kwargs)
        self.gan_config: Dict[str, Any] = self._build_gan_config(gan_config)
        self._reset_state()

    # ---------------------------------------------------------------------- #
    # Public workflow                                                        #
    # ---------------------------------------------------------------------- #

    def create_models(self, dataframes: List[DataFrame], labels: List[Any]) -> None:
        config = self.gan_config
        descriptor = config.get("description") or config.get("name") or self.__class__.__name__
        print(f"    Creating {descriptor} models using aggregate dataframe of all pairs")

        if not dataframes:
            print("    No dataframes supplied to create_models; skipping GAN creation")
            return

        if not labels or len(labels) != len(dataframes):
            print("    Label count mismatch in create_models; skipping GAN creation")
            return

        # Add sequential index to dataframes before processing
        dataframes = self.add_sequential_index(dataframes)

        full_df_norm_list: List[DataFrame] = []
        full_df_minmax_list: List[DataFrame] = []

        for df in dataframes:
            df_norm = self.scale_dataframe(df)
            full_df_norm_list.append(df_norm)

            df_minmax = self.normalise_for_gan(df_norm)
            df_minmax_df = self._ensure_dataframe(df_minmax, df_norm)
            full_df_minmax_list.append(df_minmax_df)

        combined_original = pd.concat(
            [df.reset_index(drop=True) for df in dataframes], ignore_index=True
        )
        combined_norm = pd.concat(full_df_norm_list, ignore_index=True)
        combined_minmax = pd.concat(full_df_minmax_list, ignore_index=True)

        self.print_dataframe_ranges("Original Column Ranges", combined_original)
        self.print_dataframe_ranges("Normalised Column Ranges", combined_norm)
        self.print_dataframe_ranges("MinMax Column Ranges", combined_minmax)

        train_data = combined_minmax.to_numpy(dtype=np.float32)
        train_labels = self._aggregate_labels_for_gan(labels)

        if config.get("shuffle_before_gan", False) and len(train_data) > 0:
            seed = config.get("train_shuffle_seed", 42)
            rng = np.random.RandomState(seed)
            indices = rng.permutation(len(train_data))
            train_data = train_data[indices]
            if isinstance(train_labels, dict):
                train_labels = {task: values[indices] for task, values in train_labels.items()}
            elif train_labels is not None:
                train_labels = train_labels[indices]

        self.run_gan_training(
            combined_df=combined_original,
            train_data=train_data,
            test_data=None,
            train_labels=train_labels,
            test_labels=None,
            config=config,
        )

        self.on_gan_training_complete()
        self._reset_state()

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        whitelist = self.dp.current_whitelist()
        curr_pair = metadata["pair"]

        self.iteration_init()
        dataframe = self.check_precision_columns(dataframe)
        dataframe = self.dataframePopulator.add_indicators(
            dataframe, dataset_type=DatasetType.MINIMAL
        )
        dataframe = self.add_additional_indicators(dataframe)

        labels = self.get_training_labels(dataframe)

        if not self.training_needed:
            return dataframe

        if self.aggregate_pairs:
            target_pairs = len(whitelist)
        else:
            target_pairs = 1

        if self.combined_df is None:
            self.combined_df = []
        if self.combined_labels is None:
            self.combined_labels = []

        if len(self.combined_df) == 0:
            print(f"    Init with: {curr_pair}")
        else:
            print(f"    Appending: {curr_pair}")

        self.combined_df.append(dataframe.reset_index(drop=True))
        self.combined_labels = self._extend_labels(self.combined_labels, labels)

        self.pair_count += 1

        if self.pair_count == target_pairs:
            dataframes_final = [df.copy() for df in self.combined_df]
            labels_final = self._finalize_labels(self.combined_labels)
            self.create_models(dataframes_final, labels_final)

        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        return dataframe

    # ---------------------------------------------------------------------- #
    # Hooks for subclasses                                                   #
    # ---------------------------------------------------------------------- #

    def run_gan_training(
        self,
        *,
        combined_df: DataFrame,
        train_data: np.ndarray,
        test_data: np.ndarray,
        train_labels: Any,
        test_labels: Any,
        config: Dict[str, Any],
    ) -> None:
        """
        Subclasses must implement GAN-specific training logic using the provided
        configuration dictionary.
        """
        raise NotImplementedError("run_gan_training must be implemented by subclasses")

    def on_gan_training_complete(self) -> None:
        """
        Hook for subclasses to run additional logic once GAN training completes.
        """
        pass

    # ---------------------------------------------------------------------- #
    # Helpers                                                                #
    # ---------------------------------------------------------------------- #

    def _build_gan_config(self, override: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        config = dict(self.get_default_gan_config())
        if override:
            config.update(override)
        return config

    def get_default_gan_config(self) -> Dict[str, Any]:
        return dict(self.DEFAULT_GAN_CONFIG)

    def get_gan_save_path(self, config: Optional[Dict[str, Any]] = None) -> str:
        config = config or self.gan_config
        subdir = config.get("save_subdir", "GANs")
        if getattr(self, "use_pca_reduction", False):
            subdir = subdir + "_PCA"
        return f"{self.get_storage_location()}/{subdir}"

    def _ensure_dataframe(self, data: Any, reference: DataFrame) -> DataFrame:
        if isinstance(data, pd.DataFrame):
            return data
        columns = reference.columns if isinstance(reference, pd.DataFrame) else None
        index = reference.index if isinstance(reference, pd.DataFrame) else None
        return pd.DataFrame(data, columns=columns, index=index)

    def _shuffle_split(
        self, tensor: np.ndarray, labels: Any, seed: Optional[int]
    ) -> Tuple[np.ndarray, Any]:
        if tensor is None or len(tensor) == 0 or seed is None:
            return tensor, labels

        if isinstance(labels, dict):
            rng = np.random.RandomState(seed)
            indices = rng.permutation(len(tensor))
            tensor_shuffled = tensor[indices]
            labels_shuffled = {task: values[indices] for task, values in labels.items()}
            return tensor_shuffled, labels_shuffled

        tensor_shuffled, labels_shuffled = shuffle(tensor, labels, random_state=seed)
        return tensor_shuffled, labels_shuffled

    def _clone_labels(self, labels: Any) -> Any:
        if isinstance(labels, dict):
            return {key: np.copy(value) for key, value in labels.items()}
        if hasattr(labels, "copy"):
            return labels.copy()
        return np.array(labels)

    def _extend_labels(self, combined_labels: List[Any], labels: Any) -> List[Any]:
        combined_labels.append(self._clone_labels(labels))
        return combined_labels

    def _finalize_labels(self, combined_labels: List[Any]) -> List[Any]:
        return [self._clone_labels(lbl) for lbl in combined_labels]

    def _aggregate_labels_for_gan(self, labels_list: List[Any]) -> Any:
        if not labels_list:
            return None

        first = labels_list[0]
        if isinstance(first, dict):
            aggregated: Dict[str, np.ndarray] = {}
            for task in first.keys():
                concatenated = np.concatenate(
                    [np.asarray(lbl[task]) for lbl in labels_list],
                    axis=0,
                )
                aggregated[task] = self._ensure_one_hot(concatenated)
            return aggregated

        concatenated = np.concatenate([np.asarray(lbl) for lbl in labels_list], axis=0)
        return self._ensure_one_hot(concatenated)

    def _ensure_one_hot(self, labels: np.ndarray) -> np.ndarray:
        array = np.asarray(labels)

        if array.ndim == 1:
            num_classes = int(np.max(array)) + 1 if array.size > 0 else 1
            return self.dataframeUtils.one_hot_encode(array.astype(int), num_classes)

        if array.ndim == 2:
            row_sums = np.sum(array, axis=1)
            if np.allclose(row_sums, 1.0) and np.all(array >= 0):
                return array
            if array.shape[1] == 1:
                flattened = array.reshape(-1)
                num_classes = int(np.max(flattened)) + 1 if flattened.size > 0 else 1
                return self.dataframeUtils.one_hot_encode(
                    flattened.astype(int), num_classes
                )

        return array

    def _reset_state(self) -> None:
        self.pair_count = 0
        self.combined_df: Optional[List[DataFrame]] = None
        self.combined_labels: Optional[List[Any]] = None
