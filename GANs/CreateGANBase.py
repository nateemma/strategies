# pragma pylint: disable=C0103, C0114, C0115, C0116, C0301, C0302, C0303, C0325, C0411, C0413
# pragma pylint: disable=W0105, W1203, W1309, W1514, W0613, W0621
# type: ignore
"""
CreateGANBase - shared infrastructure for every GAN creator strategy.

The GAN subsystem exposes a uniform GANInterface (fit/generate/save/load)
parameterised by GANType, so each creator variant only needs to:

  1. Aggregate training data across the configured pairs,
  2. Hand the data to GANInterface.fit(),
  3. Persist the result via GANInterface.save() with MASTER thresholds.

This base class covers everything except step 2 — concrete subclasses
implement ``run_gan_training`` to invoke the appropriate interface call.
It also owns the MASTER_* threshold values: these are forced onto the
mixed-in NN strategy during bot_start() so that the training labels
generated for GAN input are derived from the same thresholds that will
be stamped into the saved GAN metadata.
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
from GANs.GANType import GANType  # noqa: E402


class CreateGANBase:
    """
    Mixin-style base class that encapsulates the common workflow for gathering,
    normalising, and shuffling data prior to training a GAN.

    Subclasses (CreateGAN, CreateMTGAN) implement ``run_gan_training`` to
    perform the backend-specific interface call.
    """

    # --- Defaults shared across every variant --------------------------------
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

    # --- MASTER threshold values ---------------------------------------------
    # These are the single source of truth for the gain/loss/training-type
    # parameters that drive training-label generation *and* are persisted in
    # the GAN metadata.  A strategy that later loads the GAN will pick up
    # these values automatically, so the labels the classifier was trained on
    # stay consistent with the ones the GAN was trained on.
    MASTER_MIN_BUY_GAIN_THRESHOLD = 0.016
    MASTER_MIN_SELL_LOSS_THRESHOLD = 0.012
    MASTER_TRAINING_TYPE = 19

    # Concrete subclasses set this to the GANType they create.  Staying
    # here as NONE means the caller must set it (or a subclass hard-codes it).
    gan_type: GANType = GANType.NONE

    def __init__(self, gan_config: Optional[Dict[str, Any]] = None, **kwargs) -> None:
        super().__init__(**kwargs)
        self.gan_config: Dict[str, Any] = self._build_gan_config(gan_config)
        self._reset_state()

    # ---------------------------------------------------------------------- #
    # bot_start — enforces MASTER thresholds before any indicator runs       #
    # ---------------------------------------------------------------------- #

    def bot_start(self, **kwargs) -> None:
        """
        Force MASTER_* values onto the NN strategy before any populate_*
        runs.  Applies uniformly to all GAN types — previously only the
        CTAB variants set these, so the WGAN creators could silently drift
        if hyperopt mutated buy_params/sell_params.

        Marking the strategy as a GAN creator (via
        ``_is_gan_creation_strategy = True``) tells
        ``BaseNNStrategy.bot_start`` to skip its threshold overrides, so
        the MASTER values reach the GAN metadata unmodified.

        Order matters: we set the flag and the MASTER thresholds *before*
        calling ``super().bot_start()`` so BaseNNStrategy sees both when
        it runs.
        """
        self._validate_master_thresholds()

        # Only assign if the attribute exists on the mixed-in strategy.
        # CreateGANBase is sometimes used in non-NN test contexts where
        # these attrs would otherwise pollute the instance namespace.
        if hasattr(self, "MIN_BUY_GAIN_THRESHOLD"):
            self.MIN_BUY_GAIN_THRESHOLD = self.MASTER_MIN_BUY_GAIN_THRESHOLD
        if hasattr(self, "MIN_SELL_LOSS_THRESHOLD"):
            self.MIN_SELL_LOSS_THRESHOLD = self.MASTER_MIN_SELL_LOSS_THRESHOLD
        if hasattr(self, "TRAINING_TYPE"):
            self.TRAINING_TYPE = self.MASTER_TRAINING_TYPE

        self._is_gan_creation_strategy = True

        # Continue up the MRO (typically BaseNNStrategy.bot_start).
        super().bot_start(**kwargs)

    def _validate_master_thresholds(self) -> None:
        """Fail loudly if MASTER_* drift from the consumer NN strategy."""
        try:
            from Framework.BaseNNStrategy import BaseNNStrategy  # noqa: E402
        except ImportError:
            # Not mixed into an NN strategy — nothing to validate against.
            return

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
                f"{self.__class__.__name__} MASTER_* mismatch with NNStrategy:",
                *[
                    f"  {key}: NNStrategy={vals[0]} vs "
                    f"{self.__class__.__name__}={vals[1]}"
                    for key, vals in mismatches.items()
                ],
            ]
            raise ValueError("\n".join(lines))

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

    def _log_master_thresholds(self) -> None:
        """Shared announcement block printed by every run_gan_training."""
        print("    MASTER thresholds (will be stored in GAN metadata):")
        print(
            f"      MASTER_MIN_BUY_GAIN_THRESHOLD = "
            f"{self.MASTER_MIN_BUY_GAIN_THRESHOLD:.4f}"
        )
        print(
            f"      MASTER_MIN_SELL_LOSS_THRESHOLD = "
            f"{self.MASTER_MIN_SELL_LOSS_THRESHOLD:.4f}"
        )
        print(f"      MASTER_TRAINING_TYPE = {self.MASTER_TRAINING_TYPE}")

    def _master_save_kwargs(self) -> Dict[str, Any]:
        return {
            "min_buy_gain_threshold": self.MASTER_MIN_BUY_GAIN_THRESHOLD,
            "min_sell_loss_threshold": self.MASTER_MIN_SELL_LOSS_THRESHOLD,
            "training_type": self.MASTER_TRAINING_TYPE,
        }

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
