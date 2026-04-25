# pragma pylint: disable=C0103, C0114, C0115, C0116, C0301, C0302, C0303, C0325, C0411, C0413
# pragma pylint: disable=W0105, W1203, W1309, W1514, W0613, W0621,
# type: ignore
# pylint: disable=import-error
# flake8: noqa: F401, E402, F541, W0718, W0719

"""
BaseNNStrategy - Base class for all Neural Network trading strategies.

Inherits from BaseStrategy and adds:
 - NN-specific imports (tensorflow, ClassifierKeras, GANs, TrainingSignals)
 - Training parameters (seq_len, epochs, batch_size, etc.)
 - Normalization pipeline (rolling_dataframe_normalise, scalers, PCA)
 - Feature list management (include_list, pre_normalized_columns)
 - Training label generation (peak detection, signal augmentation)
 - GAN augmentation (WGAN-GP, CTAB-GAN+)
 - Model management (classifiers, storage, model paths)
 - Training / prediction pipeline
 - Aggregation helpers for multi-pair training

Subclasses (NNNCStrategy, NNMTStrategy, SklearnStrategy, etc.) override
get_classifier_type() and get_classifier() to provide specific model architectures.
"""

# --------------------------------
# Top level imports
# --------------------------------
from typing import Optional, Tuple, List, Any, Dict, Iterable, Union
import traceback

import numpy as np
import pandas as pd
from pandas import DataFrame

import tensorflow as tf

import os
import pickle
import sys
from pathlib import Path
import logging

from sklearn.preprocessing import MinMaxScaler, RobustScaler
from sklearn.decomposition import PCA
from scipy.signal import find_peaks
import pywt

try:
    import mlx.core as mx

    HAS_MLX = True
except ImportError:
    HAS_MLX = False

from sklearn.utils import shuffle

from utils.DataframeUtils import ScalerType, DataframeUtils
from utils.DataframePopulator import DataframePopulator, DatasetType
from utils.ClassifierKeras import ClassifierKeras

from utils.Scalers import scaler_exists, save_scaler, load_scaler
from GANs.GANInterface import GANInterface  # noqa: E402

import Framework.TrainingSignals as TrainingSignals

from Framework.BaseStrategy import (
    BaseStrategy,
    TradingAction,
    MarketRegime,
    RiskLevel,
    FlowDirection,
    MomentumDirection,
    StrategyConfig,
    NormalizationType,
    ModelType,
    GANType,
)

# --------------------------------
# Global setup
# --------------------------------
log = logging.getLogger(__name__)


# =========================================================================
# BaseNNStrategy
# =========================================================================

class BaseNNStrategy(BaseStrategy):

    # --------------------------------
    # NN-specific training parameters
    # --------------------------------

    seq_len = 16  # 'depth' of training sequence
    num_epochs = 256  # number of iterations for training
    batch_size = 2048  # batch size for training
    stride = 1  # stride for training

    # Common model flags
    refit_model = False  # force retraining to test class weights
    model_per_pair = False  # single model for all pairs
    combine_models = False  # combine training across all pairs

    # NN-specific debug
    dbg_scan_classifiers = False
    dbg_test_classifier = True  # test classifiers after fitting

    TRAIN_DATA_SPLIT = 0.8  # 80% of the data for training, 20% for testing
    shuffle_train_data = True
    use_markov_smoothing = False
    markov_smoothing_alpha = 0.5
    markov_transition_matrix = None

    # Use MLX accelerated tensor building (method 3) if available,
    # otherwise fallback to standard TF approach (method 0)
    tensor_method = 3 if HAS_MLX else 0

    # Training signal parameters
    filter_signals = False  # filter signals based on guard metric
    lookahead_window = BaseStrategy.PEAK_WINDOW
    HORIZON = BaseStrategy.PEAK_WINDOW
    RISK_LOOKBACK = 200
    FLOW_LOOKBACK = 200
    MIN_BUY_GAIN_THRESHOLD = 0.016  # minimum gain for buy signals
    MIN_SELL_LOSS_THRESHOLD = 0.012  # minimum loss for sell signals
    TRAINING_TYPE = 19
    augment_training_data = True

    training_needed = True  # set automatically

    # Scaler state
    main_scaler = None
    main_scaler_name = "main_scaler"
    gan_scaler_a = None
    gan_scaler_a_name = "gan_scaler_a"

    # PCA state
    pca_components = None
    pca_mean = None
    pca_n_components = None
    pca_feature_columns = None
    pca_explained_variance = None  # per-component variance (for whitening transform)
    pca_col_min = None  # per-component min from training data (for scaling to [-1, 1])
    pca_col_max = None  # per-component max from training data (for scaling to [-1, 1])
    pca_name = "pca_data"
    pca_passthrough_columns = [
        "rsi_scaled",
        "mfi_scaled",
        "ema_fast_norm",
        "fastk_scaled",
        "di_diff_scaled",
    ]
    use_pca_reduction = False
    classifier = None
    classifier_type = None

    # Aggregation
    aggregate_pairs = True  # use all pairs for training (in backtest)
    df_array: [DataFrame] = []
    label_array: List[Any] = []
    pair_count = 0

    # Hyperopt param for training type
    from freqtrade.strategy import IntParameter
    training_type = IntParameter(
        0,
        19,
        default=16,
        space="buy",
        load=True,
        optimize=False,
    )

    # --------------------------------
    # Feature lists
    # --------------------------------

    # indicators to include in normalisation. Anything not in the list will be dropped.
    include_list = [
        # "ad_scaled",
        "adx_scaled",
        "aroonosc_scaled",
        "atr_norm",
        # "bb_position",
        "bb_width",
        # "cci_scaled",
        # "close_norm",
        "di_diff_scaled",
        "doy_cos",
        "doy_sin",
        "dow_cos",
        "dow_sin",
        "ema_fast_norm",
        "fast_diff",
        # "fastd_scaled",
        "fastk_scaled",
        "fisher_ss",
        "cg_ss",
        # "dymi_scaled",
        # "fisher_wr",
        # "flow",
        "gain_norm",
        # "guard_metric",
        # "hour_cos",
        # "hour_sin",
        "log_volume_norm",
        "macd_norm",
        "macdhist_norm",
        # "macdsignal_norm",
        "mfi_scaled",
        # "minus_di_scaled",
        "mod_cos",
        "mod_sin",
        # "momentum",
        # "obv_scaled",
        # "plus_di_scaled",
        # "pv_trend",
        # "regime",
        # "risk",
        "rsi_scaled",
        # "rsi_sma",
        "sar_ratio",
        "spread_ma",
        # "volume_sma_norm",
        "vwap_ratio",
        # "willr_scaled",
    ]

    # columns that are already normalized and should not be scaled
    pre_normalized_columns = [
        "ad_scaled",
        "adx_scaled",
        "aroonosc_scaled",
        "atr_norm",
        "bb_position",
        "bb_width",
        "cci_scaled",
        "close_norm",
        "doy_cos",
        "doy_sin",
        "dow_cos",
        "dow_sin",
        "ema_fast_norm",
        "fastd_scaled",
        "fastk_scaled",
        "fast_diff",
        "fisher_ss",
        "cg_ss",
        "dymi_scaled",
        "gain_norm",
        "guard_metric",
        # "hour_cos",
        # "hour_sin",
        "mod_sin",
        "mod_cos",
        "log_volume_norm",
        "macd_norm",
        "macdhist_norm",
        "mfi_scaled",
        "minus_di_scaled",
        "obv_scaled",
        "plus_di_scaled",
        "pv_trend",
        "rsi_scaled",
        "rsi_sma",
        "willr_scaled",
    ]

    # override in subclass to add one-hot encoded columns
    one_hot_columns = []
    # one_hot_columns = ["regime", "risk", "flow", "momentum"]
    # one_hot_columns = ["regime", "flow"]

    # =========================================================================
    # Strategy info override
    # =========================================================================

    def print_strategy_info(self):
        """Print strategy information"""
        print("")
        print("Strategy Parameters/Flags")
        print("")

        print(f"    refit_model:    {self.refit_model}")
        print(f"    model_per_pair: {self.model_per_pair}")
        print(f"    combine_models: {self.combine_models}")
        print("")

    # =========================================================================
    # Aggregation helpers
    # =========================================================================

    @staticmethod
    def aggregate_dataframes(dataframes: Iterable[DataFrame]) -> DataFrame:
        """Concatenate multiple dataframes, resetting indices to avoid duplicates."""
        frames = [df.reset_index(drop=True) for df in dataframes]
        if not frames:
            return DataFrame()
        return pd.concat(frames, ignore_index=True)

    @staticmethod
    def aggregate_single_labels(
        labels: Iterable[Union[np.ndarray, List[int]]],
    ) -> np.ndarray:
        """Concatenate single-task label arrays, preserving original dtype."""
        arrays = [np.asarray(lbl) for lbl in labels]
        if not arrays:
            return np.array([], dtype=np.int64)
        dtype = arrays[0].dtype
        return np.concatenate(arrays, axis=0).astype(dtype, copy=False)

    @staticmethod
    def aggregate_multi_labels(
        labels: Iterable[Dict[str, Union[np.ndarray, List[int]]]],
    ) -> Dict[str, np.ndarray]:
        """Merge multi-task label dictionaries by concatenating each task."""
        merged: Dict[str, List[np.ndarray]] = {}
        for label_dict in labels:
            for task, values in label_dict.items():
                merged.setdefault(task, []).append(np.asarray(values))

        result: Dict[str, np.ndarray] = {}
        for task, parts in merged.items():
            if not parts:
                result[task] = np.array([])
                continue
            dtype = parts[0].dtype
            result[task] = np.concatenate(parts, axis=0).astype(dtype, copy=False)
        return result

    # =========================================================================
    # Storage / Model management
    # =========================================================================

    def get_model_path(self) -> str:
        """Get the model path for saving/loading"""
        name = self.__class__.__name__
        root_dir = self.get_storage_location()
        model_path = root_dir + name + "/" + name + ".keras"
        return model_path

    def get_markov_matrix_path(self) -> str:
        """Get the path for saving/loading the Markov transition matrix."""
        model_path = self.get_model_path()
        return model_path.replace(".keras", "_markov.npy")

    @staticmethod
    def get_pca_path(storage_location: str, name: str) -> Path:
        """Path for PCA artifact (same directory as scalers)."""
        return Path(storage_location) / f"{name}.pkl"

    @staticmethod
    def pca_data_exists(storage_location: str, name: str) -> bool:
        """Check if PCA data exists at the given location."""
        return BaseNNStrategy.get_pca_path(storage_location, name).exists()

    @staticmethod
    def save_pca_data(
        storage_location: str,
        name: str,
        components: np.ndarray,
        mean: np.ndarray,
        n_components: int,
        feature_columns: Optional[List[str]] = None,
        explained_variance_ratio: Optional[np.ndarray] = None,
        explained_variance: Optional[np.ndarray] = None,
        col_min: Optional[np.ndarray] = None,
        col_max: Optional[np.ndarray] = None,
    ) -> None:
        """Save PCA artifacts for later load."""
        path = BaseNNStrategy.get_pca_path(storage_location, name)
        path.parent.mkdir(parents=True, exist_ok=True)
        print(f"    Saving PCA data to {path}")
        payload = {
            "components": components,
            "mean": mean,
            "n_components": n_components,
            "feature_columns": feature_columns,
            "explained_variance_ratio": explained_variance_ratio,
            "explained_variance": explained_variance,
            "col_min": col_min,
            "col_max": col_max,
        }
        with open(path, "wb") as f:
            pickle.dump(payload, f)

    @staticmethod
    def load_pca_data(storage_location: str, name: str) -> Dict[str, Any]:
        """Load PCA artifacts."""
        path = BaseNNStrategy.get_pca_path(storage_location, name)
        with open(path, "rb") as f:
            payload = pickle.load(f)
        return payload

    def model_exists(self) -> bool:
        """Check if model exists on disk"""
        if self.classifier is not None:
            return self.classifier.model_exists()

        model_path_keras = self.get_model_path()
        model_path_sav = model_path_keras.replace(".keras", ".sav")

        model_found = os.path.exists(model_path_keras) or os.path.exists(model_path_sav)
        return model_found

    # =========================================================================
    # One-hot encoding
    # =========================================================================

    def process_one_hot_columns(self, dataframe: DataFrame) -> DataFrame:
        """Process one-hot encoded columns"""
        fixed_class_counts = {
            "regime": len(MarketRegime),
            "risk": len(RiskLevel),
            "flow": len(FlowDirection),
            "momentum": len(MomentumDirection),
        }
        for col in self.one_hot_columns:
            if col in dataframe.columns:
                num_classes = fixed_class_counts.get(col, len(dataframe[col].unique()))
                ohe = self.dataframeUtils.one_hot_encode(dataframe[col], num_classes)
                for i in sorted(set([0, num_classes - 1])):
                    new_col = f"{col}_{i}"
                    dataframe[new_col] = ohe[:, i]
                    if not new_col in self.pre_normalized_columns:
                        self.pre_normalized_columns.append(new_col)
                    if not new_col in self.include_list:
                        self.include_list.append(new_col)

                if col in dataframe.columns:
                    dataframe = dataframe.drop(columns=[col])
                if col in self.include_list:
                    self.include_list.remove(col)
            else:
                print(f"    Warning: Column {col} not found in dataframe")
        return dataframe

    # =========================================================================
    # Dataframe analysis
    # =========================================================================

    def print_dataframe_ranges(self, title: str, dataframe: DataFrame):
        """Print statistics of the columns in the dataframe as a formatted table"""
        print(f"    {title}")
        print(
            f"{'Column':<24} {'Min':>12} {'Max':>12} {'Mean':>12} {'Median':>12} {'Std':>12}"
        )
        print("-" * 96)
        for col in dataframe.columns:
            if not pd.api.types.is_numeric_dtype(dataframe[col]):
                continue
            col_min = dataframe[col].min()
            col_max = dataframe[col].max()
            col_mean = dataframe[col].mean()
            col_median = dataframe[col].median()
            col_std = dataframe[col].std()
            print(
                f"{col:<24} {col_min:>12.4f} {col_max:>12.4f} {col_mean:>12.4f} {col_median:>12.4f} {col_std:>12.4f}"
            )
        print("-" * 96)

    def examine_correlation(self, dataframe: DataFrame, col_name: str):
        """Examine the correlation between the features and the labels"""
        correlation_matrix = dataframe.corr(method="pearson")
        feature_target_correlation = correlation_matrix[col_name].sort_values(
            ascending=False
        )
        feature_target_correlation = feature_target_correlation.drop(
            col_name, errors="ignore"
        )
        print()
        print("\n--- Pearson Correlation with {col_name} ---")
        print(feature_target_correlation)
        print()

    def examine_full_correlation_matrix(self, dataframe: DataFrame):
        """Calculates and prints the full feature-to-feature Pearson correlation matrix."""
        correlation_matrix = dataframe.corr(method="pearson")

        print()
        print("--- Feature-to-Feature Correlation Matrix ---")
        print(correlation_matrix.round(2))
        print()

        abs_correlation_matrix = correlation_matrix.abs()
        lower_triangle_mask = np.triu(
            np.ones(abs_correlation_matrix.shape), k=1
        ).astype(bool)
        high_corr_pairs = abs_correlation_matrix.where(lower_triangle_mask)

        threshold = 0.6
        high_corr_pairs = high_corr_pairs.stack().sort_values(ascending=False)[
            high_corr_pairs.stack() > threshold
        ]

        if not high_corr_pairs.empty:
            print(
                f"\n--- Highly Correlated Feature Pairs (Absolute Value > {threshold}) ---"
            )
            print(high_corr_pairs)
        else:
            print(f"\nNo feature pairs found with absolute correlation > {threshold}.")

    def get_normalized_size(self, dataframe: DataFrame):
        """Get the size of the normalized dataframe"""

        if self.use_pca_reduction and self.pca_n_components is not None:
            return len(self.pca_passthrough_columns) + self.pca_n_components

        norm_size = len(self.include_list)
        for col in self.one_hot_columns:
            one_hot_exists = f"{col}_0" in dataframe.columns
            original_exists = col in dataframe.columns

            if one_hot_exists:
                pass
            elif original_exists:
                norm_size += 2
            else:
                print(
                    f"WARNING: {col} not found in dataframe when calculating normalized size. "
                    f"This column should be added by add_additional_indicators() before get_normalized_size() is called."
                )
                norm_size += 2

        for col in self.include_list:
            if col not in dataframe.columns:
                print(f"WARNING: column {col} not found in dataframe")
                norm_size -= 1

        return norm_size

    # =========================================================================
    # Category helpers
    # =========================================================================

    task_thresholds = {
        "momentum": {"low": -0.5, "high": 0.6},
        "flow": {"low": -0.05, "high": 0.05},
    }

    def check_columns_included(self, required_columns: list, function_name: str):
        """Check that all required columns are in the include_list."""
        missing_columns = []
        for col in required_columns:
            if col not in self.include_list:
                missing_columns.append(col)

        if missing_columns:
            error_msg = (
                f"\n{'='*80}\n"
                f"ERROR: Missing columns in include_list for {function_name}()\n"
                f"{'='*80}\n"
                f"Missing columns: {missing_columns}\n"
                f"These columns will not be available in the normalized dataframe.\n"
                f"Please add them to self.include_list in BaseNNStrategy.\n"
                f"\nCurrent include_list: {self.include_list}\n"
                f"{'='*80}\n"
            )
            raise ValueError(error_msg)

    def get_market_regime(self, dataframe: DataFrame) -> np.ndarray:
        """Classify SHORT-TERM market regimes using HT_TRENDMODE + direction (cg_ss)."""
        regime_4 = np.ones(len(dataframe), dtype=int) * MarketRegime.SIDEWAYS

        trend_mode = dataframe["trend_mode"]
        cg_ss = dataframe["cg_ss"]

        regime_4 = np.where(
            (trend_mode == 1) & (cg_ss > 0),
            MarketRegime.BULL,
            regime_4,
        )
        regime_4 = np.where(
            (trend_mode == 1) & (cg_ss < 0),
            MarketRegime.BEAR,
            regime_4,
        )

        return regime_4

    def get_risk_level(self, dataframe: DataFrame) -> np.ndarray:
        """Calculate tri-state risk classification: LOW=0, NORMAL=1, HIGH=2"""
        self.check_columns_included(["atr_norm"], "get_risk_level")

        atr = pd.Series(
            dataframe.get("atr_norm", pd.Series(np.zeros(len(dataframe))))
        ).fillna(0)
        atr = np.nan_to_num(atr, nan=0.0, posinf=1.0, neginf=-1.0)

        risk_class = np.ones(len(atr), dtype=int) * RiskLevel.NORMAL
        risk_class[atr < -0.33] = RiskLevel.LOW
        risk_class[atr > 0.33] = RiskLevel.HIGH

        return risk_class

    def get_flow(self, dataframe: DataFrame) -> np.ndarray:
        """Predict future directional bias using di_diff_scaled (Plus DI - Minus DI)."""
        self.check_columns_included(["di_diff_scaled"], "get_flow")

        di_diff = dataframe.get("di_diff_scaled", pd.Series(np.zeros(len(dataframe))))
        di_diff = pd.Series(di_diff).fillna(0)
        di_diff = np.nan_to_num(di_diff.values, nan=0.0, posinf=1.0, neginf=-1.0)

        di_diff_future = (
            pd.Series(di_diff).shift(-self.lookahead_window).fillna(0).values
        )

        flow_classes = np.ones(len(dataframe), dtype=int) * FlowDirection.NEUTRAL
        flow_classes[di_diff_future < -0.15] = FlowDirection.DECREASE
        flow_classes[di_diff_future > 0.15] = FlowDirection.INCREASE

        dataframe["flow"] = flow_classes

        return flow_classes

    def get_momentum(self, dataframe: DataFrame) -> np.ndarray:
        """Calculate momentum using normalized aroonosc (pair-agnostic)"""
        self.check_columns_included(["aroonosc_scaled"], "get_momentum")

        momentum = dataframe.get("aroonosc_scaled", pd.Series(np.zeros(len(dataframe))))
        momentum = pd.Series(momentum).fillna(0)
        momentum = np.nan_to_num(momentum, nan=0.0, posinf=1.0, neginf=-1.0)

        momentum_classes = np.ones(len(dataframe), dtype=int) * MomentumDirection.STABLE
        momentum_classes[momentum < self.task_thresholds["momentum"]["low"]] = (
            MomentumDirection.NEGATIVE
        )
        momentum_classes[momentum > self.task_thresholds["momentum"]["high"]] = (
            MomentumDirection.POSITIVE
        )

        return momentum_classes

    # =========================================================================
    # Normalisation pipeline
    # =========================================================================

    def rolling_dataframe_normalise(self, df: DataFrame):
        """Normalise the dataframe using the rolling robust scaler"""

        df_to_scale = df.copy()

        # process one-hot encoded columns (modifies lists so do first)
        df_to_scale = self.process_one_hot_columns(df_to_scale)

        # drop any columns that are not in the include list
        drop_list = []
        for col in df_to_scale.columns:
            if col not in self.include_list:
                drop_list.append(col)
        if len(drop_list) > 0:
            df_to_scale = df_to_scale.drop(columns=drop_list)

        # should be redundant, but just in case drop debug columns
        df_to_scale = self.dataframeUtils.remove_debug_columns(df_to_scale)

        # get list of columns not in pre-norm list:
        needs_norm_columns = [
            col for col in df_to_scale.columns if col not in self.pre_normalized_columns
        ]

        # Clean data BEFORE fitting scaler (replace NaNs and infs with 0)
        df_to_scale = df_to_scale.replace([np.inf, -np.inf], 0)
        df_to_scale = df_to_scale.fillna(0)

        # Use saved scaler or create new one
        if self.main_scaler is None:
            scaler_dir = self.get_storage_location()
            if scaler_exists(scaler_dir, self.main_scaler_name):
                self.main_scaler = load_scaler(scaler_dir, self.main_scaler_name)
                self.debug_print(f"    Main scaler loaded from {scaler_dir}")
                if self.pca_data_exists(scaler_dir, self.pca_name):
                    pca_payload = self.load_pca_data(scaler_dir, self.pca_name)
                    self.pca_components = pca_payload["components"]
                    self.pca_mean = pca_payload["mean"]
                    self.pca_n_components = pca_payload["n_components"]
                    self.pca_feature_columns = pca_payload.get("feature_columns")
                    self.pca_explained_variance = pca_payload.get("explained_variance")
                    self.pca_col_min = pca_payload.get("col_min")
                    self.pca_col_max = pca_payload.get("col_max")
                    self.debug_print(
                        f"    PCA data loaded: n_components={self.pca_n_components}"
                    )
            else:
                self.main_scaler = RobustScaler()

                self.debug_print(f"    Fitting columns:  {needs_norm_columns}")
                self.debug_print(f"    Skipping columns: {self.pre_normalized_columns}")

                if needs_norm_columns:
                    self.main_scaler.fit(df_to_scale[needs_norm_columns])
                    save_scaler(self.main_scaler, scaler_dir, self.main_scaler_name)
                    self.debug_print(
                        f"    Created and fitted main scaler with {len(needs_norm_columns)} features (skipping {len(self.pre_normalized_columns)} pre-normalized columns)"
                    )
                else:
                    self.debug_print(
                        "    No columns require fitting (all pre-normalized or one-hot)."
                    )

        # Transform using the fitted scaler
        df_scaled = df_to_scale.copy()
        df_scaled[needs_norm_columns] = self.main_scaler.transform(
            df_to_scale[needs_norm_columns]
        )

        # Clip extreme values to prevent NaN
        extreme_cols = []
        for col in df_scaled.columns:
            col_min = df_scaled[col].min()
            col_max = df_scaled[col].max()
            if col_min < -10 or col_max > 10:
                extreme_cols.append((col, float(col_min), float(col_max)))
        if extreme_cols:
            print(f"    Columns exceeding ±10 before clipping: {extreme_cols}")

        df_scaled = np.clip(df_scaled, -10, 10)

        if "date" in df_scaled.columns:
            raise ValueError(
                "ERROR: Date column found in scaled data. This indicates a bug in scaling."
            )

        df = df_scaled
        df = df.fillna(0)

        return df

    def normalise_for_gan(self, dataframe: np.ndarray | pd.DataFrame):
        """Performs selective minmax scaling suitable for use with a GAN"""

        if self.use_pca_reduction:
            return (
                dataframe.copy() if isinstance(dataframe, pd.DataFrame) else dataframe
            )

        if self.gan_scaler_a is None:
            scaler_dir = self.get_storage_location()
            if scaler_exists(scaler_dir, self.gan_scaler_a_name):
                self.gan_scaler_a = load_scaler(scaler_dir, self.gan_scaler_a_name)
                self.debug_print(f"    GAN scaler A loaded from {scaler_dir}")
                self.debug_print(f"      columns: {self.gan_scaler_a.feature_columns}")
                if not isinstance(dataframe, pd.DataFrame):
                    gan_df = pd.DataFrame(
                        dataframe, columns=self.gan_scaler_a.feature_columns
                    )
                else:
                    gan_df = dataframe.copy()
                non_norm_cols = [
                    col
                    for col in gan_df.columns
                    if col not in self.pre_normalized_columns
                ]

            else:
                if not isinstance(dataframe, pd.DataFrame):
                    raise ValueError("Dataframe must be a pandas DataFrame")

                gan_df = dataframe.copy()
                non_norm_cols = [
                    col
                    for col in gan_df.columns
                    if col not in self.pre_normalized_columns
                ]

                self.gan_scaler_a = MinMaxScaler(feature_range=(-1, 1))
                self.debug_print(f"    GAN Scaler A, fitting columns:  {non_norm_cols}")
                self.gan_scaler_a.fit(gan_df[non_norm_cols])
                self.gan_scaler_a.feature_columns = list(dataframe.columns)
                self.debug_print(f"      columns: {self.gan_scaler_a.feature_columns}")
                save_scaler(self.gan_scaler_a, scaler_dir, self.gan_scaler_a_name)
        else:
            if not isinstance(dataframe, pd.DataFrame):
                gan_df = pd.DataFrame(
                    dataframe, columns=self.gan_scaler_a.feature_columns
                )
            else:
                gan_df = dataframe.copy()

        non_norm_cols = [
            col for col in gan_df.columns if col not in self.pre_normalized_columns
        ]

        gan_df[non_norm_cols] = self.gan_scaler_a.transform(gan_df[non_norm_cols])

        return gan_df

    def denormalise_from_gan(self, gan_df: np.ndarray | pd.DataFrame):
        """Denormalise the dataframe from the GAN scaler"""

        if self.use_pca_reduction:
            return gan_df.copy() if isinstance(gan_df, pd.DataFrame) else gan_df

        if self.gan_scaler_a is None:
            raise ValueError("GAN scaler A is not loaded")

        if isinstance(gan_df, pd.DataFrame):
            dataframe = gan_df.copy()
        else:
            columns = self.gan_scaler_a.feature_columns
            dataframe = pd.DataFrame(gan_df, columns=columns)

        non_norm_cols = [
            col for col in dataframe.columns if col not in self.pre_normalized_columns
        ]

        dataframe[non_norm_cols] = self.gan_scaler_a.inverse_transform(
            dataframe[non_norm_cols]
        )

        return dataframe

    def scale_dataframe(self, dataframe: DataFrame):
        full_df_norm = self.rolling_dataframe_normalise(dataframe)
        if self.use_pca_reduction:
            full_df_norm = self.apply_pca(full_df_norm)
        return full_df_norm

    def descale_dataframe(self, dataframe: DataFrame):
        return dataframe

    def apply_pca(self, norm_df: pd.DataFrame) -> pd.DataFrame:
        """Apply saved whitened PCA transform to a normalised dataframe."""
        if self.pca_components is None or self.pca_feature_columns is None:
            self.debug_print(
                "    apply_pca: no PCA data loaded, returning original dataframe"
            )
            return norm_df

        passthrough = [c for c in self.pca_passthrough_columns if c in norm_df.columns]
        result = norm_df[passthrough].copy()

        missing = [c for c in self.pca_feature_columns if c not in norm_df.columns]
        if missing:
            print(f"    apply_pca WARNING: missing columns for PCA: {missing}")
            return norm_df

        X = norm_df[self.pca_feature_columns].to_numpy()
        X_centered = X - self.pca_mean
        X_pca = X_centered @ self.pca_components.T

        if self.pca_explained_variance is not None:
            X_pca = X_pca / np.sqrt(self.pca_explained_variance)

        if self.pca_col_min is not None and self.pca_col_max is not None:
            col_range = self.pca_col_max - self.pca_col_min
            col_range[col_range == 0] = 1.0
            X_pca = 2.0 * (X_pca - self.pca_col_min) / col_range - 1.0
            X_pca = np.clip(X_pca, -1.0, 1.0)

        pca_cols = [f"pca_{i}" for i in range(X_pca.shape[1])]
        pca_df = pd.DataFrame(X_pca, index=norm_df.index, columns=pca_cols)
        result = pd.concat(
            [result.reset_index(drop=True), pca_df.reset_index(drop=True)], axis=1
        )
        result.index = norm_df.index

        self.debug_print(
            f"    apply_pca: {len(passthrough)} passthrough + {len(pca_cols)} PCA columns "
            f"= {result.shape[1]} total"
        )
        return result

    # =========================================================================
    # Data transforms (tabular GAN helpers)
    # =========================================================================

    def window_and_flatten(self, dataframe: DataFrame, seq_len: int) -> pd.DataFrame:
        data = dataframe.to_numpy()
        num_rows = len(data)
        num_features = data.shape[1]

        num_sequences = num_rows - seq_len + 1
        sequences = np.zeros((num_sequences, seq_len, num_features))

        for i in range(num_sequences):
            sequences[i] = data[i : i + seq_len]

        flattened_data = sequences.reshape(
            num_rows - seq_len + 1, seq_len * num_features
        )

        col_names = []
        original_cols = dataframe.columns
        for t in range(
            seq_len - 1, -1, -1
        ):
            time_tag = f"t-{t}" if t > 0 else "t0"
            for col in original_cols:
                col_names.append(f"{col}_{time_tag}")

        return pd.DataFrame(flattened_data, columns=col_names)

    def unflatten_to_tensor(
        self, x_flat: np.ndarray, seq_len: int, num_features: int
    ) -> np.ndarray:
        """Convert results from GAN to tensor format (NOT Dataframe)"""
        num_samples = np.shape(x_flat)[0]
        x_tensor = x_flat.reshape(num_samples, seq_len, num_features)
        return x_tensor

    # =========================================================================
    # Peak detection / training signals
    # =========================================================================

    def filter_peaks_by_future_performance(
        self, future_df: DataFrame, peaks, signal_type="buy", window=64
    ):
        """Filter peaks based on future performance requirements"""
        filtered_peaks = []
        try:
            peaks = np.asarray(peaks, dtype=int)
        except Exception:
            peaks = np.array(
                [int(p) for p in np.asarray(peaks).tolist() if pd.notna(p)], dtype=int
            )
        future_window = window
        price_col = "close"

        if signal_type == "buy":
            min_gain_threshold = self.MIN_BUY_GAIN_THRESHOLD
        else:
            min_gain_threshold = self.MIN_SELL_LOSS_THRESHOLD

        for peak_idx in peaks:
            if peak_idx + future_window >= len(future_df):
                continue

            current_price = future_df[price_col].iloc[peak_idx]
            future_window_data = future_df[price_col].iloc[
                peak_idx + 1 : peak_idx + future_window + 1
            ]

            if signal_type == "buy":
                future_high = future_window_data.max()
                gain_percentage = (future_high - current_price) / current_price
                if gain_percentage >= min_gain_threshold:
                    filtered_peaks.append(peak_idx)
            else:
                future_low = future_window_data.min()
                loss_percentage = (current_price - future_low) / current_price
                if loss_percentage >= min_gain_threshold:
                    filtered_peaks.append(peak_idx)

        return filtered_peaks

    def dwt_smooth(self, data: np.array) -> np.array:
        """Apply DWT smoothing to the data"""
        wavelet = "db4"
        level = 2
        threshold = 0.4

        coeffs = pywt.wavedec(data, wavelet, level=level, mode="per")
        thresh = threshold * np.nanmax(data)
        coeffs[1:] = [pywt.threshold(c, thresh, "soft") for c in coeffs[1:]]
        poly = pywt.waverec(coeffs, wavelet, mode="per")

        if len(data) != len(poly):
            dlen = min(len(data), len(poly))
            smoothed = data.copy()
            smoothed[-dlen:] = poly[-dlen:]
        else:
            smoothed = poly
        return smoothed

    def ema_smooth(self, data: np.array) -> np.array:
        """Apply EMA smoothing to the data"""
        return data.ewm(span=5, adjust=False).mean()

    def get_train_buy_signals(self, future_df: DataFrame):
        """Generate buy signals based on peak detection"""
        signals = None

        buy_labels = TrainingSignals.get_train_buy_signals(
            future_df,
            method=self.TRAINING_TYPE,
            params={
                "horizon": self.HORIZON,
                "min_gain": self.MIN_BUY_GAIN_THRESHOLD,
                "min_loss": self.MIN_SELL_LOSS_THRESHOLD,
            },
        )
        peaks = np.where(buy_labels.values > 0.5)[0]

        filtered_peaks = self.filter_peaks_by_future_performance(
            future_df, peaks, "buy", window=self.PEAK_WINDOW
        )

        signals = pd.Series(np.zeros(np.shape(future_df)[0], dtype=float))
        signals[filtered_peaks] = 1

        if self.dbg_verbose:
            self.debug_print(
                f"        Buy signals: found {len(peaks)} initial peaks, filtered to {len(filtered_peaks)} peaks with >={self.MIN_BUY_GAIN_THRESHOLD*100:.1f}% future gain"
            )

        if signals is None:
            signals = pd.Series(np.zeros(np.shape(future_df)[0], dtype=float))

        return signals

    def get_train_sell_signals(self, future_df: DataFrame):
        """Generate sell signals based on peak detection"""
        signals = None

        sell_labels = TrainingSignals.get_train_sell_signals(
            future_df,
            method=self.TRAINING_TYPE,
            params={
                "horizon": self.HORIZON,
                "min_loss": self.MIN_SELL_LOSS_THRESHOLD,
                "min_gain": self.MIN_BUY_GAIN_THRESHOLD,
            },
        )

        peaks = np.where(sell_labels.values > 0.5)[0]

        filtered_peaks = self.filter_peaks_by_future_performance(
            future_df, peaks, "sell", window=self.PEAK_WINDOW
        )

        signals = pd.Series(np.zeros(np.shape(future_df)[0], dtype=float), dtype=float)
        if len(filtered_peaks) > 0:
            signals.iloc[filtered_peaks] = 1.0

        if self.dbg_verbose:
            self.debug_print(
                f"        Sell signals: found {len(peaks)} initial peaks, filtered to {len(filtered_peaks)} peaks with >={self.MIN_SELL_LOSS_THRESHOLD*100:.1f}% future loss"
            )

        if signals is None:
            signals = pd.Series(np.zeros(np.shape(future_df)[0], dtype=float))

        return signals

    def augment_training_signals(self, buys, sells):
        """Run data augmentation techniques"""
        # Trick 1: artificially extend positive signals two entries earlier
        bidx = np.where(buys > 0)[0]
        if len(bidx) > 0:
            start = 2 if (bidx[0] >= 2) else bidx[0]
            if len(bidx) >= 1:
                for i in range(1, 3):
                    valid_indices = bidx[bidx >= i]
                    if len(valid_indices) > 0:
                        buys[valid_indices - i] = 1.0

        sidx = np.where(sells > 0)[0]
        if len(sidx) > 0:
            start = 2 if (sidx[0] >= 2) else sidx[0]
            if len(sidx) >= 1:
                for i in range(1, 3):
                    valid_indices = sidx[sidx >= i]
                    if len(valid_indices) > 0:
                        sells[valid_indices - i] = 1.0

        # Trick 2: sells override buys
        buys[np.where(sells > 0)[0]] = 0.0

        # Trick 3: if a buy is followed by a sell, override the buy
        sell_positions = np.where(sells == 1)[0]
        N = 4
        valid_sell_positions = sell_positions[sell_positions >= N]
        if len(valid_sell_positions) > 0:
            indices = np.arange(N).reshape(1, -1) + (valid_sell_positions - N).reshape(
                -1, 1
            )
            buys[indices.flatten()] = 0

        return buys, sells

    # =========================================================================
    # Probability utils
    # =========================================================================

    def ratio_to_weights(self, ratios: [float]) -> list[float]:
        """Convert a list of class distribution ratios (percentages) to balanced class weights."""
        if not ratios or sum(ratios) <= 0:
            num_classes = len(ratios) if ratios else 3
            return [1.0 / num_classes] * num_classes

        num_classes = len(ratios)
        total_samples = 100.0

        balanced_weights = []
        for ratio in ratios:
            if ratio > 0:
                weight = total_samples / (num_classes * ratio)
            else:
                weight = 0.0
            balanced_weights.append(weight)

        total_weight = sum(balanced_weights)
        if total_weight > 0:
            normalized_weights = [w / total_weight for w in balanced_weights]
        else:
            normalized_weights = [1.0 / num_classes] * num_classes

        return normalized_weights

    def argmax_with_threshold(self, predictions, threshold=0.5, default_class=1):
        """Select class with max probability only if max prob > threshold, else return default_class"""
        max_probs = np.max(predictions, axis=1)
        argmax_classes = np.argmax(predictions, axis=1)
        return np.where(max_probs > threshold, argmax_classes, default_class)

    def argmax_with_bias(
        self, predictions, bias_map=None, threshold=0.5, default_class=1
    ):
        """Select class with max probability after applying a negative bias (penalty)."""
        if bias_map is None:
            bias_map = {}

        biased_predictions = predictions.copy()

        for class_index, penalty in bias_map.items():
            biased_predictions[:, class_index] -= penalty
            biased_predictions[:, class_index] = np.clip(
                biased_predictions[:, class_index], a_min=0, a_max=1
            )

        max_biased_probs = np.max(biased_predictions, axis=1)
        argmax_classes = np.argmax(biased_predictions, axis=1)

        return np.where(max_biased_probs > threshold, argmax_classes, default_class)

    # =========================================================================
    # GAN augmentation
    # =========================================================================

    # WGAN configuration
    wgan_epochs = 100
    wgan_batch_size = 2048
    wgan_n_critic = 5
    wgan_target_ratio = 1.0
    wgan_noise_std = 0.02

    def wgan_enhance_training_data(
        self, train_df: DataFrame, train_labels: np.ndarray
    ) -> Tuple[DataFrame, np.ndarray]:
        """WGAN-GP augmentation for 2D training data via GANInterface."""
        try:
            if train_df.empty or len(train_labels) == 0:
                print(
                    "    No training data supplied to enhance_training_data; skipping GAN augmentation"
                )
                return train_df, train_labels

            labels_int = train_labels.astype(int)
            classes, counts = np.unique(labels_int, return_counts=True)
            classes_sorted = np.sort(classes)
            if classes_sorted.size == 0:
                print("    No classes found in training labels; skipping WGAN-GP")
                return train_df, train_labels
            class_counts = dict(zip(classes.tolist(), counts.tolist()))
            self.debug_print(
                f"    Enhancing 2D training data with WGAN-GP  Class counts: {class_counts}"
            )
            original_counts_map = {int(c): int(n) for c, n in class_counts.items()}

            current_max = int(counts.max()) if counts.size > 0 else 0
            target = (
                int(current_max * self.wgan_target_ratio) if current_max > 0 else None
            )
            if target is None or target <= 0:
                self.debug_print("    No majority class found, skipping WGAN-GP")
                return train_df, train_labels

            needs_map = {
                int(c): max(0, target - original_counts_map.get(int(c), 0))
                for c in classes_sorted
            }
            self.debug_print(
                f"    WGAN target per class: {target} "
                f"(ratio={self.wgan_target_ratio})  Planned adds: {needs_map}"
            )
            if all(v <= 0 for v in needs_map.values()):
                self.debug_print("    Already at or above target; skipping WGAN-GP")
                return train_df, train_labels

            subdir = "GANs_PCA" if self.use_pca_reduction else "GANs"
            save_location = self.get_storage_location() + "/" + subdir

            train_minmax = self.normalise_for_gan(train_df)
            if isinstance(train_minmax, pd.DataFrame):
                train_minmax_values = train_minmax.to_numpy()
            else:
                train_minmax_values = train_minmax

            train_minmax_values = train_minmax_values.astype("float32")

            class_to_index = {int(cls): idx for idx, cls in enumerate(classes_sorted)}
            index_to_class = {idx: int(cls) for idx, cls in enumerate(classes_sorted)}
            label_indices = np.array(
                [class_to_index[int(cls)] for cls in labels_int], dtype=int
            )
            num_classes = len(classes_sorted)
            train_labels_one_hot = np.eye(num_classes, dtype="float32")[label_indices]

            self.debug_print("    WGAN-GP 2D augmentation starting (via GANInterface)...")

            interface = GANInterface(GANType.WGAN, save_path=save_location)
            try:
                interface.load()
                self.debug_print(f"    Loaded existing WGAN model from {save_location}.")
            except Exception as load_err:
                raise RuntimeError(
                    f"WGAN model not found at {save_location}. "
                    f"Run CreateWGAN first to train and save the model. Error: {load_err}"
                ) from load_err

            # Generate per-class augmentation samples
            aug_data_list: list = []
            aug_labels_list: list = []
            for class_idx in classes_sorted:
                need_count = needs_map.get(int(class_idx), 0)
                if need_count <= 0:
                    continue
                one_hot = np.zeros((need_count, num_classes), dtype="float32")
                one_hot[:, int(class_idx)] = 1.0
                # generate() returns (n, 1, F); squeeze seq dim for 2D augmentation
                gen_3d = interface.generate(n=need_count, one_hot=one_hot)
                aug_data_list.append(gen_3d[:, 0, :])
                aug_labels_list.append(one_hot)

            if aug_data_list:
                aug_x = np.concatenate(
                    [train_minmax_values] + aug_data_list, axis=0
                )
                aug_y = np.concatenate(
                    [train_labels_one_hot] + aug_labels_list, axis=0
                )
            else:
                aug_x = train_minmax_values
                aug_y = train_labels_one_hot

            aug_idx = aug_y.argmax(axis=1)
            aug_classes, aug_counts = np.unique(aug_idx, return_counts=True)
            new_counts_map = {
                int(c): int(n)
                for c, n in zip(aug_classes.tolist(), aug_counts.tolist())
            }
            self.debug_print("    WGAN-GP 2D augmentation complete")
            self.debug_print(
                "    WGAN-GP effect: rows "
                f"{len(train_minmax_values)} -> {len(aug_x)}; "
                f"counts {original_counts_map} -> {new_counts_map}"
            )

            aug_minmax_df = self._format_for_gan_scaler(aug_x)
            aug_normalized = self.denormalise_from_gan(aug_minmax_df)
            if isinstance(aug_normalized, pd.DataFrame):
                aug_df = aug_normalized.reset_index(drop=True)
            else:
                aug_df = pd.DataFrame(aug_normalized, columns=train_df.columns)

            aug_labels = np.array(
                [index_to_class[int(idx)] for idx in aug_idx], dtype=train_labels.dtype
            )
            return aug_df, aug_labels
        except Exception as err:
            print(
                "    WGAN-GP encountered an error in enhance_training_data; stopping strategy"
            )
            print(f"      Error: {err}")
            print(traceback.format_exc())
            raise

    def wgan_preprocess_training_data(
        self, dataframe: DataFrame, train_data, test_data, train_labels, test_labels
    ):
        """Balance training data with WGAN-GP via GANInterface (supports 2D and 3D inputs)."""
        try:
            original_shape = np.shape(train_data)
            self.debug_print("    Balancing training data with WGAN-GP")
            if len(train_data) == 0:
                print("    No training data to balance")
                return train_data, test_data, train_labels, test_labels

            train_idx = train_labels.argmax(axis=1)
            classes, counts = np.unique(train_idx, return_counts=True)
            class_counts = dict(zip(classes.tolist(), counts.tolist()))
            self.debug_print(
                f"    Train set size: {len(train_data)}  Class counts: {class_counts}"
            )
            original_counts_map = {int(c): int(n) for c, n in class_counts.items()}
            current_max = int(counts.max()) if counts.size > 0 else 0
            target = int(current_max * self.wgan_target_ratio) if current_max > 0 else 0
            if target <= 0:
                print("    No majority class found, skipping balancing")
                print(
                    f"    WGAN-GP effect: shape {original_shape} -> {np.shape(train_data)}; "
                    f"counts {original_counts_map} -> {original_counts_map}"
                )
                return train_data, test_data, train_labels, test_labels

            have_map = {
                int(c): int(n) for c, n in zip(classes.tolist(), counts.tolist())
            }
            num_classes = train_labels.shape[1]
            needs_map = {
                c: max(0, target - have_map.get(c, 0))
                for c in range(num_classes)
            }
            print(
                f"    WGAN target per class: {target} (ratio={self.wgan_target_ratio})  "
                f"Planned adds: {needs_map}"
            )
            if all(v <= 0 for v in needs_map.values()):
                print("    Already at or above target; skipping WGAN-GP")
                print(
                    f"    WGAN-GP effect: shape {original_shape} -> {np.shape(train_data)}; "
                    f"counts {original_counts_map} -> {original_counts_map}"
                )
                return train_data, test_data, train_labels, test_labels

            save_location = self.get_storage_location() + "/GANs"

            train_data_shape = train_data.shape
            num_features = train_data.shape[-1]
            train_data_2d = train_data.reshape(-1, num_features)
            gan_input = self._format_for_gan_scaler(train_data_2d)
            train_data_minmax_2d = self.normalise_for_gan(gan_input)
            if isinstance(train_data_minmax_2d, pd.DataFrame):
                train_data_minmax_2d = train_data_minmax_2d.to_numpy()
            train_data_minmax = train_data_minmax_2d.reshape(train_data_shape)

            interface = GANInterface(GANType.WGAN, save_path=save_location)
            try:
                interface.load()
                print(f"    Loaded existing WGAN model from {save_location}.")
            except Exception as load_err:
                raise RuntimeError(
                    f"WGAN model not found at {save_location}. "
                    f"Run CreateWGAN first to train and save the model. Error: {load_err}"
                ) from load_err

            # Generate per-class augmentation samples; generate() returns (n, 1, F)
            aug_data_list: list = []
            aug_labels_list: list = []
            for class_c in range(num_classes):
                need_count = needs_map.get(class_c, 0)
                if need_count <= 0:
                    continue
                one_hot = np.zeros((need_count, num_classes), dtype="float32")
                one_hot[:, class_c] = 1.0
                gen_3d = interface.generate(n=need_count, one_hot=one_hot)
                # Reshape to match original train_data_shape dimensionality
                if train_data.ndim == 3:
                    aug_data_list.append(gen_3d)           # already (n, 1, F)
                else:
                    aug_data_list.append(gen_3d[:, 0, :])  # squeeze to (n, F)
                aug_labels_list.append(one_hot)

            if aug_data_list:
                aug_x = np.concatenate(
                    [train_data_minmax] + aug_data_list, axis=0
                )
                aug_y = np.concatenate(
                    [train_labels] + aug_labels_list, axis=0
                )
            else:
                aug_x = train_data_minmax
                aug_y = train_labels

            aug_idx = aug_y.argmax(axis=1)
            aug_classes, aug_counts = np.unique(aug_idx, return_counts=True)
            aug_class_counts = dict(zip(aug_classes.tolist(), aug_counts.tolist()))
            new_counts_map = {int(c): int(n) for c, n in aug_class_counts.items()}
            print("    WGAN-GP training complete")
            print(
                f"    Augmented train size: {len(aug_x)}  Class counts: {aug_class_counts}"
            )
            print(
                f"    WGAN-GP effect: shape {original_shape} -> {np.shape(aug_x)}; "
                f"counts {original_counts_map} -> {new_counts_map}"
            )

            aug_x_shape = aug_x.shape
            aug_x_2d = aug_x.reshape(-1, num_features)
            aug_input = self._format_for_gan_scaler(aug_x_2d)
            aug_x_2d = self.denormalise_from_gan(aug_input)

            if isinstance(aug_x_2d, pd.DataFrame):
                aug_x_2d = aug_x_2d.to_numpy()

            aug_x_normalized = aug_x_2d.reshape(aug_x_shape)

            return aug_x_normalized, test_data, aug_y, test_labels
        except Exception as e:
            print("    WGAN-GP encountered an error; stopping strategy")
            print(f"      Error: {e}")
            print(traceback.format_exc())
            raise

    def _format_for_gan_scaler(self, array_2d):
        if isinstance(array_2d, pd.DataFrame):
            return array_2d
        columns = []
        if hasattr(self, "_get_gan_feature_columns"):
            columns = self._get_gan_feature_columns()
        if columns and array_2d.shape[1] == len(columns):
            return pd.DataFrame(array_2d, columns=columns)
        return array_2d

    # CTAB-GAN+ configuration
    cgp_augmentation_target_ratio = 0.5

    def ctab_gan_enhance_training_data(
        self, train_df: DataFrame, train_labels: np.ndarray
    ) -> Tuple[DataFrame, np.ndarray]:
        """Uses CTAB-GAN+ to generate more training data via GANInterface."""
        try:
            if train_df.empty or len(train_labels) == 0:
                self.debug_print(
                    "    No training data supplied to enhance_training_data; "
                    "skipping CTAB-GAN+ augmentation"
                )
                return train_df, train_labels

            if len(train_labels.shape) > 1 and train_labels.shape[1] > 1:
                train_labels_one_hot = train_labels.astype(np.float32)
                labels_int = np.argmax(train_labels_one_hot, axis=1)
                num_classes = train_labels_one_hot.shape[1]
            else:
                labels_int = train_labels.astype(int).flatten()
                num_classes = int(labels_int.max()) + 1
                train_labels_one_hot = np.eye(num_classes, dtype=np.float32)[labels_int]

            classes, counts = np.unique(labels_int, return_counts=True)
            classes_sorted = np.sort(classes)
            if classes_sorted.size == 0:
                print("    No classes found in training labels; skipping CTAB-GAN+")
                return train_df, train_labels
            class_counts = dict(zip(classes.tolist(), counts.tolist()))
            self.debug_print(
                f"    Enhancing training data with CTAB-GAN+  Class counts: {class_counts}"
            )
            original_counts_map = {int(c): int(n) for c, n in class_counts.items()}

            current_max = int(counts.max()) if counts.size > 0 else 0
            target = (
                int(current_max * self.cgp_augmentation_target_ratio)
                if current_max > 0
                else None
            )
            if target is None or target <= 0:
                print("    No majority class found, skipping CTAB-GAN+")
                return train_df, train_labels

            needs_map = {
                int(c): max(0, target - original_counts_map.get(int(c), 0))
                for c in classes_sorted
            }
            self.debug_print(
                f"    CTAB-GAN+ target per class: {target} "
                f"(ratio={self.cgp_augmentation_target_ratio})  Planned adds: {needs_map}"
            )
            if all(v <= 0 for v in needs_map.values()):
                self.debug_print("    Already at or above target; skipping CTAB-GAN+")
                return train_df, train_labels

            subdir = "CTABGANs_PCA" if self.use_pca_reduction else "CTABGANs"
            save_location = os.path.join(self.get_storage_location(), subdir)

            self.debug_print(f"    Loading CTAB-GAN+ model from {save_location}")
            interface = GANInterface(GANType.CTAB_GAN, save_path=save_location)
            try:
                thresholds = interface.load()
            except Exception as load_err:
                raise RuntimeError(
                    f"CTAB-GAN+ model not found at {save_location}. "
                    f"Run CreateCtabGanPlus first to train and save the model. Error: {load_err}"
                ) from load_err

            if thresholds.get("min_buy_gain_threshold") is not None:
                self.MIN_BUY_GAIN_THRESHOLD = thresholds["min_buy_gain_threshold"]
                self._thresholds_from_gan = True
                self.debug_print(
                    f"    Loaded threshold from GAN: MIN_BUY_GAIN_THRESHOLD={self.MIN_BUY_GAIN_THRESHOLD:.4f}"
                )
            if thresholds.get("min_sell_loss_threshold") is not None:
                self.MIN_SELL_LOSS_THRESHOLD = thresholds["min_sell_loss_threshold"]
                self._thresholds_from_gan = True
                self.debug_print(
                    f"    Loaded threshold from GAN: MIN_SELL_LOSS_THRESHOLD={self.MIN_SELL_LOSS_THRESHOLD:.4f}"
                )

            train_minmax = self.normalise_for_gan(train_df)

            ctab_gan = interface._model
            if hasattr(ctab_gan, "column_order") and ctab_gan.column_order:
                gan_columns = set(ctab_gan.column_order)
                train_columns = set(train_minmax.columns)
                if gan_columns != train_columns:
                    expected_size = self.get_normalized_size(train_df)
                    error_msg = (
                        f"GAN model feature mismatch detected before generation:\n"
                        f"  GAN model columns ({len(gan_columns)}): {sorted(gan_columns)}\n"
                        f"  Training data columns ({len(train_columns)}): {sorted(train_columns)}\n"
                        f"  Expected normalized size: {expected_size}\n"
                        f"  Missing in GAN: {sorted(train_columns - gan_columns)}\n"
                        f"  Extra in GAN: {sorted(gan_columns - train_columns)}\n"
                        f"  The GAN model must be retrained with the current feature set."
                    )
                    raise ValueError(error_msg)
            if isinstance(train_minmax, pd.DataFrame):
                train_minmax_values = train_minmax.to_numpy()
            else:
                train_minmax_values = train_minmax

            train_minmax_values = train_minmax_values.astype("float32")

            aug_data_list = []
            aug_labels_list = []

            for class_idx, need_count in needs_map.items():
                if need_count <= 0:
                    continue

                self.debug_print(
                    f"    Generating {need_count} samples for class {class_idx}"
                )
                generated_df = interface.generate(
                    n=need_count,
                    class_label=int(class_idx),
                )

                missing_cols = set(train_minmax.columns) - set(generated_df.columns)
                extra_cols = set(generated_df.columns) - set(train_minmax.columns)

                if missing_cols or extra_cols:
                    expected_size = self.get_normalized_size(train_df)
                    error_msg = (
                        f"Column mismatch between generated data and training data:\n"
                        f"  Training data columns: {len(train_minmax.columns)}\n"
                        f"  Generated data columns: {len(generated_df.columns)}\n"
                        f"  Expected normalized size: {expected_size}\n"
                    )
                    if missing_cols:
                        error_msg += (
                            f"  Missing columns in generated data: {missing_cols}\n"
                        )
                    if extra_cols:
                        error_msg += (
                            f"  Extra columns in generated data: {extra_cols}\n"
                        )
                    error_msg += (
                        f"  The GAN model must be retrained with the current feature set.\n"
                        f"  Training data columns: {list(train_minmax.columns)}\n"
                        f"  Generated data columns: {list(generated_df.columns)}"
                    )
                    raise ValueError(error_msg)

                generated_array = generated_df[train_minmax.columns].values.astype(
                    np.float32
                )

                expected_size = self.get_normalized_size(train_df)
                if generated_array.shape[1] != train_minmax_values.shape[1]:
                    raise ValueError(
                        f"Column count mismatch: "
                        f"generated has {generated_array.shape[1]} columns, "
                        f"training data has {train_minmax_values.shape[1]} columns, "
                        f"expected normalized size: {expected_size}"
                    )
                aug_data_list.append(generated_array)

                aug_labels_one_hot = np.zeros(
                    (need_count, num_classes), dtype=train_labels_one_hot.dtype
                )
                aug_labels_one_hot[:, int(class_idx)] = 1.0
                aug_labels_list.append(aug_labels_one_hot)

            if aug_data_list:
                aug_x = np.concatenate(aug_data_list, axis=0)
                aug_y = np.concatenate(aug_labels_list, axis=0)

                combined_x = np.concatenate([train_minmax_values, aug_x], axis=0)
                combined_y = np.concatenate([train_labels_one_hot, aug_y], axis=0)

                aug_minmax_df = self._format_for_gan_scaler(combined_x)
                aug_normalized = self.denormalise_from_gan(aug_minmax_df)
                if isinstance(aug_normalized, pd.DataFrame):
                    aug_df = aug_normalized.reset_index(drop=True)
                else:
                    aug_df = pd.DataFrame(aug_normalized, columns=train_df.columns)

                if list(aug_df.columns) != list(train_df.columns):
                    expected_size = self.get_normalized_size(train_df)
                    error_msg = (
                        f"Column mismatch after denormalization:\n"
                        f"  Original train_df columns ({len(train_df.columns)}): {list(train_df.columns)}\n"
                        f"  Augmented aug_df columns ({len(aug_df.columns)}): {list(aug_df.columns)}\n"
                        f"  Expected normalized size: {expected_size}\n"
                        f"  The GAN model must be retrained with the current feature set."
                    )
                    raise ValueError(error_msg)

                aug_df = aug_df[train_df.columns]

                aug_idx = np.argmax(combined_y, axis=1)
                aug_classes, aug_counts = np.unique(aug_idx, return_counts=True)
                new_counts_map = {
                    int(c): int(n)
                    for c, n in zip(aug_classes.tolist(), aug_counts.tolist())
                }
                self.debug_print("    CTAB-GAN+ augmentation complete")
                self.debug_print(
                    "    CTAB-GAN+ effect: rows "
                    f"{len(train_minmax_values)} -> {len(combined_x)}; "
                    f"counts {original_counts_map} -> {new_counts_map}"
                )

                combined_labels_class_indices = np.argmax(combined_y, axis=1)

                return aug_df, combined_labels_class_indices
            else:
                print("    CTAB-GAN+ augmentation: no samples generated")
                return train_df, train_labels

        except Exception as err:
            print(
                "    CTAB-GAN+ encountered an error in enhance_training_data; "
                "stopping strategy"
            )
            print(f"      Error: {err}")
            print(traceback.format_exc())
            raise

    # =========================================================================
    # Virtual methods (NN-specific)
    # =========================================================================

    def get_classifier_type(self):
        """Return the type of classifier used for training/predicting"""
        raise NotImplementedError("get_classifier_type() not implemented")
        return None

    def get_classifier(
        self, classifier_type, pair, seq_len, num_features
    ) -> ClassifierKeras:
        """Return the classifier used for training/predicting"""
        raise NotImplementedError("get_classifier() not implemented")
        return None

    def add_additional_indicators(self, dataframe: DataFrame):
        """Add any additional indicators to the dataframe"""
        dataframe["regime"] = self.get_market_regime(dataframe)
        dataframe["risk"] = self.get_risk_level(dataframe)
        dataframe["flow"] = self.get_flow(dataframe)
        dataframe["momentum"] = self.get_momentum(dataframe)
        return dataframe

    def add_debug_indicators(self, dataframe: DataFrame):
        """Add any debug indicators to the dataframe. Must start with '%'"""
        self.dbg_curr_df = dataframe
        return dataframe

    # =========================================================================
    # Training labels
    # =========================================================================

    def get_training_labels(self, dataframe: DataFrame):
        """Convert buy/sell signals into tri-state classification"""

        self.dbg_curr_df = dataframe

        dataframe["%train_buy"] = 0.0
        dataframe["%train_sell"] = 0.0

        buys = self.get_train_buy_signals(dataframe)
        sells = self.get_train_sell_signals(dataframe)

        dataframe["%train_buy"] = np.where(buys > 0, 1.0, 0.0)
        dataframe["%train_sell"] = np.where(sells > 0, 1.0, 0.0)

        if self.augment_training_data:
            buys, sells = self.augment_training_signals(buys, sells)

        self.dbg_curr_df = dataframe

        trading_classes = (
            np.ones(np.shape(dataframe)[0], dtype=int) * TradingAction.HOLD
        )

        df_len = np.shape(dataframe)[0]

        if isinstance(buys, pd.Series):
            buys_array = (
                buys.values[:df_len]
                if len(buys) >= df_len
                else np.pad(buys.values, (0, df_len - len(buys)), "constant")
            )
        else:
            buys_array = (
                np.asarray(buys)[:df_len]
                if len(buys) >= df_len
                else np.pad(np.asarray(buys), (0, df_len - len(buys)), "constant")
            )

        if isinstance(sells, pd.Series):
            sells_array = (
                sells.values[:df_len]
                if len(sells) >= df_len
                else np.pad(sells.values, (0, df_len - len(sells)), "constant")
            )
        else:
            sells_array = (
                np.asarray(sells)[:df_len]
                if len(sells) >= df_len
                else np.pad(np.asarray(sells), (0, df_len - len(sells)), "constant")
            )

        buys_bool = (buys_array > 0.5).astype(bool)
        sells_bool = (sells_array > 0.5).astype(bool)

        num_buys_before = np.sum(buys_bool)
        num_sells_before = np.sum(sells_bool)
        num_conflicts = np.sum(buys_bool & sells_bool)

        if self.dbg_verbose:
            if num_conflicts > 0:
                self.debug_print(
                    f"        Signal conflicts: {num_conflicts} positions have both buy and sell signals"
                )
            self.debug_print(
                f"        Signal counts: buys={num_buys_before}, sells={num_sells_before}, conflicts={num_conflicts}"
            )

        trading_classes[buys_bool] = TradingAction.BUY
        trading_classes[sells_bool] = TradingAction.SELL

        num_buys_after = np.sum(trading_classes == TradingAction.BUY)
        num_sells_after = np.sum(trading_classes == TradingAction.SELL)

        if self.dbg_verbose:
            self.debug_print(
                f"        Final trading_classes counts: buys={num_buys_after}, sells={num_sells_after}"
            )
            if not self.augment_training_data and (
                num_buys_before != num_buys_after or num_sells_before != num_sells_after
            ):
                self.debug_print(
                    f"    WARNING: Count mismatch! Expected buys={num_buys_before}, got {num_buys_after}; Expected sells={num_sells_before}, got {num_sells_after}"
                )

        self.print_distribution_compact("  Trading", trading_classes)

        dataframe["%train_buy"] = np.where(
            trading_classes == TradingAction.BUY, 1.0, 0.0
        )
        dataframe["%train_sell"] = np.where(
            trading_classes == TradingAction.SELL, 1.0, 0.0
        )

        if self.dbg_verbose:
            num_train_buy = np.sum(dataframe["%train_buy"] > 0.5)
            num_train_sell = np.sum(dataframe["%train_sell"] > 0.5)
            num_train_hold = len(dataframe) - num_train_buy - num_train_sell

            both_set = np.sum(
                (dataframe["%train_buy"] > 0.5) & (dataframe["%train_sell"] > 0.5)
            )
            if both_set > 0:
                self.debug_print(
                    f"    WARNING: {both_set} positions have both %train_buy and %train_sell set!"
                )

            self.debug_print(
                f"        %train_buy/%train_sell counts: buys={num_train_buy}, sells={num_train_sell}, holds={num_train_hold}"
            )

            if num_train_buy != num_buys_after or num_train_sell != num_sells_after:
                self.debug_print(
                    f"    WARNING: Mismatch between trading_classes and %train columns!"
                )
                self.debug_print(
                    f"      trading_classes: buys={num_buys_after}, sells={num_sells_after}"
                )
                self.debug_print(
                    f"      %train columns: buys={num_train_buy}, sells={num_train_sell}"
                )

        return trading_classes

    # =========================================================================
    # Class weights
    # =========================================================================

    def get_training_class_weights(self, train_labels=None, validation_labels=None):
        """Get the class weights for the training data"""
        if self.augment_training_data and train_labels is not None:
            labels_to_use = train_labels
        else:
            labels_to_use = (
                validation_labels if validation_labels is not None else train_labels
            )

        if labels_to_use is None:
            return [1.0, 1.0, 1.0]

        # Handle multi-task case (dictionary)
        if isinstance(labels_to_use, dict):
            class_weights_dict = {}
            for task_name, task_labels in labels_to_use.items():
                if task_labels.ndim == 2:
                    class_indices = np.argmax(task_labels, axis=1)
                else:
                    class_indices = task_labels.astype(int)

                unique_classes, class_counts = np.unique(
                    class_indices, return_counts=True
                )
                total_samples = len(class_indices)
                num_classes = len(np.unique(class_indices))

                balanced_weights = np.zeros(num_classes)
                for i, count in zip(unique_classes, class_counts):
                    if count > 0:
                        balanced_weights[i] = total_samples / (num_classes * count)
                    else:
                        balanced_weights[i] = 0.0

                class_weights_dict[task_name] = balanced_weights.tolist()

            return class_weights_dict

        # Handle single-task case (numpy array)
        else:
            labels_array = np.asarray(labels_to_use)

            if labels_array.ndim == 2:
                class_indices = np.argmax(labels_array, axis=1)
            else:
                class_indices = labels_array.astype(int)

            class_counts = np.bincount(class_indices, minlength=3)
            total_samples = len(class_indices)
            num_classes = 3

            class_weights = np.zeros(num_classes)
            for i in range(num_classes):
                if class_counts[i] > 0:
                    class_weights[i] = total_samples / (num_classes * class_counts[i])
                else:
                    class_weights[i] = 0.0

            return class_weights.tolist()

    # =========================================================================
    # Training pipeline
    # =========================================================================

    def prepare_training_data(
        self, dataframe: List[DataFrame], labels: List[Any], norm: bool = True
    ):
        """Prepare the training data"""

        if len(dataframe) < 1:
            raise ValueError("No dataframes passed in")

        num_pairs = len(dataframe)
        num_rows = np.shape(dataframe[0])[0]
        max_index = num_pairs * num_rows

        if max_index <= 1:
            raise ValueError(
                f"Insufficient data for training: max_index={max_index} "
                f"(num_pairs={num_pairs}, num_rows={num_rows}). "
                f"Need at least 2 total rows across all pairs."
            )

        aggr_tsr_train = None
        aggr_tsr_test = None
        aggr_train_labels = None
        aggr_test_labels = None

        for i in range(num_pairs):

            pair_labels = np.asarray(labels[i])
            if norm:
                df_norm = self.scale_dataframe(dataframe[i])
            else:
                df_norm = dataframe[i]

            split_idx = int(self.TRAIN_DATA_SPLIT * len(df_norm))

            buffer_size = self.seq_len - 1

            train_end = split_idx - buffer_size
            train_df = df_norm[:train_end]

            test_start = train_end
            test_df = df_norm[test_start:]

            train_labels = pair_labels[:train_end]
            test_labels = pair_labels[test_start:]

            train_df, train_labels = self.enhance_training_data(train_df, train_labels)

            train_labels = self.dataframeUtils.one_hot_encode(train_labels, 3)
            test_labels = self.dataframeUtils.one_hot_encode(test_labels, 3)

            tsr_train = self.dataframeUtils.df_to_tensor(
                train_df, self.seq_len, method=self.tensor_method
            )
            tsr_test = self.dataframeUtils.df_to_tensor(
                test_df, self.seq_len, method=self.tensor_method
            )

            if self.tensor_method == 0:
                offset = self.seq_len - 1
            else:
                offset = 0

            train_labels = train_labels[offset:]
            test_labels = test_labels[offset:]

            if aggr_tsr_train is None:
                aggr_tsr_train = tsr_train
                aggr_tsr_test = tsr_test
                aggr_train_labels = train_labels
                aggr_test_labels = test_labels
            else:
                aggr_tsr_train = np.concatenate([aggr_tsr_train, tsr_train], axis=0)
                aggr_tsr_test = np.concatenate([aggr_tsr_test, tsr_test], axis=0)
                aggr_train_labels = np.concatenate(
                    [aggr_train_labels, train_labels], axis=0
                )
                aggr_test_labels = np.concatenate(
                    [aggr_test_labels, test_labels], axis=0
                )

        return aggr_tsr_train, aggr_tsr_test, aggr_train_labels, aggr_test_labels

    def enhance_training_data(
        self, train_df: DataFrame, train_labels: np.ndarray
    ) -> Tuple[DataFrame, np.ndarray]:
        """Optional hook to modify training data before training. Override in subclass for GAN augmentation."""
        return train_df, train_labels

    def preprocess_training_data(
        self, dataframe: DataFrame, train_data, test_data, train_labels, test_labels
    ):
        """Optional hook to modify train/test tensors and labels before training."""
        return train_data, test_data, train_labels, test_labels

    def train_model(
        self, dataframes: [DataFrame], labels: [Any], classifier: ClassifierKeras
    ):
        """Train the model - default implementation"""

        tsr_train, tsr_test, train_labels, test_labels = self.prepare_training_data(
            dataframes, labels
        )

        if tsr_train is not None and len(tsr_train.shape) >= 2:
            actual_features = tsr_train.shape[-1]
            if hasattr(classifier, "model") and classifier.model is not None:
                model_input_shape = classifier.model.input_shape
                if (
                    isinstance(model_input_shape, (list, tuple))
                    and len(model_input_shape) > 0
                ):
                    input_shape = (
                        model_input_shape[0]
                        if isinstance(model_input_shape, list)
                        else model_input_shape
                    )
                else:
                    input_shape = model_input_shape

                if input_shape and len(input_shape) >= 2:
                    expected_features = input_shape[-1]
                    if actual_features != expected_features:
                        expected_size = (
                            self.get_normalized_size(dataframes[0])
                            if dataframes
                            else "unknown"
                        )
                        raise ValueError(
                            f"Feature count mismatch: Model expects {expected_features} features, "
                            f"but training data has {actual_features} features after augmentation.\n"
                            f"  Expected normalized size: {expected_size}\n"
                            f"  This usually means the GAN model was trained with a different feature set.\n"
                            f"  Please retrain the GAN model with the current feature set."
                        )

        tsr_train, tsr_test, train_labels, test_labels = self.preprocess_training_data(
            dataframes[0], tsr_train, tsr_test, train_labels, test_labels
        )

        class_weights = self.get_training_class_weights(
            train_labels=train_labels, validation_labels=test_labels
        )

        if self.use_markov_smoothing:
            label_seq = self._labels_to_class_indices(test_labels)
            self.markov_transition_matrix = self._compute_markov_transition_matrix(
                label_seq, num_classes=3
            )

        if self.shuffle_train_data:
            if isinstance(train_labels, dict):
                rng = np.random.RandomState(42)
                indices = rng.permutation(len(tsr_train))
                tsr_train = tsr_train[indices]
                train_labels = {
                    key: value[indices] for key, value in train_labels.items()
                }
            else:
                tsr_train, train_labels = shuffle(
                    tsr_train, train_labels, random_state=42
                )

        classifier.train(
            tsr_train, tsr_test, train_labels, test_labels, class_weights=class_weights
        )

        if self.use_markov_smoothing and self.markov_transition_matrix is not None:
            markov_path = self.get_markov_matrix_path()
            markov_dir = os.path.dirname(markov_path)
            if not os.path.exists(markov_dir):
                os.makedirs(markov_dir)
            np.save(markov_path, self.markov_transition_matrix)

        return None

    # =========================================================================
    # Prediction pipeline
    # =========================================================================

    def get_predictions(self, dataframe: DataFrame, classifier: ClassifierKeras):
        """Get the predictions from the model"""

        pred_threshold = self.prediction_threshold.value

        missing_columns = []
        required_columns = ["regime", "flow"]
        for col in required_columns:
            if col not in dataframe.columns:
                missing_columns.append(col)

        if missing_columns:
            raise ValueError(
                f"Missing required columns for prediction: {missing_columns}. "
                f"These columns should be added by populate_indicators() before prediction. "
                f"Please ensure add_additional_indicators() and add_sequential_index() are called."
            )

        df_norm = self.scale_dataframe(dataframe)
        df_tensor = self.dataframeUtils.df_to_tensor(
            df_norm, self.seq_len, method=self.tensor_method
        )

        if hasattr(classifier, "model") and classifier.model is not None:
            model_input_shape = classifier.model.input_shape
            if isinstance(model_input_shape, list):
                input_shape = (
                    model_input_shape[0] if len(model_input_shape) > 0 else None
                )
            else:
                input_shape = model_input_shape

            if input_shape and len(input_shape) >= 2:
                expected_features = input_shape[-1]
                actual_features = df_tensor.shape[-1]
                if actual_features != expected_features:
                    expected_size = self.get_normalized_size(dataframe)
                    raise ValueError(
                        f"Feature count mismatch during prediction: Model expects {expected_features} features, "
                        f"but prediction data has {actual_features} features.\n"
                        f"  Expected normalized size: {expected_size}\n"
                        f"  This usually means the model was trained with a different feature set than the current dataframe.\n"
                        f"  Please retrain the model with the current feature set, or ensure the dataframe has all required columns."
                    )

        pred_probs = classifier.predict(df_tensor)
        if self.use_markov_smoothing and self.markov_transition_matrix is None:
            markov_path = self.get_markov_matrix_path()
            if os.path.exists(markov_path):
                try:
                    self.markov_transition_matrix = np.load(markov_path)
                except Exception:
                    self.markov_transition_matrix = None

        if (
            self.use_markov_smoothing
            and self.markov_transition_matrix is not None
            and pred_probs is not None
            and pred_probs.ndim == 2
            and self.markov_transition_matrix.shape[0] == pred_probs.shape[1]
        ):
            smoothed = np.matmul(pred_probs, self.markov_transition_matrix)
            alpha = float(getattr(self, "markov_smoothing_alpha", 0.5))
            pred_probs = (alpha * pred_probs) + ((1.0 - alpha) * smoothed)
            if self.dbg_verbose:
                self.debug_print(
                    f"    Markov smoothing enabled: alpha={alpha:.2f}, "
                    f"matrix_shape={self.markov_transition_matrix.shape}"
                )
        predictions = self.argmax_with_threshold(
            pred_probs,
            threshold=pred_threshold,
            default_class=TradingAction.HOLD,
        )
        original_length = len(dataframe)
        predicted_length = len(predictions)

        if predicted_length < original_length:
            offset = original_length - predicted_length
            preds = np.zeros(original_length, dtype=int)
            preds[offset:] = predictions
            predictions = preds

        self.print_probability_stats(
            "Trading", "Sell", pred_probs[:, TradingAction.SELL], pred_threshold
        )
        self.print_probability_stats(
            "Trading", "Hold", pred_probs[:, TradingAction.HOLD], pred_threshold
        )
        self.print_probability_stats(
            "Trading", "Buy", pred_probs[:, TradingAction.BUY], pred_threshold
        )
        return predictions

    def _labels_to_class_indices(self, labels) -> np.ndarray:
        """Normalize label formats to 1D class index array."""
        if isinstance(labels, dict):
            if "trading" in labels:
                labels = labels["trading"]
            else:
                labels = next(iter(labels.values()))

        arr = np.asarray(labels)
        if arr.ndim > 1:
            return np.argmax(arr, axis=1).astype(int)
        return arr.astype(int)

    @staticmethod
    def _compute_markov_transition_matrix(
        label_seq: np.ndarray, num_classes: int
    ) -> np.ndarray:
        """Compute transition probabilities P(next_state | current_state)."""
        if label_seq is None or len(label_seq) < 2:
            return np.eye(num_classes, dtype=float)

        counts = np.zeros((num_classes, num_classes), dtype=float)
        prev = label_seq[:-1]
        nxt = label_seq[1:]

        for a, b in zip(prev, nxt):
            if 0 <= a < num_classes and 0 <= b < num_classes:
                counts[int(a), int(b)] += 1.0

        row_sums = counts.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1.0
        return counts / row_sums

    def process_predictions(self, dataframe: DataFrame, predictions):
        """Process the predictions."""
        dataframe["predict_buy"] = np.where(predictions == TradingAction.BUY, 1, 0)
        dataframe["predict_sell"] = np.where(predictions == TradingAction.SELL, 1, 0)
        return dataframe

    # =========================================================================
    # Entry/Exit conditions (NN-specific)
    # =========================================================================

    def get_entry_conditions(self, dataframe: DataFrame):
        """Add strategy-specific entry conditions."""
        predictions = self.get_predictions(dataframe, self.classifier)
        dataframe = self.process_predictions(dataframe, predictions)

        if "predict_buy" in dataframe.columns:
            conditions = dataframe["predict_buy"] > 0.5
        else:
            conditions = None

        return conditions

    def get_exit_conditions(self, dataframe: DataFrame):
        """Add strategy-specific exit conditions."""
        if "predict_sell" in dataframe.columns:
            conditions = dataframe["predict_sell"] > 0.5
        else:
            conditions = None
        return conditions

    # =========================================================================
    # Iteration init (NN-specific extension)
    # =========================================================================

    # =========================================================================
    # bot_start — NN-specific one-time setup
    # =========================================================================

    def bot_start(self, **kwargs) -> None:
        """
        Called once after the strategy is instantiated.  Handles the
        NN-specific setup that used to live behind a ``first_time`` gate
        inside ``iteration_init`` — TF/MLX device configuration, threshold
        loading from saved GAN metadata, and ``buy_params`` / ``sell_params``
        overrides.

        Subclasses that override this MUST call
        ``super().bot_start(**kwargs)`` so the NN setup runs.
        """
        # Run BaseStrategy.bot_start first (banner, environment, helpers).
        super().bot_start(**kwargs)

        if self.dp is not None and self.dp.runmode.value in ("util_no_exchange"):
            os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
            tf.config.set_visible_devices([], "GPU")
            print("    Forced CPU mode for lookahead analysis")

        # Print NN-specific info
        self.print_strategy_info()

        # GAN creators set this flag on themselves before bot_start runs
        # (via CreateGANBase.bot_start setting it on super()).  When the
        # current strategy *consumes* a GAN we'd like to load the thresholds
        # that the GAN was trained with — when it *creates* a GAN we leave
        # the MASTER values alone.
        is_gan_creation = getattr(self, "_is_gan_creation_strategy", False)

        if is_gan_creation:
            return

        if hasattr(self, "_load_gan_thresholds_early"):
            self._load_gan_thresholds_early()

        thresholds_from_gan = getattr(self, "_thresholds_from_gan", False)

        if (
            hasattr(self, "buy_params")
            and self.buy_params
            and "min_buy_gain_threshold" in self.buy_params
            and not thresholds_from_gan
        ):
            self.MIN_BUY_GAIN_THRESHOLD = self.buy_params["min_buy_gain_threshold"]

        if (
            hasattr(self, "sell_params")
            and self.sell_params
            and "min_sell_loss_threshold" in self.sell_params
            and not thresholds_from_gan
        ):
            self.MIN_SELL_LOSS_THRESHOLD = self.sell_params["min_sell_loss_threshold"]

        if not thresholds_from_gan:
            if not hasattr(self, "TRAINING_TYPE") or self.TRAINING_TYPE == 13:
                self.TRAINING_TYPE = self.training_type.value

    def iteration_init(self):
        """Called at the start of each populate_indicators() cycle.

        The bulk of the NN-specific setup now lives in :meth:`bot_start`.
        This hook only refreshes per-iteration state — namely whether the
        model still needs training.
        """
        super().iteration_init()
        self.training_needed = not self.model_exists()

    # =========================================================================
    # populate_indicators — NN version (override)
    # =========================================================================

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """NN-specific indicator population with training loop and aggregation."""

        # check that main scaler is present
        scaler_dir = self.get_storage_location()
        if not scaler_exists(scaler_dir, self.main_scaler_name):
            print(f"    Main scaler {self.main_scaler_name} not found in {scaler_dir}")
            print("    You must create the main scaler before running the strategy")
            raise ValueError(
                f"Main scaler {self.main_scaler_name} not found in {scaler_dir}"
            )

        curr_pair = metadata["pair"]
        self.curr_pair = curr_pair

        # some flags only make sense in backtest mode
        if self.dp.runmode.value not in ("backtest"):
            self.combine_models = False
            self.aggregate_pairs = False
            self.training_needed = False

        self.dbg_curr_df = dataframe

        self.iteration_init()

        if self.dbg_verbose:
            self.debug_print(f"    {curr_pair} - adding indicators...")

        dataframe = self.check_precision_columns(dataframe)
        dataframe = self.dataframePopulator.add_indicators(
            dataframe, dataset_type=DatasetType.MINIMAL
        )
        dataframe = self.add_additional_indicators(dataframe)
        dataframe = self.add_debug_indicators(dataframe)
        self.dbg_curr_df = dataframe

        labels = self.get_training_labels(dataframe)

        # set up the classifier if it doesn't already exist
        if self.classifier is None:
            num_features = self.get_normalized_size(dataframe)
            self.classifier_type = self.get_classifier_type()
            self.classifier = self.get_classifier(
                self.classifier_type, self.curr_pair, self.seq_len, num_features
            )
            self.classifier.set_model_path(self.get_model_path())
            self.classifier.set_batch_size(self.batch_size)

        if self.aggregate_pairs:
            whitelist = self.dp.current_whitelist()

            self.df_array.append(dataframe)
            self.label_array.append(labels)
            self.pair_count += 1

            if self.pair_count == len(whitelist):
                self.df_array = self.add_sequential_index(self.df_array)

                if self.training_needed and not self.model_exists():
                    self.debug_print(f"    Training model on {self.pair_count} pairs")
                    self.train_model(self.df_array, self.label_array, self.classifier)

            if self.pair_count == len(whitelist):
                pair_index = (
                    whitelist.index(curr_pair) if curr_pair in whitelist else -1
                )
                if pair_index >= 0 and pair_index < len(self.df_array):
                    dataframe = self.df_array[pair_index]
                else:
                    dataframe = self.df_array[-1]

        else:
            if not self.model_exists():
                self.train_model([dataframe], [labels], self.classifier)

            self.dbg_curr_df = dataframe
            dataframe = self.add_debug_indicators(dataframe)

        return dataframe

    def add_sequential_index(self, dataframe_array):
        # seq_index removed as per USER request (redundant/harmful for GAN and multi-task)
        return dataframe_array
