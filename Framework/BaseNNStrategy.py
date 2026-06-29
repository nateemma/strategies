# pragma pylint: disable=C0103, C0114, C0115, C0116, C0301, C0302, C0303, C0325, C0411, C0413
# pragma pylint: disable=W0105, W1203, W1309, W1514, W0613, W0621,
# type: ignore
# pylint: disable=import-error
# flake8: noqa: F401, E402, F541, W0718, W0719

"""
BaseNNStrategy - Base class for all Neural Network trading strategies.

Inherits from BaseStrategy and adds:
 - NN-specific imports (tensorflow, KerasBasePredictor, GANs, TrainingSignals)
 - Training parameters (seq_len, epochs, batch_size, etc.)
 - Normalization pipeline (rolling_dataframe_normalise, scalers, PCA)
 - Feature list management (include_list, pre_normalized_columns)
 - Training label generation (peak detection, signal augmentation)
 - GAN augmentation (WGAN-GP, CTAB-GAN+)
 - Model management (classifiers, storage, model paths)
 - Training / prediction pipeline
 - Aggregation helpers for multi-pair training

Subclasses override get_classifier_type() and get_classifier() to provide
specific model architectures.
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
from Predictors.KerasBasePredictor import KerasBasePredictor

from utils.Scalers import scaler_exists, save_scaler, load_scaler
from GANs.GANInterface import GANInterface  # noqa: E402

import Framework.TrainingSignals as TrainingSignals

from Framework.FeatureNormalizer import FeatureNormalizer
from Framework.TrainingEngine import TrainingEngine
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
from Framework.TrainingConfig import TrainingConfig

# --------------------------------
# Global setup
# --------------------------------
log = logging.getLogger(__name__)


# =========================================================================
# BaseNNStrategy
# =========================================================================

class BaseNNStrategy(TrainingEngine, FeatureNormalizer, BaseStrategy):

    buy_params = { **BaseStrategy.buy_params,
        "prediction_threshold": 0.6,
        "profit_prediction_threshold": 0.3,
        "apply_task_filters": True}

    # ATR-adaptive initial stoploss enabled by default for every NN
    # variant — definitions of the flag and its companion attrs
    # (multiplier, floor, cap) live in BaseStrategy. Subclasses can
    # override the multiplier / floor / cap as needed; setting this
    # flag back to False here will turn off adaptive stops for the
    # whole NN family.
    use_atr_adaptive_stoploss = True

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
    RISK_LOOKBACK = 200
    FLOW_LOOKBACK = 200
    # Pulled from Framework.TrainingConfig so the strategy, the GAN trainer
    # (CreateGANBase), and the GAN-metadata validator can never silently
    # drift. Override on a subclass to customise. HORIZON moved here from
    # ``BaseStrategy.PEAK_WINDOW`` so it stays coupled with the gain/loss
    # thresholds — both are labeling-time parameters and must be retuned
    # together (see project_horizon_threshold_learnability_finding.md).
    MIN_BUY_GAIN_THRESHOLD = TrainingConfig.MIN_BUY_GAIN_THRESHOLD
    MIN_SELL_LOSS_THRESHOLD = TrainingConfig.MIN_SELL_LOSS_THRESHOLD
    TRAINING_TYPE = TrainingConfig.TRAINING_TYPE
    HORIZON = TrainingConfig.HORIZON

    # controls the profit estimation approach
    use_forward_peak_profit_label = True

    # Signal augmentation gate — used by ``augment_training_signals``,
    # the peak-finding / wavelet-smoothed buy-sell pair generator.
    # Independent of GAN augmentation: strategies that GAN-augment
    # often set this to False so the classifier sees only the GAN's
    # synthetic samples on top of the original real signals.
    augment_training_data = True

    # GAN augmentation knobs — see StrategyConfig docstring.  Strategies
    # opt in by overriding ``gan_type`` and (optionally) the ratio /
    # diagnostics flags; everything else is dispatched by the base class
    # based on ``gan_type``.
    gan_type: GANType = GANType.NONE
    gan_augment: bool = True
    gan_target_ratio: Any = 0.8         # float or Dict — see balance.py
    gan_run_diagnostics: bool = False
    # Seed for synthetic-sample generation. A fixed value makes augmented
    # training runs reproducible (default 42, matching the codebase's
    # train/test_shuffle_seed convention); set to None for non-deterministic
    # augmentation.
    gan_augment_seed: Optional[int] = 42

    # When True, route the GAN augmentation through the post-GAN scaling
    # pipeline: the GAN sees RAW (B, T, F) tensors, does its own internal
    # z-score, and a polymorphic tensor scaler is applied to the augmented
    # tensor before the classifier sees it. Only supported for MT_DDPM with
    # single-task labels currently; other gan_type values fall back to the
    # pre-GAN scaling pipeline regardless of this flag.
    use_post_gan_scaling: bool = False

    training_needed = True  # set automatically

    classifier = None
    classifier_type = None

    # Aggregation
    aggregate_pairs = True  # use all pairs for training (in backtest)
    df_array: [DataFrame] = []
    label_array: List[Any] = []
    pair_count = 0

    # Hyperopt param for training type
    from freqtrade.strategy import IntParameter
    # training_type = IntParameter(
    #     0,
    #     19,
    #     default=16,
    #     space="buy",
    #     load=True,
    #     optimize=False,
    # )

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





    def model_exists(self) -> bool:
        """Check if model exists on disk"""
        if self.classifier is not None:
            return self.classifier.model_exists()

        model_path_keras = self.get_model_path()
        model_path_sav = model_path_keras.replace(".keras", ".sav")

        model_found = os.path.exists(model_path_keras) or os.path.exists(model_path_sav)
        return model_found

    # =========================================================================
    # Category helpers
    # =========================================================================

    task_thresholds = {
        "momentum": {"low": -0.5, "high": 0.6},
        "flow": {"low": -0.05, "high": 0.05},
    }

    # EMA spans for smoothing the underlying indicator before tri-state
    # thresholding. The raw indicators (atr_norm, di_diff_scaled,
    # aroonosc_scaled) update per-bar and flip on noise; smoothing turns
    # them into slow-moving regime labels. Tune via subclass.
    risk_smoothing_span: int = 50
    flow_smoothing_span: int = 24
    momentum_smoothing_span: int = 24

    # Slow-moving regime parameters — close vs EMA with a symmetric deadband.
    # 200-bar EMA gives multi-day persistence on hourly candles; ±1.5% deadband
    # prevents flips when price crosses the EMA on noise. Tune via subclass.
    regime_ema_period: int = 200
    regime_deadband: float = 0.015

    def get_market_regime(self, dataframe: DataFrame) -> np.ndarray:
        """Classify slow-moving bull/bear/sideways regimes via close vs a centered rolling mean.

        BULL when close is ≥``regime_deadband`` above the smoothed reference;
        BEAR when ≥deadband below; SIDEWAYS otherwise. Uses a CENTERED
        rolling mean (window = ``regime_ema_period``) so the label at bar t
        represents the regime around t with zero net lag — fine because this
        is a training target (lookahead is allowed for labels, not features).
        """
        close = dataframe["close"]
        smoothed = close.rolling(
            window=self.regime_ema_period, center=True, min_periods=1
        ).mean()
        upper = smoothed * (1.0 + self.regime_deadband)
        lower = smoothed * (1.0 - self.regime_deadband)

        regime = np.ones(len(dataframe), dtype=int) * MarketRegime.SIDEWAYS
        regime = np.where(close > upper, MarketRegime.BULL, regime)
        regime = np.where(close < lower, MarketRegime.BEAR, regime)

        return regime

    def get_risk_level(self, dataframe: DataFrame) -> np.ndarray:
        """Calculate tri-state risk classification: LOW=0, NORMAL=1, HIGH=2.

        Per-bar atr_norm flips on individual volatile/quiet candles; an EMA
        over ``risk_smoothing_span`` bars produces a slow-moving vol-regime
        label that aligns with how volatility regimes actually persist.
        """
        self.check_columns_included(["atr_norm"], "get_risk_level")

        atr = pd.Series(
            dataframe.get("atr_norm", pd.Series(np.zeros(len(dataframe))))
        ).fillna(0)
        atr = np.nan_to_num(atr, nan=0.0, posinf=1.0, neginf=-1.0)
        atr = (
            pd.Series(atr)
            .rolling(window=self.risk_smoothing_span, center=True, min_periods=1)
            .mean()
            .to_numpy()
        )

        risk_class = np.ones(len(atr), dtype=int) * RiskLevel.NORMAL
        risk_class[atr < -0.33] = RiskLevel.LOW
        risk_class[atr > 0.33] = RiskLevel.HIGH

        return risk_class

    def get_flow(self, dataframe: DataFrame) -> np.ndarray:
        """Predict future directional bias using di_diff_scaled (Plus DI - Minus DI).

        DI difference is a per-bar directional indicator; an EMA over
        ``flow_smoothing_span`` bars stops the label flipping on each
        short-term direction reversal.
        """
        self.check_columns_included(["di_diff_scaled"], "get_flow")

        di_diff = dataframe.get("di_diff_scaled", pd.Series(np.zeros(len(dataframe))))
        di_diff = pd.Series(di_diff).fillna(0)
        di_diff = np.nan_to_num(di_diff.values, nan=0.0, posinf=1.0, neginf=-1.0)
        di_diff = (
            pd.Series(di_diff)
            .rolling(window=self.flow_smoothing_span, center=True, min_periods=1)
            .mean()
            .to_numpy()
        )

        # Forward-shift is applied by the outer target layer (get_flow_target);
        # threshold the current smoothed value here so the two shifts don't
        # compound.
        flow_classes = np.ones(len(dataframe), dtype=int) * FlowDirection.NEUTRAL
        flow_classes[di_diff < -0.15] = FlowDirection.DECREASE
        flow_classes[di_diff > 0.15] = FlowDirection.INCREASE

        dataframe["flow"] = flow_classes

        return flow_classes

    def get_momentum(self, dataframe: DataFrame) -> np.ndarray:
        """Calculate momentum using normalized aroonosc (pair-agnostic).

        Aroon Oscillator updates whenever the within-period high/low moves;
        an EMA over ``momentum_smoothing_span`` bars produces a slow-moving
        momentum-regime label instead of a per-bar oscillator state.
        """
        self.check_columns_included(["aroonosc_scaled"], "get_momentum")

        momentum = dataframe.get("aroonosc_scaled", pd.Series(np.zeros(len(dataframe))))
        momentum = pd.Series(momentum).fillna(0)
        momentum = np.nan_to_num(momentum, nan=0.0, posinf=1.0, neginf=-1.0)
        momentum = (
            pd.Series(momentum)
            .rolling(window=self.momentum_smoothing_span, center=True, min_periods=1)
            .mean()
            .to_numpy()
        )

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
    # GAN augmentation config (logic lives in TrainingEngine + GANs.balance)
    # =========================================================================

    # Columns the GAN should NOT generate — values for these are copied
    # from a random real sample at augmentation time.  Use for features
    # with rigid structure that GANs reliably mis-reproduce (calendar
    # sin/cos pairs, one-hot categoricals).  Empty default — the
    # calendar features that previously lived here have been removed
    # from include_list entirely (low signal didn't justify the GAN
    # modeling overhead). Subclasses set this when they have specific
    # features the GAN can't fit; see NNNC_DDPM_MLX_LSTM for an example.
    gan_passthrough_columns: List[str] = []

    # Inference-time overrides applied to the loaded GAN model after
    # ``interface.load()``. Default ``None`` preserves whatever the model
    # was saved with — used to A/B-test sampling-side changes (more
    # denoising steps, classifier-free guidance scale) without retraining.
    # Currently honoured by DDPM-family backends (TAB_DDPM, MT_DDPM);
    # GAN-family backends (WGAN, CTAB_GAN) silently ignore the knobs that
    # don't apply to them.
    gan_inference_sample_steps: Optional[int] = None
    gan_inference_guidance_scale: Optional[float] = None

    # Density-based rejection sampling on generated synth bars. Fits a
    # per-class Gaussian mixture on the real same-class pool and drops
    # the lowest-density synth samples (those that fell in low-likelihood
    # regions of the real distribution). ``0.0`` disables the filter.
    # The generator is called with an inflated count so the post-filter
    # output still hits the requested ``need`` for each class.
    gan_synth_density_reject_pct: float = 0.0
    gan_synth_density_components: int = 8

    # Discriminator-based rejection sampling on generated synth bars.
    # Trains a binary classifier on (real_pool, synth) per class and
    # drops the bottom-fraction by P(real). Catches joint-correlation
    # drift the density filter (diagonal-cov GMM) misses. ``0.0``
    # disables; the generator is called with an inflated count so the
    # post-filter output still hits each class's requested ``need``.
    gan_synth_discrim_reject_pct: float = 0.0

    # Neural-discriminator rejection sampling. Uses the unified
    # RealnessDiscriminator trained once by the discriminator-creator strategy across
    # synth from every saved GAN. Independent of the in-loop HistGB
    # filter above — both can run, inflates multiply. ``0.0`` disables.
    # ``gan_synth_neural_discrim_model_path`` defaults to the
    # conventional save location written by that creator strategy (under
    # the strategy's storage location).
    gan_synth_neural_discrim_reject_pct: float = 0.0
    gan_synth_neural_discrim_model_path: Optional[str] = None

    # Realsignal rejection sampling. Loads per-class binary classifiers
    # trained ONLY on real data (no GAN samples) — drop synth rows whose
    # features don't look like a real example of the class they claim
    # to be. Default save root is ``saved_data/Discriminators/realsignal/``
    # written by the realsignal-discriminator creator strategy. ``0.0`` disables.
    gan_synth_realsignal_reject_pct: float = 0.0
    gan_synth_realsignal_model_root: Optional[str] = None

    # Confidence-threshold variants of the two NN filters. When set,
    # the filter keeps every synth row scoring above the threshold
    # (variable output count) instead of the rank-based bottom-fraction
    # drop. Mutually exclusive with the matching reject_pct — set only
    # one of (reject_pct, threshold) per filter. ``None`` disables.
    gan_synth_neural_discrim_threshold: Optional[float] = None
    gan_synth_realsignal_threshold: Optional[float] = None

    # Mahalanobis-distance rejection. Fits a Ledoit-Wolf shrinkage
    # multivariate Gaussian on the real same-class pool and scores
    # each synth row by squared Mahalanobis distance. Lower = closer
    # to the real distribution centroid. Captures joint structure
    # (covariance) which the realsignal classifier and diagonal-cov
    # density filter both miss. ``_reject_pct`` drops the top-fraction
    # by distance (highest-distance = most off-distribution); ``_threshold``
    # keeps rows with d² below the cutoff (under MVN, d² is
    # χ²-distributed with F degrees of freedom, so for F=24 a threshold
    # in [20, 50] is sensible).
    gan_synth_mahalanobis_reject_pct: float = 0.0
    gan_synth_mahalanobis_threshold: Optional[float] = None

    # Autoencoder rejection sampling. Loads per-class MLP autoencoders
    # trained ONLY on real same-class data (no GAN samples) by the
    # autoencoder-filter creator strategy. Scores each synth row by reconstruction
    # MSE — low MSE = on the real manifold = keep; high MSE = off
    # manifold = drop. Manifold-aware: unlike Mahalanobis it doesn't
    # reward centroid clustering, so real tail samples reconstruct well
    # while off-distribution synth (broken joints, missing structure)
    # have high error regardless of their distance from the centroid.
    # Default save root is ``saved_data/Discriminators/autoencoder/``.
    # ``_reject_pct`` drops the top-fraction by MSE; ``_threshold``
    # keeps rows below the cutoff (typical scale: 0.005-0.05 depending
    # on normalization).
    gan_synth_autoencoder_reject_pct: float = 0.0
    gan_synth_autoencoder_model_root: Optional[str] = None
    gan_synth_autoencoder_threshold: Optional[float] = None








    def get_classifier_type(self):
        """Return the type of classifier used for training/predicting"""
        raise NotImplementedError("get_classifier_type() not implemented")
        return None

    def get_classifier(
        self, classifier_type, pair, seq_len, num_features
    ) -> KerasBasePredictor:
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
    # Prediction pipeline
    # =========================================================================

    def get_predictions(self, dataframe: DataFrame, classifier: KerasBasePredictor):
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

        if getattr(self, "use_post_gan_scaling", False):
            # Post-GAN scaling path: skip DataFrame-level normalization but
            # still do the column filtering (drop non-numeric / debug cols)
            # so df_to_tensor doesn't trip on object dtypes. Then apply the
            # polymorphic tensor scaler to the 3D tensor.
            from utils.Scalers import load_scaler  # noqa: E402
            df_clean = self.clean_for_tensor(dataframe)
            df_tensor = self.dataframeUtils.df_to_tensor(
                df_clean, self.seq_len, method=self.tensor_method
            )
            tensor_scaler = load_scaler(self.get_storage_location(), "main_tensor_scaler")
            df_tensor = tensor_scaler.transform(np.asarray(df_tensor))
        else:
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
        # (via CreateGANBase.bot_start setting it on super()) so the
        # MASTER thresholds reach the GAN metadata unmodified by hyperopt.
        is_gan_creation = getattr(self, "_is_gan_creation_strategy", False)

        if is_gan_creation:
            return

        # The strategy is the source of truth for thresholds and
        # training_type.  Hyperopt overrides apply unconditionally;
        # any GAN that was trained against different values will fail
        # ``GANInterface.load(expected=…)`` with a loud diff so the
        # operator can decide whether to retrain or pin the threshold.
        # (The previous "GAN's saved thresholds silently win" path
        # masked label/GAN drift — see the threshold validation in
        # GANInterface.)
        # if (
        #     hasattr(self, "buy_params")
        #     and self.buy_params
        #     and "min_buy_gain_threshold" in self.buy_params
        # ):
        #     self.MIN_BUY_GAIN_THRESHOLD = self.buy_params["min_buy_gain_threshold"]

        # if (
        #     hasattr(self, "sell_params")
        #     and self.sell_params
        #     and "min_sell_loss_threshold" in self.sell_params
        # ):
        #     self.MIN_SELL_LOSS_THRESHOLD = self.sell_params["min_sell_loss_threshold"]

        # if not hasattr(self, "TRAINING_TYPE") or self.TRAINING_TYPE == 13:
        #     self.TRAINING_TYPE = self.training_type.value

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

        # Debug
        dataframe["%train_labels"] = labels

        # Classifier setup + training trigger now live on TrainingEngine.
        dataframe = self.maybe_train(dataframe, labels, curr_pair)

        return dataframe

    def add_sequential_index(self, dataframe_array):
        # seq_index removed as per USER request (redundant/harmful for GAN and multi-task)
        return dataframe_array
