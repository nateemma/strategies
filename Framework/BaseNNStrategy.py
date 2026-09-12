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


# --------------------------------------------------------------------------
# Parallel-hyperopt pickling fix (same guard as TSPredict/TSPredict.py).
#
# freqtrade's ``hyperopt_pickle_magic`` (optimize/hyperopt/hyperopt_optimizer.py)
# recursively walks the strategy's base-class MRO and registers each base's
# module for cloudpickle pickle-by-value, so strategies subclassed across files
# reach the hyperopt workers. That walk reaches ``object`` (via mixin bases such
# as ``StrategyDiagnostics.__bases__ == (object,)``) and registers
# ``object.__module__ == "builtins"``. Once ``builtins`` is pickled by value,
# cloudpickle serialises ``type`` by value too; because ``type``'s metaclass is
# ``type`` this recurses FOREVER and parallel hyperopt (-j>1) dies with
# "Could not pickle object as excessively deep recursion required" (works at -j 1).
#
# Guard the registration so builtins / stdlib modules are never registered
# by-value — they are always importable by reference, so this is always safe and
# just undoes the accidental over-registration. Idempotent (``_ts_guarded``
# flag, shared with TSPredict). Applied here, not freqtrade core (upstream).
from joblib.externals import cloudpickle as _ftcp

if not getattr(_ftcp.register_pickle_by_value, "_ts_guarded", False):
    _orig_register_pbv = _ftcp.register_pickle_by_value

    def _guarded_register_pbv(module):
        name = getattr(module, "__name__", "")
        if name == "builtins" or name in getattr(sys, "stdlib_module_names", ()):
            return
        return _orig_register_pbv(module)

    _guarded_register_pbv._ts_guarded = True
    _ftcp.register_pickle_by_value = _guarded_register_pbv


# =========================================================================
# BaseNNStrategy
# =========================================================================

class BaseNNStrategy(TrainingEngine, FeatureNormalizer, BaseStrategy):

    # apply_task_filters is MULTI-TASK ONLY — declared as a Parameter in
    # NNMTStrategy and read nowhere else. It used to be set True here, which
    # never reached NNMT (NNMTStrategy defines a fresh buy_params rather than
    # merging) but did propagate a dead key to every other NN strategy.
    # 2026-09-06: 0.6 -> 0.8. With the guards open the model does the selecting,
    # and its confidence only discriminates near the top of the distribution:
    # guards-off W1 goes -24.63% at 0.0, -12.71% at 0.69, +6.94% at 0.80.
    # Located across three windows; per-window optima are 0.82/0.80/0.75 so the
    # value is NOT sharply identified -- treat it as a fixed choice, never tune
    # it per era. See regime/CLOSING_THE_LEARNABILITY_GAP.md.
    buy_params = { **BaseStrategy.buy_params,
        "prediction_threshold": 0.8}

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
    # Window for the peak filter. None => HORIZON, which is correct: the filter
    # re-applies the labeler's own forward-gain test, so any other window
    # silently overrides the label horizon. Set explicitly ONLY to reproduce the
    # pre-2026-09-03 behaviour (PEAK_WINDOW=6) for a controlled comparison.
    LABEL_FILTER_WINDOW = None

    MIN_BUY_GAIN_THRESHOLD = TrainingConfig.MIN_BUY_GAIN_THRESHOLD
    MIN_SELL_LOSS_THRESHOLD = TrainingConfig.MIN_SELL_LOSS_THRESHOLD
    TRAINING_TYPE = TrainingConfig.TRAINING_TYPE

    # Label-side rvol floor, mirroring the ENTRY gate's `rvol > entry_rvol_threshold`.
    # -1.0 = off (rvol is non-negative), which reproduces the pre-2026-09-08 label
    # exactly. Families opt in by overriding; the right value is NOT automatically the
    # family's gate value, because the label also has to stay learnable:
    #   label + rvol>0.2 -> 5.59% of bars | >0.5 -> 4.10% | >1.0 -> 2.52% | >2.0 -> 1.09%
    # against ~5.5% as this repo's observed all-Hold collapse line.
    LABEL_RVOL_THRESHOLD = -1.0
    HORIZON = TrainingConfig.HORIZON

    # controls the profit estimation approach
    use_forward_peak_profit_label = True

    # Signal augmentation gate — used by ``augment_training_signals``,
    # the peak-finding / wavelet-smoothed buy-sell pair generator.
    # Independent of GAN augmentation: strategies that GAN-augment
    # often set this to False so the classifier sees only the GAN's
    # synthetic samples on top of the original real signals.
    apply_label_conflict_resolution = True

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

    # Restrict which task labels the multi-task GAN CONDITIONS on. ``None``
    # (default) ⇒ the GAN conditions on every task the label dict carries
    # (current behavior — byte-identical). When set to a list of task names
    # (e.g. ``["trading", "risk", "momentum", "flow"]``), the GAN-training
    # path filters the label dict to those tasks before the GAN sees it, so
    # the trained GAN's ``task_label_dims`` includes ONLY those tasks. The
    # classifier still trains on all its heads — the aug/generation path
    # reconciles the passed task_labels to whatever tasks the loaded GAN was
    # trained on (dropping tasks the GAN doesn't know, padding any it expects
    # but the classifier doesn't supply). Used to drop noisy conditioning
    # tasks (profit / regime) from the GAN without changing the classifier.
    gan_condition_tasks: Optional[list] = None

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
                "rvol_threshold": self.LABEL_RVOL_THRESHOLD,
            },
        )
        peaks = np.where(buy_labels.values > 0.5)[0]

        # Window is HORIZON, not PEAK_WINDOW. The filter re-applies the SAME
        # forward-gain test the labeler already applied, so using a different
        # window silently OVERRIDES the label horizon: with PEAK_WINDOW=6 and
        # HORIZON=48, every HORIZON >= 6 produced byte-identical labels because
        # the 6-bar test is strictly tighter and its output is a subset.
        # 18 of 27 labelers self-filter on gain, making this a no-op for them;
        # the other 9 (triple_barrier, local_extrema, geometry, future_*,
        # optimal_signals, multi_horizon_vote, breakout_tb) have no gain test,
        # which is what this filter was originally for -- and they now get it at
        # the intended horizon. PEAK_WINDOW is deliberately NOT changed: it is
        # aliased as lookahead_window and shifts NNMT's four aux-task labels.
        filtered_peaks = self.filter_peaks_by_future_performance(
            future_df, peaks, "buy",
            window=(self.LABEL_FILTER_WINDOW or self.HORIZON)
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
            future_df, peaks, "sell",
            window=(self.LABEL_FILTER_WINDOW or self.HORIZON)
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
        """Resolve buy/sell label conflicts. NO LONGER DILATES.

        REMOVED 2026-09-08 -- "Trick 1", which copied every buy and sell label
        onto the TWO PRECEDING bars. Those bars do not meet the label's own
        condition, so the dilation deliberately blurred the decision boundary.

        Two measurements motivated the removal. Precision is the EXPENSIVE axis
        on this data: buying recall by lowering the confidence threshold costs
        -53.2pp in bad picks against +31.7pp recovered
        (regime/RECALL_NOT_PRECISION.md). And removing the dilation measurably
        sharpens the boundary -- precision +2.8/+4.7pp AND recall +7.1/+4.5pp
        across two seeds (regime/DILATION_WIDTH.md).

        Realised return was NOT resolved either way (-2.64pp on the mean, inside
        10-14pp per-seed bands, seeds disagreeing in sign), so this is adopted on
        the boundary evidence rather than on P&L.

        It also removes a trap: the dilation leaked into any arm that read
        get_training_labels() as an entry signal, which is how every oracle in
        this repository came to be entering two bars early
        (regime/ORACLE_CORRECTION_2026-09-07.md).

        The conflict-resolution rules below are label LOGIC, not augmentation,
        and are retained unchanged.
        """

        # Trick 2: sells override buys
        buys[np.where(sells > 0)[0]] = 0.0

        # Trick 3: if a buy is followed closely by a sell, override the buy --
        # the trade would reverse before its target arrived.
        #
        # N: 4 -> 2 on 2026-09-08. Measured how many of the buy labels this
        # deletes actually reach the +2% target BEFORE the sell bar (path-aware;
        # forward-gain alone is max-favourable-excursion over the whole 48-bar
        # horizon and is path-blind, so it flatters these bars at ~5.5%):
        #
        #     N   window   deleted   reached +2% first   share
        #     1      15m     2,563                 481   18.8%
        #     2      30m     7,803               2,388   30.6%
        #     3      45m    16,031               6,546   40.8%
        #     4      60m    27,651              13,827   50.0%   <- was here
        #     6      90m    61,708              39,790   64.5%
        #
        # At N=4 it is a coin flip: half the deletions were reachable buys, so
        # the rule was destroying valid training examples rather than
        # suppressing whipsaw. At N=2 ~70% of deletions are correct. The rise is
        # monotone because bars further from the sell have more room to reach
        # target -- consistent with the ~7-bar median time-to-target on this
        # label. (Counts tally per sell-bar pair, so a buy near two sells counts
        # twice; the SHARE is the meaningful column.)
        sell_positions = np.where(sells == 1)[0]
        N = 2
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

    # --- Feedback-guided entropy SELECTION (multi-task GAN aug) ---------- #
    # After the AE fidelity filter culls off-manifold synth, keep the
    # synthetic samples a frozen real-data classifier is most UNCERTAIN
    # about (highest trading-head Shannon entropy = nearest the decision
    # boundary). Fidelity is already solved by the AE guardrail; the
    # remaining gap is downstream trading utility, which uncertain samples
    # target directly. Master gate OFF ⇒ byte-identical to current behavior.
    gan_entropy_guidance: bool = False
    # Fraction of the AE-passed synth to KEEP, highest-entropy first.
    gan_entropy_select_fraction: float = 0.5
    # Trained model dir under saved_data/ used as the frozen scorer.
    gan_entropy_guidance_model: str = "NNMT_MLX"







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

        if self.apply_label_conflict_resolution:
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
            if not self.apply_label_conflict_resolution and (
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

        # The post-GAN tensor scaler is only APPLIED AT TRAINING for multi-task
        # GAN types (preprocess_training_data is multi-task-only; single-task and
        # non-GAN strategies train in scale_dataframe / RobustScaler space). So
        # predicting through the tensor scaler for anything but a multi-task GAN
        # would mismatch train/predict normalization and degrade results. Gate on
        # multi-task membership so single-task GANs and non-GAN strategies predict
        # through the scale_dataframe space they were trained in.
        if (
            getattr(self, "use_post_gan_scaling", False)
            and self.gan_type in self._MULTI_TASK_GAN_TYPES
        ):
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
        else:
            offset = 0

        # Debug columns carrying the per-class probabilities. get_predictions() is the only
        # place they exist: process_predictions() stores just the THRESHOLDED predict_buy /
        # predict_sell, so a trade opened at confidence 0.81 and one at 0.99 are otherwise
        # indistinguishable — and 0.8 is the production prediction_threshold, so nearly every
        # trade sits where the discarded digits matter most.
        #
        # The "%" prefix is load-bearing, not cosmetic: DataframeUtils.remove_debug_columns()
        # drops every ^% column, and it runs before scaling (FeatureNormalizer) and in scaler
        # creation (CreateScalers). Without it these columns would be picked up as training
        # features, which is a predict-your-own-prediction leak.
        #
        # NaN, never 0.0, outside the predicted region: 0.0 is a legitimate probability
        # meaning "confidently not this class", and using it as padding would mark every
        # warm-up bar a confident non-buy. A consumer must treat NaN as "not scored".
        #
        # Written AFTER the Markov branch so the columns always hold what
        # argmax_with_threshold actually consumed. That branch is inert by default
        # (use_markov_smoothing = False) but writing before it would silently start recording
        # a different quantity than the one traded on if it were ever enabled.
        for _cls, _name in (
            (TradingAction.SELL, "%predict_prob_sell"),
            (TradingAction.HOLD, "%predict_prob_hold"),
            (TradingAction.BUY, "%predict_prob_buy"),
        ):
            _col = np.full(original_length, np.nan, dtype=float)
            if pred_probs is not None and pred_probs.ndim == 2:
                _col[offset:offset + predicted_length] = pred_probs[:predicted_length, _cls]
            dataframe[_name] = _col

        # Optional side-car, for consumers in ANOTHER PROCESS.
        #
        # The columns above cannot reach freqtrade's `--export signals` artefact: that
        # artefact is `preprocessed_tmp`, built from advise_all_indicators(), whose docstring
        # states "Does not run advise_entry or advise_exit!" and that it copies on input and
        # output. Predictions are model OUTPUTS and belong to the entry/exit phase, which runs
        # later against a different object — moving prediction into populate_indicators, where
        # the model's INPUTS are built, would break the general freqtrade flow.
        #
        # So a backtest subprocess needs an explicit channel. This is it, and it is INERT
        # unless FT_PREDICT_PROB_SINK names a directory: one `os.environ.get` otherwise, no
        # I/O, no behaviour change in production. One feather per pair per run; callers
        # running several strategies must give each its own sink directory or the later run
        # overwrites the earlier.
        _sink = os.environ.get("FT_PREDICT_PROB_SINK")
        if _sink:
            try:
                _sdir = Path(_sink)
                _sdir.mkdir(parents=True, exist_ok=True)
                _pair = getattr(self, "curr_pair", "") or "UNKNOWN"
                _out = DataFrame({
                    "date": dataframe["date"].to_numpy(),
                    "%predict_prob_sell": dataframe["%predict_prob_sell"].to_numpy(),
                    "%predict_prob_hold": dataframe["%predict_prob_hold"].to_numpy(),
                    "%predict_prob_buy": dataframe["%predict_prob_buy"].to_numpy(),
                })
                _out.to_feather(_sdir / f"{_pair.replace('/', '_')}.feather")
            except Exception as _e:                       # never break a run for telemetry
                self.debug_print(f"    prob sink write failed: {_e}")

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
