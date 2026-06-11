# pragma pylint: disable=C0103, C0114, C0115, C0116, C0301, C0302, C0303, C0325, C0411, C0413
# pragma pylint: disable=W0105, W1203, W1309, W1514, W0613, W0621,
# type: ignore
# pylint: disable=import-error
# flake8: noqa: F401, E402, F541, W0718, W0719

"""
BaseNNMTStrategy - shared scaffolding for Neural Network Multi-Task strategies.

Sits between BaseNNStrategy (single-task defaults + shared pipeline) and the
concrete NNMTStrategy. Multi-task class attributes, target calculators, and
overridden pipeline methods belong here so a second multi-task strategy can
inherit them without duplicating NNMTStrategy.
"""

import sys
from pathlib import Path
from enum import IntEnum
from typing import Dict, List, Optional, Tuple
from pandas import DataFrame
import numpy as np
import pandas as pd
import traceback
import logging

log = logging.getLogger(__name__)

# Match NNMTStrategy's sys.path setup so sibling-module imports resolve
group_dir = str(Path(__file__).parent)
sys.path.append(group_dir)

from Framework.BaseNNStrategy import BaseNNStrategy
from Framework.BaseStrategy import MarketRegime, TradingAction, FlowDirection, MomentumDirection, RiskLevel
from utils.ClassifierKeras import ClassifierKeras
from utils import ForwardPeakRegressor as _FPR
from freqtrade.strategy import DecimalParameter, IntParameter, BooleanParameter
from GANs.GANType import GANType


class ProfitDirection(IntEnum):
    LOSS = 0
    NEUTRAL = 1
    PROFIT = 2


class _UnflattenedGenerateWrapper:
    """Wraps a GANInterface so generate() returns (n, T, F) ndarrays.

    Multi-task tabular GAN backends (e.g. MT_CTAB_GAN) are trained on
    flattened sequence windows and natively produce (n, T*F) DataFrames
    with column names like ``<feature>_<t>``. preprocess_training_data
    needs to mix those with the real (n, T, F) tensor that arrives from
    prepare_training_data, so we reshape the synth output here once,
    then balance_multi_task and swap_passthrough_columns can run on the
    3-D path uniformly.

    The reshape uses C-order, which is the same convention used to
    flatten during GAN training: index ``t*F + f`` on the flat axis
    maps to ``(t, f)`` on the un-flattened tensor.
    """

    def __init__(self, interface, T: int, F: int):
        self._interface = interface
        self._T = T
        self._F = F
        # Mirror commonly-inspected attributes so callers can still read
        # gan_type / save_path / model through the wrapper.
        self.gan_type = getattr(interface, "gan_type", None)

    def __getattr__(self, name):
        return getattr(self._interface, name)

    def generate(self, n, **kwargs):
        result = self._interface.generate(n=n, **kwargs)
        if isinstance(result, tuple) and len(result) == 2:
            data, labels = result
            return self._unflatten(data), labels
        return self._unflatten(result)

    def _unflatten(self, data):
        if isinstance(data, pd.DataFrame):
            data = data.to_numpy()
        if not isinstance(data, np.ndarray):
            return data
        if data.ndim == 2 and data.shape[1] == self._T * self._F:
            return data.reshape(data.shape[0], self._T, self._F)
        if data.ndim == 3:
            return data
        # Anything else -- leave as-is and let the caller's shape check
        # surface the mismatch with an informative error.
        return data


class _PadMissingTaskLabelsWrapper:
    """Wraps a GANInterface so generate() pads missing task labels with
    uniform-random one-hot encodings.

    Used by single-task NNNC strategies running against a multi-task GAN:
    the strategy provides only ``{"trading": one_hot}``, but the loaded GAN
    was trained conditioned on N tasks. Calling generate() with only one
    task is an OOD input regime for the model — diagnostics showed this
    degrades sample quality (synth lag-1 autocorrelation flips negative
    against a real autocorrelation of +0.98 — see the NNNC_DDPM_MLX_LSTM_MT
    diagnostic at 2026-05-13). Filling the missing task slots with
    uniform-random one-hots restores in-distribution conditioning.

    The "uniform-random" choice is a deliberate non-informative prior on
    the auxiliary tasks; we don't want their values to systematically bias
    the trading-task generation.
    """

    def __init__(self, interface, expected_task_label_dims: Dict[str, int]):
        self._interface = interface
        self._task_dims = dict(expected_task_label_dims)
        self.gan_type = getattr(interface, "gan_type", None)

    def __getattr__(self, name):
        return getattr(self._interface, name)

    def generate(self, n, **kwargs):
        task_labels = dict(kwargs.get("task_labels", {}))
        for task, dim in self._task_dims.items():
            if task in task_labels:
                continue
            idx = np.random.randint(0, dim, size=n)
            task_labels[task] = np.eye(dim, dtype=np.float32)[idx]
        kwargs["task_labels"] = task_labels
        return self._interface.generate(n, **kwargs)


class BaseNNMTStrategy(BaseNNStrategy):
    """
    Multi-task neural network strategy base.

    Empty in this commit; subsequent phases move attributes and methods up from
    NNMTStrategy. NNMTStrategy still inherits the full multi-task surface area
    via this class — behavior is unchanged.
    """

    buy_params = { **BaseNNStrategy.buy_params,
        "prediction_threshold": 0.6,
        "profit_prediction_threshold": 0.3
        }

    profit_conflict_to_neutral = True
    # EMA span used to smooth the close before measuring forward gain. Short
    # span (≤2) preserves the moves we want the label to capture; longer
    # spans hide them by averaging across the lookahead window.
    PROFIT_EMA_SPAN = 2
    # Scale factor on ATR% when computing the volatility-adjusted threshold.
    # The effective threshold is max(MIN_GAIN, PROFIT_ATR_SCALE * atr_pct),
    # so a smaller scale lets the fixed floor win more often and keeps
    # volatile periods from ratcheting the threshold out of reach.
    PROFIT_ATR_SCALE = 0.5
    # Forward window the profit label looks at. Independent of HORIZON so
    # tightening trading-peak detection doesn't also shorten the profit
    # horizon; defaults to a multiple of the trading horizon so PROFIT/LOSS
    # have a realistic chance of exceeding the threshold.
    PROFIT_HORIZON = 8

    # When True, ``get_profit_target`` returns path-independent peak-based
    # labels (PROFIT = window max forward gain ≥ threshold regardless of
    # whether downside hit first, LOSS = only downside, NEUTRAL = neither).
    # Validated via DebugSignalLearnability as ~1.5× more learnable than the
    # first-touch triple-barrier label. Default False preserves existing
    # comparison runs; opt in per-subclass.
    use_forward_peak_profit_label = True

                                                                                                                       
    use_forward_peak_regressor_filter = False 

    stoploss = -0.05

    # ATR-adaptive stoploss now lives in BaseStrategy (flag + attrs) and
    # is enabled by default in BaseNNStrategy. Override the multiplier /
    # floor / cap here if NNMT-specific tuning diverges.

    # -----------
    # Hyperopt parameters
    # -----------

    # Consecutive signal filter (Note: increasing causes delay in real-time detection)
    min_consecutive_buys = IntParameter(
        1, 2, default=1, space="buy", optimize=True, load=True
    )

    # prediction

    optimize_bias = False

    bias_trading_sell = DecimalParameter(
        0.01,
        0.06,
        default=0.03,
        decimals=2,
        space="buy",
        optimize=optimize_bias,
        load=True,
    )
    bias_trading_buy = DecimalParameter(
        0.01,
        0.06,
        default=0.05,
        decimals=2,
        space="buy",
        optimize=optimize_bias,
        load=True,
    )
    bias_profit_low = DecimalParameter(
        0.05,
        0.18,
        default=0.09,
        decimals=2,
        space="buy",
        optimize=optimize_bias,
        load=True,
    )
    bias_profit_high = DecimalParameter(
        0.05,
        0.18,
        default=0.08,
        decimals=2,
        space="buy",
        optimize=optimize_bias,
        load=True,
    )

    # Confidence floor applied to the profit-head argmax. The shared
    # ``prediction_threshold`` (default 0.5) is too high for the profit head
    # because the profit task has a much lower learnability ceiling (MCC ~0.10
    # vs trading's 0.6+), so the model's max-class probability rarely clears
    # 0.5 and every bar falls back to NEUTRAL.  Default 0.4 is the value that
    # produced usable profit signals in manual testing.
    profit_prediction_threshold = DecimalParameter(
        0.2,
        0.5,
        default=0.4,
        decimals=2,
        space="buy",
        optimize=True,
        load=True,
    )

    apply_task_filters = BooleanParameter(
        default=False,
        space="buy",
        optimize=True,
        load=True,
    )

    # External forward-peak-gain regressor filter — opt-in.  When True,
    # ``process_predictions`` loads a per-pair RandomForestRegressor from
    # saved_data/ForwardPeakRegressor/<pair>.joblib (trained offline via
    # Debug/TrainForwardPeakRegressor.py) and zeros out trading BUY
    # predictions whose predicted vol-adjusted forward peak gain falls
    # below ``forward_peak_regressor_threshold``.  Stacks on top of
    # apply_task_filters — both can be enabled simultaneously.
    use_forward_peak_regressor_filter = False
    # Per-pair percentile cutoff applied to the regressor predictions.
    # 0.0 = never reject; 0.5 = reject bars below this pair's median
    # predicted forward peak; 0.9 = keep only the top 10%.  The percentile
    # mapping is calibrated at training time against each pair's own
    # score distribution, so the threshold has the same meaning across
    # pairs with very different vol-adjusted score scales.
    forward_peak_regressor_threshold = DecimalParameter(
        0.0,
        0.9,
        default=0.5,
        decimals=2,
        space="buy",
        optimize=True,
        load=True,
    )

    # Per-pair regressor cache.  Models load lazily on first prediction;
    # value ``False`` is sentinel for "tried and missing on disk" so we
    # don't hit the filesystem on every call.
    _forward_peak_regressor_cache: Dict[str, object] = {}
    # -----------
    # Class level parameters
    # -----------

    augment_training_data = True  # signal augmentation; GAN augmentation gates on gan_augment

    filter_signals = False  # don't double filter

    regime_lookback = 20  # Periods for regime detection
    volatility_lookback = 10  # Periods to calculate volatility
    risk_threshold = 0.02  # Risk threshold for binary classification

    PROFIT_TAKE_THRESHOLD = 0.02
    PROFIT_STOP_LOSS_THRESHOLD = 0.015

    task_thresholds = {
        "momentum": {"low": -0.5, "high": 0.6},
        "flow": {"low": -5.0, "high": 5.0},
        "profit": {"low": -0.006, "high": 0.006},
    }

    # -----------
    # Utility functions
    # -----------

    def get_class_weights(self, category_array):
        """Get class weights for a given category array (each entry is a class identifier)"""

        # Get unique classes and their counts
        unique_classes, class_counts = np.unique(category_array, return_counts=True)

        # The correct class counts: [10101 (Sell), 188300 (Hold), 9088 (Buy)]
        counts = np.array(class_counts)
        total_samples = np.sum(counts)
        num_classes = len(counts)

        # Calculate the balanced weight for all classes using the standard formula
        balanced_weights = total_samples / (num_classes * counts)

        # Map the calculated weights to the cweights_array based on class index
        # [6.847, 0.367, 7.610]
        cweights_array = np.zeros(num_classes)
        for i, weight in zip(unique_classes, balanced_weights):
            cweights_array[i] = weight

        # normalise the weights to sum to 1
        cweights_array = cweights_array / np.sum(cweights_array)
        return cweights_array

    def get_cumulative_distribution(self, probabilities: np.ndarray) -> np.ndarray:
        """Utility function to get cumulative distribution of probabilities"""
        bins = np.bincount(
            (probabilities * 10).astype(int), minlength=11
        )  # 0.0 to 1.0 in 0.1 steps
        percentages = bins / len(probabilities)
        cumulative = np.cumsum(percentages)
        return cumulative

    # -----------
    # Task-specific functions
    # -----------

    # Market Regime

    def get_market_target(self, dataframe: DataFrame) -> np.ndarray:
        """Classify market regimes using ADX and SMA alignment for better accuracy"""

        if "regime" not in dataframe.columns:
            raise ValueError("Regime column not found in dataframe")

        # save (non-lookahead) regime to dataframe
        regime = dataframe["regime"]

        # shift so that we get the future values for training
        regime = pd.Series(regime).shift(-self.lookahead_window + 1)
        regime = np.nan_to_num(
            regime,
            nan=MarketRegime.SIDEWAYS,
            posinf=MarketRegime.SIDEWAYS,
            neginf=MarketRegime.SIDEWAYS,
        ).astype(int)

        return regime

    # -----------

    # Risk Level

    def get_risk_target(self, dataframe: DataFrame) -> np.ndarray:
        """Calculate tri-state risk classification: LOW=0, NORMAL=1, HIGH=2"""

        if "risk" not in dataframe.columns:
            raise ValueError("Risk column not found in dataframe")

        risk_class = dataframe["risk"]

        # shift so that we get the future values for training
        risk_class = pd.Series(risk_class).shift(-self.lookahead_window + 1)
        risk_class = np.nan_to_num(
            risk_class,
            nan=RiskLevel.NORMAL,
            posinf=RiskLevel.NORMAL,
            neginf=RiskLevel.NORMAL,
        ).astype(int)

        return risk_class

    # -----------

    # Flow

    def get_flow_target(self, dataframe: DataFrame) -> np.ndarray:

        if "flow" not in dataframe.columns:
            raise ValueError("Flow column not found in dataframe")

        flow_classes = dataframe["flow"]

        # shift so that we get the future values for training
        flow_classes = pd.Series(flow_classes).shift(-self.lookahead_window + 1)
        flow_classes = flow_classes.fillna(FlowDirection.NEUTRAL)
        flow_classes = flow_classes.astype(int)

        # Optional: You may want to analyze the distribution of adx_change_target to set robust thresholds.
        # self.analyze_distribution("Flow_ADX_Change", adx_change_target)

        return flow_classes

    # -----------

    # Momentum

    def get_momentum_target(self, dataframe: DataFrame) -> np.ndarray:
        """Calculate momentum using indicators from dataframe"""

        if "momentum" not in dataframe.columns:
            raise ValueError("Momentum column not found in dataframe")

        momentum_classes = dataframe["momentum"]

        # shift so that we get the future values for training
        momentum_classes = pd.Series(momentum_classes).shift(-self.lookahead_window + 1)
        momentum_classes = momentum_classes.fillna(MomentumDirection.STABLE)
        momentum_classes = momentum_classes.astype(int)

        return momentum_classes

    # -----------

    # Profit

    PROFIT_RANGE = 0.15  # Use this to cap the max expected profit magnitude

    def get_profit_target(self, dataframe: DataFrame) -> np.ndarray:
        """Triple-barrier profit label.

        Scan the next ``PROFIT_HORIZON`` bars; label by which barrier the
        smoothed close hits FIRST:
          - PROFIT if forward gain reaches +pt_eff before forward loss reaches -sl_eff
          - LOSS   if forward loss reaches -sl_eff first
          - NEUTRAL if neither barrier is hit anywhere in the window

        Compared to the previous endpoint-at-t+H comparison: adjacent bars
        scan windows that overlap by ~96%, so labels cluster into runs
        instead of flipping bar-to-bar on a noisy endpoint sample. This is
        the standard "triple-barrier" labeling from Lopez de Prado.

        Set ``use_forward_peak_profit_label = True`` to use the
        path-independent peak-based labeling instead.
        """
        if getattr(self, "use_forward_peak_profit_label", False):
            return self._get_profit_target_peak(dataframe)

        # Note: "close", "high", "low" are raw columns, not normalized, so not checked
        df = dataframe
        n = len(df)
        classes = np.ones(n, dtype=int) * ProfitDirection.NEUTRAL

        horizon = int(self.PROFIT_HORIZON)
        pt = float(self.MIN_BUY_GAIN_THRESHOLD)
        sl = float(self.MIN_SELL_LOSS_THRESHOLD)

        close_raw = np.asarray(df["close"], dtype=float)
        close = (
            pd.Series(close_raw)
            .ewm(span=int(self.PROFIT_EMA_SPAN), adjust=False)
            .mean()
            .to_numpy()
        )

        # Volatility-adjusted thresholds using ATR percentage
        high = np.asarray(df["high"], dtype=float)
        low = np.asarray(df["low"], dtype=float)
        prev_close = np.roll(close_raw, 1)
        prev_close[0] = close_raw[0]
        tr1 = high - low
        tr2 = np.abs(high - prev_close)
        tr3 = np.abs(low - prev_close)
        tr = np.maximum(tr1, np.maximum(tr2, tr3))
        atr = pd.Series(tr).ewm(alpha=1.0 / 14, adjust=False).mean().to_numpy()
        atr_pct = atr / np.maximum(close_raw, 1e-12)

        atr_scale = float(self.PROFIT_ATR_SCALE)
        for t in range(n - 1):
            end_idx = min(t + horizon + 1, n)
            window = close[t + 1 : end_idx]
            if window.size == 0:
                continue

            entry = close[t]
            gains = (window - entry) / entry

            vol_adj = atr_pct[t] * atr_scale
            pt_eff = max(pt, vol_adj)
            sl_eff = max(sl, vol_adj)

            profit_hits = np.flatnonzero(gains >= pt_eff)
            loss_hits = np.flatnonzero(gains <= -sl_eff)
            first_profit = profit_hits[0] if profit_hits.size else np.iinfo(np.int64).max
            first_loss = loss_hits[0] if loss_hits.size else np.iinfo(np.int64).max

            if first_profit == first_loss:  # neither hit (both sentinel)
                classes[t] = ProfitDirection.NEUTRAL
            elif first_profit < first_loss:
                classes[t] = ProfitDirection.PROFIT
            else:
                classes[t] = ProfitDirection.LOSS

        # Note that we cannot add this to the main dataframe
        # because it is inherently looking ahead in time
        return classes

    def _get_profit_target_peak(self, dataframe: DataFrame) -> np.ndarray:
        """Path-independent peak-based profit label.

        For each bar t, look at the next ``PROFIT_HORIZON`` bars:
          - PROFIT  if max forward gain ≥ +pt_eff (window had upside,
                    regardless of whether downside also occurred)
          - LOSS    if max forward gain < +pt_eff AND min forward gain
                    ≤ -sl_eff (window had ONLY downside)
          - NEUTRAL otherwise (neither barrier reached)

        Both-hit bars are labeled PROFIT — the upside was capturable, so a
        buy-side filter should let those through; the strategy's stop-loss
        handles the downside.
        """
        df = dataframe
        n = len(df)
        classes = np.ones(n, dtype=int) * ProfitDirection.NEUTRAL

        horizon = int(self.PROFIT_HORIZON)
        pt = float(self.MIN_BUY_GAIN_THRESHOLD)
        sl = float(self.MIN_SELL_LOSS_THRESHOLD)

        close_raw = np.asarray(df["close"], dtype=float)
        close = (
            pd.Series(close_raw)
            .ewm(span=int(self.PROFIT_EMA_SPAN), adjust=False)
            .mean()
            .to_numpy()
        )

        high = np.asarray(df["high"], dtype=float)
        low = np.asarray(df["low"], dtype=float)
        prev_close = np.roll(close_raw, 1)
        prev_close[0] = close_raw[0]
        tr = np.maximum(
            high - low,
            np.maximum(np.abs(high - prev_close), np.abs(low - prev_close)),
        )
        atr = pd.Series(tr).ewm(alpha=1.0 / 14, adjust=False).mean().to_numpy()
        atr_pct = atr / np.maximum(close_raw, 1e-12)

        atr_scale = float(self.PROFIT_ATR_SCALE)
        for t in range(n - 1):
            end_idx = min(t + horizon + 1, n)
            window = close[t + 1 : end_idx]
            if window.size == 0:
                continue

            entry = close[t]
            gains = (window - entry) / entry

            vol_adj = atr_pct[t] * atr_scale
            pt_eff = max(pt, vol_adj)
            sl_eff = max(sl, vol_adj)

            max_gain = float(gains.max())
            max_loss = float(gains.min())

            if max_gain >= pt_eff:
                classes[t] = ProfitDirection.PROFIT
            elif max_loss <= -sl_eff:
                classes[t] = ProfitDirection.LOSS

        return classes

    # -----------

    def add_additional_indicators(self, dataframe: DataFrame):
        """Add any additional indicators to the dataframe"""

        # we also want the parent class to add its indicators
        dataframe = super().add_additional_indicators(dataframe)

        # Initialize training indicators
        dataframe["%train_buy"] = 0
        dataframe["%train_sell"] = 0

        # Add missing indicators that lookahead analysis expects (with default values)
        dataframe["%train_trading"] = 1
        dataframe["%train_risk"] = 1
        dataframe["%train_momentum"] = 1
        dataframe["%train_regime"] = 1
        dataframe["%train_flow"] = 1
        dataframe["%train_profit"] = 1
        dataframe["%trading"] = 1
        dataframe["%risk"] = 1
        dataframe["%momentum"] = 1
        dataframe["%regime"] = 1
        dataframe["%flow"] = 1
        dataframe["%profit"] = 1
        dataframe["enter_tag"] = ""
        dataframe["enter_long"] = 0
        dataframe["exit_tag"] = ""
        dataframe["exit_long"] = 0
        dataframe["predict_buy"] = 0
        dataframe["predict_sell"] = 0
        self.dbg_curr_df = dataframe

        # if "profit" not in self.include_list:
        #     self.include_list.append("profit")

        # print(f"DEBUG: dataframe columns: {dataframe.columns}")
        return dataframe

    # -----------

    class_weights = {}

    def get_training_labels(self, dataframe: DataFrame):

        profit_targets = self.get_profit_target(dataframe)
        regime_targets = self.get_market_target(dataframe)
        momentum_targets = self.get_momentum_target(dataframe)
        risk_targets = self.get_risk_target(dataframe)
        flow_targets = self.get_flow_target(dataframe)
        trading_targets = self.get_trading_classes(
            dataframe,
            profit_targets,
            regime_targets,
            momentum_targets,
            risk_targets,
            flow_targets,
        )

        # DEBUG: check that all tasks have the same length:
        if (
            len(profit_targets) != len(regime_targets)
            or len(profit_targets) != len(momentum_targets)
            or len(profit_targets) != len(risk_targets)
            or len(profit_targets) != len(flow_targets)
            or len(profit_targets) != len(trading_targets)
        ):
            self.debug_print("All tasks must have the same length")
            self.debug_print(f"profit_targets length: {len(profit_targets)}")
            self.debug_print(f"regime_targets length: {len(regime_targets)}")
            self.debug_print(f"momentum_targets length: {len(momentum_targets)}")
            self.debug_print(f"risk_targets length: {len(risk_targets)}")
            self.debug_print(f"flow_targets length: {len(flow_targets)}")
            self.debug_print(f"trading_targets length: {len(trading_targets)}")

        labels = {}
        labels["profit"] = profit_targets
        labels["regime"] = regime_targets
        labels["momentum"] = momentum_targets
        labels["risk"] = risk_targets
        labels["flow"] = flow_targets
        labels["trading"] = trading_targets

        # save class weights for later
        self.class_weights = {}
        self.class_weights["profit"] = self.get_class_weights(profit_targets)
        self.class_weights["regime"] = self.get_class_weights(regime_targets)
        self.class_weights["momentum"] = self.get_class_weights(momentum_targets)
        self.class_weights["risk"] = self.get_class_weights(risk_targets)
        self.class_weights["flow"] = self.get_class_weights(flow_targets)
        self.class_weights["trading"] = self.get_class_weights(trading_targets)

        # DEBUG: copy training data into debug columns of the main dataframe
        self.dbg_curr_df = dataframe
        offset = np.shape(self.dbg_curr_df)[0] - len(profit_targets)
        self.dbg_curr_df.loc[offset:, "%train_profit"] = profit_targets
        self.dbg_curr_df.loc[offset:, "%train_regime"] = regime_targets
        self.dbg_curr_df.loc[offset:, "%train_momentum"] = momentum_targets
        self.dbg_curr_df.loc[offset:, "%train_risk"] = risk_targets
        self.dbg_curr_df.loc[offset:, "%train_flow"] = flow_targets
        self.dbg_curr_df.loc[offset:, "%train_trading"] = trading_targets

        return labels

    # -----------

    def get_training_class_weights(self, train_labels=None, validation_labels=None):
        """Get the class weights for the training data

        Args:
            train_labels: Augmented training labels (balanced distribution) - used for weight calculation
            validation_labels: Validation/test labels (real-world imbalanced distribution) - NOT used for weight calculation

        Note: We calculate weights from train_labels (after augmentation) because they represent the
        actual distribution the model will see during training. This ensures Focal Loss is
        calibrated for the data it's actually processing.
        """
        labels_to_use = train_labels

        if labels_to_use is None:
            # Fall back to train_labels if validation_labels not provided (backward compatibility)
            labels_to_use = train_labels

        if labels_to_use is None:
            # Fall back to stored labels if available
            if (
                hasattr(self, "_augmented_labels")
                and self._augmented_labels is not None
            ):
                labels_to_use = self._augmented_labels

        # If we have labels (either from parameter or stored), recalculate weights from them
        if labels_to_use is not None and isinstance(labels_to_use, dict):
            # Determine which labels we're using for clear logging
            print(f"    Calculating class weights from augmented training data...")
            # Recalculate weights based on labels
            calculated_weights = {}
            for task_name, task_labels in labels_to_use.items():
                # Convert one-hot to class indices
                class_indices = task_labels.argmax(axis=1)
                counts = np.bincount(class_indices, minlength=3)
                print(
                    f"      {task_name} distribution: {counts} [{counts[0]/len(class_indices)*100:.1f}%, {counts[1]/len(class_indices)*100:.1f}%, {counts[2]/len(class_indices)*100:.1f}%]"
                )
                calculated_weights[task_name] = self.get_class_weights(class_indices)
            self.class_weights = calculated_weights

        return self.class_weights

    # -----------

    def prepare_training_data(
        self,
        dataframes: List[DataFrame],
        labels_list,
        norm: bool = True,
        pair_names: Optional[List[str]] = None,
    ):
        """Prepare the training data"""

        if not isinstance(dataframes, (list, tuple)):
            dataframes = [dataframes]
        if not isinstance(labels_list, (list, tuple)):
            labels_list = [labels_list]

        if len(dataframes) == 0:
            raise ValueError("No dataframes supplied to prepare_training_data")

        if len(dataframes) != len(labels_list):
            raise ValueError(
                "Mismatched dataframe/label counts in prepare_training_data"
            )

        aggr_tsr_train = None
        aggr_tsr_test = None
        aggr_train_labels: Optional[Dict[str, np.ndarray]] = None
        aggr_test_labels: Optional[Dict[str, np.ndarray]] = None

        for pair_idx, dataframe in enumerate(dataframes):
            labels = labels_list[pair_idx]
            if not isinstance(labels, dict):
                raise ValueError("Multi-task labels must be a dictionary per pair")

            pair_labels = {task: np.asarray(values) for task, values in labels.items()}
            if norm:
                df_norm = self.scale_dataframe(dataframe)
            else:
                df_norm = dataframe.copy()

            min_length = min(
                [len(df_norm)] + [len(values) for values in pair_labels.values()]
            )
            if min_length <= self.seq_len:
                self.debug_print(
                    f"    Skipping pair {pair_idx} due to insufficient data ({min_length} rows)"
                )
                continue

            df_norm = df_norm.iloc[:min_length].reset_index(drop=True)
            for task in pair_labels:
                pair_labels[task] = pair_labels[task][:min_length]

            split_idx = int(self.TRAIN_DATA_SPLIT * len(df_norm))
            buffer_size = self.seq_len - 1
            train_end = max(split_idx - buffer_size, self.seq_len)
            test_start = train_end

            train_df = df_norm.iloc[:train_end].reset_index(drop=True)
            test_df = df_norm.iloc[test_start:].reset_index(drop=True)

            train_label_segment = {
                task: values[:train_end] for task, values in pair_labels.items()
            }
            test_label_segment = {
                task: values[test_start:] for task, values in pair_labels.items()
            }

            train_df, train_label_segment = self.enhance_training_data(
                train_df, train_label_segment
            )

            train_one_hot = {
                task: self.dataframeUtils.one_hot_encode(
                    np.asarray(vals).astype(int), 3
                )
                for task, vals in train_label_segment.items()
            }
            test_one_hot = {
                task: self.dataframeUtils.one_hot_encode(
                    np.asarray(vals).astype(int), 3
                )
                for task, vals in test_label_segment.items()
            }

            tsr_train = self.dataframeUtils.df_to_tensor(
                train_df, self.seq_len, method=0
            )
            tsr_test = self.dataframeUtils.df_to_tensor(test_df, self.seq_len, method=0)

            offset = self.seq_len - 1
            for task in train_one_hot:
                train_one_hot[task] = train_one_hot[task][offset:]
                test_one_hot[task] = test_one_hot[task][offset:]

            if len(tsr_train) == 0 or len(tsr_test) == 0:
                self.debug_print(
                    f"    Skipping pair {pair_idx} due to zero-length tensors after windowing"
                )
                continue

            if aggr_tsr_train is None:
                aggr_tsr_train = tsr_train
                aggr_tsr_test = tsr_test
                aggr_train_labels = {
                    task: train_one_hot[task] for task in train_one_hot
                }
                aggr_test_labels = {task: test_one_hot[task] for task in test_one_hot}
            else:
                aggr_tsr_train = np.concatenate([aggr_tsr_train, tsr_train], axis=0)
                aggr_tsr_test = np.concatenate([aggr_tsr_test, tsr_test], axis=0)
                for task in train_one_hot:
                    aggr_train_labels[task] = np.concatenate(
                        [aggr_train_labels[task], train_one_hot[task]], axis=0
                    )
                    aggr_test_labels[task] = np.concatenate(
                        [aggr_test_labels[task], test_one_hot[task]], axis=0
                    )

        if (
            aggr_tsr_train is None
            or aggr_tsr_test is None
            or aggr_train_labels is None
            or aggr_test_labels is None
        ):
            raise ValueError("No valid training data produced in prepare_training_data")

        self._augmented_labels = aggr_train_labels

        return aggr_tsr_train, aggr_tsr_test, aggr_train_labels, aggr_test_labels

    # -----------

    def preprocess_training_data(
        self, dataframe, train_data, test_data, train_labels, test_labels
    ):
        """Multi-task GAN augmentation against the windowed 3-D tensor.

        Single-task strategies augment in BaseNNStrategy.enhance_training_data
        (operating on the pre-windowed 2-D DataFrame). Multi-task strategies
        cannot do that because the GAN backends (e.g. MT_CTAB_GAN) are
        trained on flattened 16-step windows: each generated row is a
        whole sequence, shape ``(n, T*F)``. The natural pairing is the
        post-windowing 3-D tensor produced by prepare_training_data,
        which is what arrives here.

        Flow:
          1. Bail out unless gan_augment is on, gan_type is multi-task,
             labels look like a multi-task dict, and train_data is a 3-D
             tensor.
          2. Load the GAN with strict metadata validation (same checks
             BaseNNStrategy.enhance_training_data uses).
          3. Wrap the interface so generate() returns (n, T, F) ndarrays
             instead of the (n, T*F) DataFrames the backend natively
             produces. The reshape interprets the flat columns as
             ``[t=0 features..., t=1 features..., ..., t=T-1 features]``,
             matching how the GAN was trained on flattened windows.
          4. Resolve passthrough column names to integer indices into
             the F axis -- the 3-D path of swap_passthrough_columns
             swaps whole (T,) sequences per index, preserving temporal
             structure of deterministic features (calendar sin/cos).
          5. Hand the 3-D real tensor + dict-of-one-hot labels to
             balance_multi_task. It iterates, calling the wrapped
             interface, swapping passthrough columns, and concatenating
             along the leading axis so the augmented output stays in
             (N_total, T, F) form.

        Returns the augmented (train_data, test_data, train_labels,
        test_labels) -- only train_data and train_labels change; test
        data passes through untouched.
        """
        # Pass-through if augmentation is disabled or this isn't the
        # multi-task path. Anything we don't handle here keeps the
        # parent's no-op behavior.
        if self.gan_type == GANType.NONE or not getattr(self, "gan_augment", False):
            return super().preprocess_training_data(
                dataframe, train_data, test_data, train_labels, test_labels
            )
        if self.gan_type not in self._MULTI_TASK_GAN_TYPES:
            return super().preprocess_training_data(
                dataframe, train_data, test_data, train_labels, test_labels
            )
        if not isinstance(train_labels, dict) or not train_labels:
            return super().preprocess_training_data(
                dataframe, train_data, test_data, train_labels, test_labels
            )
        if not isinstance(train_data, np.ndarray) or train_data.ndim != 3:
            self.debug_print(
                f"    preprocess_training_data: expected 3-D ndarray train_data, "
                f"got {type(train_data).__name__} "
                f"shape={getattr(train_data, 'shape', None)} -- skipping multi-task GAN balance"
            )
            return super().preprocess_training_data(
                dataframe, train_data, test_data, train_labels, test_labels
            )
        if train_data.shape[0] == 0:
            return super().preprocess_training_data(
                dataframe, train_data, test_data, train_labels, test_labels
            )

        # Lazy imports -- the GAN stack pulls in TF / MLX which we don't
        # want to import for strategies that never enable augmentation.
        from GANs.GANInterface import GANInterface, GANMetadataMismatchError  # noqa: E402
        from GANs.paths import gan_save_path  # noqa: E402

        save_path = gan_save_path(
            self.get_storage_location(),
            self.gan_type,
            use_pca=bool(getattr(self, "use_pca_reduction", False)),
            post_gan_scaling=bool(getattr(self, "use_post_gan_scaling", False)),
        )
        interface = GANInterface(self.gan_type, save_path=save_path)
        expected = self._gan_expected_metadata(dataframe)
        try:
            interface.load(expected=expected)
        except GANMetadataMismatchError:
            raise
        except FileNotFoundError as load_err:
            raise RuntimeError(
                f"GAN model not found at {save_path}. "
                f"Train it first via the corresponding Create* strategy "
                f"(gan_type={self.gan_type.name}). "
                f"Underlying error: {load_err}"
            ) from load_err
        except Exception as load_err:
            raise RuntimeError(
                f"Failed to load GAN model at {save_path}: {load_err}"
            ) from load_err

        T, F = int(train_data.shape[1]), int(train_data.shape[2])
        wrapped_interface = _UnflattenedGenerateWrapper(interface, T=T, F=F)

        aug_train_data, aug_train_labels = self._invoke_balance_multi_task(
            wrapped_interface,
            train_data,
            train_labels,
            dataframe=dataframe,
        )

        return aug_train_data, test_data, aug_train_labels, test_labels

    # -----------

    def get_predictions(self, dataframe: DataFrame, classifier: ClassifierKeras):
        """Get the predictions from the model"""

        # empty dictionary for use in error cases
        dlen = np.shape(dataframe)[0]
        predictions_dict = {}
        predictions_dict["trading"] = np.ones(dlen, dtype=int)
        predictions_dict["regime"] = np.ones(dlen, dtype=int)
        predictions_dict["risk"] = np.ones(dlen, dtype=int)
        predictions_dict["momentum"] = np.ones(dlen, dtype=float)
        predictions_dict["flow"] = np.ones(dlen, dtype=float)
        predictions_dict["profit"] = np.ones(dlen, dtype=float)

        if classifier is None:
            # print("    no classifier for predictions")
            raise Exception("No classifier for predictions")

        # Get multi-task predictions
        try:
            # # Debug: print data shape before prediction
            # self.debug_print(f"    DEBUG: Input data shape: {data.shape}")
            # self.debug_print(f"    DEBUG: Actual features: {data.shape[-1]}")
            df_norm = self.scale_dataframe(dataframe)
            df_tensor = self.dataframeUtils.df_to_tensor(
                df_norm, self.seq_len, method=0
            )
            multi_predictions = classifier.predict(df_tensor)

        except Exception as e:
            log.error(f"    Prediction failed: {e}")
            self.debug_print(f"    ERROR: Prediction failed: {e}")
            self.debug_print(f"    Exception type: {type(e)}")
            self.debug_print(f"    Traceback: {traceback.format_exc()}")
            return predictions_dict

        # Debug: print what we're getting

        self.debug_print(
            f"    DEBUG: multi_predictions keys: {list(multi_predictions.keys())}"
        )

        # extract arrays from dictionary
        try:
            trading_predictions = multi_predictions["trading"]
            regime_predictions = multi_predictions["regime"]
            risk_predictions = multi_predictions["risk"]
            momentum_predictions = multi_predictions["momentum"]
            flow_predictions = multi_predictions["flow"]
            profit_predictions = multi_predictions["profit"]

        except (KeyError, IndexError, TypeError) as e:
            log.error(f"    Failed to extract predictions: {e}")
            self.debug_print(f"    ERROR: Failed to extract predictions: {e}")
            self.debug_print(f"    multi_predictions: {multi_predictions}")
            return predictions_dict

        # Ensure predictions are numpy arrays and properly shaped
        trading_predictions = np.array(trading_predictions)
        regime_predictions = np.array(regime_predictions)
        risk_predictions = np.array(risk_predictions)
        momentum_predictions = np.array(momentum_predictions)
        flow_predictions = np.array(flow_predictions)
        profit_predictions = np.array(profit_predictions)

        # Clean predictions (handle NaN/Inf only)
        trading_predictions = np.nan_to_num(
            trading_predictions, nan=0.0, posinf=0.0, neginf=0.0
        )
        regime_predictions = np.nan_to_num(
            regime_predictions, nan=0.0, posinf=0.0, neginf=0.0
        )
        risk_predictions = np.nan_to_num(
            risk_predictions, nan=0.0, posinf=0.0, neginf=0.0
        )
        profit_predictions = np.nan_to_num(
            profit_predictions, nan=0.0, posinf=0.0, neginf=0.0
        )
        momentum_predictions = np.nan_to_num(
            momentum_predictions, nan=0.0, posinf=0.0, neginf=0.0
        )
        flow_predictions = np.nan_to_num(
            flow_predictions, nan=0.0, posinf=0.0, neginf=0.0
        )

        # convert the probability matrices into classes
        # Note the use of hyperparameters for the tasks that use bias

        trading_threshold = self.prediction_threshold.value
        pred_threshold = min(trading_threshold, self.profit_prediction_threshold.value)
        # print(f"    prediction threshold: {pred_threshold}")

        predictions_dict["trading"] = self.argmax_with_threshold(
            trading_predictions,
            threshold=trading_threshold,
            default_class=TradingAction.HOLD,
        )
        # predictions_dict["trading"] = self.argmax_with_bias(
        #     trading_predictions,
        #     bias_map={0: self.bias_trading_sell.value, 2: self.bias_trading_buy.value},
        #     threshold=pred_threshold,
        #     default_class=TradingAction.HOLD,
        # )
        predictions_dict["regime"] = self.argmax_with_threshold(
            regime_predictions,
            threshold=pred_threshold,
            default_class=MarketRegime.SIDEWAYS,
        )
        predictions_dict["risk"] = self.argmax_with_threshold(
            risk_predictions, threshold=pred_threshold, default_class=RiskLevel.NORMAL
        )
        predictions_dict["momentum"] = self.argmax_with_threshold(
            momentum_predictions,
            threshold=pred_threshold,
            default_class=MomentumDirection.STABLE,
        )
        predictions_dict["flow"] = self.argmax_with_threshold(
            flow_predictions,
            threshold=pred_threshold,
            default_class=FlowDirection.NEUTRAL,
        )
        predictions_dict["profit"] = self.argmax_with_threshold(
            profit_predictions,
            threshold=pred_threshold,
            default_class=ProfitDirection.NEUTRAL,
        )
        # predictions_dict["profit"] = self.argmax_with_bias(
        #     profit_predictions,
        #     bias_map={0: self.bias_profit_low.value, 2: self.bias_profit_high.value},
        #     threshold=pred_threshold,
        #     default_class=ProfitDirection.NEUTRAL,
        # )

        # DEBUG:
        self.print_probability_stats(
            "Trading", "Sell", profit_predictions[:, TradingAction.SELL], trading_threshold
        )
        self.print_probability_stats(
            "Trading", "Hold", profit_predictions[:, TradingAction.HOLD], trading_threshold
        )
        self.print_probability_stats(
            "Trading", "Buy", profit_predictions[:, TradingAction.BUY], trading_threshold
        )
        return predictions_dict

    # -----------

    def process_predictions(self, dataframe: DataFrame, predictions):
        """Process the predictions. Ideally, set up dataframe["predict_buy"] and dataframe["predict_sell"]"""

        trading_predictions = predictions["trading"]
        regime_predictions = predictions["regime"]
        risk_predictions = predictions["risk"]
        momentum_predictions = predictions["momentum"]
        flow_predictions = predictions["flow"]
        profit_predictions = predictions["profit"]

        apply_task_filters = self.apply_task_filters.value
        if apply_task_filters:
            trading_predictions = self._filter_trading_by_tasks(
                trading_predictions,
                profit_predictions,
                regime_predictions,
                momentum_predictions,
                flow_predictions,
                risk_predictions,
            )

        if getattr(self, "use_forward_peak_regressor_filter", False):
            trading_predictions = self._filter_trading_by_forward_peak(
                trading_predictions, dataframe,
            )

        # Filter for consecutive buy predictions to reduce noise (VECTORIZED)
        buy_signals = trading_predictions == TradingAction.BUY
        min_consecutive = self.min_consecutive_buys.value

        if min_consecutive > 0:
            # Find the start and end of each consecutive sequence
            # Pad with False to handle edge cases
            padded_signals = np.concatenate([[False], buy_signals, [False]])

            # Find transitions: True->False and False->True
            transitions = np.diff(padded_signals.astype(int))
            starts = np.where(transitions == 1)[0]  # False->True
            ends = np.where(transitions == -1)[0]  # True->False

            # Create mask for valid sequences (length >= min_consecutive)
            valid_buy_mask = np.zeros_like(buy_signals, dtype=bool)

            for start, end in zip(starts, ends):
                seq_length = end - start
                if seq_length >= min_consecutive:
                    # Mark all positions in this sequence as valid
                    valid_buy_mask[start:end] = True

            # Convert isolated single buy signals back to hold
            trading_predictions[buy_signals & ~valid_buy_mask] = TradingAction.HOLD

        # DEBUG: this is a bit naughty, but add results to the current dataframe (not supposed to know this)
        offset = len(dataframe) - len(trading_predictions)
        dataframe["%trading"] = 0.0
        dataframe["%regime"] = 0.0
        dataframe["%momentum"] = 0.0
        dataframe["%risk"] = 0.0
        dataframe["%flow"] = 0.0
        dataframe["%profit"] = 0.0
        # Use the FINAL predictions (after all filtering and conversions), not raw argmax
        dataframe.loc[offset:, "%trading"] = trading_predictions
        dataframe.loc[offset:, "%regime"] = regime_predictions
        dataframe.loc[offset:, "%risk"] = risk_predictions

        dataframe.loc[offset:, "%momentum"] = momentum_predictions
        dataframe.loc[offset:, "%flow"] = flow_predictions
        dataframe.loc[offset:, "%profit"] = profit_predictions

        self.debug_print("    Model predictions, after filtering:")
        self.print_distribution_compact("  Trading", trading_predictions)
        self.print_distribution_compact("  Risk", risk_predictions)
        self.print_distribution_compact("  Market Regime", regime_predictions)
        self.print_distribution_compact("  Momentum", momentum_predictions)
        self.print_distribution_compact("  Flow", flow_predictions)
        self.print_distribution_compact("  Profit", profit_predictions)

        # add results to the main dataframe
        dataframe.loc[offset:, "predict_buy"] = np.where(
            trading_predictions == TradingAction.BUY, 1, 0
        )
        dataframe.loc[offset:, "predict_sell"] = np.where(
            trading_predictions == TradingAction.SELL, 1, 0
        )
        return dataframe

    # -----------
    # Forward-peak-gain regressor filter
    # -----------

    def _get_forward_peak_regressor(self, pair: str):
        """Lazy-load the per-pair regressor; cache the result.

        Caches ``False`` for pairs whose model file is missing so we don't
        hit the filesystem on every call. Returns the sklearn model or
        ``None`` if unavailable.
        """
        cached = self._forward_peak_regressor_cache.get(pair, None)
        if cached is False:
            return None
        if cached is not None:
            return cached

        storage_root = Path(self.get_storage_location())
        model = _FPR.load(_FPR.model_path(storage_root, pair))
        if model is None:
            log.warning(
                f"ForwardPeakRegressor: no model on disk for {pair} at "
                f"{_FPR.model_path(storage_root, pair)} — filter disabled "
                f"for this pair. Run TrainForwardPeakRegressor.py first."
            )
            self._forward_peak_regressor_cache[pair] = False
            return None
        self._forward_peak_regressor_cache[pair] = model
        return model

    def _filter_trading_by_forward_peak(
        self, trading_predictions: np.ndarray, dataframe: DataFrame,
    ) -> np.ndarray:
        """Zero out BUY predictions whose regressor score is below threshold.

        Sells are not filtered — the regressor is trained on upside peaks
        only (path-independent forward-max gain), so it has no view on the
        downside that would justify a sell.
        """
        pair = getattr(self, "curr_pair", None)
        if pair is None:
            return trading_predictions
        model = self._get_forward_peak_regressor(pair)
        if model is None:
            return trading_predictions

        # Use percentile ranks so the threshold has the same meaning across
        # pairs with very different vol-adjusted score scales.
        ranks = _FPR.predict_percentile(model, dataframe)
        # Predictions are bar-aligned to the tail of the dataframe (see
        # offset handling in process_predictions). Align here too.
        offset = len(ranks) - len(trading_predictions)
        if offset > 0:
            ranks = ranks[offset:]
        elif offset < 0:
            # Shouldn't happen — predictions vector is longer than dataframe.
            return trading_predictions

        threshold = float(self.forward_peak_regressor_threshold.value)
        weak_buy = (trading_predictions == TradingAction.BUY) & (ranks < threshold)
        out = trading_predictions.copy()
        out[weak_buy] = TradingAction.HOLD

        if int(weak_buy.sum()) > 0:
            self.debug_print(
                f"    ForwardPeakRegressor: rejected {int(weak_buy.sum())}/"
                f"{int((trading_predictions == TradingAction.BUY).sum())} BUY "
                f"signals on {pair} (percentile threshold={threshold:.2f})"
            )
        return out

    # ------------------------------------------------------------------ #
    # Model-prediction bailout                                            #
    # ------------------------------------------------------------------ #
    # When a trade is underwater, the model's per-bar predictions can flag
    # that the setup has turned bad before the trailing-stop triggers.
    # The hook below reads the latest %profit and %regime predictions
    # (written into the dataframe by process_predictions) and exits when
    # either flips to its adverse class. Catches falling-knife buys that
    # the entry-side filters can't pre-screen — the leverage is on the
    # exit side, where the model gets fresh evidence after entry.

    # Only consider bailout when the trade is at least this deep underwater.
    # Above this, normal profit-based exits handle things. Too tight and
    # bailouts fire on tick noise; too loose and the trailing-stop fires
    # first anyway. -1.5% is a starting point — trailing-stop bleeds at
    # -5.23%, so saving even -3% per stopped trade is a meaningful lever.
    bailout_min_loss: float = -1.0

    # Minimum bars in trade before bailout is allowed. Prevents bailing
    # on tick-zero noise where the model briefly disagrees with itself
    # immediately after entry. 4 bars on 15m = 1 hour.
    bailout_min_bars: int = 2

    # When True, require BOTH profit==LOSS AND regime==BEAR before
    # bailing (consensus mode). When False, either signal triggers exit
    # (more permissive — fires earlier but with more false positives
    # given the profit head's MCC ceiling of ~0.20).
    bailout_require_consensus: bool = False

    def strategy_custom_exit(
        self,
        pair: str,
        trade,
        current_time,
        current_rate: float,
        current_profit: float,
        dataframe,
        last_candle,
        **kwargs,
    ) -> Optional[str]:
        """Model-prediction bailout for underwater trades.

        Reads ``%profit`` and ``%regime`` from the latest candle (written
        by process_predictions). Returns an exit-reason string when the
        model's view of the setup has turned adverse; otherwise None to
        let the static stoploss / trailing-stop handle it.
        """
        if current_profit > self.bailout_min_loss:
            return None

        # Bars-in-trade gate — 15m candles assumed.
        time_delta = current_time - trade.open_date_utc
        minutes_in_trade = time_delta.total_seconds() / 60.0
        if minutes_in_trade < self.bailout_min_bars * 15:
            return None

        try:
            profit_pred = int(last_candle.get("%profit", ProfitDirection.NEUTRAL))
            regime_pred = int(last_candle.get("%regime", MarketRegime.SIDEWAYS))
        except (TypeError, ValueError):
            return None

        profit_says_loss = profit_pred == int(ProfitDirection.LOSS)
        regime_says_bear = regime_pred == int(MarketRegime.BEAR)

        if self.bailout_require_consensus:
            should_bail = profit_says_loss and regime_says_bear
        else:
            should_bail = profit_says_loss or regime_says_bear

        if should_bail:
            tag = "bailout_consensus" if (profit_says_loss and regime_says_bear) else (
                "bailout_profit" if profit_says_loss else "bailout_regime"
            )
            return tag
        return None

    def _filter_trading_by_tasks(
        self,
        trading_predictions: np.ndarray,
        profit_predictions: np.ndarray,
        regime_predictions: np.ndarray,
        momentum_predictions: np.ndarray,
        flow_predictions: np.ndarray,
        risk_predictions: np.ndarray,
    ) -> np.ndarray:
        """Apply task-aligned filters to trading predictions."""

        """
        trading_buy_mask = (trading_predictions == TradingAction.BUY)
        trading_sell_mask = (trading_predictions == TradingAction.SELL)

        required_matches = 3

        buy_conditions = np.stack(
            [
                profit_predictions == ProfitDirection.PROFIT,
                regime_predictions == MarketRegime.BULL,
                momentum_predictions == MomentumDirection.POSITIVE,
                flow_predictions == FlowDirection.INCREASE,
                risk_predictions == RiskLevel.LOW,
            ],
            axis=0,
        )
        sell_conditions = np.stack(
            [
                profit_predictions == ProfitDirection.LOSS,
                regime_predictions == MarketRegime.BEAR,
                momentum_predictions == MomentumDirection.NEGATIVE,
                flow_predictions == FlowDirection.DECREASE,
                risk_predictions == RiskLevel.HIGH,
            ],
            axis=0,
        )

        buy_mask = trading_buy_mask & (buy_conditions.sum(axis=0) >= required_matches)
        sell_mask = trading_sell_mask & (sell_conditions.sum(axis=0) >= required_matches)
        # buy_mask = (buy_conditions.sum(axis=0) >= required_matches)
        # sell_mask = (sell_conditions.sum(axis=0) >= required_matches-1)
        """

        # Magnitude gate: the profit head explicitly predicts whether the
        # forward window will reach the gain/loss thresholds — it's the
        # closest thing we have to a per-trade magnitude predictor without
        # adding a separate model. trading=BUY tells us the model thinks
        # "this is a buy setup at the current label margin (0.3%)", but
        # profit=PROFIT tells us "the forward window will reach a profitable
        # magnitude" — that's the second-stage filter we want.
        #
        # See feedback in the May 2026 backtest comparison: raising
        # prediction_threshold didn't help because it filters by confidence,
        # not by predicted magnitude. The profit head is the magnitude proxy.
        # Buy: 3-of-4 majority — at most one head may contradict.
        # Sell: 2-of-4 majority — looser, so the model_exit can fire
        # more often (an early exit at small profit/loss is preferable
        # to letting trades run all the way to trailing_stop).
        buy_conditions = np.stack(
            [
                profit_predictions != ProfitDirection.LOSS,
                regime_predictions != MarketRegime.BEAR,
                momentum_predictions != MomentumDirection.NEGATIVE,
                flow_predictions != FlowDirection.DECREASE,
            ],
            axis=0,
        )
        sell_conditions = np.stack(
            [
                profit_predictions != ProfitDirection.PROFIT,
                regime_predictions != MarketRegime.BULL,
                momentum_predictions != MomentumDirection.POSITIVE,
                flow_predictions != FlowDirection.INCREASE,
            ],
            axis=0,
        )

        buy_mask = (
            (trading_predictions == TradingAction.BUY)
            & (buy_conditions.sum(axis=0) >= 3)
        )
        # sell_mask = (
        #     (trading_predictions == TradingAction.SELL)
        #     & (sell_conditions.sum(axis=0) >= 2)
        # )
        sell_mask = (
            (trading_predictions == TradingAction.SELL)
        )


        # Reset all to HOLD, then set BUY/SELL only where both conditions are met
        filtered = np.full_like(trading_predictions, TradingAction.HOLD)
        filtered[buy_mask] = TradingAction.BUY
        filtered[sell_mask] = TradingAction.SELL
        return filtered

    def get_trading_classes(
        self,
        dataframe: DataFrame,
        profit_targets: np.ndarray,
        regime_targets: np.ndarray,
        momentum_targets: np.ndarray,
        risk_targets: np.ndarray,
        flow_targets: np.ndarray,
    ) -> np.ndarray:

        # Calculate profit potential (similar to original NNNC)
        future_df = dataframe.copy()

        profit_series = profit_targets

        # Initialize profit class array (HOLD by default)
        trading_classes = np.ones(len(profit_series), dtype=int) * TradingAction.HOLD

        buy_signals = self.get_train_buy_signals(future_df)
        sell_signals = self.get_train_sell_signals(future_df)

        # augment, if needed
        buy_signals, sell_signals = self.augment_training_signals(
            buy_signals, sell_signals
        )

        # set initially basd on training signals
        trading_classes[buy_signals.astype(bool)] = TradingAction.BUY
        trading_classes[sell_signals.astype(bool)] = TradingAction.SELL

        # apply any filters based on the other tasks
        filtered_trading_classes = self._filter_trading_by_tasks(
            trading_classes,
            np.asarray(profit_targets),
            np.asarray(regime_targets),
            np.asarray(momentum_targets),
            np.asarray(flow_targets),
            np.asarray(risk_targets),
        )

        buy_signals = filtered_trading_classes == TradingAction.BUY
        sell_signals = filtered_trading_classes == TradingAction.SELL

        self.dbg_curr_df["%train_buy"] = buy_signals.astype(int)
        self.dbg_curr_df["%train_sell"] = sell_signals.astype(int)

        # TODO: use other tasks to filter trading decisions (?!)

        # Note that we cannot add this to the main dataframe
        # because it is inherently looking ahead in time
        return trading_classes

    # -----------
    # Shared GAN-augmentation helpers (hoisted from the identical copies in
    # NNMT_DDPM / NNMT_WGAN). Subclasses with richer classifier wiring (e.g.
    # NNMT_MLX) override _apply_classifier_overrides; the 3-D preprocess hooks
    # call these.
    # -----------

    def _apply_classifier_overrides(self, clf) -> None:
        """Push strategy-level classifier overrides onto a freshly-built
        classifier. Each knob reads a ``_CLASSIFIER_*`` class attribute (or
        ``learning_rate``) off the strategy and is skipped when unset, so
        subclasses tune by setting attributes rather than re-implementing this.
        ``_CLASSIFIER_ENTROPY_PENALTY`` is only used by the MLX multi-task
        classifier; strategies that don't set it are unaffected.
        """
        if clf is None:
            return
        lr = getattr(self, "learning_rate", None)
        if lr is not None:
            clf.learning_rate = lr
        weights = getattr(self, "_CLASSIFIER_TASK_WEIGHTS", None)
        if weights:
            clf.task_weights_override = weights
        max_epochs = getattr(self, "_CLASSIFIER_MAX_EPOCHS", None)
        if max_epochs is not None:
            clf.max_epochs = int(max_epochs)
        entropy_penalty = getattr(self, "_CLASSIFIER_ENTROPY_PENALTY", None)
        if entropy_penalty is not None:
            # Pass dict through unchanged so the classifier can route
            # per-task; coerce scalars to float for the uniform form.
            if isinstance(entropy_penalty, dict):
                clf.entropy_penalty_weight = dict(entropy_penalty)
            else:
                clf.entropy_penalty_weight = float(entropy_penalty)

    def _balance_iteratively(
        self,
        interface,
        train_minmax: np.ndarray,
        train_labels: Dict[str, np.ndarray],
    ) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
        """Delegate to ``BaseNNStrategy._invoke_balance_multi_task``.

        Thin wrapper kept so subclasses can override scheduling without
        touching the augmentation policy itself.
        """
        return self._invoke_balance_multi_task(
            interface,
            train_minmax,
            train_labels,
        )

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
