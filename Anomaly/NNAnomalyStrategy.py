# pragma pylint: disable=C0103, C0114, C0115, C0116, C0301, C0302, C0303, C0325, C0411, C0413
# pragma pylint: disable=W0105, W1203, W1309, W1514, W0613, W0621,
# type: ignore
# pylint: disable=import-error
# flake8: noqa: F401, E402, F541, W0718, W0719

"""
NNAnomalyStrategy - Base class for Neural Network Anomaly-based strategies
"""


import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from pandas import DataFrame
import numpy as np
import pandas as pd
import traceback
from enum import IntEnum

from freqtrade.strategy import DecimalParameter, IntParameter

import logging

log = logging.getLogger(__name__)

# Add parent directory to path to import NNNC
group_dir = str(Path(__file__).parent)
# Add parent directory to path to import NNNC
sys.path.append(group_dir)

from Framework.BaseNNStrategy import BaseNNStrategy, HAS_MLX, StrategyConfig
from utils.Environment import Environment
from Framework.BaseStrategy import BaseStrategy, ScalerType, MarketRegime, TradingAction, FlowDirection, MomentumDirection, RiskLevel, GANType
from Framework.BaseNNStrategy import BaseNNStrategy, HAS_MLX, StrategyConfig
from utils.Environment import Environment
from Framework.BaseStrategy import BaseStrategy, ScalerType, MarketRegime, TradingAction, FlowDirection, MomentumDirection, RiskLevel, GANType

from utils.ClassifierKerasAnomaly import ClassifierKerasAnomaly
import NNAnomalyClassifier


# -----------


class NNAnomalyStrategy(BaseNNStrategy):
    """
    Neural Network Multi-Task (NNMT) base strategy that predicts:
    - Latent space (for anomaly score)
    - Trading action (Sell/Hold/Buy) for trading decisions

    This is a base class for different multi-task model variants (LSTM, Transformer, etc.)
    Inherits from Framework.BaseNNStrategy import BaseNNStrategy, HAS_MLX, StrategyConfig
from utils.Environment import Environment
from Framework.BaseStrategy import BaseStrategy, ScalerType, MarketRegime, TradingAction, FlowDirection, MomentumDirection, RiskLevel, GANType
from Framework.BaseNNStrategy import BaseNNStrategy, HAS_MLX, StrategyConfig
from utils.Environment import Environment
from Framework.BaseStrategy import BaseStrategy, ScalerType, MarketRegime, TradingAction, FlowDirection, MomentumDirection, RiskLevel, GANType

    plot_config = {
        "main_plot": {
            "close": {"color": "lightsteelblue"},
        },
        "subplots": {
            "Diff": {
                "predict_buy": {"color": "blue"},
                "predict_sell": {"color": "purple"},
                "%train_trading": {"color": "green"},
                "%trading": {"color": "cyan"},
                "anomaly_threshold": {"color": "red"},
                "reconstruction_error": {"color": "orange"},
            },
        },
    }

    # -----------
    # Hyperopt parameters
    # -----------

    # Buy hyperspace params:
    buy_params = {
        # "entry_adx_threshold": 50.0,
        # "entry_bb_width_threshold": 0.05,
        "entry_adx_threshold": 0.0,
        "entry_bb_width_threshold": 0.0,
        "entry_guard_threshold": 1.0,
        "entry_close_norm_threshold": 0.0,
        "min_buy_gain_threshold": 0.007,
        "prediction_threshold": 0.5,
        "training_type": 16,
        "anomaly_threshold_multiplier": 1.8,
        "min_anomaly_duration": 2,
        "entry_error_threshold": 0.035,
    }

    # Sell hyperspace params:
    sell_params = {
        "cexit_enable_profit_checks": True,
        "cexit_max_days": 14,
        "cexit_take_profit": 0.024,
        "enable_exit_signal": True,
        "exit_guard_threshold": 0.5,
        "exit_close_norm_threshold": 0.0,
        "min_sell_loss_threshold": 0.009,
    }

    # Anomaly-specific hyperparams
    anomaly_threshold_multiplier = DecimalParameter(
        1.0, 2.5, default=1.8, decimals=1, space="buy", load=True, optimize=True
    )

    min_anomaly_duration = IntParameter(
        2, 8, default=2, space="buy", load=True, optimize=True
    )

    entry_error_threshold = DecimalParameter(
        0.01, 3.0, default=0.035, decimals=3, space="buy", load=True, optimize=True
    )

    # -----------
    # Class level parameters
    # -----------

    augment_training_data = False  # no GAn, so augment signals

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

        return cweights_array

    def print_distribution_compact(self, name: str, distribution: np.ndarray) -> None:
        counts = np.bincount(distribution, minlength=3)
        percentages = counts / len(distribution) * 100
        percent_str = (
            f"[{percentages[0]:.1f}%, {percentages[1]:.1f}%, {percentages[2]:.1f}%]"
        )
        self.debug_print(f"      {name} distribution: {counts} {percent_str}")

    def analyze_distribution(self, name: str, distribution: np.ndarray) -> None:
        """
        Calculates and prints key statistics for a distribution
        """

        self.debug_print("")

        # raw_min = np.min(distribution)
        # raw_max = np.max(distribution)
        # raw_mean = np.mean(distribution)
        # raw_std = np.std(distribution)
        # raw_5th = np.percentile(distribution, 5)
        # raw_95th = np.percentile(distribution, 95)

        self.debug_print(f"\n    {name} distribution:")
        self.debug_print(
            f"      Range (Min/Max): {np.min(distribution):.6f} to {np.max(distribution):.6f}"
        )
        self.debug_print(
            f"      Mean / Std Dev:  {np.mean(distribution):.6f} / {np.std(distribution):.6f}"
        )
        self.debug_print(
            f"      5th / 95th Pctl: {np.percentile(distribution, 5):.6f} / {np.percentile(distribution, 95):.6f}"
        )
        self.debug_print(
            f"      10th / 90th Pctl: {np.percentile(distribution, 10):.6f} / {np.percentile(distribution, 90):.6f}"
        )
        self.debug_print(
            f"      25th / 75th Pctl: {np.percentile(distribution, 25):.6f} / {np.percentile(distribution, 75):.6f}"
        )
        self.debug_print(
            f"      3rd / 2/3rd Pctl: {np.percentile(distribution, 33):.6f} / {np.percentile(distribution, 66):.6f}"
        )

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

    # Trading Action (more complicated because it uses data from the other tasks)

    def get_trading_classes(self, dataframe: DataFrame) -> np.ndarray:

        # Calculate profit potential
        future_df = dataframe.copy()

        df_len = np.shape(future_df)[0]
        # Initialize trading class array (HOLD by default)
        trading_classes = np.ones(df_len, dtype=int) * TradingAction.HOLD

        buy_signals = self.get_train_buy_signals(future_df)
        sell_signals = self.get_train_sell_signals(future_df)
        buy_signals, sell_signals = self.augment_training_signals(
            buy_signals, sell_signals
        )
        trading_classes[buy_signals.astype(bool)] = TradingAction.BUY
        trading_classes[sell_signals.astype(bool)] = TradingAction.SELL

        self.dbg_curr_df["%train_buy"] = buy_signals.astype(int)
        self.dbg_curr_df["%train_sell"] = sell_signals.astype(int)

        return trading_classes

    # -----------
    # Override functions
    # -----------

    def get_classifier_type(self):
        """ Return the type of classifier used for training/predicting """
        return NNAnomalyClassifier.ClassifierType.LSTM

    # -----------

    def get_classifier(self, classifier_type, pair, seq_len, num_features) -> ClassifierKerasAnomaly:
        """ Return the classifier used for training/predicting """

        classifier, _ = NNAnomalyClassifier.create_classifier(
                    classifier_type,
                    pair,
                    num_features,
                    seq_len
                )
        return classifier

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
        dataframe["%trading"] = 1 
        dataframe["enter_tag"] = ""
        dataframe["enter_long"] = 0
        dataframe["exit_tag"] = ""
        dataframe["exit_long"] = 0
        dataframe["predict_buy"] = 0
        dataframe["predict_sell"] = 0
        self.dbg_curr_df = dataframe

        # print(f"DEBUG: dataframe columns: {dataframe.columns}")
        return dataframe

    # -----------

    class_weights = {}
    def get_training_labels(self, dataframe: DataFrame):

        trading_targets = self.get_trading_classes(dataframe)

        labels = {}
        labels["reconstruction"] = None # fill in later
        labels["trading"] = trading_targets

        # save class weights for later
        self.class_weights = {}
        self.class_weights["trading"] = self.get_class_weights(trading_targets)

        # DEBUG: copy training data into debug columns of the main dataframe
        self.dbg_curr_df = dataframe
        offset = np.shape(self.dbg_curr_df)[0] - len(trading_targets)
        self.dbg_curr_df.loc[offset:, "%train_trading"] = trading_targets
        # print(f"    DEBUG: labels: {labels}")

        return labels

    # -----------

    def get_training_class_weights(self, train_labels=None, validation_labels=None):
        """Get the class weights for the training data"""
        # Use train_labels if provided (these are the augmented labels after preprocess_training_data)
        # Otherwise fall back to _augmented_labels or original class_weights
        # Note: validation_labels parameter accepted for compatibility with parent class, but not used in this implementation
        labels_to_use = train_labels

        if labels_to_use is None:
            # Fall back to stored augmented labels if available
            if hasattr(self, '_augmented_labels') and self._augmented_labels is not None:
                labels_to_use = self._augmented_labels

        # If we have labels (either from parameter or stored), recalculate weights from them
        if labels_to_use is not None and isinstance(labels_to_use, dict):
            print("    Recalculating class weights from augmented data distribution")
            # Recalculate weights based on augmented labels
            augmented_weights = {}
            for task_name, task_labels in labels_to_use.items():
                if task_name == "reconstruction":
                    continue
                # Convert one-hot to class indices
                class_indices = task_labels.argmax(axis=1)
                counts = np.bincount(class_indices, minlength=3)
                print(f"      {task_name} augmented distribution: {counts} [{counts[0]/len(class_indices)*100:.1f}%, {counts[1]/len(class_indices)*100:.1f}%, {counts[2]/len(class_indices)*100:.1f}%]")
                augmented_weights[task_name] = self.get_class_weights(class_indices)
            self.class_weights = augmented_weights
            # print(f"    Recalculated class weights: {self.class_weights}")

        #    # HACK: (?!) Artificially skew weights for difficult tasks to penalise middle class
        #     if self.augment_training_data:
        #         self.debug_print("    Augmenting training data with class weights")
        #         trading_weights = self.class_weights["trading"] * np.array([2.0, 0.5, 2.0])
        #         self.class_weights["trading"] = trading_weights

        return self.class_weights

    # -----------

    def enhance_training_data(
        self, train_df: DataFrame, train_labels: Dict[str, np.ndarray]
    ) -> Tuple[DataFrame, Dict[str, np.ndarray]]:
        """
        Hook for subclasses to augment per-pair training data. Returns the (possibly
        augmented) dataframe and label dictionary for the pair.
        """
        return train_df, train_labels

    def prepare_training_data(
        self, dataframes: List[DataFrame], labels_list: List[Any], norm: bool = True
    ):
        """Prepare the training data"""

        " A little different than most strategies - use the dataframe as one set of labels"

        if not isinstance(dataframes, (list, tuple)):
            dataframes = [dataframes]
        if not isinstance(labels_list, (list, tuple)):
            # print(f"    labels_list: {np.shape(labels_list)}")
            labels_list = [labels_list]

        if len(dataframes) == 0:
            raise ValueError("No dataframes supplied to prepare_training_data")

        if len(dataframes) != len(labels_list):
            raise ValueError("Mismatched dataframe/label counts in prepare_training_data")

        aggr_tsr_train = None
        aggr_tsr_test = None
        aggr_train_labels: Optional[Dict[str, np.ndarray]] = None
        aggr_test_labels: Optional[Dict[str, np.ndarray]] = None

        for pair_idx, dataframe in enumerate(dataframes):

            pair_labels = labels_list[pair_idx]["trading"]
            if norm:
                df_norm = self.scale_dataframe(dataframe)
            else:
                df_norm = dataframe.copy()

            # print(
            #     f"    [{pair_idx}] labels_list: {np.shape(labels_list)}, pair_labels: {np.shape(pair_labels)}, df_norm: {np.shape(df_norm)}"
            # )
            # min_length = min(
            #     [len(df_norm)] + [len(pair_labels) for values in pair_labels.values()]
            # )
            # if min_length <= self.seq_len:
            #     self.debug_print(
            #         f"    Skipping pair {pair_idx} due to insufficient data ({min_length} rows)"
            #     )
            #     continue

            # df_norm = df_norm.iloc[:min_length].reset_index(drop=True)
            # pair_labels = pair_labels[:min_length]

            split_idx = int(self.TRAIN_DATA_SPLIT * len(df_norm))
            buffer_size = self.seq_len - 1
            train_end = max(split_idx - buffer_size, self.seq_len)
            test_start = train_end

            train_df = df_norm.iloc[:train_end].reset_index(drop=True)
            test_df = df_norm.iloc[test_start:].reset_index(drop=True)

            train_label_segment = pair_labels[:train_end]
            test_label_segment = pair_labels[test_start:]

            train_df, train_label_segment = self.enhance_training_data(
                train_df, train_label_segment
            )

            train_one_hot = self.dataframeUtils.one_hot_encode(np.asarray(train_label_segment).astype(int), 3)
            test_one_hot = self.dataframeUtils.one_hot_encode(np.asarray(test_label_segment).astype(int), 3)

            tsr_train = self.dataframeUtils.df_to_tensor(train_df, self.seq_len, method=0)
            tsr_test = self.dataframeUtils.df_to_tensor(test_df, self.seq_len, method=0)

            offset = self.seq_len - 1
            train_one_hot = train_one_hot[offset:]
            test_one_hot = test_one_hot[offset:]

            if len(tsr_train) == 0 or len(tsr_test) == 0:
                self.debug_print(
                    f"    Skipping pair {pair_idx} due to zero-length tensors after windowing"
                )
                continue

            if aggr_tsr_train is None:
                aggr_tsr_train = tsr_train
                aggr_tsr_test = tsr_test
                aggr_train_labels = train_one_hot
                aggr_test_labels = test_one_hot
            else:
                aggr_tsr_train = np.concatenate([aggr_tsr_train, tsr_train], axis=0)
                aggr_tsr_test = np.concatenate([aggr_tsr_test, tsr_test], axis=0)
                aggr_train_labels = np.concatenate(
                    [aggr_train_labels, train_one_hot], axis=0
                )
                aggr_test_labels = np.concatenate(
                    [aggr_test_labels, test_one_hot], axis=0
                )

        if (
            aggr_tsr_train is None
            or aggr_tsr_test is None
            or aggr_train_labels is None
            or aggr_test_labels is None
        ):
            raise ValueError("No valid training data produced in prepare_training_data")

        final_train_labels = {}
        final_train_labels["trading"] = aggr_train_labels
        final_train_labels["reconstruction"] = aggr_tsr_train

        final_test_labels = {}
        final_test_labels["trading"] = aggr_test_labels
        final_test_labels["reconstruction"] = aggr_tsr_test

        # # only use rows where label is HOLD
        # normal_rows = final_train_labels["trading"] == TradingAction.HOLD
        # final_train_labels["trading"] = final_train_labels["trading"][normal_rows]
        # final_train_labels["reconstruction"] = final_train_labels["reconstruction"][normal_rows]

        self._augmented_labels = final_train_labels

        return aggr_tsr_train, aggr_tsr_test, final_train_labels, final_test_labels

    # -----------

    def get_predictions(self, dataframe: DataFrame, classifier: ClassifierKerasAnomaly):
        """Get the predictions from the model"""

        # empty dictionary for use in error cases
        dlen = np.shape(dataframe)[0]
        predictions_dict = {}
        predictions_dict["reconstruction"] = np.zeros(dlen, dtype=float)
        predictions_dict["trading"] = np.ones(dlen, dtype=int)

        if classifier is None:
            # print("    no classifier for predictions")
            raise Exception("No classifier for predictions")

        # Get multi-task predictions
        try:
            # # Debug: print data shape before prediction
            # self.debug_print(f"    DEBUG: Input data shape: {data.shape}")
            # self.debug_print(f"    DEBUG: Actual features: {data.shape[-1]}")
            df_norm = self.scale_dataframe(dataframe)
            df_tensor = self.dataframeUtils.df_to_tensor(df_norm, self.seq_len, method=0)
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
            reconstruction_predictions = multi_predictions["reconstruction"]
            trading_predictions = multi_predictions["trading"]

        except (KeyError, IndexError, TypeError) as e:
            log.error(f"    Failed to extract predictions: {e}")
            self.debug_print(f"    ERROR: Failed to extract predictions: {e}")
            self.debug_print(f"    multi_predictions: {multi_predictions}")
            return predictions_dict

        # Ensure predictions are numpy arrays and properly shaped
        reconstruction_predictions = np.array(reconstruction_predictions)
        trading_predictions = np.array(trading_predictions)

        # Clean predictions (handle NaN/Inf only)
        reconstruction_predictions = np.nan_to_num(
            reconstruction_predictions, nan=0.0, posinf=0.0, neginf=0.0
        )
        trading_predictions = np.nan_to_num(
            trading_predictions, nan=0.0, posinf=0.0, neginf=0.0
        )

        # convert reconstruction into mse
        # For 3D data, average across the sequence dimension
        if len(reconstruction_predictions.shape) == 3:
            errors = np.mean(
                np.square(df_tensor - reconstruction_predictions), axis=(1, 2)
            )
        else:
            errors = np.mean(np.square(df_tensor - reconstruction_predictions), axis=1)

        predictions_dict["reconstruction"] = errors

        # convert the probability matrices into classes

        pred_threshold = 0.5

        predictions_dict["trading"] = self.argmax_with_threshold(
            trading_predictions, threshold=pred_threshold, default_class=TradingAction.HOLD
        )
        # predictions_dict["trading"] = self.argmax_with_bias(
        #     trading_predictions,
        #     bias_map={0: self.bias_trading_sell.value, 2: self.bias_trading_buy.value},
        #     threshold=pred_threshold,
        #     default_class=TradingAction.HOLD,
        # )

        # DEBUG:
        self.print_probability_stats(
            "Trading",
            "Sell",
            trading_predictions[:, TradingAction.SELL],
            pred_threshold,
        )
        self.print_probability_stats(
            "Trading",
            "Hold",
            trading_predictions[:, TradingAction.HOLD],
            pred_threshold,
        )
        self.print_probability_stats(
            "Trading", "Buy", trading_predictions[:, TradingAction.BUY], pred_threshold
        )

        return predictions_dict

    # -----------

    def process_predictions(self, dataframe: DataFrame, predictions):
        """Process the predictions. Ideally, set up dataframe["predict_buy"] and dataframe["predict_sell"]"""

        reconstruction_predictions = predictions["reconstruction"]
        trading_predictions = predictions["trading"]

        # reconstruction error processing

        print(f"    DEBUG: reconstruction_predictions: {np.shape(reconstruction_predictions)}")
        print(f"    DEBUG: dataframe: {np.shape(dataframe)}")
        print(f"    DEBUG: predictions: {predictions}")
        dataframe["reconstruction_error"] = 0.0
        plen = len(reconstruction_predictions)
        # With method=0, predictions start at index seq_len-1 (first seq_len-1 rows have no predictions)
        start_idx = self.seq_len - 1
        dataframe.iloc[start_idx:start_idx + plen, dataframe.columns.get_loc("reconstruction_error")] = reconstruction_predictions
        entry_rolling_mean = (
            dataframe["reconstruction_error"]
            .rolling(window=48, min_periods=1, center=False)
            .mean()
        )
        entry_rolling_std = (
            dataframe["reconstruction_error"]
            .rolling(window=48, min_periods=1, center=False)
            .std()
        )

        # Calculate threshold using hyperopt parameters
        threshold = (
            entry_rolling_mean
            + self.anomaly_threshold_multiplier.value * entry_rolling_std
        )
        dataframe["anomaly_threshold"] = threshold

        # Find anomalies (high reconstruction error) - use dataframe column to ensure alignment
        reconstruction_error_col = dataframe["reconstruction_error"].iloc[start_idx:start_idx + plen].values
        threshold_col = threshold.iloc[start_idx:start_idx + plen].values
        anomaly_signals = (reconstruction_error_col > threshold_col) & (
            reconstruction_error_col > self.entry_error_threshold.value
        )

        # Filter for consecutive buy predictions to reduce noise (VECTORIZED)
        buy_signals = trading_predictions == TradingAction.BUY
        sell_signals = trading_predictions == TradingAction.SELL

        """
        min_consecutive = self.min_consecutive_buys.value

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
        """

        # DEBUG: this is a bit naughty, but add results to the current dataframe (not supposed to know this)
        # Use start_idx (seq_len - 1) to match where predictions start
        offset = start_idx
        dataframe["%trading"] = 0.0
        if "predict_buy" not in dataframe.columns:
            dataframe["predict_buy"] = 0.0
        if "predict_sell" not in dataframe.columns:
            dataframe["predict_sell"] = 0.0

        # Use the FINAL predictions (after all filtering and conversions), not raw argmax
        dataframe.iloc[offset:offset + len(trading_predictions), dataframe.columns.get_loc("%trading")] = trading_predictions

        self.debug_print("    Model predictions, after filtering:")
        self.print_distribution_compact("  Trading", trading_predictions)

        # add results to the main dataframe
        dataframe.iloc[offset:offset + len(anomaly_signals), dataframe.columns.get_loc("predict_buy")] = np.where(
            (anomaly_signals & buy_signals), 1, 0
        )
        dataframe.iloc[offset:offset + len(anomaly_signals), dataframe.columns.get_loc("predict_sell")] = np.where(
            (anomaly_signals & sell_signals), 1, 0
        )

        # Analyze reconstruction_error by trading signal to help set entry_error_threshold
        self._analyze_reconstruction_error_by_signal(dataframe, start_idx, plen)

        return dataframe

    def _analyze_reconstruction_error_by_signal(self, dataframe: DataFrame, start_idx: int, plen: int):
        """Analyze reconstruction_error statistics grouped by %train_trading signal"""

        # Get aligned data (only where we have predictions)
        if "%train_trading" not in dataframe.columns or "reconstruction_error" not in dataframe.columns:
            return

        # Extract aligned slices
        train_trading = dataframe["%train_trading"].iloc[start_idx:start_idx + plen].values
        reconstruction_error = dataframe["reconstruction_error"].iloc[start_idx:start_idx + plen].values

        # Filter out invalid values (NaN, inf, or where train_trading is not set)
        valid_mask = (
            np.isfinite(reconstruction_error) & 
            np.isfinite(train_trading) & 
            (train_trading >= 0) & 
            (train_trading <= 2)
        )

        if not valid_mask.any():
            return

        train_trading_valid = train_trading[valid_mask]
        reconstruction_error_valid = reconstruction_error[valid_mask]

        # Group by trading signal
        signal_names = {0: "Sell", 1: "Hold", 2: "Buy"}

        print("\n    =========================================")
        print("    Reconstruction Error Analysis by Signal")
        print("    =========================================")

        for signal_id, signal_name in signal_names.items():
            mask = train_trading_valid == signal_id
            if not mask.any():
                print(f"    {signal_name} (signal={signal_id}): No data")
                continue

            errors = reconstruction_error_valid[mask]
            count = len(errors)

            # Compute statistics
            mean_err = np.mean(errors)
            std_err = np.std(errors)
            median_err = np.median(errors)
            min_err = np.min(errors)
            max_err = np.max(errors)

            # Mode (most frequent value, rounded to 2 decimals)
            try:
                from scipy import stats
                mode_result = stats.mode(np.round(errors, 2))
                mode_err = mode_result.mode[0] if len(mode_result.mode) > 0 else np.nan
                mode_count = mode_result.count[0] if len(mode_result.count) > 0 else 0
            except (ImportError, Exception):
                # Fallback: use numpy to find mode
                rounded_errors = np.round(errors, 2)
                unique, counts = np.unique(rounded_errors, return_counts=True)
                mode_idx = np.argmax(counts)
                mode_err = unique[mode_idx]
                mode_count = counts[mode_idx]

            # Percentiles
            p25 = np.percentile(errors, 25)
            p75 = np.percentile(errors, 75)
            p90 = np.percentile(errors, 90)
            p95 = np.percentile(errors, 95)

            print(f"\n    {signal_name} (signal={signal_id}): {count} samples")
            print(f"      Mean:   {mean_err:.4f}")
            print(f"      Median: {median_err:.4f}")
            print(f"      Mode:   {mode_err:.4f} (appears {mode_count} times)")
            print(f"      Std:    {std_err:.4f}")
            print(f"      Min:    {min_err:.4f}")
            print(f"      Max:    {max_err:.4f}")
            print(f"      P25:    {p25:.4f}")
            print(f"      P75:    {p75:.4f}")
            print(f"      P90:    {p90:.4f}")
            print(f"      P95:    {p95:.4f}")

        # Overall statistics
        print(f"\n    Overall (all signals): {len(reconstruction_error_valid)} samples")
        print(f"      Mean:   {np.mean(reconstruction_error_valid):.4f}")
        print(f"      Median: {np.median(reconstruction_error_valid):.4f}")
        print(f"      Std:    {np.std(reconstruction_error_valid):.4f}")
        print(f"      Min:    {np.min(reconstruction_error_valid):.4f}")
        print(f"      Max:    {np.max(reconstruction_error_valid):.4f}")

        # Recommendation for entry_error_threshold
        buy_errors = reconstruction_error_valid[train_trading_valid == 2]
        if len(buy_errors) > 0:
            # Suggest threshold based on buy signal statistics
            buy_p25 = np.percentile(buy_errors, 25)
            buy_median = np.median(buy_errors)
            buy_mean = np.mean(buy_errors)

            print(f"\n    Suggested entry_error_threshold (based on Buy signals):")
            print(f"      Conservative (P25): {buy_p25:.4f}")
            print(f"      Moderate (Median):  {buy_median:.4f}")
            print(f"      Aggressive (Mean):  {buy_mean:.4f}")
            print(f"      Current setting:    {self.entry_error_threshold.value:.4f}")

        print("    =========================================\n")
