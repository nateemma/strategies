# pragma pylint: disable=C0103, C0114, C0115, C0116, C0301, C0302, C0303, C0325, C0411, C0413
# pragma pylint: disable=W0105, W1203, W1309, W1514, W0613, W0621,
# type: ignore
# pylint: disable=import-error
# flake8: noqa: F401, E402, F541, W0718, W0719

"""
NNMTStrategy - Base class for Neural Network Multi-Task strategies
"""


import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from pandas import DataFrame
import numpy as np
import pandas as pd
import traceback
from enum import IntEnum

from freqtrade.strategy import DecimalParameter, IntParameter, BooleanParameter

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

from utils.ClassifierKeras import ClassifierKeras
import NNMTClassifier
from BaseNNMTStrategy import BaseNNMTStrategy


# -----------

# enums for various task identifiers


class ProfitDirection(IntEnum):
    LOSS = 0
    NEUTRAL = 1
    PROFIT = 2


# -----------


class NNMTStrategy(BaseNNMTStrategy):
    """
    Neural Network Multi-Task (NNMT) base strategy that predicts:
    - Trading action (Sell/Hold/Buy) for trading decisions
    - Market regime (Bull/Bear/Sideways) for trading decisions
    - Risk level (High/Medium/Low) for risk management
    - Flow (down/neutral/up) for risk management
    - Momentum (-ve/stable/+ve) for risk management
    - Profit direction (loss/neutral/profit) for risk management

    The model is trained to predict these targets using a multi-task learning approach.
    The idea is that the model has to learn to predict all of the above features, which makes the model more
    general and the trading action decisions more usable than if we only trained for trading action.
    When I train on trading action only, the model tends to overfit the data rapidly so the predictions are
    only good for that particular training set.

    This is a base class for different multi-task model variants (LSTM, Transformer, etc.)
    Inherits from NNBase for clean(-ish) architecture
    """

    plot_config = {
        "main_plot": {
            "close": {"color": "lightsteelblue"},
        },
        "subplots": {
            "Diff": {
                # "predict_buy": {"color": "blue"},
                "%train_profit": {"color": "darkgreen"},
                "%train_trading": {"color": "green"},
                # "%train_risk": {"color": "purple"},
                "%train_flow": {"color": "orange"},
                "%train_regime": {"color": "brown"},
                "%trading": {"color": "cyan"},
                "%regime": {"color": "magenta"},
                "%flow": {"color": "yellow"},
                # "%risk": {"color": "pink"},
                "%profit": {"color": "lightseagreen"},
                "%momentum": {"color": "gray"},
                "%train_momentum": {"color": "purple"},
            },
        },
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

    def analyze_flow_distribution(
        self,
        atr_change_target: np.ndarray,
        flow_scaled: np.ndarray,
        max_atr_range: float,
    ) -> None:
        """
        Calculates and prints key statistics for raw and scaled ATR change target.
        Focuses on percentiles to check if MAX_ATR_RANGE is appropriate.
        """

        print("\n--- Flow Target Distribution Analysis ---")

        # 1. Analyze Raw ATR Change Target
        raw_min = np.min(atr_change_target)
        raw_max = np.max(atr_change_target)
        raw_mean = np.mean(atr_change_target)
        raw_std = np.std(atr_change_target)
        raw_5th = np.percentile(atr_change_target, 5)
        raw_95th = np.percentile(atr_change_target, 95)

        print("\n[RAW ATR Change Target]")
        print(f"  Range (Min/Max): {raw_min:.6f} to {raw_max:.6f}")
        print(f"  Mean / Std Dev:  {raw_mean:.6f} / {raw_std:.6f}")
        print(f"  5th / 95th Pctl: {raw_5th:.6f} / {raw_95th:.6f}")

        # Check if the 95th percentile exceeds the max_atr_range
        if raw_95th > max_atr_range or raw_5th < -max_atr_range:
            print(
                f"\n[SCALING WARNING]: The 95th percentile magnitude ({max(abs(raw_5th), abs(raw_95th)):.4f})"
            )
            print(f"  EXCEEDS the current MAX_ATR_RANGE ({max_atr_range:.4f}).")
            print(
                f"  Your scaling factor is too small and is clipping most of the signal."
            )

        else:
            print(
                f"\n[SCALING CHECK]: 95th percentile magnitude is within MAX_ATR_RANGE."
            )

        # 2. Analyze Scaled Flow Target
        scaled_min = np.min(flow_scaled)
        scaled_max = np.max(flow_scaled)
        scaled_mean = np.mean(flow_scaled)
        scaled_std = np.std(flow_scaled)
        scaled_5th = np.percentile(flow_scaled, 5)
        scaled_95th = np.percentile(flow_scaled, 95)

        self.debug_print(f"\n[SCALED Flow Target (Range [-1, 1])]")
        self.debug_print(f"  Range (Min/Max): {scaled_min:.6f} to {scaled_max:.6f}")
        self.debug_print(f"  Mean / Std Dev:  {scaled_mean:.6f} / {scaled_std:.6f}")
        self.debug_print(f"  5th / 95th Pctl: {scaled_5th:.6f} to {scaled_95th:.6f}")

        # Check if the scaled data is being squashed near zero
        if scaled_std < 0.20:
            self.debug_print(
                f"\n[RANGE WARNING]: Scaled Standard Deviation ({scaled_std:.4f}) is very low."
            )
            self.debug_print(
                f"  The vast majority of your training data is clustered near zero."
            )
            self.debug_print(
                f"  Consider decreasing MAX_ATR_RANGE to increase scaled variance."
            )

        self.debug_print("---------------------------------------")

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

    def scale_profit(self, profit: np.ndarray) -> np.ndarray:
        """Scale profit to [-1,1] range linearly based on PROFIT_RANGE."""
        # # Clip to the maximum expected range first
        # scaled_profit = np.clip(profit, -self.PROFIT_RANGE, self.PROFIT_RANGE)
        # # Scale linearly to [-1, 1]
        # scaled_profit = scaled_profit / self.PROFIT_RANGE
        scaled_profit = profit
        return scaled_profit

    def descale_profit(self, scaled_profit: np.ndarray) -> np.ndarray:
        """Descale profit from [-1,1] range back to original."""
        # Scale back linearly
        # profit = scaled_profit * self.PROFIT_RANGE
        profit = scaled_profit
        return profit

    def get_profit_target(self, dataframe: DataFrame) -> np.ndarray:

        # Simplified version: look ahead at EMA-smoothed close-to-close change only
        # Classify based on future close after horizon using thresholds
        # Note: "close", "high", "low" are raw columns, not normalized, so not checked
        df = dataframe
        n = len(df)
        classes = np.ones(n, dtype=int) * ProfitDirection.NEUTRAL

        # Parameters - use the same thresholds as buy/sell signal generation for consistency
        horizon = int(self.HORIZON)
        pt = float(
            self.MIN_BUY_GAIN_THRESHOLD
        )  # Profit threshold (same as buy signal threshold)
        sl = float(
            self.MIN_SELL_LOSS_THRESHOLD
        )  # Loss threshold (same as sell signal threshold)

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

        for t in range(n):
            future_idx = t + horizon
            if future_idx >= n:
                continue
            entry = close[t]
            future_close = close[future_idx]

            gain = (future_close - entry) / entry
            vol_adj = atr_pct[t] * float(self.PROFIT_ATR_SCALE)
            pt_eff = max(pt, vol_adj)
            sl_eff = max(sl, vol_adj)

            if gain >= pt_eff:
                classes[t] = ProfitDirection.PROFIT
            elif gain <= -sl_eff:
                classes[t] = ProfitDirection.LOSS
            else:
                classes[t] = ProfitDirection.NEUTRAL

        # Note that we cannot add this to the main dataframe
        # because it is inherently looking ahead in time
        return classes

    # -----------

    # Trading Action (more complicated because it uses data from the other tasks)

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

        buy_mask = (trading_predictions == TradingAction.BUY) & (
            momentum_predictions == MomentumDirection.POSITIVE
        )
        sell_mask = (trading_predictions == TradingAction.SELL) & (
            momentum_predictions == MomentumDirection.NEGATIVE
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
    # Override functions
    # -----------

    def get_classifier_type(self):
        """Return the type of classifier used for training/predicting"""
        return NNMTClassifier.ClassifierType.Multi_LSTM

    # -----------

    def get_classifier(
        self, classifier_type, pair, seq_len, num_features
    ) -> ClassifierKeras:
        """Return the classifier used for training/predicting"""

        classifier, _ = NNMTClassifier.create_classifier(
            classifier_type, pair, num_features, seq_len
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
        self, dataframes: List[DataFrame], labels_list, norm: bool = True
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

        pred_threshold = self.prediction_threshold.value
        # print(f"    prediction threshold: {pred_threshold}")

        predictions_dict["trading"] = self.argmax_with_threshold(
            trading_predictions,
            threshold=pred_threshold,
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
            "Trading", "Sell", profit_predictions[:, TradingAction.SELL], pred_threshold
        )
        self.print_probability_stats(
            "Trading", "Hold", profit_predictions[:, TradingAction.HOLD], pred_threshold
        )
        self.print_probability_stats(
            "Trading", "Buy", profit_predictions[:, TradingAction.BUY], pred_threshold
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
