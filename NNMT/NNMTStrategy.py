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
from pandas import DataFrame
import numpy as np

# Add parent directory to path to import NNNC
group_dir = str(Path(__file__).parent)
sys.path.append(group_dir)

from Framework.BaseStrategy import MarketRegime, TradingAction, FlowDirection, MomentumDirection, RiskLevel

from utils.ClassifierKeras import ClassifierKeras
import NNMTClassifier
from BaseNNMTStrategy import BaseNNMTStrategy, ProfitDirection


# -----------

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

    # -----------
    # Task-specific functions
    # (Target calculators moved to BaseNNMTStrategy in Phase 3)
    # -----------

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

    # Profit scaling helpers (kept in NNMTStrategy - not part of the multi-task scaffold)

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

