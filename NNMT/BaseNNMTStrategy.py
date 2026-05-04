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
from pandas import DataFrame
import numpy as np
import pandas as pd

# Match NNMTStrategy's sys.path setup so sibling-module imports resolve
group_dir = str(Path(__file__).parent)
sys.path.append(group_dir)

from Framework.BaseNNStrategy import BaseNNStrategy
from Framework.BaseStrategy import MarketRegime, FlowDirection, MomentumDirection, RiskLevel
from freqtrade.strategy import DecimalParameter, IntParameter, BooleanParameter


class ProfitDirection(IntEnum):
    LOSS = 0
    NEUTRAL = 1
    PROFIT = 2


class BaseNNMTStrategy(BaseNNStrategy):
    """
    Multi-task neural network strategy base.

    Empty in this commit; subsequent phases move attributes and methods up from
    NNMTStrategy. NNMTStrategy still inherits the full multi-task surface area
    via this class — behavior is unchanged.
    """

    profit_conflict_to_neutral = True
    PROFIT_EMA_SPAN = 5
    PROFIT_ATR_SCALE = 1.0

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

    apply_task_filters = BooleanParameter(
        default=False,
        space="buy",
        optimize=True,
        load=True,
    )
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
