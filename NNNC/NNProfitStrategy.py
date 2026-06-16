from typing import Any
# pragma pylint: disable=C0103, C0114, C0115, C0116, C0301, C0302, C0303, C0325, C0411, C0413
# pragma pylint: disable=W0105, W1203, W1309, W1514, W0613, W0621,
# type: ignore
# pylint: disable=import-error

"""
NNProfitStrategy - Base class for Neural Network N-ary Classification strategies
                   This variant uses the profit predictions instead of trading predictions
"""

import sys
from pathlib import Path
from pandas import DataFrame
import pandas as pd
import numpy as np
import traceback
from enum import IntEnum

# Add parent directory to path to import NNNC
group_dir = str(Path(__file__).parent)
sys.path.append(group_dir)

from Framework.BaseNNStrategy import BaseNNStrategy, HAS_MLX, StrategyConfig
from utils.Environment import Environment
from Framework.BaseStrategy import BaseStrategy, ScalerType, MarketRegime, TradingAction, FlowDirection, MomentumDirection, RiskLevel, GANType
from Framework.BaseNNStrategy import BaseNNStrategy, HAS_MLX, StrategyConfig
from utils.Environment import Environment
from Framework.BaseStrategy import BaseStrategy, ScalerType, MarketRegime, TradingAction, FlowDirection, MomentumDirection, RiskLevel, GANType
from Framework.BaseNNStrategy import BaseNNStrategy, HAS_MLX, StrategyConfig
from utils.Environment import Environment
from Framework.BaseStrategy import BaseStrategy, ScalerType, MarketRegime, TradingAction, FlowDirection, MomentumDirection, RiskLevel, GANType
from Framework.BaseNNStrategy import BaseNNStrategy, HAS_MLX, StrategyConfig
from utils.Environment import Environment
from Framework.BaseStrategy import BaseStrategy, ScalerType, MarketRegime, TradingAction, FlowDirection, MomentumDirection, RiskLevel, GANType
import NNNClassifier


class ProfitDirection(IntEnum):
    LOSS = 0
    NEUTRAL = 1
    PROFIT = 2

class NNProfitStrategy(BaseNNStrategy):

    profit_conflict_to_neutral = True
    # Options: "triple_barrier", "fixed_thresholds", "quantile_thresholds", "rank_based"
    profit_label_method = "triple_barrier"
    triple_barrier_vol_window = 32
    triple_barrier_vol_mult = 1.5
    profit_quantile = 0.9
    loss_quantile = 0.9
    rank_top_pct = 0.1

    def get_classifier_type(self):
        """ Return the type of classifier used for training/predicting """
        return NNNClassifier.ClassifierType.LSTM

    def get_classifier(self, classifier_type, pair, seq_len, num_features) -> "Any":
        """ Return the classifier used for training/predicting """
        clf, _ = NNNClassifier.create_classifier(
            classifier_type, pair, num_features, seq_len, 3
        )
        return clf

    # -----------

    # get (forward-looking) proit target classes
    def get_profit_target(self, dataframe: DataFrame) -> np.ndarray:
        if self.profit_label_method == "triple_barrier":
            return self.get_profit_target_triple_barrier(dataframe)
        if self.profit_label_method == "fixed_thresholds":
            return self.get_profit_target_fixed_thresholds(dataframe)
        if self.profit_label_method == "quantile_thresholds":
            return self.get_profit_target_quantile_thresholds(dataframe)
        if self.profit_label_method == "rank_based":
            return self.get_profit_target_rank_based(dataframe)

        # Default fallback
        return self.get_profit_target_triple_barrier(dataframe)

    def get_profit_target_fixed_thresholds(self, dataframe: DataFrame) -> np.ndarray:
        """Label profit using fixed thresholds and first-hit logic."""
        df = dataframe
        n = len(df)
        classes = np.ones(n, dtype=int) * ProfitDirection.NEUTRAL

        horizon = int(getattr(self, "HORIZON", 32))
        pt = float(getattr(self, "MIN_BUY_GAIN_THRESHOLD", 0.007))
        sl = float(getattr(self, "MIN_SELL_LOSS_THRESHOLD", 0.009))

        close = np.asarray(df["close"], dtype=float)
        high = np.asarray(df["high"], dtype=float)
        low = np.asarray(df["low"], dtype=float)

        for t in range(n):
            end = min(n, t + horizon + 1)
            if t + 1 >= end:
                continue

            entry = close[t]
            future_high = high[t + 1 : end]
            future_low = low[t + 1 : end]

            if len(future_high) == 0 or len(future_low) == 0:
                continue

            gain_series = (future_high - entry) / entry
            loss_series = (entry - future_low) / entry

            gain_hits = np.where(gain_series >= pt)[0]
            loss_hits = np.where(loss_series >= sl)[0]

            gain_idx = gain_hits[0] if gain_hits.size > 0 else None
            loss_idx = loss_hits[0] if loss_hits.size > 0 else None

            if gain_idx is not None and loss_idx is not None:
                if self.profit_conflict_to_neutral:
                    classes[t] = ProfitDirection.NEUTRAL
                else:
                    classes[t] = (
                        ProfitDirection.PROFIT
                        if gain_idx < loss_idx
                        else ProfitDirection.LOSS
                    )
            elif gain_idx is not None:
                classes[t] = ProfitDirection.PROFIT
            elif loss_idx is not None:
                classes[t] = ProfitDirection.LOSS
            else:
                classes[t] = ProfitDirection.NEUTRAL

        return classes

    def get_profit_target_triple_barrier(self, dataframe: DataFrame) -> np.ndarray:
        """Label profit using volatility-scaled triple-barrier thresholds."""
        df = dataframe
        n = len(df)
        classes = np.ones(n, dtype=int) * ProfitDirection.NEUTRAL

        horizon = int(getattr(self, "HORIZON", 32))
        min_pt = float(getattr(self, "MIN_BUY_GAIN_THRESHOLD", 0.007))
        min_sl = float(getattr(self, "MIN_SELL_LOSS_THRESHOLD", 0.009))

        close = np.asarray(df["close"], dtype=float)
        high = np.asarray(df["high"], dtype=float)
        low = np.asarray(df["low"], dtype=float)

        returns = np.zeros_like(close)
        returns[1:] = np.diff(close) / close[:-1]
        vol_window = int(getattr(self, "triple_barrier_vol_window", 32))
        vol = (
            pd.Series(returns)
            .rolling(window=vol_window, min_periods=1)
            .std()
            .to_numpy()
        )
        vol_mult = float(getattr(self, "triple_barrier_vol_mult", 1.5))

        for t in range(n):
            end = min(n, t + horizon + 1)
            if t + 1 >= end:
                continue

            entry = close[t]
            future_high = high[t + 1 : end]
            future_low = low[t + 1 : end]

            if len(future_high) == 0 or len(future_low) == 0:
                continue

            pt = max(min_pt, vol_mult * vol[t])
            sl = max(min_sl, vol_mult * vol[t])

            gain_series = (future_high - entry) / entry
            loss_series = (entry - future_low) / entry

            gain_hits = np.where(gain_series >= pt)[0]
            loss_hits = np.where(loss_series >= sl)[0]

            gain_idx = gain_hits[0] if gain_hits.size > 0 else None
            loss_idx = loss_hits[0] if loss_hits.size > 0 else None

            if gain_idx is not None and loss_idx is not None:
                if self.profit_conflict_to_neutral:
                    classes[t] = ProfitDirection.NEUTRAL
                else:
                    classes[t] = (
                        ProfitDirection.PROFIT
                        if gain_idx < loss_idx
                        else ProfitDirection.LOSS
                    )
            elif gain_idx is not None:
                classes[t] = ProfitDirection.PROFIT
            elif loss_idx is not None:
                classes[t] = ProfitDirection.LOSS
            else:
                classes[t] = ProfitDirection.NEUTRAL

        return classes

    def get_profit_target_quantile_thresholds(self, dataframe: DataFrame) -> np.ndarray:
        """Label profit using global quantile thresholds and first-hit logic."""
        df = dataframe
        n = len(df)
        classes = np.ones(n, dtype=int) * ProfitDirection.NEUTRAL

        horizon = int(getattr(self, "HORIZON", 32))
        min_pt = float(getattr(self, "MIN_BUY_GAIN_THRESHOLD", 0.007))
        min_sl = float(getattr(self, "MIN_SELL_LOSS_THRESHOLD", 0.009))
        profit_q = float(getattr(self, "profit_quantile", 0.9))
        loss_q = float(getattr(self, "loss_quantile", 0.9))

        close = np.asarray(df["close"], dtype=float)
        high = np.asarray(df["high"], dtype=float)
        low = np.asarray(df["low"], dtype=float)

        max_gains = np.zeros(n, dtype=float)
        max_losses = np.zeros(n, dtype=float)

        for t in range(n):
            end = min(n, t + horizon + 1)
            if t + 1 >= end:
                continue
            entry = close[t]
            future_high = high[t + 1 : end]
            future_low = low[t + 1 : end]
            if len(future_high) == 0 or len(future_low) == 0:
                continue
            max_gains[t] = (np.max(future_high) - entry) / entry
            max_losses[t] = (entry - np.min(future_low)) / entry

        pt = max(min_pt, float(np.nanquantile(max_gains, profit_q)))
        sl = max(min_sl, float(np.nanquantile(max_losses, loss_q)))

        for t in range(n):
            end = min(n, t + horizon + 1)
            if t + 1 >= end:
                continue
            entry = close[t]
            future_high = high[t + 1 : end]
            future_low = low[t + 1 : end]
            if len(future_high) == 0 or len(future_low) == 0:
                continue

            gain_series = (future_high - entry) / entry
            loss_series = (entry - future_low) / entry

            gain_hits = np.where(gain_series >= pt)[0]
            loss_hits = np.where(loss_series >= sl)[0]

            gain_idx = gain_hits[0] if gain_hits.size > 0 else None
            loss_idx = loss_hits[0] if loss_hits.size > 0 else None

            if gain_idx is not None and loss_idx is not None:
                if self.profit_conflict_to_neutral:
                    classes[t] = ProfitDirection.NEUTRAL
                else:
                    classes[t] = (
                        ProfitDirection.PROFIT
                        if gain_idx < loss_idx
                        else ProfitDirection.LOSS
                    )
            elif gain_idx is not None:
                classes[t] = ProfitDirection.PROFIT
            elif loss_idx is not None:
                classes[t] = ProfitDirection.LOSS
            else:
                classes[t] = ProfitDirection.NEUTRAL

        return classes

    def get_profit_target_rank_based(self, dataframe: DataFrame) -> np.ndarray:
        """Label profit by ranking the NET forward move (max gain - max loss)
        over the horizon.

        Distinct from quantile_thresholds, which ranks gains and losses
        *independently* and — under the default config — collapses to identical
        labels. Here a bar is PROFIT only when its upside dominates its downside
        enough to land in the top ``rank_top_pct`` of the net distribution, LOSS
        when downside dominates (bottom ``rank_top_pct``), NEUTRAL otherwise. The
        two tails are disjoint by construction, so ``profit_conflict_to_neutral``
        does not apply.
        """
        df = dataframe
        n = len(df)
        classes = np.ones(n, dtype=int) * ProfitDirection.NEUTRAL

        horizon = int(getattr(self, "HORIZON", 32))
        top_pct = float(getattr(self, "rank_top_pct", 0.1))

        close = np.asarray(df["close"], dtype=float)
        high = np.asarray(df["high"], dtype=float)
        low = np.asarray(df["low"], dtype=float)

        net = np.full(n, np.nan, dtype=float)  # NaN where no forward window exists
        for t in range(n):
            end = min(n, t + horizon + 1)
            if t + 1 >= end:
                continue
            entry = close[t]
            future_high = high[t + 1 : end]
            future_low = low[t + 1 : end]
            if len(future_high) == 0 or len(future_low) == 0:
                continue
            max_gain = (np.max(future_high) - entry) / entry
            max_loss = (entry - np.min(future_low)) / entry
            net[t] = max_gain - max_loss

        if not np.any(np.isfinite(net)):
            return classes

        profit_thr = float(np.nanquantile(net, 1.0 - top_pct))
        loss_thr = float(np.nanquantile(net, top_pct))

        # NaN comparisons evaluate False, so windowless bars stay NEUTRAL.
        classes[net >= profit_thr] = ProfitDirection.PROFIT
        classes[net <= loss_thr] = ProfitDirection.LOSS

        return classes

    # -----------

    def get_train_buy_signals(self, future_df: DataFrame):
        """Generate buy signals based on profit predictions"""

        buy_labels = np.zeros(np.shape(future_df)[0], dtype=float)
        buy_signals = pd.Series(np.zeros(np.shape(future_df)[0], dtype=float))

        # Get profit predictions
        profit_predictions = self.get_profit_target(future_df)

        # Filter buy signals based on profit predictions
        buy_labels = np.where(profit_predictions == ProfitDirection.PROFIT, 1.0, 0.0)
        buy_signals[buy_labels > 0.5] = 1.0

        return buy_signals

    def get_train_sell_signals(self, future_df: DataFrame):
        """Generate sell signals based on profit predictions"""

        sell_labels = np.zeros(np.shape(future_df)[0], dtype=float)
        sell_signals = pd.Series(np.zeros(np.shape(future_df)[0], dtype=float))

        # Get profit predictions
        profit_predictions = self.get_profit_target(future_df)

        # Filter sell signals based on profit predictions
        sell_labels = np.where(profit_predictions == ProfitDirection.LOSS, 1.0, 0.0)
        sell_signals[sell_labels > 0.5] = 1.0

        return sell_signals
