# pragma pylint: disable=C0103, C0114, C0115, C0116, C0301, C0302, C0303, C0325, C0411, C0413
# pragma pylint: disable=W0105, W1203, W1309, W1514, W0613, W0621,
# type: ignore
# pylint: disable=import-error

"""
NNMT2 - Variant of NNMTStrategy that replaces task definitions to somewhat match the trainng indicators
"""

import sys
from pathlib import Path
from pandas import DataFrame
import numpy as np
from typing import Dict, Tuple

group_dir = str(Path(__file__).parent)
sys.path.append(group_dir)

from NNMTStrategy import NNMTStrategy  # noqa: E402
from Framework.BaseNNStrategy import BaseNNStrategy, HAS_MLX, StrategyConfig
from utils.Environment import Environment
from Framework.BaseStrategy import BaseStrategy, ScalerType, MarketRegime, TradingAction, FlowDirection, MomentumDirection, RiskLevel, GANType
from Framework.BaseNNStrategy import BaseNNStrategy, HAS_MLX, StrategyConfig
from utils.Environment import Environment
from Framework.BaseStrategy import BaseStrategy, ScalerType, MarketRegime, TradingAction, FlowDirection, MomentumDirection, RiskLevel, GANType
from NNMTStrategy import TradingAction, ProfitDirection

# -----------


class NNMT2(NNMTStrategy):

    # Market Regime

    def get_market_regime(self, dataframe: DataFrame) -> np.ndarray:
        """Classify SHORT-TERM market regimes using normalized indicators (pair-agnostic)"""

        # Check that required normalized columns are in include_list
        self.check_columns_included(
            ["trend_mode", "di_diff_scaled"],
            "get_market_regime",
        )

        regime = np.ones(len(dataframe), dtype=int) * MarketRegime.SIDEWAYS

        di_diff_scaled = dataframe["di_diff_scaled"]

        # trend directio

        # 1. Calculate the Hilbert Transform Trend Mode
        # This identifies if we are in a 'Trend' (1) or 'Cycle' (0)
        trend_mode = dataframe["trend_mode"]

        # If in Trend mode, check direction
        regime = np.where(
            (trend_mode == 1) & (di_diff_scaled > 0),
            MarketRegime.BULL,
            regime,
        )
        regime = np.where(
            (trend_mode == 1) & (di_diff_scaled < 0),
            MarketRegime.BEAR,
            regime,
        )

        # regime = np.where((adxscaled < self.ADX_THRESHOLD), MarketRegime.SIDEWAYS, regime)

        return regime

    # -----------

    # Risk Level

    def get_risk_level(self, dataframe: DataFrame) -> np.ndarray:
        """Calculate tri-state risk classification: LOW=0, NORMAL=1, HIGH=2"""

        # Check that required normalized columns are in include_list
        # Note: "close" and "volume" are raw columns, not normalized, so not checked
        self.check_columns_included(["close_norm"], "get_risk_level")

        close_norm = dataframe["close_norm"]

        risk_class = np.ones(len(dataframe), dtype=int) * RiskLevel.NORMAL
        risk_class = np.where(
            (close_norm <= -0.5),
            RiskLevel.LOW,
            risk_class,
        )
        risk_class = np.where(
            (close_norm >= 0.5),
            RiskLevel.HIGH,
            risk_class,
        )

        return risk_class

    # -----------

    # Flow

    def get_flow(self, dataframe: DataFrame) -> np.ndarray:
        """
        Flow represents the change in directional movement over the lookahead window:
        - DECREASE (0): Flow is decreasing (becoming more downward or less upward)
        - NEUTRAL (1): Flow is stable (little change in direction)
        - INCREASE (2): Flow is increasing (becoming more upward or less downward)

        """

        # Check that required normalized columns are in include_list
        self.check_columns_included(["guard_metric"], "get_flow")

        guard_metric = dataframe["guard_metric"]

        flow_classes = np.ones(len(dataframe), dtype=int) * FlowDirection.NEUTRAL
        flow_classes[guard_metric < -0.5] = FlowDirection.DECREASE
        flow_classes[guard_metric > 0.5] = FlowDirection.INCREASE

        # save (non-lookahead) flow classes to dataframe
        dataframe["flow"] = flow_classes

        return flow_classes

    # -----------

    # Momentum

    def get_momentum(self, dataframe: DataFrame) -> np.ndarray:
        """Calculate momentum using normalized aroonosc (pair-agnostic)"""

        # Check that required normalized columns are in include_list
        self.check_columns_included(["adx_scaled"], "get_momentum")

        adx_scaled = dataframe["adx_scaled"]

        # Use aroonosc_scaled (already normalized [-1, 1], pair-agnostic)
        momentum_classes = np.ones(len(dataframe), dtype=int) * MomentumDirection.STABLE
        momentum_classes[adx_scaled < -0.3] = MomentumDirection.NEGATIVE
        momentum_classes[adx_scaled > 0.7] = MomentumDirection.POSITIVE

        return momentum_classes

    # -----------

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

        trading_buy_mask = trading_predictions == TradingAction.BUY
        trading_sell_mask = trading_predictions == TradingAction.SELL

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
        sell_mask = trading_sell_mask & (
            sell_conditions.sum(axis=0) >= required_matches
        )

        """

        buy_mask = (trading_predictions == TradingAction.BUY) & (
            momentum_predictions == MomentumDirection.POSITIVE
        )
        sell_mask = (trading_predictions == TradingAction.SELL) & (
            momentum_predictions == MomentumDirection.NEGATIVE
        )
        """
        # Reset all to HOLD, then set BUY/SELL only where both conditions are met
        filtered = np.full_like(trading_predictions, TradingAction.HOLD)
        filtered[buy_mask] = TradingAction.BUY
        filtered[sell_mask] = TradingAction.SELL
        return filtered
