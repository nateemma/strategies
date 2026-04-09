"""
This intended as a base class for implementing these 'simple' strategies.
Concrete strategies should inherit from SimpleStrategy and implement the functions:

get_entry_signals(self, dataframe)
get_exit_signals(self, dataframe)

Each of these should return a series suitable for use as a boolean mask. Set the elements to True for buy/sell and False for otherwise.
"""

from enum import IntEnum
import numpy as np
import pandas as pd
from pandas import DataFrame, Series
import pandas_ta as pta
import talib.abstract as ta
import freqtrade.vendor.qtpylib.indicators as qtpylib

from freqtrade.strategy import (
    CategoricalParameter,
    DecimalParameter,
    IntParameter,
    stoploss_from_open,
)

import os
import sys
from pathlib import Path

group_dir = str(Path(__file__).parent)
strat_dir = str(Path(__file__).parent.parent)
sys.path.append(strat_dir)
sys.path.append(group_dir)

import utils.custom_indicators as cta
from Framework.BaseStrategy import (
    BaseStrategy,
    MarketRegime,
    StrategyConfig,
    NormalizationType,
    ModelType,
)

import logging

log = logging.getLogger(__name__)


class SimpleStrategy(BaseStrategy):

    # Strategy configuration
    strategy_config = StrategyConfig(
        normalization=NormalizationType.NONE,
        model_type=ModelType.NONE,
        needs_training=False,
    )

    class MarketState(IntEnum):
        RANGING = 0
        TRANSITION = 1
        TRENDING = 2

    class StrategyType(IntEnum):
        TREND = 0
        REVERSION = 1
        OSCILLATOR = 2
        VOLATILITY = 3
        OTHER = 4

    strategy_type = StrategyType.OTHER

    # Unique plotting configuration for SimpleStrategy
    plot_config = {
        "main_plot": {
            "close": {"color": "lightsteelblue"},
        },
        "subplots": {
            "Indicators": {
                "roc": {"color": "lightseagreen"},
                "rsi": {"color": "lightsalmon"},
            },
            "Signals": {
                "buy_signal": {"color": "green"},
                "sell_signal": {"color": "red"},
                "enter_long": {"color": "lightseagreen"},
                "exit_long": {"color": "lightsalmon"},
            },
        },
    }

    # # Inherit all buy_params from BaseStrategy and only override the guards flag
    # buy_params = {**BaseStrategy.buy_params, "entry_enable_guards": False}

    minimal_roi = {"0": 0.04, "30": 0.02, "60": 0.01, "120": 0.002, "240": 0.001}

    # Timeframe and Stoploss overrides
    timeframe = "15m"
    inf_timeframe = "15m"
    stoploss = -0.03

    if strategy_type == StrategyType.REVERSION:
        trailing_stop = True
        trailing_stop_positive = 0.01
        trailing_stop_positive_offset = 0.005
        trailing_only_offset_is_reached = True
    else:
        trailing_stop = False
        trailing_stop_positive = None
        trailing_stop_positive_offset = 0.0
        trailing_only_offset_is_reached = False

    # ATR filter: only trade when rolling ATR (as % of price) is above this (avoids static markets)
    ATR_PERIOD = 14
    ATR_ROLL_PERIOD = 20

    min_exit_hold_hours = DecimalParameter(
        0.5, 2.0, default=0.5, decimals=1, space="sell", load=True, optimize=True
    )

    def get_market_regime(self, dataframe):
        regime = np.ones(len(dataframe)) * MarketRegime.SIDEWAYS
        bear = (dataframe["adx"] > 25) & (dataframe["close"] < dataframe["ema200"])
        bull = (dataframe["adx"] > 25) & (dataframe["close"] > dataframe["ema200"])
        regime[bear] = MarketRegime.BEAR
        regime[bull] = MarketRegime.BULL
        return regime

    def get_market_state(self, dataframe):
        state = np.ones(len(dataframe)) * self.MarketState.RANGING
        transition = (dataframe["adx"] >= 20) & (dataframe["adx"] <= 25)
        trending = dataframe["adx"] > 25
        state[transition] = self.MarketState.TRANSITION
        state[trending] = self.MarketState.TRENDING
        return state

    def add_additional_indicators(self, dataframe: DataFrame) -> DataFrame:
        guard_period = 10

        # DataFrame already has: close, open, volume, adx, ema_fast, roc from BaseStrategy (roc is missing)
        dataframe["ema200"] = ta.EMA(dataframe, timeperiod=200)
        dataframe["roc"] = ta.ROC(dataframe, timeperiod=14).fillna(0.0)
        dataframe["rsi"] = ta.RSI(dataframe, timeperiod=14).fillna(50.0)

        # Candle imbalance over guard window
        bullish = (dataframe["close"] > dataframe["open"]).astype(int)
        bearish = (dataframe["close"] < dataframe["open"]).astype(int)
        bullish_count = bullish.rolling(window=guard_period).sum()
        bearish_count = bearish.rolling(window=guard_period).sum()
        dataframe["candle_imbalance"] = (bullish_count - bearish_count) / guard_period

        # Hilbert trend mode (0=cycling, 1=trending)
        dataframe["trend_mode"] = ta.HT_TRENDMODE(dataframe)

        dataframe["market_regime"] = self.get_market_regime(dataframe)
        dataframe["market_state"] = self.get_market_state(dataframe)

        return dataframe

    def get_entry_signals(self, dataframe: DataFrame):
        """Override this in subclasses."""
        return pd.Series(0, index=dataframe.index)

    def get_exit_signals(self, dataframe: DataFrame):
        """Override this in subclasses."""
        return pd.Series(0, index=dataframe.index)

    def get_entry_conditions(self, dataframe: DataFrame):
        """Implement the abstract method from BaseStrategy."""

        # Volatility filter - common for simple strategies
        # Recalculated here so it works correctly with hyperopt
        dataframe["atr_ok"] = dataframe["atr_pct_roll"] >= self.entry_atr_pct.value

        dataframe["entry_signals"] = self.get_entry_signals(dataframe)
        strategy_signals = dataframe["entry_signals"] > 0

        if self.strategy_type == self.StrategyType.TREND:
            strategy_signals &= (
                (dataframe["market_state"] == self.MarketState.TRENDING)
                & (dataframe["market_regime"] != MarketRegime.BEAR)
                & (dataframe["trend_mode"] == 1)
            )

        elif self.strategy_type == self.StrategyType.REVERSION:
            if "roc" in dataframe.columns and "ema200" in dataframe.columns:
                strategy_signals &= (
                    (dataframe["market_state"] != self.MarketState.TRENDING)
                    & (dataframe["roc"] < -0.1)
                    & (dataframe["roc"] > dataframe["roc"].shift(1))
                    & (dataframe["close"] > dataframe["ema200"])
                )

        elif self.strategy_type == self.StrategyType.OSCILLATOR:
            strategy_signals &= (
                (dataframe["market_state"] != self.MarketState.TRENDING)
                & (dataframe["market_regime"] != MarketRegime.BEAR)
                & (dataframe["trend_mode"] == 0)
            )

        elif self.strategy_type == self.StrategyType.VOLATILITY:
            strategy_signals &= dataframe["market_state"] != self.MarketState.RANGING

        slope_col = getattr(self, "entry_slope_column", None)
        if slope_col and slope_col in dataframe:
            strategy_signals &= dataframe[slope_col] > dataframe[slope_col].shift(1)

        dataframe["buy_signal"] = np.where(strategy_signals, 1, 0)

        # # DEBUG
        # buy_count = np.sum(dataframe["buy_signal"])
        # self.debug_print(f"SimpleStrategy buy_count: {buy_count}")

        return strategy_signals

    def get_exit_conditions(self, dataframe: DataFrame):
        """Implement the abstract method from BaseStrategy."""

        # Use same volatility filter for exits (can be expanded if needed)
        dataframe["atr_ok"] = dataframe["atr_pct_roll"] >= self.entry_atr_pct.value

        dataframe["exit_signals"] = self.get_exit_signals(dataframe)
        dataframe["sell_signal"] = np.where(dataframe["exit_signals"] > 0, 1, 0)
        return dataframe["exit_signals"] > 0

    def confirm_trade_exit(
        self,
        pair: str,
        trade,
        order_type: str,
        amount: float,
        rate: float,
        time_in_force: str,
        exit_reason: str,
        current_time,
        **kwargs,
    ) -> bool:
        if exit_reason == "exit_signal":
            time_delta = current_time - trade.open_date_utc
            hours_held = time_delta.total_seconds() / 3600
            min_hold = self.min_exit_hold_hours.value
            if hours_held < min_hold:
                return False

        return super().confirm_trade_exit(
            pair,
            trade,
            order_type,
            amount,
            rate,
            time_in_force,
            exit_reason,
            current_time,
            **kwargs,
        )
