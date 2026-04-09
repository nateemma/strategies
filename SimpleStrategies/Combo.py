from freqtrade.strategy import IStrategy
import pandas as pd
import pandas_ta as pta
import numpy as np
from pandas import DataFrame, Series
import talib.abstract as ta
import freqtrade.vendor.qtpylib.indicators as qtpylib
from freqtrade.strategy import (
    CategoricalParameter,
    DecimalParameter,
    IntParameter,
    IStrategy,
    merge_informative_pair,
    stoploss_from_open,
)

# set paths so that we can find imports in parallel directories
import os
import sys
from pathlib import Path

group_dir = str(Path(__file__).parent)
strat_dir = str(Path(__file__).parent.parent)
sys.path.append(strat_dir)
sys.path.append(group_dir)

from utils import custom_indicators as cta

import warnings

warnings.filterwarnings(
    "ignore", message="The objective has been evaluated at this point before."
)

from SimpleStrategy import SimpleStrategy

from finta import TA as fta

"""
Combo - Combination of Indicators
"""


class Combo(SimpleStrategy):

    strategy_type = SimpleStrategy.StrategyType.OSCILLATOR
    # entry_slope_column = "cmo"

    plot_config = {
        "main_plot": {
            "close": {"color": "lightsteelblue"},
        },
        "subplots": {
            "Indicators": {
                "close_norm": {"color": "lightsteelblue"},
                "adx_scaled": {"color": "lightcoral"},
                "bb_width": {"color": "lightgreen"},
                "guard_metric": {"color": "lightsalmon"},
            },
            "Signals": {
                "buy_signal": {"color": "green"},
                "sell_signal": {"color": "red"},
                "enter_long": {"color": "lightgreen"},
                "exit_long": {"color": "lightsalmon"},
            },
        },
    }

    enable_guards = True  # set to True for testing, False for debug

    # Buy hyperspace params:

    # Strategy parameters

    entry_close_norm_threshold = DecimalParameter(
        -1.0, 0.0, default=0.0, decimals=1, space="buy", load=True, optimize=True
    )
    entry_adx_scaled_threshold = DecimalParameter(
        -1.0, 0.0, default=0.0, decimals=1, space="buy", load=True, optimize=True
    )
    entry_bb_width_threshold = DecimalParameter(
        -1.0, -0.01, default=-0.2, decimals=1, space="buy", load=True, optimize=True
    )
    entry_guard_metric_threshold = DecimalParameter(
        -1.0, 0.0, default=0.0, decimals=1, space="buy", load=True, optimize=True
    )
    entry_min_indicators_buy = IntParameter(
        2, 4, default=4, space="buy", load=True, optimize=True
    )

    exit_close_norm_threshold = DecimalParameter(
        0.0, 1.0, default=0.0, decimals=1, space="buy", load=True, optimize=True
    )
    exit_adx_scaled_threshold = DecimalParameter(
        0.0, 1.0, default=0.0, decimals=1, space="buy", load=True, optimize=True
    )
    exit_bb_width_threshold = DecimalParameter(
        0.01, 1.0, default=-0.2, decimals=1, space="buy", load=True, optimize=True
    )
    exit_guard_metric_threshold = DecimalParameter(
        0.0, 1.0, default=0.0, decimals=1, space="buy", load=True, optimize=True
    )
    exit_min_indicators_sell = IntParameter(
        2, 4, default=4, space="buy", load=True, optimize=True
    )

    def get_entry_signals(self, dataframe):

        min_indicators_buy = self.entry_min_indicators_buy.value

        n = len(dataframe["close"])
        labels = np.zeros(n, dtype=float)

        # Get indicator values (match utils/DataframePopulator.add_minimal_indicators)
        epsilon = 1e-6
        wlen = 48
        if "close_norm" not in dataframe.columns:
            close_safe = np.maximum(dataframe["close"].values, 1e-6)
            close_safe_series = pd.Series(close_safe, index=dataframe.index)
            rolling_min = close_safe_series.rolling(window=wlen).min()
            rolling_max = close_safe_series.rolling(window=wlen).max()
            rolling_range = rolling_max - rolling_min
            dataframe["close_norm"] = (close_safe_series - rolling_min) / (
                rolling_range + epsilon
            ) * 2.0 - 1.0
        if "adx_scaled" not in dataframe.columns:
            adx = ta.ADX(dataframe)
            dataframe["adx"] = adx
            ADX_FIXED_MAX = 55.0
            dataframe["adx_scaled"] = (adx / ADX_FIXED_MAX).clip(0, 1) * 2.0 - 1.0
        if "bb_width" not in dataframe.columns:
            bollinger = qtpylib.bollinger_bands(dataframe["close"], window=20, stds=2)
            dataframe["bb_lowerband"] = bollinger["lower"]
            dataframe["bb_middleband"] = bollinger["mid"]
            dataframe["bb_upperband"] = bollinger["upper"]
            dataframe["bb_width"] = (
                dataframe["bb_upperband"] - dataframe["bb_lowerband"]
            ) / (dataframe["bb_middleband"] + epsilon)
            BB_WIDTH_FIXED_MAX = 0.028
            dataframe["bb_width"] = (dataframe["bb_width"] / BB_WIDTH_FIXED_MAX).clip(
                0, 1
            ) * 2.0 - 1.0
        if "guard_metric" not in dataframe.columns:
            rmi = cta.RMI(dataframe, length=14, mom=5)
            dataframe["guard_metric"] = 2.0 * (rmi - 50.0) / 100.0

        # Convert to numpy arrays, handling NaN
        close_norm = np.asarray(dataframe["close_norm"], dtype=float)
        adx_scaled = np.asarray(dataframe["adx_scaled"], dtype=float)
        bb_width = np.asarray(dataframe["bb_width"], dtype=float)
        guard_metric = np.asarray(dataframe["guard_metric"], dtype=float)

        # Replace NaN with neutral values (0)
        close_norm = np.nan_to_num(close_norm, nan=0.0)
        adx_scaled = np.nan_to_num(adx_scaled, nan=0.0)
        bb_width = np.nan_to_num(bb_width, nan=0.0)
        guard_metric = np.nan_to_num(guard_metric, nan=0.0)

        # Buy signals: Oversold indicators + future gain validation

        buy_votes = np.zeros(n, dtype=int)
        buy_votes += (close_norm < self.entry_close_norm_threshold.value).astype(
            int
        )  # Close norm negative
        buy_votes += (adx_scaled > self.entry_adx_scaled_threshold.value).astype(
            int
        )  # ADX scaled positive
        buy_votes += (bb_width <= self.entry_bb_width_threshold.value).astype(
            int
        )  # BB width negative (volatility high)
        buy_votes += (guard_metric < self.entry_guard_metric_threshold.value).astype(
            int
        )  # Guard metric negative

        # Buy: enough indicators agree
        buy_mask = buy_votes >= min_indicators_buy
        labels[buy_mask] = 1.0

        return pd.Series(labels, index=dataframe.index)

    def get_exit_signals(self, dataframe):

        min_indicators_sell = self.exit_min_indicators_sell.value
        n = len(dataframe["close"])
        labels = np.zeros(n, dtype=float)

        # Convert to numpy arrays, handling NaN
        close_norm = np.asarray(dataframe["close_norm"], dtype=float)
        adx_scaled = np.asarray(dataframe["adx_scaled"], dtype=float)
        bb_width = np.asarray(dataframe["bb_width"], dtype=float)
        guard_metric = np.asarray(dataframe["guard_metric"], dtype=float)

        # Replace NaN with neutral values (0)
        close_norm = np.nan_to_num(close_norm, nan=0.0)
        adx_scaled = np.nan_to_num(adx_scaled, nan=0.0)
        bb_width = np.nan_to_num(bb_width, nan=0.0)
        guard_metric = np.nan_to_num(guard_metric, nan=0.0)

        # Sell signals: Overbought indicators + future loss validation
        sell_votes = np.zeros(n, dtype=int)
        sell_votes += (close_norm > self.exit_close_norm_threshold.value).astype(
            int
        )  # Close norm positive
        sell_votes += (adx_scaled > self.exit_adx_scaled_threshold.value).astype(
            int
        )  # ADX scaled negative
        sell_votes += (bb_width >= self.exit_bb_width_threshold.value).astype(
            int
        )  # BB width positive (volatility low)
        sell_votes += (guard_metric > self.exit_guard_metric_threshold.value).astype(
            int
        )  # Guard metric positive

        # Sell: enough indicators agree
        sell_mask = sell_votes >= min_indicators_sell
        labels[sell_mask] = 1.0

        return pd.Series(labels, index=dataframe.index)
