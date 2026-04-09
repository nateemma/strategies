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


import warnings

warnings.filterwarnings(
    "ignore", message="The objective has been evaluated at this point before."
)

from SimpleStrategy import SimpleStrategy

"""
Bolinger Band Breakout
"""


class BBBreakout(SimpleStrategy):

    strategy_type = SimpleStrategy.StrategyType.VOLATILITY

    plot_config = {
        "main_plot": {
            "close": {"color": "lightsteelblue"},
            "bb_lowerband": {"color": "green"},
            "bb_upperband": {"color": "red"},
        },
        "subplots": {
            "Diff": {},
        },
    }

    enable_guards = True  # set to True for testing, False for debug

    # Buy hyperspace params:
    buy_params = {
        **SimpleStrategy.buy_params,
        "entry_guard_metric": 0.0,
        "entry_period": 24,
    }

    # Sell hyperspace params:
    sell_params = {
        **SimpleStrategy.sell_params,
        "exit_guard_metric": 0.1,
    }

    # Strategy parameters
    entry_period = IntParameter(6, 96, default=24, space="buy")

    def get_entry_signals(self, dataframe):

        bollinger = qtpylib.bollinger_bands(
            dataframe["close"], window=int(self.entry_period.value), stds=2
        )
        dataframe["bb_lowerband"] = bollinger["lower"]
        dataframe["bb_middleband"] = bollinger["mid"]
        dataframe["bb_upperband"] = bollinger["upper"]

        series = np.where(
            (
                (
                    qtpylib.crossed_above(dataframe["close"], dataframe["bb_upperband"])
                    & (dataframe["trend_mode"] == 1)
                )
                | (
                    qtpylib.crossed_above(dataframe["close"], dataframe["bb_lowerband"])
                    & (dataframe["trend_mode"] == 0)
                )
            ),
            1,
            0,
        )

        return series

    def get_exit_signals(self, dataframe):

        # series = np.where(
        #     qtpylib.crossed_below(dataframe["close"], dataframe["bb_upperband"]), 1, 0
        # )

        series = np.where(
            (
                (
                    qtpylib.crossed_below(dataframe["close"], dataframe["bb_upperband"])
                    & (dataframe["trend_mode"] == 1)
                )
                | (
                    qtpylib.crossed_above(dataframe["close"], dataframe["bb_upperband"])
                    & (dataframe["trend_mode"] == 0)
                )
            ),
            1,
            0,
        )
        return series
