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

from finta import TA as fta

"""
ROC = Rate of Change
"""


class ROC(SimpleStrategy):

    strategy_type = SimpleStrategy.StrategyType.TREND
    entry_slope_column = "roc"
    plot_config = {
        "main_plot": {
            "close": {"color": "lightsteelblue"},
        },
        "subplots": {
            "Diff": {
                "roc": {"color": "lightskyblue"},
                "candle_imbalance": {"color": "lightcoral"},
            },
        },
    }

    enable_guards = True  # set to True for testing, False for debug

    # Buy hyperspace params:
    buy_params = {
        **SimpleStrategy.buy_params,
        "entry_enable_guards": False,
        "entry_guard_metric": -0.0,
        "entry_period": 17,
        "entry_roc": -3.0,
    }

    # Sell hyperspace params:
    sell_params = {
        **SimpleStrategy.sell_params,
        "exit_guard_metric": 0.0,
        "exit_roc": 2.8,
    }

    # Strategy parameters
    entry_roc = DecimalParameter(-5.0, -1.0, default=-3.0, decimals=1, space="buy")
    entry_period = IntParameter(
        6, 32, default=14, space="buy", load=True, optimize=True
    )

    exit_roc = DecimalParameter(0.0, 5.0, default=3.0, decimals=1, space="sell")

    def get_entry_signals(self, dataframe):

        dataframe["roc"] = ta.ROC(
            dataframe, timeperiod=int(self.entry_period.value)
        ).fillna(0.0)

        series = np.where(
            (
                (dataframe["roc"] < self.entry_roc.value)
                & (dataframe["roc"] > dataframe["roc"].shift(1))
            ),
            1,
            0,
        )

        # # DEBUG
        # entry_count = np.sum(series)
        # self.debug_print(f"ROC entry_count: {entry_count}")

        return series

    def get_exit_signals(self, dataframe):

        series = np.where(
            (dataframe["roc"] > self.exit_roc.value)
            & (dataframe["candle_imbalance"] > 0.5),
            1,
            0,
        )
        return series
