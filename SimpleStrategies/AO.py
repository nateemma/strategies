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
Awesome Oscillator
"""


class AO(SimpleStrategy):

    strategy_type = SimpleStrategy.StrategyType.OSCILLATOR
    entry_slope_column = "aosc"
    plot_config = {
        "main_plot": {
            "close": {"color": "lightsteelblue"},
        },
        "subplots": {
            "Diff": {
                "aosc": {"color": "lightskyblue"},
            },
        },
    }

    enable_guards = True  # set to True for testing, False for debug

    # Buy hyperspace params:
    buy_params = {
        **SimpleStrategy.buy_params,
        "entry_fast_period": 6,
        "entry_guard_metric": -0.2,
        "entry_osc": 0.0,
        "entry_slow_period": 30,
    }

    # Sell hyperspace params:
    sell_params = {
        **SimpleStrategy.sell_params,
        "exit_guard_metric": 0.2,
        "exit_osc": 0.05,
    }

    # Strategy parameters
    entry_fast_period = IntParameter(
        3, 12, default=5, space="buy", load=True, optimize=True
    )
    entry_slow_period = IntParameter(
        16, 64, default=34, space="buy", load=True, optimize=True
    )
    entry_osc = DecimalParameter(
        -0.05, 0.0, default=-0.02, decimals=2, space="buy", load=True, optimize=True
    )
    exit_osc = DecimalParameter(
        0.0, 0.05, default=0.03, decimals=2, space="sell", load=True, optimize=True
    )

    def get_entry_signals(self, dataframe):

        # Awesome Oscillator
        dataframe["aosc"] = qtpylib.awesome_oscillator(
            dataframe,
            fast=self.entry_fast_period.value,
            slow=self.entry_slow_period.value,
        )

        series = np.where((dataframe["aosc"] < self.entry_osc.value), 1, 0)

        return series

    def get_exit_signals(self, dataframe):

        series = np.where((dataframe["aosc"] > self.exit_osc.value), 1, 0)

        return series
