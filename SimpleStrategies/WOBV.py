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
warnings.filterwarnings("ignore", message="The objective has been evaluated at this point before.")

from SimpleStrategy import SimpleStrategy

from finta import TA as fta

'''
WOBV - Weighted On Balance Volume
'''
class WOBV(SimpleStrategy):

    strategy_type = SimpleStrategy.StrategyType.VOLATILITY
    plot_config = {
        "main_plot": {
            "close": {"color": "lightsteelblue"},
        },
        "subplots": {
            "Diff": {
                "wobv": {"color": "lightseagreen"},
                # "wobv_raw": {"color": "lightsteelblue"},
            },
        },
    }

    enable_guards = True # set to True for testing, False for debug

    # Buy hyperspace params:
    buy_params = {
        **SimpleStrategy.buy_params,
        "entry_guard_metric": -0.2,
        "entry_wobv": -20.0,
        "entry_wobv_distance": 1,
    }

    # Sell hyperspace params:
    sell_params = {
        **SimpleStrategy.sell_params,
        "exit_guard_metric": 0.7,
        "exit_wobv": 79.0,
    }

    # Strategy parameters
    entry_wobv = DecimalParameter(-100.0, 0.0, default=-20.0, decimals=0, space='buy')
    entry_wobv_distance = IntParameter(1, 8, default=1, space="buy", load=True, optimize=True)

    exit_wobv = DecimalParameter(0.0, 100.0, default=20.0, decimals=0, space='sell')

    def get_entry_signals(self, dataframe):

        # WOBV just returns a volume-derived number, so compare it to previous value and turn into percentage change
        dataframe["wobv_raw"] = fta.WOBV(dataframe)
        prev = dataframe["wobv_raw"].shift(int(self.entry_wobv_distance.value))
        dataframe["wobv"] = 100.0 * (dataframe["wobv_raw"] - prev) / prev
        dataframe["wobv"] = dataframe["wobv"].fillna(0.0)
        dataframe["wobv"] = dataframe["wobv"].clip(lower=-100.0, upper=100.0)

        series = np.where(
            (
            (dataframe['wobv'] < self.entry_wobv.value) 
            ), 1, 0)

        return series

    def get_exit_signals(self, dataframe):

        series = np.where(
            (
            (dataframe['wobv'] > self.exit_wobv.value) 
            ), 1, 0)
        return series
