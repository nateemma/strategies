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
Parabolic SAR (Stop And Reverse)
'''
class SAR(SimpleStrategy):


    strategy_type = SimpleStrategy.StrategyType.TREND
    entry_slope_column = "sar"
    plot_config = {
        'main_plot': {
            'close': {'color': 'lightsteelblue'},
            'sar': {'color': 'lightseagreen'},
        },
        'subplots': {
            "Diff": {
            },
        }
    }

    enable_guards = True # set to True for testing, False for debug

    # Buy hyperspace params:
    buy_params = {
        **SimpleStrategy.buy_params,
        "entry_guard_metric": -0.8,
        "entry_period": 12,
    }

    # Sell hyperspace params:
    sell_params = {
        **SimpleStrategy.sell_params,
        "exit_guard_metric": 0.7,
    }

    # Strategy parameters
    entry_period = IntParameter(5, 24, default=6, space='buy')

    def get_entry_signals(self, dataframe):


        dataframe['sar'] = ta.SAR(dataframe, timeperiod=int(self.entry_period.value))

        series = np.where(
            (
            (dataframe["sar"] > dataframe['close']) 
            ), 1, 0)

        return series

    def get_exit_signals(self, dataframe):

        series = np.where(
            (
            (dataframe["sar"] < dataframe['close']) 
            ), 1, 0)
        return series
    