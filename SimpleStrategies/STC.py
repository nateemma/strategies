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
Schaff Trend Cycle
'''
class STC(SimpleStrategy):


    strategy_type = SimpleStrategy.StrategyType.OSCILLATOR
    entry_slope_column = "stc"
    plot_config = {
        'main_plot': {
            'close': {'color': 'lightsteelblue'},
        },
        'subplots': {
            "Diff": {
            'stc': {'color': 'lightseagreen'},
            },
        }
    }

    enable_guards = True # set to True for testing, False for debug

    # Buy hyperspace params:
    buy_params = {
        **SimpleStrategy.buy_params,
        "entry_guard_metric": -0.5,
        "entry_period_fast": 14,
        "entry_period_slow": 58,
        "entry_stc": 12.0,
    }

    # Sell hyperspace params:
    sell_params = {
        **SimpleStrategy.sell_params,
        "exit_guard_metric": 0.6,
        "exit_stc": 81.0,
    }


    # Strategy parameters
    entry_period_fast = IntParameter(6, 32, default=23, space='buy')
    entry_period_slow = IntParameter(24, 64, default=50, space='buy')
    entry_stc = DecimalParameter(0.0, 20.0, default=0.0, decimals=0, space='buy')

    exit_stc = DecimalParameter(80.0, 100.0, default=100.0, decimals=0, space='sell')

    def get_entry_signals(self, dataframe):


        if self.entry_period_fast.value < self.entry_period_slow.value:
            dataframe['stc'] = fta.STC(dataframe, 
                                    period_fast=int(self.entry_period_fast.value), 
                                    period_slow=int(self.entry_period_slow.value))
        else:
            dataframe['stc'] = 0.0

        series = np.where(
            (
            (dataframe["stc"] > self.entry_stc.value) 
            ), 1, 0)

        return series

    def get_exit_signals(self, dataframe):

        series = np.where(
            (
            (dataframe["stc"] < self.exit_stc.value) 
            ), 1, 0)
        return series
    