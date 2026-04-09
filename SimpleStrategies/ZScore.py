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
Z-Score Mean Reversion
'''
class ZScore(SimpleStrategy):

    strategy_type = SimpleStrategy.StrategyType.OSCILLATOR

    plot_config = {
        'main_plot': {
            'close': {'color': 'lightsteelblue'},
        },
        'subplots': {
            "Diff": {
            'zscore': {'color': 'lightseagreen'},
            },
        }
    }

    enable_guards = True # set to True for testing, False for debug
    
    # Buy hyperspace params:
    buy_params = {
        **SimpleStrategy.buy_params,
        "entry_guard_metric": -0.3,
        "entry_period": 31,
        "entry_zscore": -2.6,
    }

    # Sell hyperspace params:
    sell_params = {
        **SimpleStrategy.sell_params,
        "exit_guard_metric": 0.7,
        "exit_zscore": 2.96,
    }

    enable_guards = True # set to True for testing, False for debug

    entry_period = IntParameter(12, 64, default=20, space='buy', load=True, optimize=True)
    entry_zscore = DecimalParameter(-3.0, -1.0, default=-2.0, decimals=1, space='buy', load=True, optimize=True)

    exit_zscore = DecimalParameter(1.0, 3.0, default=2.0, decimals=2, space='sell', load=True, optimize=True)

    def get_entry_signals(self, dataframe):

         # Calculate the rolling mean and standard deviation
        dataframe['rolling_mean'] = dataframe['close'].rolling(window=self.entry_period.value).mean()
        dataframe['rolling_std'] = dataframe['close'].rolling(window=self.entry_period.value).std()

        # Calculate the z-score
        dataframe['zscore'] = (dataframe['close'] - dataframe['rolling_mean']) / dataframe['rolling_std']

        series = np.where(
            (
            (dataframe['zscore'] < self.entry_zscore.value) 
            ), 1, 0)

        return series

    def get_exit_signals(self, dataframe):

        series = np.where(
            (
            (dataframe['zscore'] > self.entry_zscore.value) 
            ), 1, 0)
        return series
    