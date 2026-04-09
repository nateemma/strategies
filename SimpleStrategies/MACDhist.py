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

from finta import TA as fta

import warnings
warnings.filterwarnings("ignore", message="The objective has been evaluated at this point before.")

from SimpleStrategy import SimpleStrategy

'''
Classic MACD, using the histogram
'''
class MACDhist(SimpleStrategy):

    strategy_type = SimpleStrategy.StrategyType.OSCILLATOR

    plot_config = {
        'main_plot': {
            'close': {'color': 'lightsteelblue'},
        },
        'subplots': {
            "Diff": {
                'macdhist': {'color': 'lightskyblue'},
            },
        }
    }

    enable_guards = True # set to True for testing, False for debug

    # Buy hyperspace params:
    buy_params = {
        **SimpleStrategy.buy_params,
        "entry_fast_period": 14,
        "entry_guard_metric": -0.8,
        "entry_signal_period": 14,
        "entry_slow_period": 26,
    }

    # Sell hyperspace params:
    sell_params = {
        **SimpleStrategy.sell_params,
        "exit_guard_metric": 0.1,
    }

    # Strategy parameters
    entry_fast_period = IntParameter(6, 18, default=12, space='buy')
    entry_slow_period = IntParameter(12, 48, default=26, space='buy')
    entry_signal_period = IntParameter(6, 18, default=9, space='buy')

    def get_entry_signals(self, dataframe):

        if (self.entry_slow_period.value <= self.entry_fast_period.value):
            dataframe['macd'] = 0.0
            dataframe['macdsignal'] = 0.0
            dataframe['macdhist'] = 0.0
        else:
            macd = ta.MACD(dataframe, 
                        fastperiod=int(self.entry_fast_period.value), 
                        slowperiod=int(self.entry_slow_period.value), 
                        signalperiod=int(self.entry_signal_period.value))
            dataframe['macd'] = macd['macd']
            dataframe['macdsignal'] = macd['macdsignal']
            dataframe['macdhist'] = macd['macdhist']

        series = np.where(
            (
                qtpylib.crossed_above(dataframe['macdhist'], 0.0)
            ),
              1, 0)

        return series

    def get_exit_signals(self, dataframe):

        series = np.where(
            (
                qtpylib.crossed_below(dataframe['macdhist'], 0.0)
            ),
              1, 0)
        return series
