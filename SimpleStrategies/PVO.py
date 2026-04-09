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
Percentage Volume Oscillator - similar to On Balance Volume, but produces a normalised value (i.e. not pair-specific)
'''
class PVO(SimpleStrategy):


    strategy_type = SimpleStrategy.StrategyType.OSCILLATOR
    entry_slope_column = "pvo"
    plot_config = {
        'main_plot': {
            'close': {'color': 'lightsteelblue'},
        },
        'subplots': {
            "Diff": {
                'pvo': {'color': 'lightskyblue'},
            },
        }
    }


    enable_guards = True # set to True for testing, False for debug

    # Buy hyperspace params:
    buy_params = {
        **SimpleStrategy.buy_params,
        "entry_guard_metric": -0.3,
        "entry_pvo": -56.0,
    }

    # Sell hyperspace params:
    sell_params = {
        **SimpleStrategy.sell_params,
        "exit_guard_metric": 0.7,
        "exit_pvo": 73.0,
    }


    # Strategy parameters
    entry_pvo = DecimalParameter(-100.0, 0.0, default=-80.0, decimals=0, space='buy')
    entry_fast_period = IntParameter(2, 24, default=6, space='buy', load=True, optimize=True)
    entry_slow_period = IntParameter(6, 64, default=15, space='buy', load=True, optimize=True)

    exit_pvo = DecimalParameter(0.0, 100.0, default=80.0, decimals=0, space='sell')

    def get_entry_signals(self, dataframe):

        dataframe['pvo'] = self.calculate_pvo(dataframe, 
                                              fast_period=int(self.entry_fast_period.value), 
                                              slow_period=int(self.entry_slow_period.value))

        series = np.where(
            (dataframe['pvo'] < self.entry_pvo.value),
              1, 0)

        return series


    def get_exit_signals(self, dataframe):

        series = np.where(
            (dataframe['pvo'] > self.exit_pvo.value),
              1, 0)
        return series

    def calculate_pvo(self, dataframe, fast_period=12, slow_period=26):
        fast_ema = ta.EMA(dataframe['volume'], timeperiod=fast_period)
        slow_ema = ta.EMA(dataframe['volume'], timeperiod=slow_period)
        pvo = ((fast_ema - slow_ema) / slow_ema) * 100
        return pvo
