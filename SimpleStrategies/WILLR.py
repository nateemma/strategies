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
WILLR - Williams %R
(Careful, this generates a lot of signals)
'''
class WILLR(SimpleStrategy):


    strategy_type = SimpleStrategy.StrategyType.OSCILLATOR
    entry_slope_column = "wobv"
    plot_config = {
        'main_plot': {
            'close': {'color': 'lightsteelblue'},
        },
        'subplots': {
            "Diff": {
            'wobv': {'color': 'lightseagreen'},
            },
        }
    }

    enable_guards = True # set to True for testing, False for debug

    # Buy hyperspace params:
    buy_params = {
        **SimpleStrategy.buy_params,
        "entry_guard_metric": -0.4,
        "entry_period": 58,
        "entry_wobv": -58.0,
    }

    # Sell hyperspace params:
    sell_params = {
        **SimpleStrategy.sell_params,
        "exit_guard_metric": 0.6,
        "exit_wobv": -49.0,
    }


    # Strategy parameters
    entry_wobv = DecimalParameter(-100.0, -50.0, default=-80.0, decimals=0, space='buy')
    entry_period = IntParameter(6, 64, default=24, space='buy')

    exit_wobv = DecimalParameter(-50.0, -0.0, default=-20.0, decimals=0, space='sell')


    def get_entry_signals(self, dataframe):

        dataframe['wobv'] = fta.WILLIAMS(dataframe, period=int(self.entry_period.value))

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
    