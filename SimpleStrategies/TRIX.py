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
TRIX - 1-day Rate-Of-Change (ROC) of a Triple Smooth EMA
'''
class TRIX(SimpleStrategy):


    strategy_type = SimpleStrategy.StrategyType.OSCILLATOR
    entry_slope_column = "trix"
    plot_config = {
        'main_plot': {
            'close': {'color': 'lightsteelblue'},
        },
        'subplots': {
            "Diff": {
            'trix': {'color': 'lightseagreen'},
            },
        }
    }

    enable_guards = True # set to True for testing, False for debug

    # Buy hyperspace params:
    buy_params = {
        **SimpleStrategy.buy_params,
        "entry_guard_metric": -0.5,
        "entry_period": 11,
        "entry_trix": -0.2,
    }

    # Sell hyperspace params:
    sell_params = {
        **SimpleStrategy.sell_params,
        "exit_guard_metric": 0.8,
        "exit_trix": 0.3,
    }


    # Strategy parameters
    entry_period = IntParameter(8, 64, default=14, space='buy', load=True, optimize=True)
    entry_trix = DecimalParameter(-1.0, -0.1, default=-0.1, decimals=2, space='buy')

    exit_trix = DecimalParameter(0.1, 1.0, default=0.3, decimals=2, space='sell')


    def get_entry_signals(self, dataframe):

        dataframe['trix'] = ta.TRIX(dataframe, timeperiod=int(self.entry_period.value))

        series = np.where(
            (
            (dataframe["trix"] < self.entry_trix.value) 
            ), 1, 0)

        return series

    def get_exit_signals(self, dataframe):

        series = np.where(
            (
            (dataframe["trix"] > self.exit_trix.value) 
            ), 1, 0)
        return series
    