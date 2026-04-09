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
Chaikin Money Flow
'''
class CMF(SimpleStrategy):


    strategy_type = SimpleStrategy.StrategyType.TREND
    # entry_slope_column = "cmf"
    plot_config = {
        'main_plot': {
            'close': {'color': 'lightsteelblue'},
        },
        'subplots': {
            "Diff": {
                'cmf': {'color': 'lightskyblue'},
            },
        }
    }

    enable_guards = True # set to True for testing, False for debug
 
    # Buy hyperspace params:
    buy_params = {
        **SimpleStrategy.buy_params,
        "entry_cmf": 0.7,
        "entry_guard_metric": 0.0,
        "entry_win_size": 34,
    }

    # Sell hyperspace params:
    sell_params = {
        **SimpleStrategy.sell_params,
        "exit_cmf": -0.7,
        "exit_guard_metric": 0.6,
    }


    # Strategy parameters
    entry_win_size = IntParameter(6, 48, default=15, space='buy')
    entry_cmf = DecimalParameter(0.0, 1.0, default=0.8, decimals=1, space='buy')

    exit_cmf = DecimalParameter(-1.0, -0.0, default=-0.8, decimals=1, space='sell')


    def get_entry_signals(self, dataframe):

        dataframe['cmf'] = pta.cmf(dataframe['high'], dataframe['low'], dataframe['close'], dataframe['volume'], 
                                   length=int(self.entry_win_size.value))

        series = np.where(
            qtpylib.crossed_below(dataframe['cmf'], self.entry_cmf.value),
              1, 0)

        return series


    def get_exit_signals(self, dataframe):

        series = np.where(
            qtpylib.crossed_above(dataframe['cmf'], self.exit_cmf.value),
              1, 0)
        return series
