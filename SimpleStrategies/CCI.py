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
Commodity Channel Index: values [Oversold:-100, Overbought:100]
'''
class CCI(SimpleStrategy):


    strategy_type = SimpleStrategy.StrategyType.OSCILLATOR
    entry_slope_column = "cci"
    plot_config = {
        'main_plot': {
            'close': {'color': 'lightsteelblue'},
        },
        'subplots': {
            "Diff": {
                'cci': {'color': 'lightskyblue'},
            },
        }
    }

    enable_guards = True # set to True for testing, False for debug
 
    # Buy hyperspace params:
    buy_params = {
        **SimpleStrategy.buy_params,
        "entry_cci": -236.0,
        "entry_guard_metric": 0.0,
        "entry_period": 28,
    }

    # Sell hyperspace params:
    sell_params = {
        **SimpleStrategy.sell_params,
        "exit_cci": 159.2,
        "exit_guard_metric": 0.8,
    }

    # Strategy parameters
    entry_cci = DecimalParameter(-250.0, -140.0, default=-200.0, decimals=0, space='buy', load=True, optimize=True)
    entry_period = IntParameter(14, 50, default=14, space='buy', load=True, optimize=True)

    exit_cci = DecimalParameter(140.0, 250.0, default=200.0, decimals=1, space='sell', load=True, optimize=True)


    def get_entry_signals(self, dataframe):

        dataframe['cci'] = ta.CCI(dataframe, timeperiod=int(self.entry_period.value)) # type: ignore

        series = np.where(
            (dataframe['cci'] < self.entry_cci.value),
              1, 0)

        return series


    def get_exit_signals(self, dataframe):

        series = np.where(
            (dataframe['cci'] > self.exit_cci.value),
              1, 0)
        return series
