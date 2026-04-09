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
COPP - Coppock Curve momentum indicator
'''
class COPP(SimpleStrategy):


    strategy_type = SimpleStrategy.StrategyType.OSCILLATOR
    entry_slope_column = "copp"
    enable_guards = True # set to True for testing, False for debug
 
    # Buy hyperspace params:
   
    # Buy hyperspace params:
    buy_params = {
        **SimpleStrategy.buy_params,
        "entry_copp": -5.0,
        "entry_guard_metric": -0.0,  # value loaded from strategy
    }

    # Sell hyperspace params:
    sell_params = {
        **SimpleStrategy.sell_params,
        "exit_copp": 1.0,
        "exit_guard_metric": 0.0,  # value loaded from strategy
    }


    # Strategy parameters
    entry_copp = DecimalParameter(-15.0, -0.0, default=-5.0, decimals=0, space='buy')

    exit_copp = DecimalParameter(0.0, 15.0, default=10.0, decimals=0, space='sell')


    def get_entry_signals(self, dataframe):

        dataframe["copp"] = fta.COPP(dataframe)

        series = np.where(
            (dataframe['copp'] < self.entry_copp.value),
              1, 0)

        return series


    def get_exit_signals(self, dataframe):

        series = np.where(
            (dataframe['copp'] > self.exit_copp.value),
              1, 0)
        return series
