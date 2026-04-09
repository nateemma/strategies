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
MFI - Money Flow Indicator
'''
class MFI(SimpleStrategy):


    strategy_type = SimpleStrategy.StrategyType.OSCILLATOR
    entry_slope_column = "mfi"
    plot_config = {
        'main_plot': {
            'close': {'color': 'lightsteelblue'},
        },
        'subplots': {
            "Diff": {
                'mfi': {'color': 'lightskyblue'},
            },
        }
    }


    enable_guards = True # set to True for testing, False for debug

    # Buy hyperspace params:
    buy_params = {
        **SimpleStrategy.buy_params,
        "entry_guard_metric": -0.5,
        "entry_mfi": 18.0,
        "entry_period": 20,
    }

    # Sell hyperspace params:
    sell_params = {
        **SimpleStrategy.sell_params,
        "exit_guard_metric": 0.6,
        "exit_mfi": 99.0,
    }

    # Strategy parameters
    entry_period = IntParameter(6, 48, default=14, space='buy')
    entry_mfi = DecimalParameter(0.0, 20.0, default=10.0, decimals=0, space='buy')

    exit_mfi = DecimalParameter(80.0, 100.0, default=90.0, decimals=0, space='sell')

    def get_entry_signals(self, dataframe):

        dataframe["mfi"] = ta.MFI(dataframe, timeperiod=int(self.entry_period.value)).fillna(0.0)

        series = np.where(
            (dataframe['mfi'] < self.entry_mfi.value),
              1, 0)

        return series


    def get_exit_signals(self, dataframe):

        series = np.where(
            (dataframe['mfi'] > self.exit_mfi.value),
              1, 0)
        return series
