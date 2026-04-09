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
Volume Flow Indicator (VFI)
'''
class VFI(SimpleStrategy):


    strategy_type = SimpleStrategy.StrategyType.OSCILLATOR
    entry_slope_column = "vfi"
    plot_config = {
        'main_plot': {
            'close': {'color': 'lightsteelblue'},
        },
        'subplots': {
            "Diff": {
            'vfi': {'color': 'lightseagreen'},
            },
        }
    }

    enable_guards = True # set to True for testing, False for debug

    # Buy hyperspace params:
    buy_params = {
        **SimpleStrategy.buy_params,
        "entry_factor": 0.9,
        "entry_guard_metric": -0.1,
        "entry_period": 34,
        "entry_smooth": 12,
        "entry_vfactor": 5.0,
        "entry_vfi": -14.0,
    }

    # Sell hyperspace params:
    sell_params = {
        **SimpleStrategy.sell_params,
        "exit_guard_metric": 0.2,
        "exit_vfi": 7.0,
    }

    # Strategy parameters
    entry_vfi = DecimalParameter(-50.0, 0.0, default=-15.0, decimals=0, space='buy')
    entry_period = IntParameter(32, 160, default=130, space='buy')
    entry_smooth = IntParameter(2, 16, default=3, space='buy')
    entry_factor = DecimalParameter(0.1, 1.0, default=0.2, decimals=1, space='buy')
    entry_vfactor = DecimalParameter(1.0, 5.0, default=2.5, decimals=0, space='buy')

    exit_vfi = DecimalParameter(0.0, 50.0, default=20.0, decimals=0, space='sell')

    def get_entry_signals(self, dataframe):

        dataframe['vfi'] = fta.VFI(dataframe, 
                                   period=int(self.entry_period.value))

        series = np.where(
            (
            (dataframe['vfi'] < self.entry_vfi.value) 
            ), 1, 0)

        return series

    def get_exit_signals(self, dataframe):

        series = np.where(
            (
            (dataframe['vfi'] > self.exit_vfi.value) 
            ), 1, 0)
        return series
    