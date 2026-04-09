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
Elder's Force Index
'''
class EFI(SimpleStrategy):


    strategy_type = SimpleStrategy.StrategyType.OSCILLATOR
    entry_slope_column = "efi"
    plot_config = {
        'main_plot': {
            'close': {'color': 'lightsteelblue'},
        },
        'subplots': {
            "Diff": {
                'efi': {'color': 'lightskyblue'},
            },
        }
    }

    enable_guards = True # set to True for testing, False for debug

    # Buy hyperspace params:
    buy_params = {
        **SimpleStrategy.buy_params,
        "entry_efi": -7.0,
        "entry_guard_metric": -0.2,
        "entry_win_size": 24,
    }

    # Sell hyperspace params:
    sell_params = {
        **SimpleStrategy.sell_params,
        "exit_efi": 43.0,
        "exit_guard_metric": 0.3,
    }


    # Strategy parameters
    entry_win_size = IntParameter(6, 48, default=13, space='buy')
    entry_efi = DecimalParameter(-10.0, -0.0, default=-8.0, decimals=0, space='buy')

    exit_efi = DecimalParameter(40.0, 60.0, default=8.0, decimals=0, space='sell')


    def get_entry_signals(self, dataframe):

        dataframe['efi'] = fta.EFI(dataframe, period=int(self.entry_win_size.value)) / 20.0

        series = np.where(
            (dataframe['efi'] > self.entry_efi.value),
              1, 0)

        return series


    def get_exit_signals(self, dataframe):

        series = np.where(
            (dataframe['efi'] < self.exit_efi.value),
              1, 0)
        return series
