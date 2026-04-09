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
VZO - Volume Zone Oscillator
'''
class VZO(SimpleStrategy):


    strategy_type = SimpleStrategy.StrategyType.OSCILLATOR
    entry_slope_column = "vzo"
    plot_config = {
        'main_plot': {
            'close': {'color': 'lightsteelblue'},
        },
        'subplots': {
            "Diff": {
            'vzo': {'color': 'lightseagreen'},
            },
        }
    }

    enable_guards = True # set to True for testing, False for debug

    # Buy hyperspace params:
    buy_params = {
        **SimpleStrategy.buy_params,
        "entry_guard_metric": -0.8,
        "entry_period": 9,
        "entry_vzo": -67.0,
    }

    # Sell hyperspace params:
    sell_params = {
        **SimpleStrategy.sell_params,
        "exit_guard_metric": 0.8,
        "exit_vzo": 83.0,
    }

    # Strategy parameters
    entry_vzo = DecimalParameter(-100.0, -40.0, default=-80.0, decimals=0, space='buy')
    entry_period = IntParameter(6, 48, default=14, space='buy', load=True, optimize=True)

    exit_vzo = DecimalParameter(40.0, 100.0, default=80.0, decimals=0, space='sell')


    def get_entry_signals(self, dataframe):

        dataframe["vzo"] = fta.VZO(dataframe, period=int(self.entry_period.value))

        series = np.where(
            (
            (dataframe['vzo'] < self.entry_vzo.value) 
            ), 1, 0)

        return series

    def get_exit_signals(self, dataframe):

        series = np.where(
            (
            (dataframe['vzo'] > self.exit_vzo.value) 
            ), 1, 0)
        return series
    