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
VORTEX Indicator/signal
'''
class VORTEX(SimpleStrategy):

    strategy_type = SimpleStrategy.StrategyType.OSCILLATOR

    plot_config = {
        'main_plot': {
            'close': {'color': 'lightsteelblue'},
        },
        'subplots': {
            "Diff": {
            'vortex': {'color': 'lightseagreen'},
            },
        }
    }

    enable_guards = True # set to True for testing, False for debug

    # Buy hyperspace params:
    buy_params = {
        **SimpleStrategy.buy_params,
        "entry_guard_metric": -0.6,
        "entry_period": 25,
        "entry_vortex": -0.5,
    }

    # Sell hyperspace params:
    sell_params = {
        **SimpleStrategy.sell_params,
        "exit_guard_metric": 0.8,
        "exit_vortex": 0.1,
    }

    # Strategy parameters
    entry_vortex = DecimalParameter(-1.0, 0.0, default=-0.5, decimals=1, space='buy', load=True, optimize=True)
    entry_period = IntParameter(6, 48, default=14, space='buy', load=True, optimize=True)

    exit_vortex = DecimalParameter(0.0, 1.0, default=0.8, decimals=1, space='sell', load=True, optimize=True)

    def get_entry_signals(self, dataframe):

        vortex = fta.VORTEX(dataframe, period=int(self.entry_period.value))
        dataframe["VIp"] = vortex['VIp']
        dataframe["VIm"] = vortex['VIm']
        dataframe["vortex"] = vortex['VIp'] - vortex['VIm']

        series = np.where(
            (
            (dataframe['vortex'] < self.entry_vortex.value) 
            ), 1, 0)

        return series

    def get_exit_signals(self, dataframe):

        series = np.where(
            (
            (dataframe['vortex'] > self.exit_vortex.value) 
            ), 1, 0)
        return series
    