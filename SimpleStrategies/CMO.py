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
CMO - Claude Momentum Oscillator
'''
class CMO(SimpleStrategy):


    strategy_type = SimpleStrategy.StrategyType.OSCILLATOR
    # entry_slope_column = "cmo"

    plot_config = {
        'main_plot': {
            'close': {'color': 'lightsteelblue'},
        },
        'subplots': {
            "Diff": {
                'cmo': {'color': 'lightskyblue'},
                'candle_imbalance': {'color': 'lightcoral'},
            },
        }
    }

    enable_guards = True # set to True for testing, False for debug

    # Buy hyperspace params:
    buy_params = {
        **SimpleStrategy.buy_params,
        "entry_cmo": -66.0,
        "entry_guard_metric": -0.5,
        "entry_period": 16,
    }

    # Sell hyperspace params:
    sell_params = {
        **SimpleStrategy.sell_params,
        "exit_cmo": 68.0,
        "exit_guard_metric": 0.5,
    }


    # Strategy parameters
    entry_cmo = DecimalParameter(-70.0, -50.0, default=-60.0, decimals=0, space='buy')
    entry_period = IntParameter(6, 24, default=9, space='buy', load=True, optimize=True)

    exit_cmo = DecimalParameter(60.0, 90.0, default=80.0, decimals=0, space='sell')


    def get_entry_signals(self, dataframe):

        dataframe["cmo"] = fta.CMO(dataframe, period=int(self.entry_period.value))

        series = np.where(
            (dataframe['cmo'] < self.entry_cmo.value),
              1, 0)

        return series


    def get_exit_signals(self, dataframe):

        series = np.where(
            (dataframe['cmo'] > self.exit_cmo.value),
              1, 0)
        return series
