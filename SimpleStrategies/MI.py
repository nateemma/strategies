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
Dorsey Mass Index (MI)
'''
class MI(SimpleStrategy):


    strategy_type = SimpleStrategy.StrategyType.OSCILLATOR
    entry_slope_column = "mi"
    plot_config = {
        'main_plot': {
            'close': {'color': 'lightsteelblue'},
        },
        'subplots': {
            "Diff": {
                'mi': {'color': 'lightskyblue'},
            },
        }
    }


    enable_guards = True # set to True for testing, False for debug

    # Buy hyperspace params:
    buy_params = {
        **SimpleStrategy.buy_params,
        "entry_guard_metric": -0.8,
        "entry_mi": 46.0,
        "entry_period": 47,
    }

    # Sell hyperspace params:
    sell_params = {
        **SimpleStrategy.sell_params,
        "exit_guard_metric": 0.0,
        "exit_mi": 15.0,
    }


    # Strategy parameters
    entry_period = IntParameter(6, 48, default=14, space='buy')
    entry_mi = DecimalParameter(0.0, 50.0, default=25.0, decimals=0, space='buy')

    exit_mi = DecimalParameter(0.0, 50.0, default=25.0, decimals=0, space='sell')

    def get_entry_signals(self, dataframe):

        dataframe["mi"] = fta.MI(dataframe, period=int(self.entry_period.value)).fillna(0.0)

        series = np.where(
            (dataframe['mi'] < self.entry_mi.value),
              1, 0)

        return series


    def get_exit_signals(self, dataframe):

        series = np.where(
            (dataframe['mi'] > self.exit_mi.value),
              1, 0)
        return series

