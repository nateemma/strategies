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
RVOL = Relative Volume
'''
class RVOL(SimpleStrategy):


    strategy_type = SimpleStrategy.StrategyType.OSCILLATOR
    plot_config = {
        'main_plot': {
            'close': {'color': 'lightsteelblue'},
        },
        'subplots': {
            "Diff": {
                'rvol': {'color': 'lightskyblue'},
            },
        }
    }


    enable_guards = True # set to True for testing, False for debug

    # Buy hyperspace params:
    buy_params = {
        **SimpleStrategy.buy_params,
        "entry_guard_metric": 0.0,
        "entry_period": 21,
        "entry_rvol": 3.0,
    }

    # Sell hyperspace params:
    sell_params = {
        **SimpleStrategy.sell_params,
        "exit_guard_metric": 0.3,
        "exit_rvol": 1.0,
    }

    # Strategy parameters
    entry_rvol = DecimalParameter(2.0, 5.0, default=2.0, decimals=1, space='buy')
    entry_period = IntParameter(4, 24, default=5, space='buy', load=True, optimize=True)

    exit_rvol = DecimalParameter(0.0, 2.0, default=0.5, decimals=1, space='sell')

    def get_entry_signals(self, dataframe):

        dataframe["rvol"] = dataframe["volume"] / dataframe["volume"].ewm(span=int(self.entry_period.value), adjust=False).mean() 

        series = np.where(
            qtpylib.crossed_above(dataframe['rvol'], self.entry_rvol.value),
              1, 0)

        return series


    def get_exit_signals(self, dataframe):

        series = np.where(
            qtpylib.crossed_below(dataframe['rvol'], self.exit_rvol.value),
              1, 0)
        return series

