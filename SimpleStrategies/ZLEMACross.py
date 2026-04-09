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
Classic ZLEMA (Zero Lag EMA) Crossing
'''
class ZLEMACross(SimpleStrategy):

    strategy_type = SimpleStrategy.StrategyType.TREND

    plot_config = {
        'main_plot': {
            'close': {'color': 'lightsteelblue'},
            'zlema_short': {'color': 'lightseagreen'},
            'zlema_long': {'color': 'lightsalmon'},
        },
        'subplots': {
            "Diff": {
            },
        }
    }

    enable_guards = True # set to True for testing, False for debug
 
    # Buy hyperspace params:
    buy_params = {
        **SimpleStrategy.buy_params,
        "entry_guard_metric": -0.1,
        "entry_long_period": 115,
        "entry_short_period": 23,
    }

    # Sell hyperspace params:
    sell_params = {
        **SimpleStrategy.sell_params,
        "exit_guard_metric": 0.6,
    }

    # Strategy parameters

    entry_short_period = IntParameter(6, 24, default=6, space='buy', load=True, optimize=True)
    entry_long_period = IntParameter(12, 128, default=48, space='buy', load=True, optimize=True)

    def get_entry_signals(self, dataframe):

        if self.entry_short_period.value >= self.entry_long_period.value:
            dataframe['zlema_short'] = 0.0
            dataframe['zlema_long'] = 0.0
        else:
            dataframe['zlema_short'] = fta.ZLEMA(dataframe, period=self.entry_short_period.value) # type: ignore
            dataframe['zlema_long'] = fta.ZLEMA(dataframe, period=self.entry_long_period.value) # type: ignore

        series = np.where(
            (
                (dataframe['zlema_short'] > dataframe['zlema_long']) &
                (dataframe['close'] < dataframe['zlema_short'])
            ),
            1, 0)

        return series

    def get_exit_signals(self, dataframe):

        series = np.where(
            (
                (dataframe['zlema_short'] < dataframe['zlema_long'])
            ),
            1, 0)
        return series
