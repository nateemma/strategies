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
Classic EMA Crossing
'''
class EMACross(SimpleStrategy):


    strategy_type = SimpleStrategy.StrategyType.OTHER

    plot_config = {
        'main_plot': {
            'close': {'color': 'lightsteelblue'},
            'ema_short': {'color': 'lightseagreen'},
            'ema_long': {'color': 'lightsalmon'},
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
        "entry_long_period": 117,
        "entry_short_period": 22,
    }

    # Sell hyperspace params:
    sell_params = {
        **SimpleStrategy.sell_params,
        "exit_guard_metric": 0.7,
    }

    # Strategy parameters

    entry_short_period = IntParameter(6, 24, default=6, space='buy', load=True, optimize=True)
    entry_long_period = IntParameter(24, 256, default=48, space='buy', load=True, optimize=True)

    def get_entry_signals(self, dataframe):

        if self.entry_short_period.value >= self.entry_long_period.value:
            dataframe['ema_short'] = 0.0
            dataframe['ema_long'] = 0.0
        else:
            dataframe['ema_short'] = fta.EMA(dataframe, period=self.entry_short_period.value) # type: ignore
            dataframe['ema_long'] = fta.EMA(dataframe, period=self.entry_long_period.value) # type: ignore

        series = np.where(
            (
                (dataframe['ema_short'] > dataframe['ema_long']) &
                (dataframe['close'] < dataframe['ema_short'])
            ),
            1, 0)

        return series


    def get_exit_signals(self, dataframe):

        series = np.where(
            (
                (dataframe['ema_short'] < dataframe['ema_long'])
            ),
            1, 0)
        return series
