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
Classic SMA Crossing
'''
class SMACross(SimpleStrategy):

    strategy_type = SimpleStrategy.StrategyType.TREND

    plot_config = {
        'main_plot': {
            'close': {'color': 'lightsteelblue'},
            'sma_short': {'color': 'lightseagreen'},
            'sma_long': {'color': 'lightsalmon'},
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
        "entry_long_period": 32,
        "entry_short_period": 6,
        "entry_guard_metric": -0.0,  # value loaded from strategy
    }

    # Sell hyperspace params:
    sell_params = {
        **SimpleStrategy.sell_params,
        "exit_guard_metric": 0.0,  # value loaded from strategy
    }

    # Strategy parameters

    entry_short_period = IntParameter(6, 24, default=6, space='buy', load=True, optimize=True)
    entry_long_period = IntParameter(12, 128, default=48, space='buy', load=True, optimize=True)

    def get_entry_signals(self, dataframe):

        if self.entry_short_period.value >= self.entry_long_period.value:
            dataframe['sma_short'] = 0.0
            dataframe['sma_long'] = 0.0
        else:
            dataframe['sma_short'] = fta.SMA(dataframe, period=self.entry_short_period.value) # type: ignore
            dataframe['sma_long'] = fta.SMA(dataframe, period=self.entry_long_period.value) # type: ignore

        series = np.where(
            (
                (dataframe['sma_short'] > dataframe['sma_long']) &
                (dataframe['close'] < dataframe['sma_short'])
            ),
            1, 0)

        return series

    def get_exit_signals(self, dataframe):

        series = np.where(
            (
                (dataframe['sma_short'] < dataframe['sma_long'])
            ),
            1, 0)
        return series
