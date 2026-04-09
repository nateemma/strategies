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
Wave Trend Oscillator crossing strategy

Notes: 
    - periods must be long
    - The win% will be low, but it does tend to catch big jumps
'''
class WTO(SimpleStrategy):

    strategy_type = SimpleStrategy.StrategyType.OSCILLATOR

    plot_config = {
        'main_plot': {
            'close': {'color': 'lightsteelblue'},
        },
        'subplots': {
            "Diff": {
            'wt1': {'color': 'lightseagreen'},
            'wt2': {'color': 'lightcoral'},
            },
        }
    }

    enable_guards = True # set to True for testing, False for debug

    # Buy hyperspace params:
    buy_params = {
        **SimpleStrategy.buy_params,
        "entry_average_len": 21,
        "entry_channel_len": 38,
        "entry_guard_metric": -0.2,
    }

    # Sell hyperspace params:
    sell_params = {
        **SimpleStrategy.sell_params,
        "exit_guard_metric": 0.3,
    }

    enable_guards = True # set to True for testing, False for debug

    entry_average_len = IntParameter(6, 24, default=10, space='buy', load=True, optimize=True)
    entry_channel_len = IntParameter(12, 48, default=21, space='buy', load=True, optimize=True)

    def get_entry_signals(self, dataframe):

        wto = fta.WTO(dataframe, 
                      channel_lenght=self.entry_channel_len.value,
                      average_lenght=self.entry_average_len.value)
        dataframe['wt1'] = wto['WT1.']
        dataframe['wt2'] = wto['WT2.']

        series = np.where(
            (
            qtpylib.crossed_above(dataframe['wt2'], dataframe['wt1']) 
            ), 1, 0)

        return series

    def get_exit_signals(self, dataframe):

        series = np.where(
            (
            qtpylib.crossed_below(dataframe['wt2'], dataframe['wt1']) 
            ), 1, 0)
        return series
    