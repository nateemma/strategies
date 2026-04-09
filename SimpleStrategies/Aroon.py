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

'''
Aroon Oscillator
'''
class Aroon(SimpleStrategy):

    strategy_type = SimpleStrategy.StrategyType.OSCILLATOR

    plot_config = {
        'main_plot': {
            'close': {'color': 'lightsteelblue'},
        },
        'subplots': {
            "Diff": {
                'aroonup': {'color': 'lightseagreen'},
                'aroondown': {'color': 'lightsalmon'},
                'aroonosc': {'color': 'lightskyblue'},
            },
        }
    }

    enable_guards = True # set to True for testing, False for debug

    # Buy hyperspace params:
    buy_params = {
        **SimpleStrategy.buy_params,
        "entry_aroon_up": 58.0,
        "entry_guard_metric": 0.0,
        "entry_osc": 62.0,
        "entry_win_size": 91,
        "exit_aroon_down": 59.0,
    }

    # Sell hyperspace params:
    sell_params = {
        **SimpleStrategy.sell_params,
        "exit_guard_metric": 0.1,
        "exit_osc": -45.0,
    }

    # Strategy parameters
    entry_osc = DecimalParameter(25.0, 99.0, default=50.0, decimals=0, space='buy')
    entry_aroon_up = DecimalParameter(50.0, 99.0, default=70.0, decimals=0, space='buy')
    entry_win_size = IntParameter(6, 96, default=14, space='buy')

    exit_osc = DecimalParameter(-99.0, -25.0, default=-50.0, decimals=0, space='sell')
    exit_aroon_down = DecimalParameter(50.0, 99.0, default=70.0, decimals=0, space='buy')

    def get_entry_signals(self, dataframe):

        # Aroon, Aroon Oscillator
        aroon = ta.AROON(dataframe, window=int(self.entry_win_size.value))
        dataframe['aroonup'] = aroon['aroonup']
        dataframe['aroondown'] = aroon['aroondown']
        dataframe['aroonosc'] = ta.AROONOSC(dataframe, timeperiod=int(self.entry_win_size.value))

        series = np.where(
            (dataframe["aroonup"] > self.entry_aroon_up.value) & 
            (dataframe["aroonosc"] > self.entry_osc.value), 
            1, 0
        )

        return series

    def get_exit_signals(self, dataframe):

        series = np.where(
            (dataframe["aroondown"] > self.exit_aroon_down.value) & 
            (dataframe["aroonosc"] < self.entry_osc.value), 
            1, 0
        )

        return series
