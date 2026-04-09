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
Stochasic RSI
'''
class STOCHRSI(SimpleStrategy):


    strategy_type = SimpleStrategy.StrategyType.OSCILLATOR
    entry_slope_column = "stochrsi"
    plot_config = {
        'main_plot': {
            'close': {'color': 'lightsteelblue'},
        },
        'subplots': {
            "Diff": {
            'stochrsi': {'color': 'lightseagreen'},
            },
        }
    }

    enable_guards = True # set to True for testing, False for debug

    # Buy hyperspace params:
    buy_params = {
        **SimpleStrategy.buy_params,
        "entry_guard_metric": -0.5,
        "entry_rsi_period": 11,
        "entry_stoch_period": 22,
        "entry_stochrsi": 0.71,
    }

    # Sell hyperspace params:
    sell_params = {
        **SimpleStrategy.sell_params,
        "exit_guard_metric": 0.5,
        "exit_stochrsi": 0.34,
    }


    # Strategy parameters
    entry_stochrsi = DecimalParameter(0.0, 1.0, default=0.2, decimals=2, space='buy', load=True, optimize=True)
    entry_rsi_period = IntParameter(8, 64, default=14, space='buy', load=True, optimize=True)
    entry_stoch_period = IntParameter(8, 64, default=14, space='buy', load=True, optimize=True)

    exit_stochrsi = DecimalParameter(0.0, 1.0, default=0.8, decimals=2, space='sell', load=True, optimize=True)


    def get_entry_signals(self, dataframe):

        dataframe['stochrsi'] = fta.STOCHRSI(dataframe, 
                                             rsi_period=int(self.entry_rsi_period.value), 
                                             stoch_period=int(self.entry_stoch_period.value)
                                             )

        series = np.where(
            (
            # qtpylib.crossed_below(dataframe["stochrsi"], self.entry_stochrsi.value) 
            dataframe["stochrsi"] < self.entry_stochrsi.value
            ), 1, 0)

        return series

    def get_exit_signals(self, dataframe):

        series = np.where(
            (
            # qtpylib.crossed_above(dataframe["stochrsi"], self.exit_stochrsi.value) 
            dataframe["stochrsi"] > self.exit_stochrsi.value
            ), 1, 0)
        return series
    