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
WFRACTAL - Williams Fractal
'''
class WFRACTAL(SimpleStrategy):


    strategy_type = SimpleStrategy.StrategyType.OSCILLATOR
    entry_slope_column = "wfractal"
    plot_config = {
        'main_plot': {
            'close': {'color': 'lightsteelblue'},
        },
        'subplots': {
            "Diff": {
            'wfractal': {'color': 'lightseagreen'},
            },
        }
    }

    enable_guards = True # set to True for testing, False for debug


    # Buy hyperspace params:
    buy_params = {
        **SimpleStrategy.buy_params,
        "entry_guard_metric": -0.3,
        "entry_period": 6,
    }

    # Sell hyperspace params:
    sell_params = {
        **SimpleStrategy.sell_params,
        "exit_guard_metric": 0.8,
    }


    # Strategy parameters
    entry_period = IntParameter(2, 32, default=8, space='buy', load=True, optimize=True)


    def get_entry_signals(self, dataframe):

        # the 'fractals' identify peaks and troughs, but the signal is 'period' candles in the past, so we need to shift the signals
        period = int(self.entry_period.value)
        fractal = fta.WILLIAMS_FRACTAL(dataframe, period=period)
        bull_fractal = fractal['BullishFractal']
        bear_fractal = fractal['BearishFractal']
        dataframe['wfractal'] = 0.0
        dataframe['wfractal'] = bull_fractal.shift(period) - bear_fractal.shift(period)

        series = np.where(
            (
            (dataframe['wfractal'] < 0.0) 
            ), 1, 0)

        return series

    def get_exit_signals(self, dataframe):

        series = np.where(
            (
            (dataframe['wfractal'] > 0.0) 
            ), 1, 0)
        return series
    