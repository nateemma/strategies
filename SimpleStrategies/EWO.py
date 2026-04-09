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
Elliot Wave Oscillator
'''
class EWO(SimpleStrategy):


    strategy_type = SimpleStrategy.StrategyType.OSCILLATOR
    entry_slope_column = "ewo"
    plot_config = {
        'main_plot': {
            'close': {'color': 'lightsteelblue'},
        },
        'subplots': {
            "Diff": {
                'ewo': {'color': 'lightskyblue'},
            },
        }
    }

    enable_guards = True # set to True for testing, False for debug

    # Buy hyperspace params:
    buy_params = {
        **SimpleStrategy.buy_params,
        "entry_guard_metric": -0.2,
        "entry_long_period": 62,
        "entry_osc": -2.3,
        "entry_short_period": 4,
    }

    # Sell hyperspace params:
    sell_params = {
        **SimpleStrategy.sell_params,
        "exit_guard_metric": 0.7,
        "exit_osc": 1.2,
    }

    # Strategy parameters
    entry_short_period = IntParameter(2, 24, default=5, space='buy', load=True, optimize=True)
    entry_long_period = IntParameter(6, 64, default=35, space='buy', load=True, optimize=True)
    entry_osc = DecimalParameter(-5.0, -1.0, default=-1.0, decimals=1, space='buy')
    exit_osc = DecimalParameter(1.0, 5.0, default=5.0, decimals=1, space='sell')
    


    def get_entry_signals(self, dataframe):

        # EW Oscillator
        if self.entry_short_period.value < self.entry_long_period.value:
            dataframe['ewo'] = self.ewo(dataframe, 
                                        int(self.entry_short_period.value), 
                                        int(self.entry_long_period.value))
        else:
            dataframe['ewo'] = 0.0

        series = np.where((dataframe['ewo'] < self.entry_osc.value), 1, 0)

        return series

    def get_exit_signals(self, dataframe):

        series = np.where((dataframe['ewo'] > self.exit_osc.value), 1, 0)

        return series
    

    # Elliot Wave Oscillator
    def ewo(self, dataframe, sma1_length=5, sma2_length=35):
        sma1 = ta.EMA(dataframe, timeperiod=sma1_length)
        sma2 = ta.EMA(dataframe, timeperiod=sma2_length)
        smadif = (sma1 - sma2) / dataframe['close'] * 100
        return smadif

