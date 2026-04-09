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
Schaff Trend Cycle
'''
class TSI(SimpleStrategy):

    strategy_type = SimpleStrategy.StrategyType.OSCILLATOR

    plot_config = {
        'main_plot': {
            'close': {'color': 'lightsteelblue'},
        },
        'subplots': {
            "Diff": {
            'tsi': {'color': 'lightseagreen'},
            'signal': {'color': 'lightsalmon'},
            },
        }
    }

    enable_guards = True # set to True for testing, False for debug

    # Buy hyperspace params:
    buy_params = {
        **SimpleStrategy.buy_params,
        "entry_guard_metric": -0.8,
        "entry_period_long": 39,
        "entry_period_short": 6,
        "entry_period_signal": 12,
        "entry_tsi": -35.7,
    }

    # Sell hyperspace params:
    sell_params = {
        **SimpleStrategy.sell_params,
        "exit_guard_metric": 0.6,
        "exit_tsi": 63.1,
    }

    # Strategy parameters
    entry_tsi = DecimalParameter(-100.0, -20.0, default=-30.0, decimals=1, space='buy')
    entry_period_long = IntParameter(12, 64, default=25, space='buy')
    entry_period_short = IntParameter(6, 24, default=13, space='buy')
    entry_period_signal = IntParameter(6, 24, default=13, space='buy')

    exit_tsi = DecimalParameter(50.0, 100.0, default=90.0, decimals=1, space='sell')

    def get_entry_signals(self, dataframe):

        if int(self.entry_period_long.value) > int(self.entry_period_short.value):
            tsi = fta.TSI(dataframe, 
                                    long=int(self.entry_period_long.value), 
                                    short=int(self.entry_period_short.value),
                                    signal=int(self.entry_period_signal.value)
                                    )
            dataframe['tsi'] = tsi['TSI']
            dataframe['signal'] = tsi['signal']
        else:
            # print(f'entry_period_long;{self.entry_period_long.value} entry_period_short:{self.entry_period_short.value}')
            dataframe['tsi'] = 0.0
            dataframe['signal'] = 0.0

        series = np.where(
            (
            (dataframe["tsi"] < self.entry_tsi.value) &
            (dataframe["tsi"] < dataframe["signal"]) 
            ), 1, 0)

        return series

    def get_exit_signals(self, dataframe):

        series = np.where(
            (
            (dataframe["tsi"] > self.exit_tsi.value) &
            (dataframe["tsi"] > dataframe["signal"])
            ), 1, 0)
        return series
    