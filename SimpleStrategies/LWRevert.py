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


from finta import TA as fta

import warnings
warnings.filterwarnings("ignore", message="The objective has been evaluated at this point before.")

from SimpleStrategy import SimpleStrategy

'''
Larry Williams Volatility Revert
'''
class LWRevert(SimpleStrategy):


    strategy_type = SimpleStrategy.StrategyType.REVERSION

    plot_config = {
        'main_plot': {
            'close': {'color': 'lightsteelblue'},
            'buy_level': {'color': 'lightseagreen'},
            'sell_level': {'color': 'lightsalmon'}
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
        "entry_guard_metric": -0.2,
        "entry_multiplier": 2.4,
        "entry_win_size": 94,
    }

    # Sell hyperspace params:
    sell_params = {
        **SimpleStrategy.sell_params,
        "exit_guard_metric": 0.3,
        "exit_multiplier": 0.5,
    }

    # Strategy parameters
    entry_win_size = IntParameter(14, 96, default=24, space='buy')
    entry_multiplier = DecimalParameter(0.5, 3.0, default=1.0, decimals=1, space='buy')
    
    exit_multiplier = DecimalParameter(0.5, 3.0, default=1.0, decimals=1, space='sell')


    def get_entry_signals(self, dataframe):

        self.window_size = int(self.entry_win_size.value)
        # dataframe['prev_day_high'] = dataframe['high'].rolling(window=self.window_size, min_periods=1).max()
        # dataframe['prev_day_low'] = dataframe['low'].rolling(window=self.window_size, min_periods=1).min()

        # Calculate breakout range
        # dataframe['breakout_range'] = dataframe['prev_day_high'] - dataframe['prev_day_low']
        dataframe['atr'] = fta.ATR(dataframe, period=self.window_size)
        
        # Calculate buy and sell levels
        # dataframe['buy_level'] = dataframe['open'] + (self.entry_multiplier.value * dataframe['breakout_range'])
        # dataframe['sell_level'] = dataframe['open'] - (self.exit_multiplier.value * dataframe['breakout_range'])
        dataframe['buy_level'] = dataframe['open'] - (self.entry_multiplier.value * dataframe['atr'])
        dataframe['sell_level'] = dataframe['open'] + (self.exit_multiplier.value * dataframe['atr'])


        series = np.where(
            (
                (dataframe['close'] <= dataframe['buy_level'])
            ),
              1, 0)

        return series

    def get_exit_signals(self, dataframe):

        series = np.where(
            (
                (dataframe['close'] >= dataframe['sell_level'])
            ),
              1, 0)
        return series
