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
Donchian Channel Breakout
'''
class DCRevert(SimpleStrategy):


    strategy_type = SimpleStrategy.StrategyType.REVERSION

    plot_config = {
        'main_plot': {
            'close': {'color': 'lightsteelblue'},
            'dc_lower': {'color': 'lightsalmon'},
            'dc_upper': {'color': 'lightgreen'}
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
        "entry_period": 96,
    }

    # Sell hyperspace params:
    sell_params = {
        **SimpleStrategy.sell_params,
        "exit_guard_metric": 0.3,
    }


    # Strategy parameters
    entry_period = IntParameter(6, 96, default=24, space='buy')


    def get_entry_signals(self, dataframe):

        self.win_size = int(self.entry_period.value)
        dataframe['dc_upper'] = ta.MAX(dataframe['high'], timeperiod=self.win_size) # type: ignore
        dataframe['dc_lower'] = ta.MIN(dataframe['low'], timeperiod=self.win_size) # type: ignore

        # close moves away from dc_lower
        series = np.where(
            (
                (dataframe['close'] >= dataframe['dc_lower']) &
                (dataframe['close'].shift() <= dataframe['dc_lower'].shift())
            ),
              1, 0)

        return series

    def get_exit_signals(self, dataframe):

        # close moves away from dc_upper
        series = np.where(
            (
                (dataframe['close'] <= dataframe['dc_upper']) &
                (dataframe['close'].shift() >= dataframe['dc_upper'].shift())
            ),
              1, 0)
        return series
