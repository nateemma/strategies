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
Keltner Channel Breakout
'''
class KCBreakout(SimpleStrategy):


    strategy_type = SimpleStrategy.StrategyType.VOLATILITY

    plot_config = {
        'main_plot': {
            'close': {'color': 'lightsteelblue'},
            'kc_lower': {'color': 'lightsalmon'},
            'kc_upper': {'color': 'lightgreen'}
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
        "entry_guard_metric": -0.4,
        "entry_period": 90,
    }

    # Sell hyperspace params:
    sell_params = {
        **SimpleStrategy.sell_params,
        "exit_guard_metric": 0.6,
    }


    # Strategy parameters
    entry_period = IntParameter(6, 96, default=24, space='buy')


    def get_entry_signals(self, dataframe):

        keltner = qtpylib.keltner_channel(dataframe, window=int(self.entry_period.value))
        dataframe["kc_upper"] = keltner["upper"].fillna(0)
        dataframe["kc_lower"] = keltner["lower"].fillna(0)

        series = np.where(
            (
                # close reaches kc_upper
                (dataframe['close'] >= dataframe['kc_upper']) 
            ),
              1, 0)

        return series

    def get_exit_signals(self, dataframe):

        series = np.where(
            (
                # close reaches kc_lower
                (dataframe['close'] <= dataframe['kc_lower'])
            ),
              1, 0)
        return series
