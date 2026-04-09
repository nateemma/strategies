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
Percentage Price Oscillator
'''
class PPO(SimpleStrategy):


    strategy_type = SimpleStrategy.StrategyType.OSCILLATOR
    entry_slope_column = "ppo"
    plot_config = {
        'main_plot': {
            'close': {'color': 'lightsteelblue'},
        },
        'subplots': {
            "Diff": {
                "ppo": {'color': 'lightskyblue'},
            },
        }
    }

    enable_guards = True # set to True for testing, False for debug
 
    # Buy hyperspace params:
    buy_params = {
        **SimpleStrategy.buy_params,
        "entry_fast_period": 9,
        "entry_guard_metric": -0.2,
        "entry_matype": 3,
        "entry_ppo": -2.0,
        "entry_slow_period": 48,
    }

    # Sell hyperspace params:
    sell_params = {
        **SimpleStrategy.sell_params,
        "exit_guard_metric": 0.6,
        "exit_ppo": 1.5,
    }

    # Strategy parameters
    entry_ppo = DecimalParameter(-2.0, 0.0, default=-0.8, decimals=1, space='buy')
    entry_fast_period = IntParameter(2, 24, default=12, space='buy', load=True, optimize=True)
    entry_slow_period = IntParameter(6, 64, default=24, space='buy', load=True, optimize=True)
    entry_matype = IntParameter(0, 8, default=0, space='buy', load=True, optimize=True)

    exit_ppo = DecimalParameter(0.0, 2.0, default=0.8, decimals=1, space='sell')


    def get_entry_signals(self, dataframe):

        dataframe['ppo'] = ta.PPO(dataframe, 
                                  fastperiod=int(self.entry_fast_period.value), 
                                  slowperiod=int(self.entry_slow_period.value), 
                                  matype=int(self.entry_matype.value))

        series = np.where(
            (dataframe["ppo"] < self.entry_ppo.value),
              1, 0)

        return series


    def get_exit_signals(self, dataframe):

        series = np.where(
            (dataframe["ppo"] > self.exit_ppo.value),
              1, 0)
        return series
