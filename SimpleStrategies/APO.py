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
Absolute Price Oscillator
'''
class APO(SimpleStrategy):


    strategy_type = SimpleStrategy.StrategyType.OSCILLATOR
    entry_slope_column = "apo"
    plot_config = {
        'main_plot': {
            'close': {'color': 'lightsteelblue'},
        },
        'subplots': {
            "Diff": {
                'enter_long': {'color': 'lightseagreen'},
                'exit_long': {'color': 'lightsalmon'},
                'apo': {'color': 'lightsteelblue'},
            },
        }
    }


    enable_guards = True # set to True for testing, False for debug


    # Buy hyperspace params:
    buy_params = {
        **SimpleStrategy.buy_params,
        "entry_apo": -0.27,
        "entry_fast_period": 14,
        "entry_guard_metric": -0.8,
        "entry_matype": 3,
        "entry_slow_period": 61,
    }

    # Sell hyperspace params:
    sell_params = {
        **SimpleStrategy.sell_params,
        "exit_apo": 0.88,
        "exit_guard_metric": 0.4,
    }


    # Strategy parameters

    entry_fast_period = IntParameter(2, 24, default=12, space='buy', load=True, optimize=True)
    entry_slow_period = IntParameter(6, 64, default=24, space='buy', load=True, optimize=True)
    entry_matype = IntParameter(0, 8, default=0, space='buy', load=True, optimize=True)
    entry_apo = DecimalParameter(-1.0, 0.0, default=-0.01, decimals=2, space='buy', load=True, optimize=True)

    exit_apo = DecimalParameter(0.0, 1.0, default=0.01, decimals=2, space='sell', load=True, optimize=True)


    def get_entry_signals(self, dataframe):
 
        if self.entry_fast_period.value < self.entry_slow_period.value:
                dataframe['apo'] = ta.APO(dataframe, 
                                        fastperiod=int(self.entry_fast_period.value),
                                        slowperiod=int(self.entry_slow_period.value),
                                        matype=int(self.entry_matype.value))
                series = np.where((dataframe['apo'] < self.entry_apo.value), 1, 0)
        else:
            dataframe['apo'] = 0.0
            series = pd.Series(0, index=dataframe.index)

        return series


    def get_exit_signals(self, dataframe):
 
        if self.entry_fast_period.value < self.entry_slow_period.value:
                series = np.where((dataframe['apo'] > self.exit_apo.value), 1, 0)
        else:
            series = pd.Series(0, index=dataframe.index)

        return series
