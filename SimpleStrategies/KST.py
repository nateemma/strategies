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
KST - Know Sure Thing
'''
class KST(SimpleStrategy):


    strategy_type = SimpleStrategy.StrategyType.OSCILLATOR
    entry_slope_column = "kst"
    plot_config = {
        'main_plot': {
            'close': {'color': 'lightsteelblue'},
        },
        'subplots': {
            "Diff": {
                'kst': {'color': 'lightskyblue'},
            },
        }
    }

    enable_guards = True # set to True for testing, False for debug

    # Buy hyperspace params:
    buy_params = {
        **SimpleStrategy.buy_params,
        "entry_guard_metric": -0.2,
        "entry_kst": -31.0,
        "entry_r1": 24,
        "entry_r2": 36,
        "entry_r3": 48,
        "entry_r4": 54,
    }

    # Sell hyperspace params:
    sell_params = {
        **SimpleStrategy.sell_params,
        "exit_guard_metric": 0.1,
        "exit_kst": 28.0,
    }


    # Strategy parameters    
    entry_r1 = IntParameter(4, 24, default=10, space='buy', load=True, optimize=True)
    entry_r2 = IntParameter(12, 36, default=15, space='buy', load=True, optimize=True)
    entry_r3 = IntParameter(16, 48, default=20, space='buy', load=True, optimize=True)
    entry_r4 = IntParameter(26, 72, default=30, space='buy', load=True, optimize=True)
    entry_kst = DecimalParameter(-60.0, -20.0, default=-30.0, decimals=0, space='buy', load=True, optimize=True)

    exit_kst = DecimalParameter(20.0, 60.0, default=90.0, decimals=0, space='sell', load=True, optimize=True)


    def get_entry_signals(self, dataframe):

        # periods must be progressively longer
        if (
            (self.entry_r2.value > self.entry_r1.value) &
            (self.entry_r3.value > self.entry_r2.value) &
            (self.entry_r4.value > self.entry_r3.value) 
        ):
            kst = fta.KST(dataframe, 
                        r1=int(self.entry_r1.value), 
                        r2=int(self.entry_r2.value), 
                        r3=int(self.entry_r3.value), 
                        r4=int(self.entry_r4.value))
            dataframe["kst"] = kst['KST']
        else:
            dataframe["kst"] = 0.0


        series = np.where(
            (
                (dataframe['kst'] < self.entry_kst.value)
            ),
              1, 0)

        return series

    def get_exit_signals(self, dataframe):

        series = np.where(
            (
                (dataframe['kst'] > self.exit_kst.value)
            ),
              1, 0)
        return series
