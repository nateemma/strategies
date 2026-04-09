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
Volatility-Based-Momentum (VBM)
'''
class VBM(SimpleStrategy):


    strategy_type = SimpleStrategy.StrategyType.OSCILLATOR
    # entry_slope_column = "vbm"
    plot_config = {
        'main_plot': {
            'close': {'color': 'lightsteelblue'},
        },
        'subplots': {
            "Diff": {
            'vbm': {'color': 'lightseagreen'},
            'vbm_zscore': {'color': 'lightskyblue'},
            },
        }
    }

    enable_guards = True # set to True for testing, False for debug

    # Buy hyperspace params:
    buy_params = {
        **SimpleStrategy.buy_params,
        "entry_atr_period": 8,
        "entry_guard_metric": -0.3,
        "entry_roc_period": 28,
        "entry_vbm": -217.0,
        "entry_vbm_mode": "zscore",
        "entry_vbm_zscore": -1.5,
        "entry_vbm_zscore_period": 50,
    }

    # Sell hyperspace params:
    sell_params = {
        **SimpleStrategy.sell_params,
        "exit_guard_metric": 0.2,
        "exit_vbm": -191.0,
        "exit_vbm_zscore": 0.5,
    }


    # Strategy parameters
    entry_vbm = DecimalParameter(-500.0, -200.0, default=-300.0, decimals=0, space='buy')
    entry_roc_period = IntParameter(6, 32, default=12, space='buy')
    entry_atr_period = IntParameter(6, 32, default=26, space='buy')
    entry_vbm_mode = CategoricalParameter(["raw", "zscore"], default="zscore", space="buy")
    entry_vbm_zscore = DecimalParameter(-3.0, 0.0, default=-1.5, decimals=2, space='buy')
    entry_vbm_zscore_period = IntParameter(20, 200, default=50, space='buy')

    exit_vbm = DecimalParameter(-200.0, -0.0, default=-100.0, decimals=0, space='sell')
    exit_vbm_zscore = DecimalParameter(0.0, 3.0, default=0.5, decimals=2, space='sell')

    def get_entry_signals(self, dataframe):

        dataframe['vbm'] = fta.VBM(dataframe, 
                                   roc_period=int(self.entry_roc_period.value), 
                                   atr_period=int(self.entry_atr_period.value))
        vbm_mean = dataframe['vbm'].rolling(window=int(self.entry_vbm_zscore_period.value)).mean()
        vbm_std = dataframe['vbm'].rolling(window=int(self.entry_vbm_zscore_period.value)).std()
        dataframe['vbm_zscore'] = (dataframe['vbm'] - vbm_mean) / vbm_std.replace(0, np.nan)
        dataframe['vbm_zscore'] = dataframe['vbm_zscore'].fillna(0.0)

        series = np.where(
            (
            (
                (self.entry_vbm_mode.value == "raw")
                & (dataframe['vbm'] < self.entry_vbm.value)
            )
            | (
                (self.entry_vbm_mode.value == "zscore")
                & (dataframe['vbm_zscore'] < self.entry_vbm_zscore.value)
            )
            ), 1, 0)

        return series

    def get_exit_signals(self, dataframe):

        series = np.where(
            (
            (
                (self.entry_vbm_mode.value == "raw")
                & (dataframe['vbm'] > self.exit_vbm.value)
            )
            | (
                (self.entry_vbm_mode.value == "zscore")
                & (dataframe['vbm_zscore'] > self.exit_vbm_zscore.value)
            )
            ), 1, 0)
        return series
    