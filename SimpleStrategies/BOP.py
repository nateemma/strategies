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
Balance Of Power indicator (BOP)
(Lots of signals, needs strong guards)
'''
class BOP(SimpleStrategy):

    strategy_type = SimpleStrategy.StrategyType.TREND
    # entry_slope_column = "bop"
    plot_config = {
        'main_plot': {
            'close': {'color': 'lightsteelblue'},
        },
        'subplots': {
            "Diff": {
                'bop': {'color': 'lightskyblue'},
                'candle_imbalance': {'color': 'lightcoral'},
            },
        }
    }

    enable_guards = True # set to True for testing, False for debug

    # Buy hyperspace params:
    buy_params = {
        **SimpleStrategy.buy_params,
        "entry_bop": 0.91,
        "entry_guard_metric": -0.8,
    }

    # Sell hyperspace params:
    sell_params = {
        **SimpleStrategy.sell_params,
        "exit_bop": -1.0,
        "exit_guard_metric": 0.6,
    }

    # Strategy parameters
    entry_bop = DecimalParameter(0.9, 1.0, default=0.99, decimals=2, space='buy')

    exit_bop = DecimalParameter(-1.0, -0.9, default=-0.99, decimals=2, space='sell')

    def get_entry_signals(self, dataframe):

        dataframe['bop'] = fta.BOP(dataframe).fillna(0.0)

        series = np.where(
            (
                (dataframe["bop"] < self.entry_bop.value)
                & (dataframe["candle_imbalance"] < -0.5)
            ),
            1,
            0,
        )

        return series

    def get_exit_signals(self, dataframe):

        series = np.where(
            (
                (dataframe["bop"] > self.exit_bop.value)
                & (dataframe["candle_imbalance"] > 0.5)
            ),
            1,
            0,
        )
        return series
