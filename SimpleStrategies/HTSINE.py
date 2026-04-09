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
HTSINE - Hilbert Transform, Sine Wave
'''
class HTSINE(SimpleStrategy):

    strategy_type = SimpleStrategy.StrategyType.OSCILLATOR

    plot_config = {
        'main_plot': {
            'close': {'color': 'lightsteelblue'},
        },
        'subplots': {
            "Diff": {
                'sine': {'color': 'lightseagreen'},
                'lead_sine': {'color': 'lightskyblue'},
            },
        }
    }

    enable_guards = True # set to True for testing, False for debug
 
    # Buy hyperspace params:
    buy_params = {
        **SimpleStrategy.buy_params,
        "entry_guard_metric": -0.1,
    }

    # Sell hyperspace params:
    sell_params = {
        **SimpleStrategy.sell_params,
        "exit_guard_metric": 0.2,
    }

    # ROI table:  # value loaded from strategy
    minimal_roi = {
        "0": 0.12
    }

    # Strategy parameters

    def get_entry_signals(self, dataframe):

        sine, leadsine = ta.HT_SINE(dataframe['close'])
        dataframe['sine'] = sine
        dataframe['lead_sine'] = leadsine

        series = np.where(
            (dataframe['sine'] > dataframe['lead_sine']),
              1, 0)

        return series

    def get_exit_signals(self, dataframe):

        series = np.where(
            (dataframe['sine'] < dataframe['lead_sine']),
              1, 0)
        return series
