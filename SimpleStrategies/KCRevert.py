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
Keltner Channel Revert
'''
class KCRevert(SimpleStrategy):

    plot_config = {
        'main_plot': {
            'close': {'color': 'lightsteelblue'},
            'kc_lower': {'color': 'lightsalmon'},
            'kc_upper': {'color': 'lightgreen'}
        },
        'subplots': {
            "Diff": {
                "rvol": {'color': 'lightcoral'},
                "roc": {'color': 'lightblue'},
            },
        }
    }

    enable_guards = True # set to True for testing, False for debug

    # Buy hyperspace params:
    buy_params = {
        **SimpleStrategy.buy_params,
        "entry_guard_metric": -0.0,
        "entry_period": 93,
    }

    # Sell hyperspace params:
    sell_params = {
        **SimpleStrategy.sell_params,
        "exit_guard_metric": 0.3,
    }

    # Strategy parameters
    entry_period = IntParameter(6, 96, default=24, space='buy')

    strategy_type = SimpleStrategy.StrategyType.REVERSION

    def get_entry_signals(self, dataframe):

        keltner = qtpylib.keltner_channel(dataframe, window=int(self.entry_period.value))
        dataframe["kc_upper"] = keltner["upper"].fillna(0)
        dataframe["kc_middle"] = keltner["mid"].fillna(0)
        dataframe["kc_lower"] = keltner["lower"].fillna(0)

        # Requirement:
        # 1. Price is below lower band
        # 2. Price is ABOVE the EMA200 (Long-term trend is UP)
        # 3. Guard metric (RMI/RSI) is below -0.6 (Extreme oversold)

        series = np.where(
            (
                (dataframe["close"] < dataframe["kc_lower"])
            ),
            1,
            0,
        )
        return series

    def get_exit_signals(self, dataframe):

        series = np.where(
            (
                (dataframe["close"] >= dataframe["kc_middle"])
            ),
            1, 0,
        )
        return series
