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

warnings.filterwarnings(
    "ignore", message="The objective has been evaluated at this point before."
)

from EhlersBase import EhlersBase

"""
Bolinger Band Breakout/Reversion
"""


class Ehlers_BB(EhlersBase):

    # Buy hyperspace params:
    buy_params = {
        **EhlersBase.buy_params,
        "entry_period": 24,
        "entry_wobv_pct": 0.0,
    }

    # Sell hyperspace params:
    sell_params = {
        **EhlersBase.sell_params,
        "exit_guard_metric": 0.1,
    }

    # Strategy parameters
    entry_period = IntParameter(6, 96, default=24, space="buy")
    exit_bb_factor = DecimalParameter(0.5, 1.0, default=0.85, decimals=2, space="sell")

    # cycle = mean reversion, trend = breakout

    def add_strategy_indicators(self, dataframe):
        if "bb_lowerband" not in dataframe.columns:
            bollinger = qtpylib.bollinger_bands(
                dataframe["close"], window=int(self.entry_period.value), stds=2
            )
            dataframe["bb_lowerband"] = bollinger["lower"]
            dataframe["bb_middleband"] = bollinger["mid"]
            dataframe["bb_upperband"] = bollinger["upper"]
        return dataframe

    def get_cycle_entry_trigger(self, dataframe):

        entry_trigger = np.where(
            (qtpylib.crossed_above(dataframe["close"], dataframe["bb_lowerband"])),
            1,
            0,
        )

        return entry_trigger

    def get_trend_entry_trigger(self, dataframe):

        entry_trigger = np.where(
            (dataframe["dsp"] > 0.0)
            & qtpylib.crossed_above(dataframe["close"], dataframe["bb_upperband"]),
            1,
            0,
        )
        return entry_trigger

    def get_cycle_exit_trigger(self, dataframe):

        exit_trigger = np.where(
            (qtpylib.crossed_below(dataframe["close"], dataframe["bb_upperband"])),
            1,
            0,
        )
        return exit_trigger

    def get_trend_exit_trigger(self, dataframe):

        # Exit if Price crosses below (Middle + N * (Upper - Middle))
        # This is equivalent to %B < 0.5 + 0.5*N (since Upper is 1.0 and Mid is 0.5 in standard %B terms)
        # Standard %B: Lower=0, Mid=0.5, Upper=1.0.

        factor = self.exit_bb_factor.value
        upper_half_gap = dataframe["bb_upperband"] - dataframe["bb_middleband"]
        exit_threshold = dataframe["bb_middleband"] + (factor * upper_half_gap)

        exit_trigger = np.where(
            qtpylib.crossed_below(dataframe["close"], exit_threshold),
            1,
            0,
        )
        return exit_trigger
