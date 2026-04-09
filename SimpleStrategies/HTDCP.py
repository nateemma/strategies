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
HTDCP - Hilbert Transform Dominant Cycle Period strategy.

HT_DCPERIOD measures the number of bars in the current dominant cycle.

Regime semantics:
  Cycle mode: dcperiod is reliable. A short, stable period means fast tight cycles
              are running — ideal for mean reversion. Entry when period is low/falling,
              exit when period rises (cycle lengthening / trend forming).
  Trend mode: dcperiod becomes noisy and inflated. Falls back to EhlersBase defaults
              (ht_sine crossover), since dcperiod is not a useful trend trigger.
"""


class HTDCP(EhlersBase):

    strategy_type = EhlersBase.StrategyType.OTHER

    @property
    def plot_config(self):
        return {
            "main_plot": {
                "close": {"color": "lightsteelblue"},
                "itrend": {"color": "lightseagreen"},
                "ss": {"color": "lightsalmon"},
            },
            "subplots": {
                "Indicators": {
                    "htdcp": {"color": "lightskyblue"},
                    "htdcp_smooth": {"color": "lightyellow"},
                },
                "Signals": {
                    "cycle_buy": {"color": "lightseagreen"},
                    "trend_buy": {"color": "green"},
                    "cycle_sell": {"color": "orange"},
                    "trend_sell": {"color": "red"},
                },
            },
        }

    enable_guards = True

    # Cycle entry: enter when dcperiod drops below this (tight fast cycle active)
    cycle_entry_period = DecimalParameter(
        10.0, 30.0, default=20.0, decimals=0, space="buy", load=True, optimize=True
    )
    # Cycle exit: exit when dcperiod rises above this (cycle lengthening, trend forming)
    cycle_exit_period = DecimalParameter(
        20.0, 40.0, default=30.0, decimals=0, space="sell", load=True, optimize=True
    )

    def add_strategy_indicators(self, dataframe):
        # dcperiod already computed in EhlersBase.get_entry_signals as "dcperiod"
        # Smooth it over 3 bars to reduce single-candle noise in the period estimate
        dataframe["htdcp"] = dataframe["dcperiod"]
        dataframe["htdcp_smooth"] = dataframe["htdcp"].rolling(3, min_periods=1).mean()

        # Instantaneous Trendline for Trend Mode participation
        dataframe["itrend"] = ta.HT_TRENDLINE(dataframe["close"])

        return dataframe

    def get_cycle_entry_trigger(self, dataframe):
        """
        Enter cycle trade when:
        - dcperiod crosses below threshold (cycle is short and active)
        - price is below super-smoother (buying the dip within the cycle)
        """
        period_active = qtpylib.crossed_below(
            # dataframe["htdcp_smooth"], self.cycle_entry_period.value
            dataframe["htdcp"],
            self.cycle_entry_period.value,
        )
        buying_the_dip = dataframe["dsp"] < 0.0

        return np.where(period_active & buying_the_dip, 1, 0)

    def get_cycle_exit_trigger(self, dataframe):
        """
        Exit cycle trade when dcperiod rises above threshold —
        the cycle is lengthening, signalling a potential trend transition.
        """
        period_dying = qtpylib.crossed_above(
            # dataframe["htdcp_smooth"], self.cycle_exit_period.value
            dataframe["htdcp"],
            self.cycle_exit_period.value,
        )
        return np.where(period_dying, 1, 0)

    # Trend participation using Instantaneous Trendline
    def get_trend_entry_trigger(self, dataframe):
        trend_cross_up = qtpylib.crossed_above(dataframe["close"], dataframe["itrend"])
        below_average = dataframe["dsp"] < 0.0
        return np.where(trend_cross_up & below_average, 1, 0)

    def get_trend_exit_trigger(self, dataframe):
        trend_cross_down = qtpylib.crossed_above(
            dataframe["close"], dataframe["itrend"]
        )
        above_average = dataframe["dsp"] > 0.0
        return np.where(trend_cross_down & above_average, 1, 0)
