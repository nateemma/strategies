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
Base class for single idicator reversion using Fisher Transform to find momentum changes
To subclass, override add_main_indicator() and define indicator (tehe name of the indicator column)
"""


class Ehlers_Indicator(EhlersBase):

    indicator: str = "close_norm"
    fisher_ind: str = f"{indicator}_fisher"
    slope: str = f"{indicator}_slope"

    @property
    def plot_config(self):
        return {
            "main_plot": {
                "close": {"color": "lightsteelblue"},
            },
            "subplots": {
                "Indicators": {
                    self.fisher_ind: {"color": "lightseagreen"},
                },
                "Signals": {
                    "cycle_buy": {"color": "lightseagreen"},
                    "trend_buy": {"color": "green"},
                    "cycle_sell": {"color": "orange"},
                    "trend_sell": {"color": "red"},
                },
            },
        }

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
    cycle_entry_threshold = DecimalParameter(
        -1.0, -0.5, default=-0.8, decimals=1, space="buy", load=True, optimize=True
    )
    cycle_exit_threshold = DecimalParameter(
        0.0, 1.0, default=0.8, decimals=1, space="sell", load=True, optimize=True
    )

    trend_entry_threshold = DecimalParameter(
        -1.0, 1.0, default=0.0, decimals=1, space="buy", load=True, optimize=True
    )
    trend_exit_threshold = DecimalParameter(
        -1.0, 1.0, default=0.0, decimals=1, space="sell", load=True, optimize=True
    )

    def add_main_indicator(self, dataframe):
        # populate the needed indicator(s) here. In this case it should already be there
        raise ValueError(f"{self.indicator} not in dataframe")
        return dataframe

    def add_strategy_indicators(self, dataframe):

        self.fisher_ind = f"{self.indicator}_fisher"
        self.slope = f"{self.indicator}_slope"

        if self.indicator not in dataframe.columns:
            dataframe = self.add_main_indicator(dataframe)

        # 1. Detrend the indicator (using a longer 20-period smoother for stability)
        valid_ind = dataframe[self.indicator].ffill().fillna(0.0)
        smoothed_ind = self.super_smoother(
            valid_ind, period=20
        )  # Longer period = smoother baseline
        detrended_ind = dataframe[self.indicator] - smoothed_ind

        # 2. Apply Fisher and then smooth the Fisher itself to kill the jitter
        raw_fisher = self.safe_fisher(detrended_ind, length=9).fillna(0.0)
        dataframe[self.fisher_ind] = self.super_smoother(raw_fisher, period=5)

        # 3. Calculate a smoothed slope
        dataframe[self.slope] = (
            dataframe[self.fisher_ind].diff().rolling(window=3).mean().fillna(0.0)
        )
        return dataframe

    def get_cycle_entry_trigger(self, dataframe):
        slope = dataframe[self.slope]

        # 2-candle confirmation: slope must be positive for 2 consecutive candles
        # after being non-positive. This prevents single-candle false hooks during crashes.
        slope_confirmed = (slope > 0) & (slope.shift(1) > 0) & (slope.shift(2) <= 0)

        entry_trigger = np.where(
            (dataframe[self.fisher_ind] < self.cycle_entry_threshold.value)
            & slope_confirmed,
            1,
            0,
        )

        return entry_trigger

    def get_trend_entry_trigger(self, dataframe):

        entry_trigger = np.where(
            (dataframe["dsp"] > 0.0)
            & qtpylib.crossed_above(
                dataframe[self.fisher_ind], self.trend_entry_threshold.value
            ),
            1,
            0,
        )
        return entry_trigger

    def get_cycle_exit_trigger(self, dataframe):

        exit_trigger = np.where(
            (dataframe[self.fisher_ind] > self.cycle_exit_threshold.value)
            & (qtpylib.crossed_below(dataframe[self.slope], 0)),
            1,
            0,
        )
        return exit_trigger

    def get_trend_exit_trigger(self, dataframe):
        # Exit the trend only when BOTH structural breakdown and momentum failure occur:
        # 1. dsp crosses below zero (price fell back below the super smoother —
        #    trend genuinely failing)
        # 2. fisher confirms momentum is fading (crossed below exit threshold)
        # Using a 3-candle window so a single choppy candle doesn't exit prematurely.
        dsp_breakdown = qtpylib.crossed_below(dataframe["dsp"], 0.0)
        dsp_breakdown_recent = (
            dsp_breakdown | dsp_breakdown.shift(1) | dsp_breakdown.shift(2)
        )

        fisher_fading = qtpylib.crossed_below(
            dataframe[self.fisher_ind], self.trend_exit_threshold.value
        )

        exit_trigger = np.where(
            dsp_breakdown_recent & fisher_fading,
            1,
            0,
        )
        return exit_trigger
