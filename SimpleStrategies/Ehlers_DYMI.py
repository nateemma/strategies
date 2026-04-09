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

from Ehlers_Indicator import Ehlers_Indicator

"""
Base class for reversion using Fisher Transform to find momentum changes
"""


class Ehlers_DYMI(Ehlers_Indicator):

    indicator: str = "dymi"

    def _calc_dymi_fast(self, dataframe: DataFrame) -> Series:
        """
        Vectorized DYMI approximation using precomputed RSI(5..30).
        Matches the finta.DYMI period selection logic without per-row RSI calls.
        """
        close = dataframe["close"]
        sd = close.rolling(5).std()
        asd = sd.rolling(10).mean()
        v = sd / asd
        t = (14 / v.round()).fillna(0)
        t = t.clip(5, 30).astype(int)

        rsi_by_period = {}
        for period in range(5, 31):
            rsi_by_period[period] = ta.RSI(dataframe, timeperiod=period).fillna(0)

        dymi = pd.Series(np.nan, index=dataframe.index, dtype=float)
        for period, rsi in rsi_by_period.items():
            mask = t == period
            if mask.any():
                dymi.loc[mask] = rsi.loc[mask]

        return dymi.fillna(0), t

    def add_main_indicator(self, dataframe):
        dataframe["dymi"], dataframe["dymi_period"] = self._calc_dymi_fast(dataframe)
        return dataframe

    def add_strategy_indicators(self, dataframe):
        self.fisher_ind = f"{self.indicator}_fisher"
        self.slope = f"{self.indicator}_slope"

        if self.indicator not in dataframe.columns:
            dataframe = self.add_main_indicator(dataframe)

        # 1. Adaptive Detrending
        # Use the DYMI period itself to drive the smoother speed
        valid_ind = dataframe[self.indicator].ffill().fillna(0.0)
        smoothed_ind = self.adaptive_super_smoother(valid_ind, dataframe["dymi_period"])
        detrended_ind = dataframe[self.indicator] - smoothed_ind

        # 2. Apply Fisher and then smooth
        raw_fisher = self.safe_fisher(detrended_ind, length=9).fillna(0.0)
        dataframe[self.fisher_ind] = self.super_smoother(raw_fisher, period=5)

        # 3. Calculate a smoothed slope
        dataframe[self.slope] = (
            dataframe[self.fisher_ind].diff().rolling(window=3).mean().fillna(0.0)
        )
        return dataframe

    def get_cycle_entry_trigger(self, dataframe):
        # Base logic: Fisher Threshold + Slope turn-up
        base_trigger = super().get_cycle_entry_trigger(dataframe)

        # Secondary Filter: Convengence
        # Only buy if the raw DYMI is also in oversold territory (< 30)
        return np.where((base_trigger == 1) & (dataframe["dymi"] < 30), 1, 0)

    def get_cycle_exit_trigger(self, dataframe):
        # Base logic: Fisher Threshold + Slope turn-down
        base_trigger = super().get_cycle_exit_trigger(dataframe)

        # Only sell if the raw DYMI is also in overbought territory (> 70)
        return np.where((base_trigger == 1) & (dataframe["dymi"] > 70), 1, 0)
