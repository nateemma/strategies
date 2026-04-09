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
Dynamic Momentum Index (DYMI)

NOTE: this is very slow to calculate for some reason

'''
class DYMI(SimpleStrategy):

    strategy_type = SimpleStrategy.StrategyType.OSCILLATOR

    plot_config = {
        'main_plot': {
            'close': {'color': 'lightsteelblue'},
        },
        'subplots': {
            "Diff": {
                'dymi_slow': {'color': 'lightskyblue'},
                'dymi_fast': {'color': 'lightgreen'},
            },
        }
    }

    enable_guards = True # set to True for testing, False for debug
    use_fast_dymi = True
 
    # Buy hyperspace params:
    buy_params = {
        **SimpleStrategy.buy_params,
        "entry_dymi": 45.0,
        "entry_guard_metric": 0.0,
    }

    # Sell hyperspace params:
    sell_params = {
        **SimpleStrategy.sell_params,
        "exit_dymi": 96.0,
        "exit_guard_metric": 0.6,
    }

    # Strategy parameters
    entry_dymi = DecimalParameter(0.0, 50.0, default=-20.0, decimals=0, space='buy')

    exit_dymi = DecimalParameter(50.0, 100.0, default=-80.0, decimals=0, space='sell')

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

        return dymi.fillna(0)

    def get_entry_signals(self, dataframe):

        # dataframe["dymi_slow"] = fta.DYMI(dataframe)
        dataframe["dymi_fast"] = self._calc_dymi_fast(dataframe)
        dataframe["dymi"] = (
            dataframe["dymi_fast"] if self.use_fast_dymi else dataframe["dymi_slow"]
        )

        series = np.where(
            (dataframe['dymi'] < self.entry_dymi.value),
              1, 0)

        return series

    def get_exit_signals(self, dataframe):

        series = np.where(
            (dataframe['dymi'] > self.exit_dymi.value),
              1, 0)
        return series
