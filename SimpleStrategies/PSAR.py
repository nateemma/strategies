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
Parabolic SAR (Stop And Reverse)
'''
class PSAR(SimpleStrategy):

    strategy_type = SimpleStrategy.StrategyType.TREND

    plot_config = {
        'main_plot': {
            'close': {'color': 'lightsteelblue'},
            'psar': {'color': 'lightseagreen'},
        },
        'subplots': {
            "Diff": {
                # 'psarbull': {'color': 'lightseagreen'},
                # 'psarbear': {'color': 'lightsalmon'},
                'trend': {'color': 'lightsalmon'},
            },
        }
    }

    enable_guards = True # set to True for testing, False for debug

    # Buy hyperspace params:
    buy_params = {
        **SimpleStrategy.buy_params,
        "entry_af_increment": 0.15,
        "entry_af_start": 0.2,
        "entry_guard_metric": -0.0,
    }

    # Sell hyperspace params:
    sell_params = {
        **SimpleStrategy.sell_params,
        "exit_guard_metric": 0.5,
    }

    # Strategy parameters
    entry_af_start = DecimalParameter(0.0, 0.2, default=0.02, decimals=2, space='buy')
    entry_af_increment = DecimalParameter(0.0, 0.2, default=0.02, decimals=2, space='buy')

    def get_entry_signals(self, dataframe):

        psar = self.calculate_psar(dataframe, af_start=float(self.entry_af_start.value), af_increment=float(self.entry_af_increment.value))
        dataframe['psar'] = psar['psar']
        dataframe['trend'] = psar['trend']
        dataframe.fillna(0.0, inplace=True)

        series = np.where(
            ((dataframe["psar"] < dataframe["close"]) & (dataframe["trend"] > 0)), 1, 0
        )

        return series

    def get_exit_signals(self, dataframe):

        series = np.where(
            ((dataframe["psar"] > dataframe["close"]) & (dataframe["trend"] < 0)), 1, 0
        )
        return series

    def calculate_psar(self, ohlc, af_start=0.02, af_increment=0.02, af_max=0.2):
        """
        Calculate the Parabolic SAR (PSAR) indicator.
        
        Parameters:
            ohlc: dataframe
            af_start (float): Starting Acceleration Factor (default=0.02).
            af_increment (float): Increment for AF (default=0.02).
            af_max (float): Maximum Acceleration Factor (default=0.2).
            
        Returns:
            psar (np.ndarray): Calculated PSAR values.
            trend (np.ndarray): Trend direction (1 for bullish, -1 for bearish).
        """
        high, low, close = np.array(ohlc.high), np.array(ohlc.low), np.array(ohlc.close)
        psar = np.zeros_like(high)
        trend = np.zeros_like(high, dtype=int)

        # Initialization
        psar[0] = low[0]  # Start PSAR with the first low
        # psar[0] = close[0]  # Start PSAR with the first low
        ep = high[0]      # Extreme Price
        af = af_start      # Initial Acceleration Factor
        current_trend = 1  # Start with bullish trend (1 for bullish, -1 for bearish)

        for i in range(1, len(high)):
            prev_psar = psar[i - 1]

            # Calculate the unconstrained PSAR
            if current_trend == 1:  # Bullish trend
                psar[i] = prev_psar + af * (ep - prev_psar)

                # Check for trend reversal
                if low[i] < psar[i]:
                    # Switch to bearish trend
                    current_trend = -1
                    psar[i] = ep  # Reset PSAR to EP
                    ep = low[i]   # Reset EP to current low
                    af = af_start  # Reset AF
                else:
                    # No reversal: update EP and AF
                    if high[i] > ep:
                        ep = high[i]
                        af = min(af + af_increment, af_max)

                    # Apply the PSAR boundary
                    psar[i] = min(psar[i], low[i - 1], low[i])

            elif current_trend == -1:  # Bearish trend
                psar[i] = prev_psar - af * (prev_psar - ep)

                # Check for trend reversal
                if high[i] > psar[i]:
                    # Switch to bullish trend
                    current_trend = 1
                    psar[i] = ep  # Reset PSAR to EP
                    ep = high[i]  # Reset EP to current high
                    af = af_start  # Reset AF
                else:
                    # No reversal: update EP and AF
                    if low[i] < ep:
                        ep = low[i]
                        af = min(af + af_increment, af_max)

                    # Apply the PSAR boundary
                    psar[i] = max(psar[i], high[i - 1], high[i])

            # Track trend direction
            trend[i] = current_trend

        psar = pd.Series(psar, name="psar", index=ohlc.index)
        trend = pd.Series(trend, name="trend", index=ohlc.index)
        return pd.concat([psar, trend], axis=1)
