import numpy as np
import scipy.fft
from scipy.fft import rfft, irfft
import talib.abstract as ta
import freqtrade.vendor.qtpylib.indicators as qtpylib
import arrow

from freqtrade.strategy import (IStrategy, merge_informative_pair, stoploss_from_open,
                                IntParameter, DecimalParameter, CategoricalParameter)

from typing import Dict, List, Optional, Tuple, Union
from pandas import DataFrame, Series
from functools import reduce
from datetime import datetime, timedelta
from freqtrade.persistence import Trade

# Get rid of pandas warnings during backtesting
import pandas as pd

pd.options.mode.chained_assignment = None  # default='warn'

# Strategy specific imports, files must reside in same folder as strategy
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent))

import logging
import warnings

log = logging.getLogger(__name__)
# log.setLevel(logging.DEBUG)
warnings.simplefilter(action='ignore', category=pd.errors.PerformanceWarning)



"""
####################################################################################
FFT - use a Fast Fourier Transform (FFT) to estimate future price movements

####################################################################################
"""


class FFT(IStrategy):
    # Do *not* hyperopt for the roi and stoploss spaces

    # ROI table:
    minimal_roi = {
        "0": 100
    }

    # Stoploss:
    stoploss = -0.10

    # Trailing stop:
    trailing_stop = False
    trailing_stop_positive = None
    trailing_stop_positive_offset = 0.0
    trailing_only_offset_is_reached = False

    timeframe = '5m'
    inf_timeframe = '1h'

    use_custom_stoploss = False

    # Recommended
    use_sell_signal = True
    sell_profit_only = False
    ignore_roi_if_buy_signal = True

    # Required
    startup_candle_count: int = 48
    process_only_new_candles = True

    ###################################

    # Strategy Specific Variable Storage

    ## Hyperopt Variables

    # FFT limits
    buy_fft_gain = DecimalParameter(0.000, 0.050, decimals=3, default=0.015, space='buy', load=True, optimize=True)
    sell_fft_loss = DecimalParameter(-0.050, 0.000, decimals=3, default=-0.005, space='sell', load=True, optimize=True)

    ###################################

    """
    Informative Pair Definitions
    """

    def informative_pairs(self):
        return []

    ###################################

    """
    Indicator Definitions
    """

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:

        # FFT

        # get the FFT
        yf = scipy.fft.rfft(np.array(dataframe['close']))

        # zero out frequencies beyond 'cutoff'
        cutoff:int = int(len(yf) / 6)
        yf[(cutoff-1):] = 0
        
        # inverse transform
        dataframe['ifft'] = pd.Series(scipy.fft.irfft(yf))

        dataframe['fft_mean'] = dataframe['ifft']
        dataframe['fft_predict'] = dataframe['ifft']

        dataframe['fft_predict_diff'] = (dataframe['fft_predict'] - dataframe['close']) / dataframe['close']

        return dataframe


    ###################################

    """
    Buy Signal
    """


    def populate_buy_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        conditions = []
        dataframe.loc[:, 'buy_tag'] = ''

        # conditions.append(dataframe['volume'] > 0)

        # Kalman triggers
        fft_cond = (
            qtpylib.crossed_above(dataframe['fft_predict_diff'], self.buy_fft_gain.value)
        )

        conditions.append(fft_cond)

        # set buy tags
        dataframe.loc[fft_cond, 'buy_tag'] += 'fft_buy_1 '

        if conditions:
            dataframe.loc[reduce(lambda x, y: x & y, conditions), 'buy'] = 1

        return dataframe


    ###################################

    """
    Sell Signal
    """


    def populate_sell_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        conditions = []
        dataframe.loc[:, 'exit_tag'] = ''

        # Kalman triggers
        fft_cond = (
            qtpylib.crossed_below(dataframe['fft_predict_diff'], self.sell_fft_loss.value)
        )

        conditions.append(fft_cond)

        # set buy tags
        dataframe.loc[fft_cond, 'exit_tag'] += 'fft_sell_1 '

        if conditions:
            dataframe.loc[reduce(lambda x, y: x & y, conditions), 'sell'] = 1

        return dataframe
