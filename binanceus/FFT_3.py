import numpy as np
import scipy.fft
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

This variant uses the FFT to predict 'just' the turnaround points'

####################################################################################
"""


class FFT_3(IStrategy):
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
    startup_candle_count: int = 12
    process_only_new_candles = True

    ###################################

    # Strategy Specific Variable Storage

    ## Hyperopt Variables

    # FFT limits
    buy_fft_predict = IntParameter(1, 16, default=4, space='buy', load=True, optimize=True)

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

        length = len(dataframe['close'])
        nsteps = self.buy_fft_predict.value
        prediction = self.fourierExtrapolation(np.array(dataframe['close']), nsteps)
        dataframe['fft_predict'] = prediction[nsteps:]

        dataframe['fft_predict_diff'] = (dataframe['fft_predict'] - dataframe['close']) / dataframe['close']

        dataframe['fft_angle'] = ta.LINEARREG_SLOPE(dataframe['fft_predict'], timeperiod=3)

        return dataframe

    # function to predict future steps. x is the raw data, not the FFT
    def fourierExtrapolation(self, x, n_predict):
        n = x.size
        n_harm = 64  # number of harmonics in model
        t = np.arange(0, n)
        p = np.polyfit(t, x, 1)  # find linear trend in x
        x_notrend = x - p[0] * t  # detrended x
        x_freqdom = scipy.fft.fft(x_notrend)  # detrended x in frequency domain
        f = scipy.fft.fftfreq(n)  # frequencies

        # h = np.sort(x_freqdom)[-n_param]
        # x_freqdom = [x_freqdom[i] if np.absolute(x_freqdom[i]) >= h else 0 for i in range(len(x_freqdom))]

        indexes = list(range(n))
        # sort indexes by frequency, lower -> higher
        indexes.sort(key=lambda i: np.absolute(f[i]))

        t = np.arange(0, n + n_predict)
        restored_sig = np.zeros(t.size)
        for i in indexes[:1 + n_harm * 2]:
            ampli = np.absolute(x_freqdom[i]) / n  # amplitude
            phase = np.angle(x_freqdom[i])  # phase
            restored_sig += ampli * np.cos(2 * np.pi * f[i] * t + phase)
        return restored_sig + p[0] * t

    ###################################

    """
    Buy Signal
    """


    def populate_buy_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        conditions = []
        dataframe.loc[:, 'buy_tag'] = ''

        # conditions.append(dataframe['volume'] > 0)

        # FFT triggers

        angle_cond = (
            qtpylib.crossed_above(dataframe['fft_angle'], 0)
        )

        conditions.append(angle_cond)

        # set buy tags
        dataframe.loc[angle_cond, 'buy_tag'] += 'fft_buy_low '

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

        # FFT triggers

        angle_cond = (
            qtpylib.crossed_below(dataframe['fft_angle'], 0)
        )

        conditions.append(angle_cond)

        # set sell tags
        dataframe.loc[angle_cond, 'exit_tag'] += 'fft_sell_high '

        if conditions:
            dataframe.loc[reduce(lambda x, y: x & y, conditions), 'sell'] = 1

        return dataframe
