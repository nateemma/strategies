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
    startup_candle_count: int = 12
    process_only_new_candles = True

    ###################################

    # Strategy Specific Variable Storage

    ## Hyperopt Variables

    # FFT limits
    buy_fft_gain = DecimalParameter(0.000, 0.050, decimals=3, default=0.015, space='buy', load=True, optimize=True)
    buy_fft_cutoff = DecimalParameter(1/8.0, 1/2.0, decimals=2, default=1/6.0, space='buy', load=True, optimize=True)
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
        cutoff:int = int(len(yf) * self.buy_fft_cutoff.value)
        yf[(cutoff-1):] = 0
        
        # inverse transform
        # dataframe['ifft'] = pd.Series(scipy.fft.irfft(yf)[:-1])

        dataframe['fft_mean'] = dataframe['close']
        model = scipy.fft.irfft(yf)

        lc = len(dataframe['close'])
        lm = len(model)
        if (lc == lm):
            dataframe['fft_mean'] = model
        elif (lc > lm):
            dataframe['fft_mean'][(lc-lm):] = model
        else:
            dataframe['fft_mean'] = model[(lm-lc):]

        # predict next candle (simple linear extrapolation for now)
        # Note: use the 'mean' values because we expect prices to oscillate around that model
        dataframe['fft_predict'] = dataframe['fft_mean'] + dataframe['fft_mean'].shift(1) - dataframe['fft_mean'].shift(2)

        dataframe['fft_predict_diff'] = (dataframe['fft_predict'] - dataframe['close']) / dataframe['close']

        # exploring prediction:
        length = len(dataframe['close'])
        nsteps=8
        prediction = self.fourierExtrapolation(np.array(dataframe['close']), nsteps, 10)
        dataframe['fft_predict2'] = prediction[nsteps:]

        return dataframe

    # function to predict future steps. x is the raw data, not the FFT
    def fourierExtrapolation(self, x, n_predict, n_param):
        n = x.size
        n_harm = 10  # number of harmonics in model
        t = np.arange(0, n)
        p = np.polyfit(t, x, 1)  # find linear trend in x
        x_notrend = x - p[0] * t  # detrended x
        x_freqdom = scipy.fft.fft(x_notrend)  # detrended x in frequency domain
        f = scipy.fft.fftfreq(n)  # frequencies

        h = np.sort(x_freqdom)[-n_param]
        x_freqdom = [x_freqdom[i] if np.absolute(x_freqdom[i]) >= h else 0 for i in range(len(x_freqdom))]

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
        fft_cond = (
            qtpylib.crossed_above(dataframe['fft_predict_diff'], self.buy_fft_gain.value)
        )

        conditions.append(fft_cond)

        # going up?
        conditions.append(dataframe['fft_predict'] > dataframe['fft_mean'])

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

        # FFT triggers
        fft_cond = (
            qtpylib.crossed_below(dataframe['fft_predict_diff'], self.sell_fft_loss.value)
        )

        conditions.append(fft_cond)

        # going down?
        # conditions.append(dataframe['fft_predict'] < dataframe['fft_mean'])

        # set buy tags
        dataframe.loc[fft_cond, 'exit_tag'] += 'fft_sell_1 '

        if conditions:
            dataframe.loc[reduce(lambda x, y: x & y, conditions), 'sell'] = 1

        return dataframe
