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
    inf_timeframe = '5m'

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
    buy_fft_window = IntParameter(1, 128, default=64, space='buy', load=True, optimize=True)
    buy_fft_nharmonics = IntParameter(6, 16, default=8, space='buy', load=True, optimize=True)
    buy_fft_predict = IntParameter(1, 16, default=4, space='buy', load=True, optimize=True)
    buy_fft_invert = CategoricalParameter([True, False], default=False, space="buy")

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

        dataframe['fft_predict'] = dataframe['close'].rolling(window=self.buy_fft_window.value).apply(self.runFourier)
        dataframe['fft_angle'] = ta.LINEARREG_SLOPE(dataframe['fft_predict'], timeperiod=3)

        if self.buy_fft_invert.value:
            dataframe['buy_region'] = np.where(qtpylib.crossed_below(dataframe['fft_angle'], 0), 1, 0)
            dataframe['sell_region'] = np.where(qtpylib.crossed_above(dataframe['fft_angle'], 0), 1, 0)
        else:
            dataframe['buy_region'] = np.where(qtpylib.crossed_above(dataframe['fft_angle'], 0), 1, 0)
            dataframe['sell_region'] = np.where(qtpylib.crossed_below(dataframe['fft_angle'], 0), 1, 0)

        return dataframe


    def runFourier(self, a: np.ndarray) -> np.float:
        #must return scalar, so just calculate prediction and take last value
        npredict = self.buy_fft_predict.value
        prediction = self.fourierExtrapolation(np.array(a),npredict )
        length = len(prediction)
        return prediction[length-1]

    # function to predict future steps. x is the raw data, not the FFT
    def fourierExtrapolation(self, x, n_predict):
        n = x.size
        n_harm = self.buy_fft_nharmonics.value  # number of harmonics in model
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

        buy_cond = (
            qtpylib.crossed_above(dataframe['buy_region'], 0.9)
        )

        conditions.append(buy_cond)

        # set buy tags
        dataframe.loc[buy_cond, 'buy_tag'] += 'fft_buy_low '

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

        sell_cond = (
            qtpylib.crossed_above(dataframe['sell_region'], 0.9)
        )

        conditions.append(sell_cond)

        # set sell tags
        dataframe.loc[sell_cond, 'exit_tag'] += 'fft_sell_high '

        if conditions:
            dataframe.loc[reduce(lambda x, y: x & y, conditions), 'sell'] = 1

        return dataframe
