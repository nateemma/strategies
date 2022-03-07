import numpy as np
import talib.abstract as ta
import freqtrade.vendor.qtpylib.indicators as qtpylib
import arrow

from freqtrade.exchange import timeframe_to_minutes
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

import custom_indicators as cta
import scipy

import pywt

'''
####################################################################################
DWT_1 - use a Discreet Wavelet Transform (DWT) to estimate future price movements

####################################################################################
'''


class DWT_1(IStrategy):
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
    startup_candle_count: int = 128
    process_only_new_candles = True

    ###################################

    # Strategy Specific Variable Storage

    ## Hyperopt Variables

    # DWT Filter limits
    buy_dwt_diff = DecimalParameter(0.000, 0.050, decimals=3, default=0.001, space='buy', load=True, optimize=True)
    buy_dwt_window = IntParameter(8, 164, default=64, space='buy', load=True, optimize=True)
    buy_dwt_lookahead = IntParameter(0, 64, default=0, space='buy', load=True, optimize=True)

    sell_dwt_diff = DecimalParameter(-0.050, 0.000, decimals=3, default=-0.001, space='sell', load=True, optimize=True)


    ###################################

    """
    Informative Pair Definitions
    """

    def informative_pairs(self):

        pairs = self.dp.current_whitelist()
        informative_pairs = [(pair, self.inf_timeframe) for pair in pairs]
        return informative_pairs

    ###################################

    """
    Indicator Definitions
    """

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:

        # Base pair informative timeframe indicators
        curr_pair = metadata['pair']
        informative = self.dp.get_pair_dataframe(pair=curr_pair, timeframe=self.inf_timeframe)

        # DWT filter
        # slightly smoothed closing prices
        # informative['smoothed'] = ta.LINEARREG(informative['close'], timeperiod=3)
        informative['tema'] = ta.TEMA(informative, timeperiod=6)

        # run the filter on a rolling window
        informative['dwt_predict'] = informative['tema'].rolling(window=self.buy_dwt_window.value).apply(self.runDWT)

        # merge into normal timeframe
        dataframe = merge_informative_pair(dataframe, informative, self.timeframe, self.inf_timeframe, ffill=True)

        # calculate predictive indicators in shorter timeframe (not informative)


        # reference
        dataframe['tema'] = ta.TEMA(dataframe, timeperiod=6)

        # dataframe['dwt_predict'] = ta.LINEARREG(dataframe[f"dwt_predict_{self.inf_timeframe}"], timeperiod=12)
        # dataframe['dwt_predict'] = dataframe[f"dwt_predict_{self.inf_timeframe}"]
        dataframe['dwt_predict'] = ta.LINEARREG(dataframe[f"dwt_predict_{self.inf_timeframe}"], timeperiod=12)
        # dataframe['smoothed'] = dataframe[f"smoothed_{self.inf_timeframe}"]
        dataframe['dwt_predict_diff'] = (dataframe['tema'] - dataframe['dwt_predict']) / dataframe['dwt_predict']
        # dataframe['dwt_angle'] = ta.LINEARREG_SLOPE(dataframe['dwt_predict'], timeperiod=3)

        dataframe['buy_region'] = np.where(dataframe['dwt_predict_diff'] > self.buy_dwt_diff.value, 1, 0)
        dataframe['sell_region'] = np.where(dataframe['dwt_predict_diff'] < self.sell_dwt_diff.value, 1, 0)

        return dataframe

    def madev(self, d, axis=None):
        """ Mean absolute deviation of a signal """
        return np.mean(np.absolute(d - np.mean(d, axis)), axis)

    def runDWT(self, data: np.ndarray) -> np.float:

        wavelet = 'db9'
        level = 1
        # wmode = "periodization"
        wmode = "constant"
        length = len(data)

        coeff = pywt.wavedec(data, wavelet, mode=wmode)
        # coeff = pywt.dwt(data, wavelet, mode=wmode)
        sigma = (1 / 0.6745) * self.madev(coeff[-level])
        uthresh = sigma * np.sqrt(2 * np.log(length))
        coeff[1:] = (pywt.threshold(i, value=uthresh, mode='hard') for i in coeff[1:])
        y = pywt.waverec(coeff, wavelet, mode=wmode)
        # y = pywt.idwt(coeff, wavelet, mode=wmode)

        # predict the next value 'n' steps away
        # NOTE: currently not working well for n > 1
        nsteps = self.buy_dwt_lookahead.value

        x = np.linspace(0, length-1, num=length)

        # print("x: ", x)
        # print("y: ", y)

        f = scipy.interpolate.interp1d(x, y, fill_value='extrapolate')

        predict = f(length-1+nsteps)

        # print("Predict: ", predict)

        return predict

    ###################################

    """
    Buy Signal
    """

    def populate_buy_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        conditions = []
        dataframe.loc[:, 'buy_tag'] = ''

        # conditions.append(dataframe['volume'] > 0)

        # DWT triggers
        dwt_cond = (
            qtpylib.crossed_above(dataframe['buy_region'], 0.9)
        )

        # angle_cond = (
        #     (dataframe['dwt_angle'] >= 0.0) &
        #     (dataframe['dwt_angle'].shift(1) < 0.0)
        # )

        conditions.append(dwt_cond)
        # conditions.append(DWT_cond & angle_cond)
        # conditions.append(DWT_cond | angle_cond)
        # conditions.append(angle_cond)

        # set buy tags
        dataframe.loc[dwt_cond, 'buy_tag'] += 'dwt_buy_1 '
        # dataframe.loc[angle_cond, 'buy_tag'] += 'dwt_bottom '

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

        # DWT triggers
        dwt_cond = (
            qtpylib.crossed_above(dataframe['sell_region'], 0.9)
        )

        # angle_cond = (
        #     (dataframe['dwt_angle'] <= 0.0) &
        #     (dataframe['dwt_angle'].shift(1) > 0.0)
        # )

        conditions.append(dwt_cond)
        # conditions.append(DWT_cond & angle_cond)
        # conditions.append(DWT_cond | angle_cond)
        # conditions.append(angle_cond)

        # set buy tags
        dataframe.loc[dwt_cond, 'exit_tag'] += 'dwt_sell_1 '
        # dataframe.loc[angle_cond, 'exit_tag'] += 'dwt_top '

        if conditions:
            dataframe.loc[reduce(lambda x, y: x & y, conditions), 'sell'] = 1

        return dataframe
