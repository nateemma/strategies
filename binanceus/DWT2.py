import numpy as np
import scipy.fft
from scipy.fft import rfft, irfft
import talib.abstract as ta
from scipy.ndimage import gaussian_filter1d

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

import custom_indicators as cta

import pywt
import scipy


"""
####################################################################################
DWT - use a Discreet Wavelet Transform to estimate future price movements

####################################################################################
"""


class DWT2(IStrategy):
    # Do *not* hyperopt for the roi and stoploss spaces

    # ROI table:
    minimal_roi = {
        "0": 0.1
    }

    # Stoploss:
    stoploss = -0.10

    # Trailing stop:
    trailing_stop = False
    trailing_stop_positive = None
    trailing_stop_positive_offset = 0.0
    trailing_only_offset_is_reached = False

    timeframe = '5m'
    inf_timeframe = '15m'

    use_custom_stoploss = True

    # Recommended
    use_exit_signal = True
    exit_profit_only = False
    ignore_roi_if_entry_signal = True

    # Required
    startup_candle_count: int = 128 # must be power of 2

    process_only_new_candles = True

    custom_trade_info = {}

    ###################################

    # Strategy Specific Variable Storage

    ## Hyperopt Variables

    dwt_window = startup_candle_count

    # DWT  hyperparams
    entry_dwt_diff = DecimalParameter(-3.0, 0.0, decimals=1, default=-1.0, space='buy', load=True, optimize=True)
    exit_dwt_diff = DecimalParameter(0.0, 3.0, decimals=1, default=1.0, space='sell', load=True, optimize=True)


    # Custom exit Profit (formerly Dynamic ROI)
    cexit_roi_type = CategoricalParameter(['static', 'decay', 'step'], default='step', space='sell', load=True,
                                          optimize=True)
    cexit_roi_time = IntParameter(720, 1440, default=720, space='sell', load=True, optimize=True)
    cexit_roi_start = DecimalParameter(0.01, 0.05, default=0.01, space='sell', load=True, optimize=True)
    cexit_roi_end = DecimalParameter(0.0, 0.01, default=0, space='sell', load=True, optimize=True)
    cexit_trend_type = CategoricalParameter(['rmi', 'ssl', 'candle', 'any', 'none'], default='any', space='sell',
                                            load=True, optimize=True)
    cexit_pullback = CategoricalParameter([True, False], default=True, space='sell', load=True, optimize=True)
    cexit_pullback_amount = DecimalParameter(0.005, 0.03, default=0.01, space='sell', load=True, optimize=True)
    cexit_pullback_respect_roi = CategoricalParameter([True, False], default=False, space='sell', load=True,
                                                      optimize=True)
    cexit_endtrend_respect_roi = CategoricalParameter([True, False], default=False, space='sell', load=True,
                                                      optimize=True)

    # Custom Stoploss
    cstop_loss_threshold = DecimalParameter(-0.05, -0.01, default=-0.03, space='sell', load=True, optimize=True)
    cstop_bail_how = CategoricalParameter(['roc', 'time', 'any', 'none'], default='none', space='sell', load=True,
                                          optimize=True)
    cstop_bail_roc = DecimalParameter(-5.0, -1.0, default=-3.0, space='sell', load=True, optimize=True)
    cstop_bail_time = IntParameter(60, 1440, default=720, space='sell', load=True, optimize=True)
    cstop_bail_time_trend = CategoricalParameter([True, False], default=True, space='sell', load=True, optimize=True)
    cstop_max_stoploss =  DecimalParameter(-0.30, -0.01, default=-0.10, space='sell', load=True, optimize=True)

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


        # Base pair informative timeframe indicators
        curr_pair = metadata['pair']

        # DWT

        # DWT model of stock price. Have to use rolling window, otherwise it looks into the future
        dataframe['dwt'] = dataframe['close'].rolling(window=self.dwt_window).apply(self.roll_get_dwt)
        dataframe['smooth'] = dataframe['close'].rolling(window=self.dwt_window).apply(self.roll_smooth)

        # smoothed version - useful for trends
        # dataframe['smooth'] = gaussian_filter1d(dataframe['close'], 8)

        dataframe['dwt_model_diff'] = 100.0 * (dataframe['dwt'] - dataframe['smooth']) / dataframe['smooth']

        # Custom Stoploss

        if not metadata['pair'] in self.custom_trade_info:
            self.custom_trade_info[metadata['pair']] = {}
            if not 'had-trend' in self.custom_trade_info[metadata["pair"]]:
                self.custom_trade_info[metadata['pair']]['had-trend'] = False

        # RMI: https://www.tradingview.com/script/kwIt9OgQ-Relative-Momentum-Index/
        dataframe['rmi'] = cta.RMI(dataframe, length=24, mom=5)

        # MA Streak: https://www.tradingview.com/script/Yq1z7cIv-MA-Streak-Can-Show-When-a-Run-Is-Getting-Long-in-the-Tooth/
        dataframe['mastreak'] = cta.mastreak(dataframe, period=4)

        # Trends, Peaks and Crosses
        dataframe['candle-up'] = np.where(dataframe['close'] >= dataframe['open'], 1, 0)
        dataframe['candle-up-trend'] = np.where(dataframe['candle-up'].rolling(5).sum() >= 3, 1, 0)

        dataframe['rmi-up'] = np.where(dataframe['rmi'] >= dataframe['rmi'].shift(), 1, 0)
        dataframe['rmi-up-trend'] = np.where(dataframe['rmi-up'].rolling(5).sum() >= 3, 1, 0)

        dataframe['rmi-dn'] = np.where(dataframe['rmi'] <= dataframe['rmi'].shift(), 1, 0)
        dataframe['rmi-dn-count'] = dataframe['rmi-dn'].rolling(8).sum()

        # Indicators used only for ROI and Custom Stoploss
        ssldown, sslup = cta.SSLChannels_ATR(dataframe, length=21)
        dataframe['sroc'] = cta.SROC(dataframe, roclen=21, emalen=13, smooth=21)
        dataframe['ssl-dir'] = np.where(sslup > ssldown, 'up', 'down')

        return dataframe

    ###################################


    def get_dwt(self, col):

        a = np.array(col)

        # de-trend the data
        w_mean = a.mean()
        w_std = a.std()
        a_notrend = (a - w_mean) / w_std
        # a_notrend = a_notrend.clip(min=-3.0, max=3.0)

        # get DWT model of data
        restored_sig = self.dwtModel(a_notrend)

        # re-trend
        model = (restored_sig * w_std) + w_mean

        return model

    def roll_get_dwt(self, col) -> np.float:
        # must return scalar, so just calculate prediction and take last value

        model = self.get_dwt(col)

        length = len(model)
        if length > 0:
            return model[length - 1]
        else:
            print("model:", model)
            return 0.0

    def dwtModel(self, data):

        # the choice of wavelet makes a big difference
        # for an overview, check out: https://www.kaggle.com/theoviel/denoising-with-direct-wavelet-transform
        # wavelet = 'db1'
        wavelet = 'db8'
        # wavelet = 'bior1.1'
        # wavelet = 'haar'  # deals well with harsh transitions
        level = 1
        wmode = "smooth"
        tmode = "hard"
        length = len(data)

        # Apply DWT transform
        coeff = pywt.wavedec(data, wavelet, mode=wmode)

        # remove higher harmonics
        std = np.std(coeff[level])
        sigma = (1 / 0.6745) * self.madev(coeff[-level])
        # sigma = self.madev(coeff[-level])
        uthresh = sigma * np.sqrt(2 * np.log(length))

        coeff[1:] = (pywt.threshold(i, value=uthresh, mode=tmode) for i in coeff[1:])

        # inverse DWT transform
        model = pywt.waverec(coeff, wavelet, mode=wmode)

        # there is a known bug in waverec where odd numbered lengths result in an extra item at the end
        diff = len(model) - len(data)
        return model[0:len(model) - diff]
        # return model[diff:]

    def madev(self, d, axis=None):
        """ Mean absolute deviation of a signal """
        return np.mean(np.absolute(d - np.mean(d, axis)), axis)


    def roll_smooth(self, col) -> np.float:
        # must return scalar, so just calculate prediction and take last value

        smooth = gaussian_filter1d(col, 4)

        length = len(smooth)
        if length > 0:
            return smooth[length - 1]
        else:
            print("model:", smooth)
            return 0.0

    ###################################

    """
    entry Signal
    """


    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        conditions = []
        dataframe.loc[:, 'enter_tag'] = ''

        # conditions.append(dataframe['volume'] > 0)

        # DWT triggers
        dwt_cond = (
                qtpylib.crossed_below(dataframe['dwt_model_diff'], self.entry_dwt_diff.value)
        )

        conditions.append(dwt_cond)

        # # DWTs will spike on big gains, so try to constrain
        # spike_cond = (
        #         dataframe['dwt_model_diff'] < 2.0 * self.entry_dwt_diff.value
        # )
        # conditions.append(spike_cond)

        # set entry tags
        dataframe.loc[dwt_cond, 'enter_tag'] += 'dwt_entry '

        if conditions:
            dataframe.loc[reduce(lambda x, y: x & y, conditions), 'enter_long'] = 1

        return dataframe


    ###################################

    """
    exit Signal
    """


    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        conditions = []
        dataframe.loc[:, 'exit_tag'] = ''

        # FFT triggers
        dwt_cond = (
                qtpylib.crossed_above(dataframe['dwt_model_diff'], self.exit_dwt_diff.value)
        )

        conditions.append(dwt_cond)

        # # DWTs will spike on big gains, so try to constrain
        # spike_cond = (
        #         dataframe['dwt_model_diff'] > 2.0 * self.exit_dwt_diff.value
        # )
        # conditions.append(spike_cond)

        # set exit tags
        dataframe.loc[dwt_cond, 'exit_tag'] += 'dwt_exit '

        if conditions:
            dataframe.loc[reduce(lambda x, y: x & y, conditions), 'exit_long'] = 1

        return dataframe


    ###################################

    """
    Custom Stoploss
    """

    def custom_stoploss(self, pair: str, trade: 'Trade', current_time: datetime, current_rate: float,
                        current_profit: float, **kwargs) -> float:

        dataframe, _ = self.dp.get_analyzed_dataframe(pair=pair, timeframe=self.timeframe)
        last_candle = dataframe.iloc[-1].squeeze()
        trade_dur = int((current_time.timestamp() - trade.open_date_utc.timestamp()) // 60)
        in_trend = self.custom_trade_info[trade.pair]['had-trend']

        # limit stoploss
        if current_profit <  self.cstop_max_stoploss.value:
            return 0.01

        # Determine how we exit when we are in a loss
        if current_profit < self.cstop_loss_threshold.value:
            if self.cstop_bail_how.value == 'roc' or self.cstop_bail_how.value == 'any':
                # Dynamic bailout based on rate of change
                if last_candle['sroc'] <= self.cstop_bail_roc.value:
                    return 0.01
            if self.cstop_bail_how.value == 'time' or self.cstop_bail_how.value == 'any':
                # Dynamic bailout based on time, unless time_trend is true and there is a potential reversal
                if trade_dur > self.cstop_bail_time.value:
                    if self.cstop_bail_time_trend.value == True and in_trend == True:
                        return 1
                    else:
                        return 0.01
        return 1

    ###################################

    """
    Custom exit
    """

    def custom_exit(self, pair: str, trade: 'Trade', current_time: 'datetime', current_rate: float,
                    current_profit: float, **kwargs):

        dataframe, _ = self.dp.get_analyzed_dataframe(pair=pair, timeframe=self.timeframe)
        last_candle = dataframe.iloc[-1].squeeze()

        trade_dur = int((current_time.timestamp() - trade.open_date_utc.timestamp()) // 60)
        max_profit = max(0, trade.calc_profit_ratio(trade.max_rate))
        pullback_value = max(0, (max_profit - self.cexit_pullback_amount.value))
        in_trend = False

        # Determine our current ROI point based on the defined type
        if self.cexit_roi_type.value == 'static':
            min_roi = self.cexit_roi_start.value
        elif self.cexit_roi_type.value == 'decay':
            min_roi = cta.linear_decay(self.cexit_roi_start.value, self.cexit_roi_end.value, 0,
                                       self.cexit_roi_time.value, trade_dur)
        elif self.cexit_roi_type.value == 'step':
            if trade_dur < self.cexit_roi_time.value:
                min_roi = self.cexit_roi_start.value
            else:
                min_roi = self.cexit_roi_end.value

        # Determine if there is a trend
        if self.cexit_trend_type.value == 'rmi' or self.cexit_trend_type.value == 'any':
            if last_candle['rmi-up-trend'] == 1:
                in_trend = True
        if self.cexit_trend_type.value == 'ssl' or self.cexit_trend_type.value == 'any':
            if last_candle['ssl-dir'] == 'up':
                in_trend = True
        if self.cexit_trend_type.value == 'candle' or self.cexit_trend_type.value == 'any':
            if last_candle['candle-up-trend'] == 1:
                in_trend = True

        # Don't exit if we are in a trend unless the pullback threshold is met
        if in_trend == True and current_profit > 0:
            # Record that we were in a trend for this trade/pair for a more useful exit message later
            self.custom_trade_info[trade.pair]['had-trend'] = True
            # If pullback is enabled and profit has pulled back allow a exit, maybe
            if self.cexit_pullback.value == True and (current_profit <= pullback_value):
                if self.cexit_pullback_respect_roi.value == True and current_profit > min_roi:
                    return 'intrend_pullback_roi'
                elif self.cexit_pullback_respect_roi.value == False:
                    if current_profit > min_roi:
                        return 'intrend_pullback_roi'
                    else:
                        return 'intrend_pullback_noroi'
            # We are in a trend and pullback is disabled or has not happened or various criteria were not met, hold
            return None
        # If we are not in a trend, just use the roi value
        elif in_trend == False:
            if self.custom_trade_info[trade.pair]['had-trend']:
                if current_profit > min_roi:
                    self.custom_trade_info[trade.pair]['had-trend'] = False
                    return 'trend_roi'
                elif self.cexit_endtrend_respect_roi.value == False:
                    self.custom_trade_info[trade.pair]['had-trend'] = False
                    return 'trend_noroi'
            elif current_profit > min_roi:
                return 'notrend_roi'
        else:
            return None

