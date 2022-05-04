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

import custom_indicators as cta

import pywt
import scipy


"""
####################################################################################
DWT - use a Discreet Wavelet Transform to estimate future price movements

####################################################################################
"""


class DWT(IStrategy):
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
    use_sell_signal = True
    sell_profit_only = False
    ignore_roi_if_buy_signal = True

    # Required
    startup_candle_count: int = 128 # must be power of 2

    process_only_new_candles = True

    custom_trade_info = {}

    ###################################

    # Strategy Specific Variable Storage

    ## Hyperopt Variables

    dwt_window = startup_candle_count

    # DWT  hyperparams
    buy_dwt_diff = DecimalParameter(0.0, 5.0, decimals=1, default=1.0, space='buy', load=True, optimize=True)
    sell_dwt_diff = DecimalParameter(-5.0, 0.0, decimals=1, default=-1.0, space='sell', load=True, optimize=True)


    # Custom Sell Profit (formerly Dynamic ROI)
    csell_roi_type = CategoricalParameter(['static', 'decay', 'step'], default='step', space='sell', load=True,
                                          optimize=True)
    csell_roi_time = IntParameter(720, 1440, default=720, space='sell', load=True, optimize=True)
    csell_roi_start = DecimalParameter(0.01, 0.05, default=0.01, space='sell', load=True, optimize=True)
    csell_roi_end = DecimalParameter(0.0, 0.01, default=0, space='sell', load=True, optimize=True)
    csell_trend_type = CategoricalParameter(['rmi', 'ssl', 'candle', 'any', 'none'], default='any', space='sell',
                                            load=True, optimize=True)
    csell_pullback = CategoricalParameter([True, False], default=True, space='sell', load=True, optimize=True)
    csell_pullback_amount = DecimalParameter(0.005, 0.03, default=0.01, space='sell', load=True, optimize=True)
    csell_pullback_respect_roi = CategoricalParameter([True, False], default=False, space='sell', load=True,
                                                      optimize=True)
    csell_endtrend_respect_roi = CategoricalParameter([True, False], default=False, space='sell', load=True,
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

        # DWT

        informative['dwt_model'] = informative['close'].rolling(window=self.dwt_window).apply(self.model)
        # informative['dwt_predict'] = informative['dwt_model'].rolling(window=self.dwt_window).apply(self.predict)

        # merge into normal timeframe
        dataframe = merge_informative_pair(dataframe, informative, self.timeframe, self.inf_timeframe, ffill=True)

        # calculate predictive indicators in shorter timeframe (not informative)

        dataframe['dwt_model'] = dataframe[f"dwt_model_{self.inf_timeframe}"]
        dataframe['dwt_model_diff'] = 100.0 * (dataframe['dwt_model'] - dataframe['close']) / dataframe['close']
        # dataframe['dwt_predict'] = dataframe[f"dwt_predict_{self.inf_timeframe}"]
        # dataframe['dwt_predict_diff'] = 100.0 * (dataframe['dwt_predict'] - dataframe['dwt_model']) / dataframe['dwt_model']

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


    def madev(self, d, axis=None):
        """ Mean absolute deviation of a signal """
        return np.mean(np.absolute(d - np.mean(d, axis)), axis)

    def dwtModel(self, data):

        # the choice of wavelet makes a big difference
        # for an overview, check out: https://www.kaggle.com/theoviel/denoising-with-direct-wavelet-transform
        # wavelet = 'db1'
        # wavelet = 'bior1.1'
        wavelet = 'haar' # deals well with harsh transitions
        level = 1
        wmode = "smooth"
        length = len(data)

        coeff = pywt.wavedec(data, wavelet, mode=wmode)

        # remove higher harmonics
        sigma = (1 / 0.6745) * self.madev(coeff[-level])
        uthresh = sigma * np.sqrt(2 * np.log(length))
        coeff[1:] = (pywt.threshold(i, value=uthresh, mode='hard') for i in coeff[1:])

        # inverse transform
        model = pywt.waverec(coeff, wavelet, mode=wmode)

        return model

    def model(self, a: np.ndarray) -> np.float:
        #must return scalar, so just calculate prediction and take last value
        # model = self.dwtModel(np.array(a))

        # de-trend the data
        w_mean = a.mean()
        w_std = a.std()
        x_notrend = (a - w_mean) / w_std

        # get DWT model of data
        restored_sig = self.dwtModel(x_notrend)

        # re-trend
        model = (restored_sig * w_std) + w_mean

        length = len(model)
        return model[length-1]

    def scaledModel(self, a: np.ndarray) -> np.float:
        #must return scalar, so just calculate prediction and take last value
        # model = self.dwtModel(np.array(a))

        # de-trend the data
        w_mean = a.mean()
        w_std = a.std()
        x_notrend = (a - w_mean) / w_std

        # get DWT model of data
        model = self.dwtModel(x_notrend)

        length = len(model)
        return model[length-1]

    def scaledData(self, a: np.ndarray) -> np.float:

        # scale the data
        standardized = a.copy()
        w_mean = np.mean(standardized)
        w_std = np.std(standardized)
        scaled = (standardized - w_mean) / w_std
        # scaled.fillna(0, inplace=True)

        length = len(scaled)
        return scaled.ravel()[length-1]

    def predict(self, a: np.ndarray) -> np.float:

        # predicts the next value using polynomial extrapolation

        # a.fillna(0)

        # fit the supplied data
        # Note: extrapolation is notoriously fickle. Be careful
        length = len(a)
        x = np.arange(length)
        f = scipy.interpolate.UnivariateSpline(x, a, k=5)

        predict = f(length)

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
                qtpylib.crossed_above(dataframe['dwt_model_diff'], self.buy_dwt_diff.value)
        )

        conditions.append(dwt_cond)

        # DWTs will spike on big gains, so try to constrain
        spike_cond = (
                dataframe['dwt_model_diff'] < 2.0 * self.buy_dwt_diff.value
        )
        conditions.append(spike_cond)

        # set buy tags
        dataframe.loc[dwt_cond, 'buy_tag'] += 'dwt_buy '

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
        dwt_cond = (
                qtpylib.crossed_below(dataframe['dwt_model_diff'], self.sell_dwt_diff.value)
        )

        conditions.append(dwt_cond)

        # DWTs will spike on big gains, so try to constrain
        spike_cond = (
                dataframe['dwt_model_diff'] > 2.0 * self.sell_dwt_diff.value
        )
        conditions.append(spike_cond)

        # set sell tags
        dataframe.loc[dwt_cond, 'exit_tag'] += 'dwt_sell '

        if conditions:
            dataframe.loc[reduce(lambda x, y: x & y, conditions), 'sell'] = 1

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

        # Determine how we sell when we are in a loss
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
    Custom Sell
    """

    def custom_sell(self, pair: str, trade: 'Trade', current_time: 'datetime', current_rate: float,
                    current_profit: float, **kwargs):

        dataframe, _ = self.dp.get_analyzed_dataframe(pair=pair, timeframe=self.timeframe)
        last_candle = dataframe.iloc[-1].squeeze()

        trade_dur = int((current_time.timestamp() - trade.open_date_utc.timestamp()) // 60)
        max_profit = max(0, trade.calc_profit_ratio(trade.max_rate))
        pullback_value = max(0, (max_profit - self.csell_pullback_amount.value))
        in_trend = False

        # Determine our current ROI point based on the defined type
        if self.csell_roi_type.value == 'static':
            min_roi = self.csell_roi_start.value
        elif self.csell_roi_type.value == 'decay':
            min_roi = cta.linear_decay(self.csell_roi_start.value, self.csell_roi_end.value, 0,
                                       self.csell_roi_time.value, trade_dur)
        elif self.csell_roi_type.value == 'step':
            if trade_dur < self.csell_roi_time.value:
                min_roi = self.csell_roi_start.value
            else:
                min_roi = self.csell_roi_end.value

        # Determine if there is a trend
        if self.csell_trend_type.value == 'rmi' or self.csell_trend_type.value == 'any':
            if last_candle['rmi-up-trend'] == 1:
                in_trend = True
        if self.csell_trend_type.value == 'ssl' or self.csell_trend_type.value == 'any':
            if last_candle['ssl-dir'] == 'up':
                in_trend = True
        if self.csell_trend_type.value == 'candle' or self.csell_trend_type.value == 'any':
            if last_candle['candle-up-trend'] == 1:
                in_trend = True

        # Don't sell if we are in a trend unless the pullback threshold is met
        if in_trend == True and current_profit > 0:
            # Record that we were in a trend for this trade/pair for a more useful sell message later
            self.custom_trade_info[trade.pair]['had-trend'] = True
            # If pullback is enabled and profit has pulled back allow a sell, maybe
            if self.csell_pullback.value == True and (current_profit <= pullback_value):
                if self.csell_pullback_respect_roi.value == True and current_profit > min_roi:
                    return 'intrend_pullback_roi'
                elif self.csell_pullback_respect_roi.value == False:
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
                elif self.csell_endtrend_respect_roi.value == False:
                    self.custom_trade_info[trade.pair]['had-trend'] = False
                    return 'trend_noroi'
            elif current_profit > min_roi:
                return 'notrend_roi'
        else:
            return None

