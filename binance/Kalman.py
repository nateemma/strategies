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

from  pykalman import KalmanFilter


"""
####################################################################################
Kalman - use a Kalmqn Filter to estimate future price movements,
          but *without* Fisher/Williams/Bollinger buy/sell signals

####################################################################################
"""


class Kalman(IStrategy):
    # Do *not* hyperopt for the roi and stoploss spaces

    # ROI table:
    minimal_roi = {
        "0": 10
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

    use_custom_stoploss = True

    # Recommended
    use_exit_signal = True
    exit_profit_only = False
    ignore_roi_if_entry_signal = True

    # Required
    startup_candle_count: int = 32
    process_only_new_candles = True

    custom_trade_info = {}

    ###################################

    # Strategy Specific Variable Storage

    kf_window = startup_candle_count
    filter_list = {}
    filter_init_list = {}

    kalman_filter = KalmanFilter(
                transition_matrices=1.0,
                observation_matrices=1.0,
                initial_state_mean=0.0,
                initial_state_covariance=1.0,
                observation_covariance=0.1,
                transition_covariance=0.1
            )
    current_pair = ""


    ## Hyperopt Variables
    
    # Kalman  hyperparams
    entry_kf_diff = DecimalParameter(0.0, 5.0, decimals=1, default=1.0, space='buy', load=True, optimize=True)

    exit_kf_diff = DecimalParameter(-5.0, 0.0, decimals=1, default=-0.1, space='sell', load=True, optimize=True)


    # # FBB_ hyperparams
    # buy_bb_gain = DecimalParameter(0.01, 0.50, decimals=2, default=0.09, space='buy', load=True, optimize=True)
    # buy_fisher_wr = DecimalParameter(-0.99, -0.75, decimals=2, default=-0.75, space='buy', load=True, optimize=True)
    # # buy_force_fisher_wr = DecimalParameter(-0.99, -0.85, decimals=2, default=-0.99, space='buy', load=True, optimize=True)
    #
    # exit_bb_gain = DecimalParameter(0.7, 1.5, decimals=2, default=0.8, space='sell', load=True, optimize=True)
    # exit_fisher_wr = DecimalParameter(0.75, 0.99, decimals=2, default=0.9, space='sell', load=True, optimize=True)
    # # exit_force_fisher_wr = DecimalParameter(0.85, 0.99, decimals=2, default=0.99, space='sell', load=True, optimize=True)


    # Custom Sell Profit (formerly Dynamic ROI)
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

        # Kalman Filter

        # get filter for current pair

        self.current_pair = curr_pair

        # create if not already done
        if not curr_pair in self.filter_list:
            self.filter_list[curr_pair] = kalman_filter = KalmanFilter(
                transition_matrices=1.0,
                observation_matrices=1.0,
                initial_state_mean=0.0,
                initial_state_covariance=1.0,
                observation_covariance=0.1,
                transition_covariance=0.1
            )
            self.filter_init_list[curr_pair] = False


        # set current filter (can't pass parameter to apply())
        self.kalman_filter = self.filter_list[curr_pair]

        informative['kf_predict'] = informative['close'].rolling(window=self.kf_window).apply(self.model)

        # merge into normal timeframe
        dataframe = merge_informative_pair(dataframe, informative, self.timeframe, self.inf_timeframe, ffill=True)

        # calculate predictive indicators in shorter timeframe (not informative)

        dataframe['kf_predict'] = dataframe[f"kf_predict_{self.inf_timeframe}"]
        dataframe['kf_predict_diff'] = 100.0 * (dataframe['kf_predict'] - dataframe['close']) / dataframe['close']

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

    def model(self, a: np.ndarray) -> np.float:

        # scale the data
        standardized = a.copy()
        w_mean = np.mean(standardized)
        w_std = np.std(standardized)
        scaled = (standardized - w_mean) / w_std
        scaled.fillna(0, inplace=True)

        # init filter if needed
        if not self.filter_init_list[self.current_pair]:
            self.filter_init_list[self.current_pair] = True
            self.filter_list[self.current_pair] = self.filter_list[self.current_pair].em(scaled, n_iter=6)

        # get the Kalman model
        restored_sig = self.kalmanModel(scaled, self.kalman_filter)

        # re-trend
        model = (restored_sig * w_std) + w_mean

        length = len(model)
        return model[length-1]
    
    def scaledModel(self, a: np.ndarray) -> np.float:

        # scale the data
        standardized = a.copy()
        w_mean = np.mean(standardized)
        w_std = np.std(standardized)
        scaled = (standardized - w_mean) / w_std
        scaled.fillna(0, inplace=True)

        # get the Fourier model
        model = self.kalmanModel(scaled, self.kalman_filter)

        length = len(model)
        return model[length-1]

    def scaledData(self, a: np.ndarray) -> np.float:

        # scale the data
        standardized = a.copy()
        w_mean = np.mean(standardized)
        w_std = np.std(standardized)
        scaled = (standardized - w_mean) / w_std
        scaled.fillna(0, inplace=True)

        length = len(scaled)
        return scaled.ravel()[length-1]

    def kalmanModel(self, data, kfilter: KalmanFilter):

        n = len(data)
        x = np.array(data)

        # kfilter = kfilter.em(data, n_iter=6)

        # mean, cov = kfilter.filter(x)
        # kfilter.filter_update(mean[0], cov[0])
        #
        # mean = mean.squeeze()
        # print("model(", len(mean), "): ", mean)

        # predict next close
        pr_mean, pr_cov = kfilter.smooth(x)
        restored_sig = pr_mean.squeeze()
        # print ("Predict(", len(restored_sig), "): ", restored_sig)


        ldiff = len(restored_sig) - len(x)
        model = restored_sig[ldiff:]

        return model

    # ###################################
    # 
    # # Williams %R
    # def williams_r(self, dataframe: DataFrame, period: int = 14) -> Series:
    #     """Williams %R, or just %R, is a technical analysis oscillator showing the current closing price in relation to the high and low
    #         of the past N days (for a given N). It was developed by a publisher and promoter of trading materials, Larry Williams.
    #         Its purpose is to tell whether a stock or commodity market is trading near the high or the low, or somewhere in between,
    #         of its recent trading range.
    #         The oscillator is on a negative scale, from −100 (lowest) up to 0 (highest).
    #     """
    # 
    #     highest_high = dataframe["high"].rolling(center=False, window=period).max()
    #     lowest_low = dataframe["low"].rolling(center=False, window=period).min()
    # 
    #     WR = Series(
    #         (highest_high - dataframe["close"]) / (highest_high - lowest_low),
    #         name=f"{period} Williams %R",
    #     )
    # 
    #     return WR * -100
    ###################################

    """
    Buy Signal
    """


    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        conditions = []
        dataframe.loc[:, 'enter_tag'] = ''

        # conditions.append(dataframe['volume'] > 0)


        # Kalman triggers
        kf_cond = (
                qtpylib.crossed_above(dataframe['kf_predict_diff'], self.entry_kf_diff.value)
        )

        conditions.append(kf_cond)

        # Model will spike on big gains, so try to constrain
        spike_cond = (
                dataframe['kf_predict_diff'] < 2.0 * self.entry_kf_diff.value
        )
        conditions.append(spike_cond)

        # set buy tags
        dataframe.loc[kf_cond, 'enter_tag'] += 'kf_buy '

        if conditions:
            dataframe.loc[reduce(lambda x, y: x & y, conditions), 'enter_long'] = 1

        return dataframe


    ###################################

    """
    Sell Signal
    """


    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        conditions = []
        dataframe.loc[:, 'exit_tag'] = ''

        # Kalman triggers
        kf_cond = (
                qtpylib.crossed_below(dataframe['kf_predict_diff'], self.exit_kf_diff.value)
        )

        conditions.append(kf_cond)

        # DWTs will spike on big gains, so try to constrain
        spike_cond = (
                dataframe['kf_predict_diff'] > 2.0 * self.exit_kf_diff.value
        )
        conditions.append(spike_cond)

        # set sell tags
        dataframe.loc[kf_cond, 'exit_tag'] += 'kf_sell '



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

        # Don't sell if we are in a trend unless the pullback threshold is met
        if in_trend == True and current_profit > 0:
            # Record that we were in a trend for this trade/pair for a more useful sell message later
            self.custom_trade_info[trade.pair]['had-trend'] = True
            # If pullback is enabled and profit has pulled back allow a sell, maybe
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