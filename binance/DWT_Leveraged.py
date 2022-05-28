import numpy as np
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

from technical.indicators import hull_moving_average

pd.options.mode.chained_assignment = None  # default='warn'

# Strategy specific imports, files must reside in same folder as strategy
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent))

import re


import custom_indicators as cta

import pywt


"""

This strategy is intended to work with leveraged pairs.
It uses the buy/sell signals from DWT, looking for both uptrends and downtrends
Note that these are not reall long/short pairs, but 'long' pairs that track 
long or short 'base' pairs

Note that this strat uses the 'base' pair to trigger buys/sells, so they must be in the config file
For example, ADA3S/USDT and ADA3L/USDT would be leveraged short/long pairs, and the associated 'base' pair is ADA/USDT
"""


class DWT_Leveraged(IStrategy):

    INTERFACE_VERSION = 3

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


    # NOTE: hyperspace parameters are in the associated .json file (<clasname>.json)
    #       Values in that file will override the default values in the variable definitions below
    #       If the .json file does not exist, you will need to run hyperopt to generate it

    ## Buy Space Hyperopt Variables

    entry_long_dwt_diff = DecimalParameter(0.1, 5.0, decimals=1, default=2.0, space='buy', load=True, optimize=True)
    entry_short_dwt_diff = DecimalParameter(-5.0, -0.1, decimals=1, default=-2.0, space='buy', load=True, optimize=True)
    # entry_trend_type = CategoricalParameter(['rmi', 'ssl', 'candle', 'macd', 'adx', 'none'], default='none', space='buy',
    #                                         load=True, optimize=True)

    ## Sell Space Hyperopt Variables

    exit_long_dwt_diff = DecimalParameter(-5.0, -0.1, decimals=1, default=-2.0, space='sell', load=True, optimize=True)
    exit_short_dwt_diff = DecimalParameter(0.1, 5.0, decimals=1, default=2.0, space='sell', load=True, optimize=True)

    # Custom exit Profit (formerly Dynamic ROI)

    cexit_long_roi_type = CategoricalParameter(['static', 'decay', 'step'], default='step', space='sell', load=True,
                                          optimize=True)
    cexit_long_roi_time = IntParameter(720, 1440, default=720, space='sell', load=True, optimize=True)
    cexit_long_roi_start = DecimalParameter(0.01, 0.05, default=0.01, space='sell', load=True, optimize=True)
    cexit_long_roi_end = DecimalParameter(0.0, 0.01, default=0, space='sell', load=True, optimize=True)
    cexit_long_trend_type = CategoricalParameter(['rmi', 'ssl', 'candle', 'any', 'none'], default='any', space='sell',
                                            load=True, optimize=True)
    cexit_long_pullback = CategoricalParameter([True, False], default=True, space='sell', load=True, optimize=True)
    cexit_long_pullback_amount = DecimalParameter(0.005, 0.03, default=0.01, space='sell', load=True, optimize=True)
    cexit_long_pullback_respect_roi = CategoricalParameter([True, False], default=False, space='sell', load=True,
                                                      optimize=True)
    cexit_long_endtrend_respect_roi = CategoricalParameter([True, False], default=False, space='sell', load=True,
                                                      optimize=True)
    cexit_short_roi_type = CategoricalParameter(['static', 'decay', 'step'], default='step', space='sell', load=True,
                                          optimize=True)

    cexit_short_roi_time = IntParameter(720, 1440, default=720, space='sell', load=True, optimize=True)
    cexit_short_roi_start = DecimalParameter(0.01, 0.05, default=0.01, space='sell', load=True, optimize=True)
    cexit_short_roi_end = DecimalParameter(0.0, 0.01, default=0, space='sell', load=True, optimize=True)
    cexit_short_trend_type = CategoricalParameter(['rmi', 'ssl', 'candle', 'any', 'none'], default='any', space='sell',
                                            load=True, optimize=True)
    cexit_short_pullback = CategoricalParameter([True, False], default=True, space='sell', load=True, optimize=True)
    cexit_short_pullback_amount = DecimalParameter(0.005, 0.03, default=0.01, space='sell', load=True, optimize=True)
    cexit_short_pullback_respect_roi = CategoricalParameter([True, False], default=False, space='sell', load=True,
                                                      optimize=True)
    cexit_short_endtrend_respect_roi = CategoricalParameter([True, False], default=False, space='sell', load=True,
                                                      optimize=True)

    # Custom Stoploss

    cstop_loss_threshold = DecimalParameter(-0.05, -0.01, default=-0.03, space='sell', load=True, optimize=True)
    cstop_bail_how = CategoricalParameter(['roc', 'time', 'any', 'none'], default='none', space='sell', load=True,
                                          optimize=True)
    cstop_bail_roc = DecimalParameter(-5.0, -1.0, default=-3.0, space='sell', load=True, optimize=True)
    cstop_bail_time = IntParameter(60, 1440, default=720, space='sell', load=True, optimize=True)
    cstop_bail_time_trend = CategoricalParameter([True, False], default=True, space='sell', load=True, optimize=True)
    cstop_max_stoploss =  DecimalParameter(-0.30, -0.01, default=-0.10, space='sell', load=True, optimize=True)

    # Strategy Specific Variable Storage
    dwt_window = startup_candle_count
    custom_trade_info = {}
    custom_fiat = "USDT"  # Only relevant if stake is BTC or ETH

    ############################################################################

    """
    Informative Pair Definitions
    """

    def informative_pairs(self):
        pairs = self.dp.current_whitelist()
        informative_pairs = []
        infs = {}
        for pair in pairs:
            inf_pair = self.getInformative(pair)
            # informative_pairs += [(pair, self.inf_timeframe)]
            if (inf_pair != ""):
                infs[inf_pair] = (inf_pair, self.inf_timeframe)

        informative_pairs = list(infs.values())

        # print("informative_pairs: ", informative_pairs)

        return informative_pairs

    ############################################################################

    """
    Indicator Definitions
    """

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:

        # Base pair informative timeframe indicators
        curr_pair = metadata['pair']

        # only process if long or short (not 'normal')
        if (self.isBull(curr_pair)) or (self.isBear(curr_pair)):
            inf_pair = self.getInformative(curr_pair)
            # print("pair: ", curr_pair, " inf_pair: ", inf_pair)

            inf_slow = self.dp.get_pair_dataframe(pair=inf_pair, timeframe=self.inf_timeframe)
            inf_fast = self.dp.get_pair_dataframe(pair=inf_pair, timeframe=self.timeframe)

            # DWT

            inf_slow['dwt_model'] = inf_slow['close'].rolling(window=self.dwt_window).apply(self.model)

            # trend (in informative)
            inf_fast['candle-up'] = np.where(inf_fast['close'] >= inf_fast['open'], 1, 0)
            inf_fast['candle-up-trend'] = np.where(inf_fast['candle-up'].rolling(5).sum() >= 3, 1, 0)
            inf_fast['candle-dn-trend'] = np.where(inf_fast['candle-up'].rolling(5).sum() <= 2, 1, 0)

            # merge into normal timeframe
            dataframe = merge_informative_pair(dataframe, inf_slow, self.timeframe, self.inf_timeframe, ffill=True)
            dataframe = merge_informative_pair(dataframe, inf_fast, self.timeframe, self.timeframe, ffill=True)

            # calculate predictive indicators in shorter timeframe (not informative)

            dataframe['dwt_model'] = dataframe[f"dwt_model_{self.inf_timeframe}"]
            dataframe['inf_close'] = dataframe[f"close_{self.timeframe}"]
            dataframe['inf_close2'] = dataframe['inf_close'] / dataframe['close']
            # dataframe['dwt_model_diff'] = 100.0 * (dataframe['dwt_model'] - dataframe['close']) / dataframe['close']
            dataframe['dwt_model_diff'] = 100.0 * (dataframe['dwt_model'] - dataframe['inf_close']) / dataframe['inf_close']

            dataframe['inf_candle-up-trend'] = dataframe[f"candle-up-trend_{self.timeframe}"]
            dataframe['inf_candle-dn-trend'] = dataframe[f"candle-dn-trend_{self.timeframe}"]

            # MACD
            macd = ta.MACD(dataframe)
            dataframe['macd'] = macd['macd']
            # dataframe['macdsignal'] = macd['macdsignal']
            dataframe['macdhist'] = macd['macdhist']

            # Custom Stoploss

            if not metadata['pair'] in self.custom_trade_info:
                self.custom_trade_info[metadata['pair']] = {}
                if not 'had-trend' in self.custom_trade_info[metadata["pair"]]:
                    self.custom_trade_info[metadata['pair']]['had-trend'] = False

            # MA Streak: https://www.tradingview.com/script/Yq1z7cIv-MA-Streak-Can-Show-When-a-Run-Is-Getting-Long-in-the-Tooth/
            # dataframe['mastreak'] = cta.mastreak(dataframe, period=4)

            # Trends

            dataframe['candle-up'] = np.where(dataframe['close'] >= dataframe['open'], 1, 0)
            dataframe['candle-up-trend'] = np.where(dataframe['candle-up'].rolling(5).sum() >= 3, 1, 0)
            dataframe['candle-dn-trend'] = np.where(dataframe['candle-up'].rolling(5).sum() <= 2, 1, 0)


            # RMI: https://www.tradingview.com/script/kwIt9OgQ-Relative-Momentum-Index/
            dataframe['rmi'] = cta.RMI(dataframe, length=24, mom=5)
            dataframe['rmi-up'] = np.where(dataframe['rmi'] >= dataframe['rmi'].shift(), 1, 0)
            dataframe['rmi-up-trend'] = np.where(dataframe['rmi-up'].rolling(5).sum() >= 3, 1, 0)
            dataframe['rmi-dn-trend'] = np.where(dataframe['rmi-up'].rolling(5).sum() <= 2, 1, 0)

            # dataframe['rmi-dn'] = np.where(dataframe['rmi'] <= dataframe['rmi'].shift(), 1, 0)
            # dataframe['rmi-dn-count'] = dataframe['rmi-dn'].rolling(8).sum()
            #
            # dataframe['rmi-up'] = np.where(dataframe['rmi'] > dataframe['rmi'].shift(), 1, 0)
            # dataframe['rmi-up-count'] = dataframe['rmi-up'].rolling(8).sum()

            dataframe['adx'] = ta.ADX(dataframe)
            dataframe['dm_plus'] = ta.PLUS_DM(dataframe)
            dataframe['dm_minus'] = ta.MINUS_DM(dataframe)
            dataframe['adx-up-trend'] = np.where(
                (
                        (dataframe['adx'] > 20.0) &
                        (dataframe['dm_plus'] > dataframe['dm_minus'])
                ), 1, 0)
            dataframe['adx-dn-trend'] = np.where(
                (
                        (dataframe['adx'] > 20.0) &
                        (dataframe['dm_plus'] < dataframe['dm_minus'])
                ), 1, 0)

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

    ############################################################################

    """
    Buy Signal
    """

    def populate_buy_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:

        short_conditions = []
        long_conditions = []
        dataframe.loc[:, 'enter_tag'] = ''

        # 'Bull'/long leveraged token
        if self.isBull(metadata['pair']):

            # volume check
            long_conditions.append(dataframe['volume'] > 0)

            # Trend
            long_conditions.append(dataframe['inf_candle-dn-trend'] == 1)

            # DWT triggers
            long_dwt_cond = (
                qtpylib.crossed_above(dataframe['dwt_model_diff'], self.entry_long_dwt_diff.value)
            )

            # DWTs will spike on big gains, so try to constrain
            long_spike_cond = (
                    dataframe['dwt_model_diff'] < 2.0 * self.entry_long_dwt_diff.value
            )

            long_conditions.append(long_dwt_cond)
            long_conditions.append(long_spike_cond)

            # set entry tags
            dataframe.loc[long_dwt_cond, 'enter_tag'] += 'long_dwt_entry '

            if long_conditions:
                dataframe.loc[reduce(lambda x, y: x & y, long_conditions), 'enter_long'] = 1

        # 'Bear'/short leveraged token
        elif self.isBear(metadata['pair']):

            # volume check
            short_conditions.append(dataframe['volume'] > 0)

            # Trend
            short_conditions.append(dataframe['inf_candle-up-trend'] == 1)

            # DWT triggers
            short_dwt_cond = (
                    qtpylib.crossed_below(dataframe['dwt_model_diff'], self.entry_short_dwt_diff.value)
            )


            # DWTs will spike on big gains, so try to constrain
            short_spike_cond = (
                    dataframe['dwt_model_diff'] > 2.0 * self.entry_short_dwt_diff.value
            )

            short_conditions.append(short_dwt_cond)
            short_conditions.append(short_spike_cond)

            # set entry tags
            dataframe.loc[short_dwt_cond, 'enter_tag'] += 'short_dwt_entry '

            if short_conditions:
                dataframe.loc[reduce(lambda x, y: x & y, short_conditions), 'enter_long'] = 1

        else:
            dataframe.loc[(dataframe['close'].notnull()), 'enter_long'] = 0

        return dataframe

    ############################################################################

    """
    Sell Signal
    """

    def populate_sell_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:

        short_conditions = []
        long_conditions = []
        dataframe.loc[:, 'exit_tag'] = ''

        # 'Bull'/long leveraged token
        if self.isBull(metadata['pair']):

            # DWT triggers
            long_dwt_cond = (
                    qtpylib.crossed_below(dataframe['dwt_model_diff'], self.exit_long_dwt_diff.value)
            )

            # DWTs will spike on big gains, so try to constrain
            long_spike_cond = (
                    dataframe['dwt_model_diff'] > 2.0 * self.exit_long_dwt_diff.value
            )

            long_conditions.append(long_dwt_cond)
            long_conditions.append(long_spike_cond)

            # set exit tags
            dataframe.loc[long_dwt_cond, 'exit_tag'] += 'long_dwt_exit '

            if long_conditions:
                dataframe.loc[reduce(lambda x, y: x & y, long_conditions), 'exit_long'] = 1

        # 'Bear'/short leveraged token
        elif self.isBear(metadata['pair']):

            # note that these aren't true 'short' pairs, they just leverage in the short direction.
            # In other words, the conditions are the same as or bull/long pairs, just with independent hyperparameters

            # DWT triggers
            short_dwt_cond = (
                qtpylib.crossed_above(dataframe['dwt_model_diff'], self.exit_short_dwt_diff.value)
            )


            # DWTs will spike on big gains, so try to constrain
            short_spike_cond = (
                    dataframe['dwt_model_diff'] < 2.0 * self.exit_short_dwt_diff.value
            )

            # conditions.append(trend_cond)
            short_conditions.append(short_dwt_cond)
            short_conditions.append(short_spike_cond)

            # set exit tags
            dataframe.loc[short_dwt_cond, 'exit_tag'] += 'short_dwt_exit '

            if short_conditions:
                dataframe.loc[reduce(lambda x, y: x & y, short_conditions), 'exit_long'] = 1

        else:
            dataframe.loc[(dataframe['close'].notnull()), 'exit_long'] = 0

        return dataframe


    ############################################################################

    def isBull(self, pair):
        return re.search(".*(BULL|UP|[235]L)", pair)

    def isBear(self, pair):
        return re.search(".*(BEAR|DOWN|[235]S)", pair)

    def getInformative(self, pair) -> str:
        inf_pair = ""
        if self.isBull(pair):
            inf_pair = re.sub('(BULL|UP|[235]L)', '', pair)
        elif self.isBear(pair):
            inf_pair = re.sub('(BEAR|DOWN|[235]S)', '', pair)

        # print(pair, " -> ", inf_pair)
        return inf_pair

    ############################################################################

    # the custom stoploss/exit logic is adapted from Solipsis by werkkrew (https://github.com/werkkrew/freqtrade-strategies)

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

    def custom_exit_long(self, pair: str, trade: 'Trade', current_time: 'datetime', current_rate: float,
                             current_profit: float, **kwargs):

        dataframe, _ = self.dp.get_analyzed_dataframe(pair=pair, timeframe=self.timeframe)
        last_candle = dataframe.iloc[-1].squeeze()

        trade_dur = int((current_time.timestamp() - trade.open_date_utc.timestamp()) // 60)
        max_profit = max(0.0, trade.calc_profit_ratio(trade.max_rate))
        pullback_value = max(0.0, (max_profit - self.cexit_long_pullback_amount.value))
        in_trend = False

        # Determine our current ROI point based on the defined type
        if self.cexit_long_roi_type.value == 'static':
            min_roi = self.cexit_long_roi_start.value
        elif self.cexit_long_roi_type.value == 'decay':
            min_roi = cta.linear_decay(self.cexit_long_roi_start.value, self.cexit_long_roi_end.value, 0,
                                       self.cexit_long_roi_time.value, trade_dur)
        elif self.cexit_long_roi_type.value == 'step':
            if trade_dur < self.cexit_long_roi_time.value:
                min_roi = self.cexit_long_roi_start.value
            else:
                min_roi = self.cexit_long_roi_end.value

        # Determine if there is a trend
        if self.cexit_long_trend_type.value == 'rmi' or self.cexit_long_trend_type.value == 'any':
            if last_candle['rmi-up-trend'] == 1:
                in_trend = True
        if self.cexit_long_trend_type.value == 'ssl' or self.cexit_long_trend_type.value == 'any':
            if last_candle['ssl-dir'] == 'up':
                in_trend = True
        if self.cexit_long_trend_type.value == 'candle' or self.cexit_long_trend_type.value == 'any':
            if last_candle['candle-up-trend'] == 1:
                in_trend = True

        # Don't exit if we are in a trend unless the pullback threshold is met
        if in_trend == True and current_profit > 0:
            # Record that we were in a trend for this trade/pair for a more useful exit message later
            self.custom_trade_info[trade.pair]['had-trend'] = True
            # If pullback is enabled and profit has pulled back allow a exit, maybe
            if self.cexit_long_pullback.value == True and (current_profit <= pullback_value):
                if self.cexit_long_pullback_respect_roi.value == True and current_profit > min_roi:
                    return 'long_intrend_pullback_roi'
                elif self.cexit_long_pullback_respect_roi.value == False:
                    if current_profit > min_roi:
                        return 'long_intrend_pullback_roi'
                    else:
                        return 'long_intrend_pullback_noroi'
            # We are in a trend and pullback is disabled or has not happened or various criteria were not met, hold
            return None
        # If we are not in a trend, just use the roi value
        elif in_trend == False:
            if self.custom_trade_info[trade.pair]['had-trend']:
                if current_profit > min_roi:
                    self.custom_trade_info[trade.pair]['had-trend'] = False
                    return 'long_trend_roi'
                elif self.cexit_long_endtrend_respect_roi.value == False:
                    self.custom_trade_info[trade.pair]['had-trend'] = False
                    return 'long_trend_noroi'
            elif current_profit > min_roi:
                return 'long_notrend_roi'
        else:
            return None

    def custom_exit_short(self, pair: str, trade: 'Trade', current_time: 'datetime', current_rate: float,
                    current_profit: float, **kwargs):

        dataframe, _ = self.dp.get_analyzed_dataframe(pair=pair, timeframe=self.timeframe)
        last_candle = dataframe.iloc[-1].squeeze()

        trade_dur = int((current_time.timestamp() - trade.open_date_utc.timestamp()) // 60)
        max_profit = max(0.0, trade.calc_profit_ratio(trade.max_rate))
        pullback_value = max(0.0, (max_profit - self.cexit_short_pullback_amount.value))
        in_trend = False

        # Determine our current ROI point based on the defined type
        if self.cexit_short_roi_type.value == 'static':
            min_roi = self.cexit_short_roi_start.value
        elif self.cexit_short_roi_type.value == 'decay':
            min_roi = cta.linear_decay(self.cexit_short_roi_start.value, self.cexit_short_roi_end.value, 0,
                                       self.cexit_short_roi_time.value, trade_dur)
        elif self.cexit_short_roi_type.value == 'step':
            if trade_dur < self.cexit_short_roi_time.value:
                min_roi = self.cexit_short_roi_start.value
            else:
                min_roi = self.cexit_short_roi_end.value

        # Determine if there is a trend
        if self.cexit_short_trend_type.value == 'rmi' or self.cexit_short_trend_type.value == 'any':
            if last_candle['rmi-up-trend'] == 1:
                in_trend = True
        if self.cexit_short_trend_type.value == 'ssl' or self.cexit_short_trend_type.value == 'any':
            if last_candle['ssl-dir'] == 'up':
                in_trend = True
        if self.cexit_short_trend_type.value == 'candle' or self.cexit_short_trend_type.value == 'any':
            if last_candle['candle-up-trend'] == 1:
                in_trend = True

        # Don't exit if we are in a trend unless the pullback threshold is met
        if in_trend == True and current_profit > 0:
            # Record that we were in a trend for this trade/pair for a more useful exit message later
            self.custom_trade_info[trade.pair]['had-trend'] = True
            # If pullback is enabled and profit has pulled back allow a exit, maybe
            if self.cexit_short_pullback.value == True and (current_profit <= pullback_value):
                if self.cexit_short_pullback_respect_roi.value == True and current_profit > min_roi:
                    return 'short_intrend_pullback_roi'
                elif self.cexit_short_pullback_respect_roi.value == False:
                    if current_profit > min_roi:
                        return 'short_intrend_pullback_roi'
                    else:
                        return 'short_intrend_pullback_noroi'
            # We are in a trend and pullback is disabled or has not happened or various criteria were not met, hold
            return None
        # If we are not in a trend, just use the roi value
        elif in_trend == False:
            if self.custom_trade_info[trade.pair]['had-trend']:
                if current_profit > min_roi:
                    self.custom_trade_info[trade.pair]['had-trend'] = False
                    return 'short_trend_roi'
                elif self.cexit_short_endtrend_respect_roi.value == False:
                    self.custom_trade_info[trade.pair]['had-trend'] = False
                    return 'short_trend_noroi'
            elif current_profit > min_roi:
                return 'short_notrend_roi'
        else:
            return None


    def custom_exit(self, pair: str, trade: 'Trade', current_time: 'datetime', current_rate: float,
                    current_profit: float, **kwargs):

        if self.isBull(pair):
            return self.custom_exit_long(pair, trade, current_time, current_rate, current_profit)
        elif self.isBear(pair):
            return self.custom_exit_short(pair, trade, current_time, current_rate, current_profit)


