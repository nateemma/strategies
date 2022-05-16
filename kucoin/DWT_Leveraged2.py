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


class DWT_Leveraged2(IStrategy):

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

    entry_long_dwt_diff = DecimalParameter(0.0, 5.0, decimals=1, default=2.0, space='buy', load=True, optimize=True)
    entry_short_dwt_diff = DecimalParameter(-5.0, 0.0, decimals=1, default=-2.0, space='buy', load=True, optimize=True)
    # entry_trend_type = CategoricalParameter(['rmi', 'ssl', 'candle', 'macd', 'adx', 'none'], default='none', space='buy',
    #                                         load=True, optimize=True)

    ## Sell Space Hyperopt Variables

    exit_long_dwt_diff = DecimalParameter(-5.0, 0.0, decimals=1, default=-2.0, space='sell', load=True, optimize=True)
    exit_short_dwt_diff = DecimalParameter(0.0, 5.0, decimals=1, default=2.0, space='sell', load=True, optimize=True)

    # Custom Stoploss

    # hard stoploss profit
    pHSL = DecimalParameter(-0.200, -0.040, default=-0.08, decimals=3, space='sell', load=True)
    # profit threshold 1, trigger point, SL_1 is used
    pPF_1 = DecimalParameter(0.008, 0.020, default=0.016, decimals=3, space='sell', load=True)
    pSL_1 = DecimalParameter(0.008, 0.020, default=0.011, decimals=3, space='sell', load=True)

    # profit threshold 2, SL_2 is used
    pPF_2 = DecimalParameter(0.040, 0.100, default=0.080, decimals=3, space='sell', load=True)
    pSL_2 = DecimalParameter(0.020, 0.070, default=0.040, decimals=3, space='sell', load=True)


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
        for pair in pairs:
            inf_pair = self.getInformative(pair)
            # informative_pairs += [(pair, self.inf_timeframe)]
            if (inf_pair != ""):
                informative_pairs += [(inf_pair, self.inf_timeframe)]

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

            # merge into normal timeframe
            dataframe = merge_informative_pair(dataframe, inf_slow, self.timeframe, self.inf_timeframe, ffill=True)
            dataframe = merge_informative_pair(dataframe, inf_fast, self.timeframe, self.timeframe, ffill=True)

            # calculate predictive indicators in shorter timeframe (not informative)

            dataframe['dwt_model'] = dataframe[f"dwt_model_{self.inf_timeframe}"]
            dataframe['inf_close'] = dataframe[f"close_{self.timeframe}"]
            dataframe['inf_open'] = dataframe[f"open_{self.timeframe}"]
            # dataframe['dwt_model_diff'] = 100.0 * (dataframe['dwt_model'] - dataframe['close']) / dataframe['close']
            dataframe['dwt_model_diff'] = 100.0 * (dataframe['dwt_model'] - dataframe['inf_close']) / dataframe['inf_close']

            # Trends

            # Note that trends are based on the informative data, not the leveraged pair
            dataframe['candle-up'] = np.where(dataframe['inf_close'] >= dataframe['inf_open'], 1, 0)
            dataframe['candle-up-trend'] = np.where(dataframe['candle-up'].rolling(5).sum() >= 3, 1, 0)
            dataframe['candle-dn-trend'] = np.where(dataframe['candle-up'].rolling(5).sum() <= 2, 1, 0)

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
            trend_cond = (dataframe['candle-dn-trend'] == 1)
            long_conditions.append(trend_cond)

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
            trend_cond = (dataframe['candle-up-trend'] == 1)
            long_conditions.append(trend_cond)


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

    ## Custom Trailing stoploss ( credit to Perkmeister for this custom stoploss to help the strategy ride a green candle )
    def custom_stoploss(self, pair: str, trade: 'Trade', current_time: datetime,
                        current_rate: float, current_profit: float, **kwargs) -> float:

        # hard stoploss profit
        HSL = self.pHSL.value
        PF_1 = self.pPF_1.value
        SL_1 = self.pSL_1.value
        PF_2 = self.pPF_2.value
        SL_2 = self.pSL_2.value

        # For profits between PF_1 and PF_2 the stoploss (sl_profit) used is linearly interpolated
        # between the values of SL_1 and SL_2. For all profits above PL_2 the sl_profit value
        # rises linearly with current profit, for profits below PF_1 the hard stoploss profit is used.

        if (current_profit > PF_2):
            sl_profit = SL_2 + (current_profit - PF_2)
        elif (current_profit > PF_1):
            sl_profit = SL_1 + ((current_profit - PF_1) * (SL_2 - SL_1) / (PF_2 - PF_1))
        else:
            sl_profit = HSL

        # Only for hyperopt invalid return
        if (sl_profit >= current_profit):
            return -0.99

        return min(-0.01, max(stoploss_from_open(sl_profit, current_profit), -0.99))


