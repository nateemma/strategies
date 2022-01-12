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

import custom_indicators as cta
import re

"""

This strategy is intended to work with leveraged pairs.
It uses the buy/sell signals from FisherBB2, and checks against BTC
For 'bull' pairs, it will  buy if BTC is in an uptrend (and other signals are met)
For 'bear' pairs, it will buy if BTC is in a down trend (and other signals are met)
"""


class FisherBBBTCLeveraged(IStrategy):

    # Buy hyperspace params:
    buy_params = {
        "buy_bb_gain": 0.01,
        "buy_fisher": 0.36,
        "buy_wr": -59.0,
    }

    # Sell hyperspace params:
    sell_params = {
        "sell_bb_gain": 0.99,
    }

    # ROI table:
    minimal_roi = {
        "0": 0.228,
        "17": 0.047,
        "61": 0.014,
        "155": 0
    }

    # Stoploss:
    stoploss = -0.10

    # Trailing stop:
    trailing_stop = True
    trailing_stop_positive = 0.168
    trailing_stop_positive_offset = 0.268
    trailing_only_offset_is_reached = True


    ## Buy Space Hyperopt Variables

    # FisherBB hyperparams
    buy_bb_gain = DecimalParameter(0.01, 0.10, decimals=2, default=0.09, space="buy", load=True, optimize=True)
    buy_fisher = DecimalParameter(-0.99, 0.99, decimals=2, default=0.99, space="buy", load=True, optimize=True)
    buy_wr = DecimalParameter(-99, 0, decimals=0, default=-80, space="buy", load=True, optimize=True)

    inf_pct_adr = DecimalParameter(0.70, 0.99, default=0.80, space='buy', load=True, optimize=True)

    ## Sell Space Hyperopt Variables

    sell_bb_gain = DecimalParameter(0.7, 1.3, decimals=2, default=0.8, space="sell", load=True, optimize=True)


    timeframe = '5m'
    inf_timeframe = '5m'

    use_custom_stoploss = False

    # Recommended
    use_sell_signal = True
    sell_profit_only = False
    ignore_roi_if_buy_signal = True

    # Required
    startup_candle_count: int = 24
    process_only_new_candles = False

    # Strategy Specific Variable Storage
    custom_trade_info = {}
    custom_fiat = "USDT"  # Only relevant if stake is BTC or ETH
    custom_btc_inf = True  # This is normally False. Try True for leveraged tokens

    """
    Informative Pair Definitions
    """

    def informative_pairs(self):
        btc_stake = f"BTC/{self.config['stake_currency']}"
        return [(btc_stake, self.timeframe)]

    """
    Indicator Definitions
    """

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        if not metadata['pair'] in self.custom_trade_info:
            self.custom_trade_info[metadata['pair']] = {}
            if not 'had-trend' in self.custom_trade_info[metadata["pair"]]:
                self.custom_trade_info[metadata['pair']]['had-trend'] = False

        ## Base Timeframe / Pair

        # # Kaufmann Adaptive Moving Average
        # dataframe['kama'] = ta.KAMA(dataframe, length=233)
        #
        # # RMI: https://www.tradingview.com/script/kwIt9OgQ-Relative-Momentum-Index/
        # dataframe['rmi'] = cta.RMI(dataframe, length=24, mom=5)
        #
        # # Momentum Pinball: https://www.tradingview.com/script/fBpVB1ez-Momentum-Pinball-Indicator/
        # dataframe['roc-mp'] = ta.ROC(dataframe, timeperiod=1)
        # dataframe['mp'] = ta.RSI(dataframe['roc-mp'], timeperiod=3)
        #
        # # MA Streak: https://www.tradingview.com/script/Yq1z7cIv-MA-Streak-Can-Show-When-a-Run-Is-Getting-Long-in-the-Tooth/
        # dataframe['mastreak'] = cta.mastreak(dataframe, period=4)
        #
        # # Percent Change Channel: https://www.tradingview.com/script/6wwAWXA1-MA-Streak-Change-Channel/
        # upper, mid, lower = cta.pcc(dataframe, period=40, mult=3)
        # dataframe['pcc-lowerband'] = lower
        # dataframe['pcc-upperband'] = upper
        #
        # lookup_idxs = dataframe.index.values - (abs(dataframe['mastreak'].values) + 1)
        # valid_lookups = lookup_idxs >= 0
        # dataframe['sbc'] = np.nan
        # dataframe.loc[valid_lookups, 'sbc'] = dataframe['close'].to_numpy()[lookup_idxs[valid_lookups].astype(int)]
        #
        # dataframe['streak-roc'] = 100 * (dataframe['close'] - dataframe['sbc']) / dataframe['sbc']
        #
        # # Trends, Peaks and Crosses
        # dataframe['candle-up'] = np.where(dataframe['close'] >= dataframe['open'], 1, 0)
        # dataframe['candle-up-trend'] = np.where(dataframe['candle-up'].rolling(5).sum() >= 3, 1, 0)
        #
        # dataframe['rmi-up'] = np.where(dataframe['rmi'] >= dataframe['rmi'].shift(), 1, 0)
        # dataframe['rmi-up-trend'] = np.where(dataframe['rmi-up'].rolling(5).sum() >= 3, 1, 0)
        #
        # dataframe['rmi-dn'] = np.where(dataframe['rmi'] <= dataframe['rmi'].shift(), 1, 0)
        # dataframe['rmi-dn-count'] = dataframe['rmi-dn'].rolling(8).sum()
        #
        # dataframe['streak-bo'] = np.where(dataframe['streak-roc'] < dataframe['pcc-lowerband'], 1, 0)
        # dataframe['streak-bo-count'] = dataframe['streak-bo'].rolling(8).sum()
        #
        # # Indicators used only for ROI and Custom Stoploss
        # ssldown, sslup = cta.SSLChannels_ATR(dataframe, length=21)
        # dataframe['sroc'] = cta.SROC(dataframe, roclen=21, emalen=13, smooth=21)
        # dataframe['ssl-dir'] = np.where(sslup > ssldown, 'up', 'down')

        # FisherBB indicators
        # RSI
        dataframe['rsi'] = ta.RSI(dataframe, timeperiod=14)

        # Inverse Fisher transform on RSI, values [-1.0, 1.0] (https://goo.gl/2JGGoy)
        rsi = 0.1 * (dataframe['rsi'] - 50)
        dataframe['fisher_rsi'] = (np.exp(2 * rsi) - 1) / (np.exp(2 * rsi) + 1)

        # Bollinger bands
        bollinger = qtpylib.bollinger_bands(qtpylib.typical_price(dataframe), window=20, stds=2)
        dataframe['bb_lowerband'] = bollinger['lower']
        dataframe['bb_upperband'] = bollinger['upper']
        dataframe["bb_gain"] = ((dataframe["bb_upperband"] - dataframe["close"]) / dataframe["close"])

        # Williams %R
        dataframe['wr'] = williams_r(dataframe, period=14)

        # Base pair informative timeframe indicators
        informative = self.dp.get_pair_dataframe(pair=metadata['pair'], timeframe=self.inf_timeframe)

        # Get the "average day range" between the 1d high and 1d low to set up guards
        informative['1d-high'] = informative['close'].rolling(24).max()
        informative['1d-low'] = informative['close'].rolling(24).min()
        informative['adr'] = informative['1d-high'] - informative['1d-low']

        dataframe = merge_informative_pair(dataframe, informative, self.timeframe, self.inf_timeframe, ffill=True)

        btc_stake = f"BTC/{self.config['stake_currency']}"
        # BTC/STAKE - Base Timeframe
        btc_stake_tf = self.dp.get_pair_dataframe(pair=btc_stake, timeframe=self.timeframe)
        dataframe['BTC_rmi'] = cta.RMI(btc_stake_tf, length=55, mom=5)
        dataframe['BTC_open'] = btc_stake_tf['open']
        dataframe['BTC_close'] = btc_stake_tf['close']
        # dataframe['BTC_kama'] = ta.KAMA(btc_stake_tf, length=10)

        dataframe['BTC_up'] = np.where(dataframe['BTC_close'] > dataframe['BTC_open'], 1, 0)
        dataframe['BTC_up_trend'] = np.where(dataframe['BTC_up'].rolling(5).sum() >= 3, 1, 0)
        dataframe['BTC_down'] = np.where(dataframe['BTC_close'] < dataframe['BTC_open'], 1, 0)
        dataframe['BTC_down_trend'] = np.where(dataframe['BTC_down'].rolling(5).sum() >= 3, 1, 0)


        return dataframe

    """
    Buy Signal
    """

    def populate_buy_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        conditions = []

        # volume check
        conditions.append(dataframe['volume'] > 0)

        # FisherBB triggers
        conditions.append(self.get_buy_signal_fisher_bb(dataframe))

        conditions.append(self.get_buy_signal_wr(dataframe))

        # Bull/Bear checks
        if self.isBull(metadata['pair']):
            # print(metadata['pair'], ": BULL")
            # check that KAMA is in an uptrend
            conditions.append(
                # (dataframe['BTC_kama'] > dataframe['BTC_kama'].shift(1))
                (dataframe['BTC_up_trend'] == 1)
            )

        elif self.isBear(metadata['pair']):
            # print(metadata['pair'], ": BEAR")
            # check that KAMA is in an uptrend
            conditions.append(
                # (dataframe['BTC_kama'] < dataframe['BTC_kama'].shift(1))
                (dataframe['BTC_down_trend'] == 1)
            )

        else:
            print(metadata['pair'], ": UNKNOWN")

        if conditions:
            dataframe.loc[
                reduce(lambda x, y: x & y, conditions),
                'buy'] = 1

        return dataframe


    """
    Sell Signal
    """

    def populate_sell_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:

        conditions = []

        # Volume check
        conditions.append(dataframe['volume'] > 0)

        # Bollinger band check
        conditions.append(qtpylib.crossed_above(dataframe['close'],
                                                (dataframe['bb_upperband'] * self.sell_bb_gain.value)))

        # Bull/Bear checks
        if self.isBull(metadata['pair']):
            # check that KAMA is in a downtrend
            # conditions.append(dataframe['BTC_kama'] < dataframe['BTC_kama'].shift(1))
            conditions.append(dataframe['BTC_down_trend'] == 1)

        elif self.isBear(metadata['pair']):
            # check that KAMA is in an uptrend
            # conditions.append(dataframe['BTC_kama'] > dataframe['BTC_kama'].shift(1))
            conditions.append(dataframe['BTC_up_trend'] == 1)

        if conditions:
            dataframe.loc[
                reduce(lambda x, y: x & y, conditions),
                'sell'] = 1

        return dataframe




    def get_buy_signal_fisher_bb(self, dataframe: DataFrame):
        signal = (
            (dataframe['fisher_rsi'] <= self.buy_fisher.value) &
            (dataframe['bb_gain'] >= self.buy_bb_gain.value)
        )
        return signal


    def get_buy_signal_wr(self, dataframe: DataFrame):
        signal = (
             (qtpylib.crossed_below(dataframe['wr'], self.buy_wr.value))
        )
        return signal

    def isBull(selfself, pair):
        return re.search(".*(BULL|UP|[235]L)", pair)

    def isBear(selfself, pair):
        return re.search(".*(BEAR|DOWN|[235]S)", pair)

# Williams %R
def williams_r(dataframe: DataFrame, period: int = 14) -> Series:
    """Williams %R, or just %R, is a technical analysis oscillator showing the current closing price in relation to the high and low
        of the past N days (for a given N). It was developed by a publisher and promoter of trading materials, Larry Williams.
        Its purpose is to tell whether a stock or commodity market is trading near the high or the low, or somewhere in between,
        of its recent trading range.
        The oscillator is on a negative scale, from −100 (lowest) up to 0 (highest).
    """

    highest_high = dataframe["high"].rolling(center=False, window=period).max()
    lowest_low = dataframe["low"].rolling(center=False, window=period).min()

    WR = Series(
        (highest_high - dataframe["close"]) / (highest_high - lowest_low),
        name=f"{period} Williams %R",
        )

    return WR * -100