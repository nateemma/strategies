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
    buy_bull_params = {
        "buy_bull_bb_gain": 0.03,
        "buy_bull_fisher": 0.68,
        "buy_bull_wr": 0.0,
        "buy_bear_bb_gain": 0.03,
        "buy_bear_fisher": 0.68,
        "buy_bear_wr": 0.0,
    }

    # Sell hyperspace params:
    sell_bull_params = {
        "sell_bull_bb_gain": 0.97,
        "sell_bear_bb_gain": 0.97,
    }

    # ROI table:
    minimal_roi = {
        "0": 0.068,
        "7": 0.029,
        "18": 0.011,
        "32": 0
    }

    # Stoploss:
    stoploss = -0.1

    # Trailing stop:
    trailing_stop = True
    trailing_stop_positive = 0.013
    trailing_stop_positive_offset = 0.072
    trailing_only_offset_is_reached = True

    ## Buy Space Hyperopt Variables

    # FisherBB hyperparams
    buy_trend_method = CategoricalParameter(['price', 'ssl', 'kama', 'supertrend', 'sqzmom'],
                                            default='ssl', space='buy', load=True, optimize=True)

    buy_bull_bb_gain = DecimalParameter(0.01, 0.10, decimals=2, default=0.09, space="buy", load=True, optimize=True)
    buy_bull_fisher = DecimalParameter(-0.99, 0.99, decimals=2, default=0.99, space="buy", load=True, optimize=True)
    buy_bull_wr = DecimalParameter(-99, 0, decimals=0, default=-80, space="buy", load=True, optimize=True)
    buy_bear_bb_gain = DecimalParameter(0.01, 0.10, decimals=2, default=0.09, space="buy", load=True, optimize=True)
    buy_bear_fisher = DecimalParameter(-0.99, 0.99, decimals=2, default=0.99, space="buy", load=True, optimize=True)
    buy_bear_wr = DecimalParameter(-99, 0, decimals=0, default=-80, space="buy", load=True, optimize=True)

    ## Sell Space Hyperopt Variables

    sell_bull_bb_gain = DecimalParameter(0.7, 1.3, decimals=2, default=0.8, space="sell", load=True, optimize=True)
    sell_bear_bb_gain = DecimalParameter(0.7, 1.3, decimals=2, default=0.8, space="sell", load=True, optimize=True)

    timeframe = '5m'
    inf_timeframe = '5m'

    use_custom_stoploss = False

    # Recommended
    use_sell_signal = True
    sell_profit_only = True
    ignore_roi_if_buy_signal = True

    # Required
    startup_candle_count: int = 50
    process_only_new_candles = False

    # Strategy Specific Variable Storage
    custom_trade_info = {}
    custom_fiat = "USDT"  # Only relevant if stake is BTC or ETH

    """
    Informative Pair Definitions
    """

    def informative_pairs(self):
        # just using BTC here
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

        # Kaufmann Adaptive Moving Average
        dataframe['kama'] = ta.KAMA(dataframe, length=233)

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
        btc_dataframe = self.dp.get_pair_dataframe(pair=btc_stake, timeframe=self.timeframe)
        dataframe['BTC_rmi'] = cta.RMI(btc_dataframe, length=55, mom=5)
        dataframe['BTC_open'] = btc_dataframe['open']
        dataframe['BTC_close'] = btc_dataframe['close']
        dataframe['BTC_high'] = btc_dataframe['high']
        dataframe['BTC_low'] = btc_dataframe['low']

        # dataframe['BTC_kama'] = ta.KAMA(btc_dataframe, length=144)
        dataframe['BTC_kama'] = ta.KAMA(btc_dataframe, length=10)

        # Price Trend
        dataframe['BTC_up'] = np.where(dataframe['BTC_close'] > dataframe['BTC_open'], 1, 0)
        dataframe['BTC_up_trend'] = np.where(dataframe['BTC_up'].rolling(5).sum() >= 3, 1, 0)
        dataframe['BTC_down'] = np.where(dataframe['BTC_close'] < dataframe['BTC_open'], 1, 0)
        dataframe['BTC_down_trend'] = np.where(dataframe['BTC_down'].rolling(5).sum() >= 3, 1, 0)

        # SSL Trend
        ssldown, sslup = cta.SSLChannels_ATR(btc_dataframe, length=21)
        dataframe['BTC_ssl_down'] = ssldown
        dataframe['BTC_ssl_up'] = sslup
        dataframe['BTC_ssl_diff'] = sslup - ssldown
        dataframe['BTC_ssl_dir'] = np.where(dataframe['BTC_ssl_diff'] > 0.0, 'up', 'down')

        # KAMA Trend
        dataframe['BTC_kama_slow'] = filtered_kama(btc_dataframe, 10, 5, 30)
        # dataframe['BTC_kama_fast'] = filtered_kama(btc_dataframe, 10, 2, 30)
        dataframe['BTC_kama_up'] = np.where(dataframe['BTC_kama_slow'] > dataframe['BTC_kama_slow'].shift(1), 1, 0)
        dataframe['BTC_kama_up_trend'] = np.where(dataframe['BTC_kama_up'].rolling(5).sum() >= 3, 1, 0)
        dataframe['BTC_kama_down'] = np.where(dataframe['BTC_kama_slow'] < dataframe['BTC_kama_slow'].shift(1), 1, 0)
        dataframe['BTC_kama_down_trend'] = np.where(dataframe['BTC_kama_down'].rolling(5).sum() >= 3, 1, 0)

        # Squeeze Momentum
        # Donchian Channels
        dataframe['BTC_dc_upper'] = ta.MAX(dataframe['BTC_high'], timeperiod=21)
        dataframe['BTC_dc_lower'] = ta.MIN(dataframe['BTC_low'], timeperiod=21)
        dataframe['BTC_dc_mid'] = ta.TEMA(((dataframe['BTC_dc_upper'] + dataframe['BTC_dc_lower']) / 2), timeperiod=21)
        dataframe['BTC_tema'] = ta.TEMA(dataframe['BTC_close'], timeperiod=21)

        dataframe['BTC_sqz_ave'] = ta.TEMA(((dataframe['BTC_dc_mid'] + dataframe['BTC_tema']) / 2), timeperiod=21)
        dataframe['BTC_sqz_delta'] = ta.TEMA((dataframe['BTC_close'] - dataframe['BTC_sqz_ave']), timeperiod=21)
        dataframe['BTC_sqz_val'] = ta.LINEARREG(dataframe['BTC_sqz_delta'], timeperiod=21)
        
        # Supertrend
        # computationally intensive, so only calculate if needed
        if self.dp.runmode.value in ('hyperopt'):
            # only done once in hyperopt, so can't use value of hyperopt param here
            st = SuperTrend(btc_dataframe)
            dataframe['BTC_st'] = st['ST']
            dataframe['BTC_st_trend'] = st['STX']  # 'up', 'down'
        else:
            if self.buy_trend_method.value == 'supertrend':
                st = SuperTrend(btc_dataframe)
                dataframe['BTC_st'] = st['ST']
                dataframe['BTC_st_trend'] = st['STX']  # 'up', 'down'
            else:
                dataframe['BTC_st'] = 0.0
                dataframe['BTC_st_trend'] = ''

        return dataframe

    """
    Buy Signal
    """

    def populate_buy_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        conditions = []

        # volume check
        conditions.append(dataframe['volume'] > 0)

        # Trend checks
        if self.isBull(metadata['pair']):
            # print(metadata['pair'], ": BULL")
            # check that BTC is in an uptrend
            # conditions.append(self.inUptrend(dataframe))
            conditions.append(self.newUptrend(dataframe))

        elif self.isBear(metadata['pair']):
            # conditions.append(self.inDowntrend(dataframe))
            conditions.append(self.newDowntrend(dataframe))

            # FisherBB Triggers
        if self.isBear(metadata['pair']):
            conditions.append(
                (dataframe['fisher_rsi'] <= self.buy_bear_fisher.value) &
                (dataframe['bb_gain'] >= self.buy_bear_bb_gain.value)
            )
            conditions.append(
                (qtpylib.crossed_below(dataframe['wr'], self.buy_bear_wr.value))
            )

        else:
            # bull or neither
            conditions.append(
                (dataframe['fisher_rsi'] <= self.buy_bull_fisher.value) &
                (dataframe['bb_gain'] >= self.buy_bull_bb_gain.value)
            )
            conditions.append(
                (qtpylib.crossed_below(dataframe['wr'], self.buy_bull_wr.value))
            )

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

        # Trend checks
        if self.isBull(metadata['pair']):
            # check that BTC is in a downtrend
            # conditions.append(self.inDowntrend(dataframe))
            conditions.append(self.newDowntrend(dataframe))


        elif self.isBear(metadata['pair']):
            # check that BTC is in an uptrend
            # conditions.append(self.inUptrend(dataframe))
            conditions.append(self.newUptrend(dataframe))

        # Bollinger band check
        if self.isBear(metadata['pair']):
            conditions.append(qtpylib.crossed_above(dataframe['close'],
                                                    (dataframe['bb_upperband'] * self.sell_bear_bb_gain.value)))
        else:
            # bull or neither
            conditions.append(qtpylib.crossed_above(dataframe['close'],
                                                    (dataframe['bb_upperband'] * self.sell_bull_bb_gain.value)))

        if conditions:
            dataframe.loc[
                reduce(lambda x, y: x & y, conditions),
                'sell'] = 1

        return dataframe

    def isBull(self, pair):
        return re.search(".*(BULL|UP|[235]L)", pair)

    def isBear(self, pair):
        return re.search(".*(BEAR|DOWN|[235]S)", pair)

    def inUptrend(self, dataframe: DataFrame):
        if self.buy_trend_method.value == 'price':
            result = (dataframe['BTC_up_trend'] == 1)
        elif self.buy_trend_method.value == 'ssl':
            result = (dataframe['BTC_ssl_dir'] == 'up')
        elif self.buy_trend_method.value == 'kama':
            result = (dataframe['BTC_kama_up_trend'] == 1)
        elif self.buy_trend_method.value == 'supertrend':
            result = (dataframe['BTC_st_trend'] == 'up')
        elif self.buy_trend_method.value == 'sqzmom':
            result = (dataframe['BTC_sqz_val'] > 0)
        else:
            print("ERR: buy_trend_method is: ", self.buy_trend_method.value)
        return result

    def inDowntrend(self, dataframe: DataFrame):
        if self.buy_trend_method.value == 'price':
            result = (dataframe['BTC_down_trend'] == 1)
        elif self.buy_trend_method.value == 'ssl':
            result = (dataframe['BTC_ssl_dir'] == 'down')
        elif self.buy_trend_method.value == 'kama':
            result = (dataframe['BTC_kama_down_trend'] == 1)
        elif self.buy_trend_method.value == 'supertrend':
            result = (dataframe['BTC_st_trend'] == 'down')
        elif self.buy_trend_method.value == 'sqzmom':
            result = (dataframe['BTC_sqz_val'] < 0)
        else:
            print("ERR: buy_trend_method is: ", self.buy_trend_method.value)
        return result

    def newUptrend(self, dataframe: DataFrame):
        if self.buy_trend_method.value == 'price':
            result = ((dataframe['BTC_up_trend'] == 1) & (dataframe['BTC_up_trend'].shift(1) != 1))
        elif self.buy_trend_method.value == 'ssl':
            result = (qtpylib.crossed_above(dataframe['BTC_ssl_up'], dataframe['BTC_ssl_down']))
        elif self.buy_trend_method.value == 'kama':
            result = ((dataframe['BTC_kama_up_trend'] == 1) & (dataframe['BTC_kama_up_trend'].shift(1) != 1))
        elif self.buy_trend_method.value == 'supertrend':
            result = ((dataframe['BTC_st_trend'] == 'up') & (dataframe['BTC_st_trend'].shift(1) != 'up'))
        elif self.buy_trend_method.value == 'sqzmom':
            result = (qtpylib.crossed_above(dataframe['BTC_sqz_val'], 0))
        return result

    def newDowntrend(self, dataframe: DataFrame):
        if self.buy_trend_method.value == 'price':
            result = ((dataframe['BTC_down_trend'] == 1) & (dataframe['BTC_down_trend'].shift(1) != 1))
        elif self.buy_trend_method.value == 'ssl':
            result = (qtpylib.crossed_below(dataframe['BTC_ssl_up'], dataframe['BTC_ssl_down']))
        elif self.buy_trend_method.value == 'kama':
            result = ((dataframe['BTC_kama_down_trend'] == 1) & (dataframe['BTC_kama_down_trend'].shift(1) != 1))
        elif self.buy_trend_method.value == 'supertrend':
            result = ((dataframe['BTC_st_trend'] == 'down') & (dataframe['BTC_st_trend'].shift(1) != 'down'))
        elif self.buy_trend_method.value == 'sqzmom':
            result = (qtpylib.crossed_below(dataframe['BTC_sqz_val'], 0))
        return result


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


# Filtered Kaufmann Moving Average
def filtered_kama(dataframe: DataFrame, er_period: int, fast_period: int, slow_period: int) -> Series:
    """
    The KAMA function provided by the Technical Analysis library does not seem to work well, so this is
    an alternate version.
    see: https://school.stockcharts.com/doku.php?id=technical_indicators:kaufman_s_adaptive_moving_average

    er_period: Efficiency Ratio period (typically 10)
    fast_period: fast moving average period (typically 2 or 5)
    slow_period: slow average period (typically 30)
    """

    change = abs(dataframe['close'] - dataframe['close'].shift(er_period))
    cdiff = abs(dataframe['close'] - dataframe['close'].shift(1))
    volatility = cdiff.rolling(er_period).sum()
    er = change / volatility

    sc_fast = fast_period / (fast_period + 1)
    sc_slow = fast_period / (slow_period + 1)
    sc = np.square(er * (sc_fast - sc_slow) + sc_slow)

    kama = Series(ta.SMA(dataframe['close'], timeperiod=slow_period))
    kama = kama.shift(1) + (sc * (dataframe['close'] - kama.shift(1)))

    return kama


# Supertrend indicator
def SuperTrend(dataframe, period=10, multiplier=3):
    """
    Supertrend Indicator
    adapted for freqtrade. Matches TradingView implementation(s)
    from: https://github.com/freqtrade/freqtrade-strategies/issues/30
    """
    df = dataframe.copy()

    df['TR'] = ta.TRANGE(df)
    df['ATR'] = ta.SMA(df['TR'], period)

    st = 'ST_' + str(period) + '_' + str(multiplier)
    stx = 'STX_' + str(period) + '_' + str(multiplier)

    # Compute basic upper and lower bands
    df['basic_ub'] = (df['high'] + df['low']) / 2 + multiplier * df['ATR']
    df['basic_lb'] = (df['high'] + df['low']) / 2 - multiplier * df['ATR']

    # Compute final upper and lower bands
    df['final_ub'] = 0.00
    df['final_lb'] = 0.00
    for i in range(period, len(df)):
        df['final_ub'].iat[i] = df['basic_ub'].iat[i] if df['basic_ub'].iat[i] < df['final_ub'].iat[i - 1] or \
                                                         df['close'].iat[i - 1] > df['final_ub'].iat[i - 1] else \
            df['final_ub'].iat[i - 1]
        df['final_lb'].iat[i] = df['basic_lb'].iat[i] if df['basic_lb'].iat[i] > df['final_lb'].iat[i - 1] or \
                                                         df['close'].iat[i - 1] < df['final_lb'].iat[i - 1] else \
            df['final_lb'].iat[i - 1]

    # Set the Supertrend value
    df[st] = 0.00
    for i in range(period, len(df)):
        df[st].iat[i] = df['final_ub'].iat[i] if df[st].iat[i - 1] == df['final_ub'].iat[i - 1] and df['close'].iat[
            i] <= df['final_ub'].iat[i] else \
            df['final_lb'].iat[i] if df[st].iat[i - 1] == df['final_ub'].iat[i - 1] and df['close'].iat[i] > \
                                     df['final_ub'].iat[i] else \
                df['final_lb'].iat[i] if df[st].iat[i - 1] == df['final_lb'].iat[i - 1] and df['close'].iat[i] >= \
                                         df['final_lb'].iat[i] else \
                    df['final_ub'].iat[i] if df[st].iat[i - 1] == df['final_lb'].iat[i - 1] and df['close'].iat[i] < \
                                             df['final_lb'].iat[i] else 0.00

    # Mark the trend direction up/down
    df[stx] = np.where((df[st] > 0.00), np.where((df['close'] < df[st]), 'down', 'up'), np.NaN)

    # Remove basic and final bands from the columns
    df.drop(['basic_ub', 'basic_lb', 'final_ub', 'final_lb'], inplace=True, axis=1)

    df.fillna(0, inplace=True)

    # df.to_csv(f"user_data/Supertrend_{period}_{multiplier}.csv")
    return DataFrame(index=df.index, data={
        'ST': df[st],
        'STX': df[stx]
    })
