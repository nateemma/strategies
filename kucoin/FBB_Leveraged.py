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
It uses the buy/sell signals from FBB_2, and checks against BTC
For 'bull' pairs, it will  buy if BTC is in an uptrend (and other signals are met)
For 'bear' pairs, it will buy if BTC is in a down trend (and other signals are met)
"""


class FBB_Leveraged(IStrategy):
    # Buy hyperspace params:
    buy_params = {
        "buy_bear_bb_gain": 0.09,
        "buy_bear_fisher": -0.17,
        "buy_bear_wr": -52.0,
        "buy_bull_bb_gain": 0.08,
        "buy_bull_fisher": -0.07,
        "buy_bull_wr": -48.0
    }

    # Sell hyperspace params:
    sell_params = {
    }

    # ROI table:
    minimal_roi = {
        "0": 0.195,
        "30": 0.025,
        "70": 0.012,
        "158": 0
    }

    # Stoploss:
    stoploss = -0.1

    # Trailing stop:
    trailing_stop = True
    trailing_stop_positive = 0.013
    trailing_stop_positive_offset = 0.072
    trailing_only_offset_is_reached = True

    ## Buy Space Hyperopt Variables

    # FBB_ hyperparams

    buy_bull_bb_gain = DecimalParameter(0.01, 0.10, decimals=2, default=0.09, space="buy", load=True, optimize=True)
    buy_bull_fisher = DecimalParameter(-0.99, 0.99, decimals=2, default=0.99, space="buy", load=True, optimize=True)
    buy_bull_wr = DecimalParameter(-99, 0, decimals=0, default=-80, space="buy", load=True, optimize=True)

    buy_bear_bb_gain = DecimalParameter(0.01, 0.10, decimals=2, default=0.09, space="buy", load=True, optimize=True)
    buy_bear_fisher = DecimalParameter(-0.99, 0.99, decimals=2, default=0.99, space="buy", load=True, optimize=True)
    buy_bear_wr = DecimalParameter(-99, 0, decimals=0, default=-80, space="buy", load=True, optimize=True)

    ## Sell Space Hyperopt Variables

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

    """
    Informative Pair Definitions
    """

    def informative_pairs(self):
        return []

    """
    Indicator Definitions
    """

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:

        ## Base Timeframe / Pair

        # # Kaufmann Adaptive Moving Average
        # dataframe['kama'] = ta.KAMA(dataframe, length=233)

        # FBB_ indicators
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

        return dataframe

    """
    Buy Signal
    """

    def populate_buy_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        conditions = []

        # volume check
        conditions.append(dataframe['volume'] > 0)

        # FBB_ Triggers
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

        # no sell signals, just iuse ROI
        dataframe.loc[(dataframe['close'].notnull()), 'sell'] = 0

        return dataframe

    def isBull(self, pair):
        return re.search(".*(BULL|UP|[235]L)", pair)

    def isBear(self, pair):
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
