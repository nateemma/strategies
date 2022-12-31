
# --- Do not remove these libs ---
from freqtrade.strategy.interface import IStrategy
from typing import Dict, List
from functools import reduce
from pandas import DataFrame
# --------------------------------

import talib.abstract as ta
import freqtrade.vendor.qtpylib.indicators as qtpylib
import numpy # noqa
from freqtrade.strategy.hyper import CategoricalParameter, DecimalParameter, IntParameter

from user_data.strategies import Config


class Hammer(IStrategy):
    """
    Trades based on detection of Hammer-like candlestick patterns

    How to use it?
    > python3 ./freqtrade/main.py -s Hammer
    """

    # Buy hyperspace params:
    buy_params = {
        "buy_bb_enabled": True,
        "buy_bb_gain": 0.08,
        "buy_ema_enabled": True,
        "buy_mfi": 47.0,
        "buy_mfi_enabled": False,
        "buy_sma_enabled": True,
    }

    pattern_strength = 90
    buy_mfi = DecimalParameter(0, 50, decimals=0, default=47, space="buy")
    buy_bb_gain = DecimalParameter(0.01, 0.10, decimals=2, default=0.08, space="buy")

    buy_bb_enabled = CategoricalParameter([True, False], default=True, space="buy")
    buy_sma_enabled = CategoricalParameter([True, False], default=True, space="buy")
    buy_ema_enabled = CategoricalParameter([True, False], default=True, space="buy")
    buy_mfi_enabled = CategoricalParameter([True, False], default=False, space="buy")

    sell_hold_enabled = CategoricalParameter([True, False], default=True, space="sell")

    # set the startup candles count to the longest average used (SMA, EMA etc)
    startup_candle_count = 20

    # set common parameters
    minimal_roi = Config.minimal_roi
    trailing_stop = Config.trailing_stop
    trailing_stop_positive = Config.trailing_stop_positive
    trailing_stop_positive_offset = Config.trailing_stop_positive_offset
    trailing_only_offset_is_reached = Config.trailing_only_offset_is_reached
    stoploss = Config.stoploss
    timeframe = Config.timeframe
    process_only_new_candles = Config.process_only_new_candles
    use_sell_signal = Config.use_sell_signal
    sell_profit_only = Config.sell_profit_only
    ignore_roi_if_buy_signal = Config.ignore_roi_if_buy_signal
    order_types = Config.order_types

    def informative_pairs(self):
        """
        Define additional, informative pair/interval combinations to be cached from the exchange.
        These pair/interval combinations are non-tradeable, unless they are part
        of the whitelist as well.
        For more information, please consult the documentation
        :return: List of tuples in the format (pair, interval)
            Sample: return [("ETH/USDT", "5m"),
                            ("BTC/USDT", "15m"),
                            ]
        """
        return []

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Adds several different TA indicators to the given DataFrame

        Performance Note: For the best performance be frugal on the number of indicators
        you are using. Let uncomment only the indicator you are using in your strategies
        or your hyperopt configuration, otherwise you will waste your memory and CPU usage.
        """

        # MFI
        dataframe['mfi'] = ta.MFI(dataframe)

        # Stoch fast
        stoch_fast = ta.STOCHF(dataframe)
        dataframe['fastd'] = stoch_fast['fastd']
        dataframe['fastk'] = stoch_fast['fastk']

        # RSI
        dataframe['rsi'] = ta.RSI(dataframe)

        # Inverse Fisher transform on RSI, values [-1.0, 1.0] (https://goo.gl/2JGGoy)
        rsi = 0.1 * (dataframe['rsi'] - 50)
        dataframe['fisher_rsi'] = (numpy.exp(2 * rsi) - 1) / (numpy.exp(2 * rsi) + 1)

        # Bollinger bands
        # bollinger = qtpylib.bollinger_bands(qtpylib.typical_price(dataframe), window=20, stds=2)
        bollinger = qtpylib.weighted_bollinger_bands(qtpylib.typical_price(dataframe), window=20, stds=2)
        dataframe['bb_upperband'] = bollinger['upper']
        dataframe['bb_lowerband'] = bollinger['lower']
        dataframe["bb_gain"] = ((dataframe["bb_upperband"] - dataframe["close"]) / dataframe["close"])

        # EMA - Exponential Moving Average
        dataframe['ema5'] = ta.EMA(dataframe, timeperiod=5)
        dataframe['ema10'] = ta.EMA(dataframe, timeperiod=10)
        # dataframe['ema50'] = ta.EMA(dataframe, timeperiod=50)
        # dataframe['ema100'] = ta.EMA(dataframe, timeperiod=100)

        # SAR Parabol
        dataframe['sar'] = ta.SAR(dataframe)

        # SMA - Simple Moving Average
        dataframe['sma'] = ta.SMA(dataframe, timeperiod=40)

        # Pattern Recognition - Bullish candlestick patterns
        # ------------------------------------
        # Hammer: values [0, 100]
        dataframe['CDLHAMMER'] = ta.CDLHAMMER(dataframe)
        # Inverted Hammer: values [0, 100]
        dataframe['CDLINVERTEDHAMMER'] = ta.CDLINVERTEDHAMMER(dataframe)
        # # Dragonfly Doji: values [0, 100]
        # dataframe['CDLDRAGONFLYDOJI'] = ta.CDLDRAGONFLYDOJI(dataframe)
        # # Piercing Line: values [0, 100]
        # dataframe['CDLPIERCING'] = ta.CDLPIERCING(dataframe) # values [0, 100]
        # # Morningstar: values [0, 100]
        # dataframe['CDLMORNINGSTAR'] = ta.CDLMORNINGSTAR(dataframe) # values [0, 100]
        # # Three White Soldiers: values [0, 100]
        # dataframe['CDL3WHITESOLDIERS'] = ta.CDL3WHITESOLDIERS(dataframe) # values [0, 100]

        # Pattern Recognition - Bearish candlestick patterns
        # ------------------------------------
        # Hanging Man: values [0, 100]
        dataframe['CDLHANGINGMAN'] = ta.CDLHANGINGMAN(dataframe)
        # Shooting Star: values [0, 100]
        dataframe['CDLSHOOTINGSTAR'] = ta.CDLSHOOTINGSTAR(dataframe)
        # Gravestone Doji: values [0, 100]
        # dataframe['CDLGRAVESTONEDOJI'] = ta.CDLGRAVESTONEDOJI(dataframe)
        # # Dark Cloud Cover: values [0, 100]
        # dataframe['CDLDARKCLOUDCOVER'] = ta.CDLDARKCLOUDCOVER(dataframe)
        # # Evening Doji Star: values [0, 100]
        # dataframe['CDLEVENINGDOJISTAR'] = ta.CDLEVENINGDOJISTAR(dataframe)
        # # Evening Star: values [0, 100]
        # dataframe['CDLEVENINGSTAR'] = ta.CDLEVENINGSTAR(dataframe)
        #
        # # Pattern Recognition - Bullish/Bearish candlestick patterns
        # # ------------------------------------
        # # Three Line Strike: values [0, -100, 100]
        # dataframe['CDL3LINESTRIKE'] = ta.CDL3LINESTRIKE(dataframe)
        # # Spinning Top: values [0, -100, 100]
        # dataframe['CDLSPINNINGTOP'] = ta.CDLSPINNINGTOP(dataframe) # values [0, -100, 100]
        # # Engulfing: values [0, -100, 100]
        # dataframe['CDLENGULFING'] = ta.CDLENGULFING(dataframe) # values [0, -100, 100]
        # # Harami: values [0, -100, 100]
        # dataframe['CDLHARAMI'] = ta.CDLHARAMI(dataframe) # values [0, -100, 100]
        # # Three Outside Up/Down: values [0, -100, 100]
        # dataframe['CDL3OUTSIDE'] = ta.CDL3OUTSIDE(dataframe) # values [0, -100, 100]
        # # Three Inside Up/Down: values [0, -100, 100]
        # dataframe['CDL3INSIDE'] = ta.CDL3INSIDE(dataframe) # values [0, -100, 100]

        # Built-in candlestick patterns are not very good, so let's look for hammer-like patterns
        # We don't care which ones, just that the shadow (up or down) is 2x the body
        dataframe['height'] = abs(dataframe['close']-dataframe['open'])
        dataframe['body'] = dataframe['height'].clip(lower=0.01)
        dataframe['top'] = dataframe[['close','open']].max(axis=1)
        dataframe['bottom'] = dataframe[['close','open']].min(axis=1)
        dataframe['upper_shadow'] = dataframe['high']-dataframe['top']
        dataframe['lower_shadow'] = dataframe['bottom']-dataframe['low']
        dataframe['upper_ratio'] = (dataframe['high']-dataframe['top'])/dataframe['body']
        dataframe['upper_ratio'] = dataframe['upper_ratio'].clip(upper=10)
        dataframe['lower_ratio'] = (dataframe['bottom']-dataframe['low'])/dataframe['body']
        dataframe['lower_ratio'] = dataframe['lower_ratio'].clip(upper=10)
        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Based on TA indicators, populates the buy signal for the given dataframe
        :param dataframe: DataFrame
        :return: DataFrame with buy column
        """

        conditions = []

        # GUARDS AND TRENDS

        conditions.append(dataframe['volume'] > 0)

        if self.buy_sma_enabled.value:
            conditions.append(dataframe['close'] < dataframe['sma'])

        if self.buy_mfi_enabled.value:
            conditions.append(dataframe['mfi'] <= self.buy_mfi.value)

        if self.buy_ema_enabled.value:
            conditions.append(dataframe['close'] <= dataframe['ema10'])

        # green candle
        #conditions.append(dataframe['close'] > dataframe['open'])

        # potential gain > goal
        if self.buy_bb_enabled.value:
            conditions.append(dataframe['bb_gain'] >= self.buy_bb_gain.value)

        # TRIGGERS

        # the built-in pattern recognition doesn't work well, so check manually
        # non-zero body
        conditions.append(dataframe['body'] > 0.01)

        # shadow ratio > 2
        conditions.append(
            (dataframe['upper_ratio'] > 2) |
            (dataframe['lower_ratio'] > 2)
        )

        # # Detected one of the patterns
        # conditions.append(
        #     (dataframe['CDLHAMMER'] >= self.pattern_strength) |
        #     (dataframe['CDLINVERTEDHAMMER'] >= self.pattern_strength) |
        #     (dataframe['CDLHANGINGMAN'] >= self.pattern_strength) |
        #     (dataframe['CDLSHOOTINGSTAR'] >= self.pattern_strength)
        # )

        if conditions:
            dataframe.loc[reduce(lambda x, y: x & y, conditions), 'buy'] = 1

        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Based on TA indicators, populates the sell signal for the given dataframe
        :param dataframe: DataFrame
        :return: DataFrame with buy column
        """

        conditions = []


        # if hold, then don't set a sell signal
        if self.sell_hold_enabled.value:
            dataframe.loc[(dataframe['close'].notnull() ), 'sell'] = 0

        else:
            # GUARDS AND TRENDS

            conditions.append(dataframe['volume'] > 0)

            # red candle
            #conditions.append(dataframe['close'] < dataframe['open'])

            # conditions.append(
            #     (dataframe['rsi'] > self.rsi_limit) &
            #     (dataframe['rsi'] > 0)
            # )
            #
            # conditions.append(dataframe['close'] > dataframe['sma'])
            #
            # conditions.append(dataframe['mfi'] >= self.mfi_limit)

            # close is above EMA10
            conditions.append(dataframe['close'] >= dataframe['ema10'])

            # TRIGGERS
            conditions.append(dataframe['body'] > 0.01)

            # upper or lower shadow ratio > 2
            conditions.append(
                (dataframe['upper_ratio'] > 2)
                #(dataframe['upper_ratio'] > 2) |
                #(dataframe['lower_ratio'] > 2)
            )

            # # Detected one of the patterns
            # conditions.append(
            #     (dataframe['CDLHAMMER'] >= self.pattern_strength) |
            #     (dataframe['CDLINVERTEDHAMMER'] >= self.pattern_strength) |
            #     (dataframe['CDLHANGINGMAN'] >= self.pattern_strength) |
            #     (dataframe['CDLSHOOTINGSTAR'] >= self.pattern_strength)
            # )

            if conditions:
                dataframe.loc[reduce(lambda x, y: x & y, conditions), 'sell'] = 1

        return dataframe