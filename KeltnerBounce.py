
# --- Do not remove these libs ---
from freqtrade.strategy.interface import IStrategy
from typing import Dict, List
from functools import reduce
from pandas import DataFrame
# --------------------------------

import talib.abstract as ta
import freqtrade.vendor.qtpylib.indicators as qtpylib
import numpy # noqa


class KeltnerBounce(IStrategy):
    """
    Simple strategy based on Keltner Band Bounce from bottom

    How to use it?
    > python3 ./freqtrade/main.py -s KeltnerBounce
    """

    # ROI table:
    minimal_roi = {
        "0": 0.173,
        "35": 0.106,
        "84": 0.038,
        "189": 0
    }

    # Stoploss:
    stoploss = -0.037

    # Trailing stop:
    trailing_stop = True
    trailing_stop_positive = 0.18
    trailing_stop_positive_offset = 0.224
    trailing_only_offset_is_reached = False

    # Optimal timeframe for the strategy
    timeframe = '5m'

    # run "populate_indicators" only for new candle
    process_only_new_candles = False

    # Experimental settings (configuration will overide these if set)
    use_sell_signal = True
    sell_profit_only = True
    ignore_roi_if_buy_signal = False

    # Optional order type mapping
    order_types = {
        'buy': 'limit',
        'sell': 'limit',
        'stoploss': 'market',
        'stoploss_on_exchange': False
    }

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

        # MACD
        macd = ta.MACD(dataframe)
        dataframe['macd'] = macd['macd']
        dataframe['macdsignal'] = macd['macdsignal']

        # Stoch fast
        stoch_fast = ta.STOCHF(dataframe)
        dataframe['fastd'] = stoch_fast['fastd']
        dataframe['fastk'] = stoch_fast['fastk']

        # RSI
        dataframe['rsi'] = ta.RSI(dataframe)

        # Inverse Fisher transform on RSI, values [-1.0, 1.0] (https://goo.gl/2JGGoy)
        rsi = 0.1 * (dataframe['rsi'] - 50)
        dataframe['fisher_rsi'] = (numpy.exp(2 * rsi) - 1) / (numpy.exp(2 * rsi) + 1)

        # Keltner Channel
        keltner = qtpylib.keltner_channel(dataframe)
        dataframe["kc_upperband"] = keltner["upper"]
        dataframe["kc_lowerband"] = keltner["lower"]
        dataframe["kc_middleband"] = keltner["mid"]

        # EMA - Exponential Moving Average
        dataframe['ema5'] = ta.EMA(dataframe, timeperiod=5)
        dataframe['ema10'] = ta.EMA(dataframe, timeperiod=10)
        dataframe['ema50'] = ta.EMA(dataframe, timeperiod=50)
        dataframe['ema100'] = ta.EMA(dataframe, timeperiod=100)

        # SAR Parabol
        dataframe['sar'] = ta.SAR(dataframe)

        # SMA - Simple Moving Average
        dataframe['sma'] = ta.SMA(dataframe, timeperiod=40)

        return dataframe

    def populate_buy_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Based on TA indicators, populates the buy signal for the given dataframe
        :param dataframe: DataFrame
        :return: DataFrame with buy column

        buy when candle comes into lower keltner band and is green

        """
        dataframe.loc[
            (
                (
                        (dataframe['open'] > dataframe['kc_lowerband']) &
                        (dataframe['open'] < dataframe['kc_middleband']) &
                        (dataframe['close'] > dataframe['kc_lowerband']) &
                        (dataframe['close'] < dataframe['kc_middleband'])
                ) &
                (dataframe['close'] > dataframe['open']) &
                (
                    (dataframe['open'].shift(1) <= dataframe['kc_lowerband']) |
                    (dataframe['close'].shift(1) <= dataframe['kc_lowerband'])
                ) &
                (dataframe['mfi'] <= 16) &
                (dataframe['fisher_rsi'] < -0.94)

            ),
            'buy'] = 1

        return dataframe

    def populate_sell_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Based on TA indicators, populates the sell signal for the given dataframe
        :param dataframe: DataFrame
        :return: DataFrame with buy column

        """
        # Exit long position if price is above upper band
        dataframe.loc[
            (
                (dataframe['open'] > dataframe['kc_upperband']) |
                (dataframe['close'] > dataframe['kc_upperband'])
                #(dataframe['mfi'] > 80)
            ),
            'sell'] = 1
        return dataframe