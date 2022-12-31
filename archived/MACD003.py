
# --- Do not remove these libs ---
from freqtrade.strategy.interface import IStrategy
from typing import Dict, List
from functools import reduce
from pandas import DataFrame
# --------------------------------
from freqtrade.strategy.hyper import CategoricalParameter, DecimalParameter, IntParameter


import talib.abstract as ta
import freqtrade.vendor.qtpylib.indicators as qtpylib
import numpy # noqa


class MACD003(IStrategy):
    """
    Strategy 003 sell + MACD buy signal

    How to use it?
    > python3 ./freqtrade/main.py -s MACD003
    """
    buy_mfi = DecimalParameter(10, 50, decimals=0, default=20, space="buy")
    buy_fisher = DecimalParameter(-1, 1, decimals=2, default=-0.94, space="buy")
    #buy_fisher = DecimalParameter(-1, 1, decimals=2, default=-0.5, space="buy")
    buy_bb_gain = DecimalParameter(0, 0.20, decimals=1, default=0.04, space="buy")

    sell_mfi = DecimalParameter(1, 99, decimals=0, default=80, space="sell")
    sell_fisher = DecimalParameter(-1, 1, decimals=2, default=0.3, space="sell")

    # ROI table:
    minimal_roi = {
        "0": 0.171,
        "15": 0.08,
        "40": 0.011,
        "131": 0
    }

    # Stoploss:
    stoploss = -0.332

    # Trailing stop:
    trailing_stop = True
    trailing_stop_positive = 0.153
    trailing_stop_positive_offset = 0.219
    trailing_only_offset_is_reached = True

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

        # Bollinger bands
        bollinger = qtpylib.bollinger_bands(qtpylib.typical_price(dataframe), window=20, stds=2)
        dataframe['bb_lowerband'] = bollinger['lower']
        dataframe['bb_upperband'] = bollinger['upper']
        dataframe["bb_gain"] = ((dataframe["bb_upperband"] - dataframe["close"]) / dataframe["close"])


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

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Based on TA indicators, populates the buy signal for the given dataframe
        :param dataframe: DataFrame
        :return: DataFrame with buy column
        """
        dataframe.loc[
            (
                    (
                            (dataframe['mfi'] < self.buy_mfi.value) |
                            # (dataframe['mfi'] > 8.0) &
                            (dataframe['fisher_rsi'] < self.buy_fisher.value) |
                            # (dataframe['rsi'] < 28) &
                            # (dataframe['rsi'] > 0) &
                            (
                                    (dataframe['fastd'] > dataframe['fastk']) &
                                    (dataframe['fastk'] < 20)
                            )
                    ) &
                    (
                        (dataframe['macd'] < 0.0) &
                        (qtpylib.crossed_above(dataframe['macd'], dataframe['macdsignal'])) &
                        (dataframe['bb_gain'] >= self.buy_bb_gain.value)
                    )
            ),
            'buy'] = 1

        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Based on TA indicators, populates the sell signal for the given dataframe
        :param dataframe: DataFrame
        :return: DataFrame with buy column
        """
        dataframe.loc[
            (
                    (dataframe['mfi'] > self.sell_mfi.value) &
                    (dataframe['sar'] > dataframe['close']) &
                    (dataframe['fisher_rsi'] > self.sell_fisher.value)
            ),
            'sell'] = 1
        return dataframe