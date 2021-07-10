
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


class SimpleBollinger(IStrategy):
    """
    Simple strategy based on Bollinger Band Breakout

    How to use it?
    > python3 ./freqtrade/main.py -s SimpleBollinger
    """

    # HYPERPARAMETERS
    buy_mfi = DecimalParameter(10, 40, decimals=0, default=20.0, space="buy")
    sell_mfi = DecimalParameter(10, 40, decimals=0, default=80.0, space="sell")

    # Categorical parameters that control whether a trend/check is used or not
    buy_mfi_enabled = CategoricalParameter([True, False], default=True, space="buy")
    buy_macd_enabled = CategoricalParameter([True, False], default=True, space="buy")
    sell_mfi_enabled = CategoricalParameter([True, False], default=True, space="sell")

    # Minimal ROI designed for the strategy.
    # This attribute will be overridden if the config file contains "minimal_roi"

    # ROI table:
    minimal_roi = {
        "0": 0.173,
        "19": 0.05,
        "77": 0.04,
        "155": 0
    }

    # Stoploss:
    stoploss = -0.102

    # Trailing stop:
    trailing_stop = True
    trailing_stop_positive = 0.318
    trailing_stop_positive_offset = 0.345
    trailing_only_offset_is_reached = True

    # run "populate_indicators" only for new candle
    process_only_new_candles = False

    # Experimental settings (configuration will overide these if set)
    use_sell_signal = True
    sell_profit_only = True
    ignore_roi_if_buy_signal = False

    # Optimal timeframe for the strategy
    timeframe = '5m'

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
        dataframe['bb_upperband'] = bollinger['upper']
        dataframe['bb_lowerband'] = bollinger['lower']

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

        """

        conditions = []

        # GUARDS AND TRENDS
        if self.buy_mfi_enabled.value:
            conditions.append(dataframe['mfi'] <= self.buy_mfi.value)

        if self.buy_macd_enabled.value:
            conditions.append(dataframe['macd'] > dataframe['macdsignal'])

        # TRIGGERS
        # price closes above upper band
        conditions.append(qtpylib.crossed_above(dataframe['close'], dataframe['bb_upperband']))

        # build the dataframe using the conditions
        if conditions:
            dataframe.loc[
                reduce(lambda x, y: x & y, conditions),
                'buy'] = 1

        return dataframe

    def populate_sell_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Based on TA indicators, populates the sell signal for the given dataframe
        :param dataframe: DataFrame
        :return: DataFrame with buy column

        close if price is below  lower band
        """

        conditions = []

        # GUARDS AND TRENDS
        if self.sell_mfi_enabled.value:
            conditions.append(dataframe['mfi'] >= self.sell_mfi.value)

        # TRIGGERS
        # exit if price closes below lower band
        conditions.append(qtpylib.crossed_below(dataframe['close'], dataframe['bb_lowerband']))

        # build the dataframe using the conditions
        if conditions:
            dataframe.loc[
                reduce(lambda x, y: x & y, conditions),
                'sell'] = 1

        return dataframe
