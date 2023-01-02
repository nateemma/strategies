
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

from user_data.strategies import Config


class BuyDips(IStrategy):
    """
    Simple strategy that buys on dips and holds until ROI target is met

    How to use it?
    > python3 ./freqtrade/main.py -s BuyDips
    """

    # Buy hyperspace params:
    buy_params = {
        "buy_bb_gain": 0.07,
        "buy_bb_gain_enabled": True,
        "buy_mfi": 54.0,
        "buy_mfi_enabled": True,
        "buy_neg_macd_enabled": False,
    }

    # MFI (Money Flow Indicator) buy limit
    buy_mfi = DecimalParameter(10, 100, decimals=0, default=54, space="buy")

    # Bollinger Band 'gain' (% difference between current price and upper band).
    # Since we are looking for potential swings of >2%, we look for potential of more than that
    buy_bb_gain = DecimalParameter(0.01, 0.10, decimals=2, default=0.07, space="buy")

    # Categorical parameters that control whether a trend/check is used or not
    buy_mfi_enabled = CategoricalParameter([True, False], default=True, space="buy")
    buy_bb_gain_enabled = CategoricalParameter([True, False], default=True, space="buy")
    buy_neg_macd_enabled = CategoricalParameter([True, False], default=False, space="buy")

    sell_fisher = DecimalParameter(-1, 1, decimals=2, default=0.3, space="sell")

    sell_fisher_enabled = CategoricalParameter([True, False], default=True, space="sell")
    sell_sar_enabled = CategoricalParameter([True, False], default=True, space="sell")
    sell_hold_enabled = CategoricalParameter([True, False], default=True, space="sell")


    # set the startup candles count to the longest average used (EMA, EMA etc)
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

        conditions = []
        # GUARDS AND TRENDS
        if self.buy_mfi_enabled.value:
            conditions.append(dataframe['mfi'] >= self.buy_mfi.value)

        if self.buy_neg_macd_enabled.value:
            conditions.append(dataframe['macd'] < 0.0)

        if self.buy_bb_gain_enabled.value:
            conditions.append(dataframe['bb_gain'] >= self.buy_bb_gain.value)

        # Triggers
        conditions.append(qtpylib.crossed_above(dataframe['macd'], dataframe['macdsignal']))

        # check that volume is not 0
        conditions.append(dataframe['volume'] > 0)

        # build the dataframe using the conditions
        if conditions:
            dataframe.loc[
                reduce(lambda x, y: x & y, conditions),
                'buy'] = 1

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
            if self.sell_sar_enabled.value:
                conditions.append(dataframe['sar'] > dataframe['close'])

            if self.sell_fisher_enabled.value:
                conditions.append(dataframe['fisher_rsi'] > self.sell_fisher.value)

            if conditions:
                dataframe.loc[reduce(lambda x, y: x & y, conditions), 'buy'] = 1

        return dataframe