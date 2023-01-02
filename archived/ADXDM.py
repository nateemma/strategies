
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



class ADXDM(IStrategy):
    """
    Simple strategy based on ADX value and DM+/DM- crossing

    How to use it?
    > python3 ./freqtrade/main.py -s ADXDM
    """

    # Hyperparameters

    # Buy hyperspace params:
    buy_params = {
        "buy_adx": 60.0,
        "buy_bb_enabled": False,
        "buy_bb_gain": 0.02,
        "buy_mfi": 6.0,
        "buy_mfi_enabled": True,
        "buy_period": 12,
    }

    buy_adx = DecimalParameter(20, 60, decimals=0, default=60, space="buy")
    buy_mfi = DecimalParameter(1, 30, decimals=0, default=6, space="buy")
    buy_mfi_enabled = CategoricalParameter([True, False], default=True, space="buy")
    buy_period = IntParameter(3, 50, default=12, space="buy")
    buy_bb_gain = DecimalParameter(0.01, 0.10, decimals=2, default=0.02, space="buy")

    buy_bb_enabled = CategoricalParameter([True, False], default=False, space="buy")

    sell_hold = CategoricalParameter([True, False], default=True, space="sell")

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


        # ADX
        dataframe['adx'] = ta.ADX(dataframe)
        dataframe['dm_plus'] = ta.PLUS_DM(dataframe)
        dataframe['dm_minus'] = ta.MINUS_DM(dataframe)
        dataframe['dm_delta'] = dataframe['dm_plus'] - dataframe['dm_minus']
        dataframe['adx_delta'] = (dataframe['adx'] - self.buy_adx.value) / 100 # for display
        dataframe['adx_slope'] = ta.LINEARREG_SLOPE(dataframe['adx'], timeperiod=3)

        # Bollinger Bands
        bollinger = qtpylib.bollinger_bands(qtpylib.typical_price(dataframe), window=20, stds=2)
        #bollinger = qtpylib.weighted_bollinger_bands(qtpylib.typical_price(dataframe), window=self.buy_period.value, stds=2)
        dataframe['bb_upperband'] = bollinger['upper']
        dataframe['bb_lowerband'] = bollinger['lower']
        dataframe["bb_gain"] = ((dataframe["bb_upperband"] - dataframe["close"]) / dataframe["close"])

        # # SAR Parabolic
        # dataframe['sar'] = ta.SAR(dataframe)

        # SMA - Simple Moving Average
        dataframe['sma'] = ta.SMA(dataframe, timeperiod=20)

        # TEMA - Triple Exponential Moving Average
        dataframe['tema'] = ta.TEMA(dataframe, timeperiod=20)

        # MFI
        dataframe['mfi'] = ta.MFI(dataframe)

        # RSI
        dataframe['rsi'] = ta.RSI(dataframe)

        # Inverse Fisher transform on RSI, values [-1.0, 1.0] (https://goo.gl/2JGGoy)
        rsi = 0.1 * (dataframe['rsi'] - 50)
        dataframe['fisher_rsi'] = (numpy.exp(2 * rsi) - 1) / (numpy.exp(2 * rsi) + 1)

        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        conditions = []
        # GUARDS AND TRENDS

        # check that volume is not 0 (can happen in testing, or if there are issues with exchange data)
        #conditions.append(dataframe['volume'] > 0)

        # during back testing, data can be undefined, so check
        conditions.append(dataframe['adx'].notnull())

        if self.buy_mfi_enabled.value:
            # conditions.append(dataframe['mfi'] <= self.buy_mfi.value)
            conditions.append(dataframe['mfi'] <= dataframe['adx'])

        # SAR check
        #conditions.append(dataframe['sar'] < dataframe['close'])

        # TRIGGERS

        # Strong trend
        conditions.append(dataframe['adx'] > self.buy_adx.value)

        #ADX slope indicates changing trend
        conditions.append(qtpylib.crossed_below(dataframe['adx_slope'], 0))

        # currently in downtrend (i.e. about to reverse)
        conditions.append(dataframe['dm_delta'] < 0)

        # ADX with DM+ > DM- indicates uptrend
        # conditions.append(
        #     (dataframe['adx'] > self.buy_adx.value) &
        #     (qtpylib.crossed_above(dataframe['dm_plus'], dataframe['dm_minus']))
        # )

        # build the dataframe using the conditions
        if conditions:
            dataframe.loc[reduce(lambda x, y: x & y, conditions), 'buy'] = 1

        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Based on TA indicators, populates the sell signal for the given dataframe
        :param dataframe: DataFrame
        :return: DataFrame with buy column

        """
        # if hold flag is set then don't issue any sell signals at all (rely on ROI and stoploss)
        if self.sell_hold.value:
            dataframe.loc[(dataframe['close'].notnull()), 'sell'] = 0
            return dataframe

        conditions = []

        # Strong trend
        conditions.append(dataframe['adx'] > self.buy_adx.value)

        # ADX slope indicates changing trend
        conditions.append(qtpylib.crossed_below(dataframe['adx_slope'], 0))

        # currently in uptrend (i.e. about to reverse)
        conditions.append(dataframe['dm_delta'] > 0)

        # potential gain > goal
        if self.buy_bb_enabled.value:
            conditions.append(dataframe['bb_gain'] >= self.buy_bb_gain.value)

        # # Exit long position if price crosses below lower band
        # dataframe.loc[
        #     (
        #             (dataframe['adx'].notnull()) &
        #             (dataframe['adx'] > self.buy_adx.value) &
        #             (qtpylib.crossed_below(dataframe['dm_plus'], dataframe['dm_minus']))
        #     ),
        #     'sell'] = 1
        if conditions:
            dataframe.loc[reduce(lambda x, y: x & y, conditions), 'sell'] = 1

        return dataframe