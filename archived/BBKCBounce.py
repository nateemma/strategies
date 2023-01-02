
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



class BBKCBounce(IStrategy):
    """
    Strategy based on prices bouncing between the Bollinger and Keltner Channel bounds

    How to use it?
    > python3 ./freqtrade/main.py -s BBKCBounce
    """

    # Buy hyperspace params:
    buy_params = {
        "buy_adx": 43.0,
        "buy_adx_enabled": False,
        "buy_dm_enabled": False,
        "buy_fisher": -0.41,
        "buy_fisher_enabled": True,
        "buy_kc_gain": 0.05,
        "buy_macd_enabled": False,
        "buy_mfi": 6.0,
        "buy_mfi_enabled": True,
        "buy_period": 16,
        "buy_sar_enabled": True,
    }

    buy_period = IntParameter(3, 20, default=16, space="buy")
    buy_kc_gain = DecimalParameter(0.01, 0.10, decimals=2, default=0.05, space="buy")
    buy_adx = DecimalParameter(1, 99, decimals=0, default=43, space="buy")
    buy_mfi = DecimalParameter(1, 99, decimals=0, default=6, space="buy")
    buy_fisher = DecimalParameter(-1, 1, decimals=2, default=-0.41, space="buy")

    buy_adx_enabled = CategoricalParameter([True, False], default=False, space="buy")
    buy_dm_enabled = CategoricalParameter([True, False], default=False, space="buy")
    buy_macd_enabled = CategoricalParameter([True, False], default=False, space="buy")
    buy_mfi_enabled = CategoricalParameter([True, False], default=True, space="buy")
    buy_sar_enabled = CategoricalParameter([True, False], default=True, space="buy")
    buy_fisher_enabled = CategoricalParameter([True, False], default=True, space="buy")

    sell_hold = CategoricalParameter([True, False], default=True, space="sell")
    sell_upper_band = CategoricalParameter([True, False], default=False, space="sell")

    # set the startup candles count to the longest average used (SMA, EMA etc)
    startup_candle_count = max(buy_period.value, 20)

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

        # SMA - Simple Moving Average
        dataframe['sma'] = ta.SMA(dataframe, timeperiod=self.buy_period.value)

        dataframe['ema'] = ta.EMA(dataframe, timeperiod=self.buy_period.value)
        dataframe['tema'] = ta.TEMA(dataframe, timeperiod=self.buy_period.value)
        dataframe['ema3'] = ta.EMA(dataframe, timeperiod=3)
        dataframe['ema5'] = ta.EMA(dataframe, timeperiod=5)

        # MACD
        macd = ta.MACD(dataframe)
        dataframe['macd'] = macd['macd']
        dataframe['macdsignal'] = macd['macdsignal']
        dataframe['macdhist'] = macd['macdhist']

        # MFI
        dataframe['mfi'] = ta.MFI(dataframe)

        # ADX
        dataframe['adx'] = ta.ADX(dataframe)
        dataframe['dm_plus'] = ta.PLUS_DM(dataframe)
        dataframe['dm_minus'] = ta.MINUS_DM(dataframe)
        dataframe['dm_delta'] = dataframe['dm_plus'] - dataframe['dm_minus']

        # SAR Parabolic
        dataframe['sar'] = ta.SAR(dataframe)

        # RSI
        dataframe['rsi'] = ta.RSI(dataframe)

        # Inverse Fisher transform on RSI, values [-1.0, 1.0] (https://goo.gl/2JGGoy)
        rsi = 0.1 * (dataframe['rsi'] - 50)
        dataframe['fisher_rsi'] = (numpy.exp(2 * rsi) - 1) / (numpy.exp(2 * rsi) + 1)

        # Bollinger Bands
        bollinger = qtpylib.bollinger_bands(qtpylib.typical_price(dataframe), window=20, stds=2)
        #bollinger = qtpylib.weighted_bollinger_bands(qtpylib.typical_price(dataframe), window=self.buy_period.value, stds=2)
        dataframe['bb_upperband'] = bollinger['upper']
        dataframe['bb_mid'] = bollinger['mid']
        dataframe['bb_lowerband'] = bollinger['lower']

        # Keltner Channel
        keltner = qtpylib.keltner_channel(dataframe)
        dataframe["kc_upper"] = keltner["upper"]
        dataframe["kc_lower"] = keltner["lower"]
        dataframe["kc_middle"] = keltner["mid"]
        dataframe["kc_gain"] = ((dataframe["kc_upper"] - dataframe["close"]) / dataframe["close"])

        # # Donchian Channels
        # dataframe['dc_upper'] = ta.MAX(dataframe['high'], timeperiod=self.buy_period.value)
        # dataframe['dc_lower'] = ta.MIN(dataframe['low'], timeperiod=self.buy_period.value)
        # dataframe['dc_mid'] = ta.TEMA(((dataframe['dc_upper'] + dataframe['dc_lower']) / 2),
        #                              timeperiod=self.buy_period.value)
        # # Fibonacci Levels (of Donchian Channel)
        # dataframe['dc_dist'] = (dataframe['dc_upper']  - dataframe['dc_lower'])
        # dataframe['dc_hf'] = dataframe['dc_upper'] - dataframe['dc_dist'] * 0.236 # Highest Fib
        # dataframe['dc_chf'] = dataframe['dc_upper'] - dataframe['dc_dist'] * 0.382 # Centre High Fib
        # dataframe['dc_clf'] = dataframe['dc_upper'] - dataframe['dc_dist'] * 0.618 # Centre Low Fib
        # dataframe['dc_lf'] = dataframe['dc_upper'] - dataframe['dc_dist'] * 0.764 # Low Fib
        #
        # # Squeeze Indicators.
        # #   'on'  means Bollinger Band lies completely within the Keltner Channel
        # #   'off' means Keltner Channel lies completely within the Bollinger Band
        # #   Booleans are funky with dataframes, so just do an intermediate calculation
        # dataframe['sqz_upper'] = (dataframe['bb_upperband'] - dataframe["kc_upper"])
        # dataframe['sqz_lower'] = (dataframe['bb_lowerband'] - dataframe["kc_lower"])
        # dataframe['sqz_on'] = ((dataframe['sqz_upper'] < 0) & (dataframe['sqz_lower'] > 0))
        # dataframe['sqz_off'] = ((dataframe['sqz_upper'] > 0) & (dataframe['sqz_lower'] < 0))
        #
        # # Momentum
        # # value is: Close - Moving Average( (Donchian midline + EMA) / 2 )
        #
        # # get momentum value by running linear regression on delta
        # dataframe['sqz_ave'] = ta.TEMA(((dataframe['dc_mid'] + dataframe['tema']) / 2),
        #                               timeperiod=self.buy_period.value)
        # dataframe['sqz_delta'] = ta.TEMA((dataframe['close'] - dataframe['sqz_ave']),
        #                               timeperiod=30)
        # #                               timeperiod = self.buy_period.value)
        # dataframe['sqz_val'] = ta.LINEARREG(dataframe['sqz_delta'], timeperiod=self.buy_period.value)

        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:

        conditions = []

        # GUARDS AND TRENDS

        if self.buy_adx_enabled.value:
            conditions.append(dataframe['adx'] >= self.buy_adx.value)

        if self.buy_dm_enabled.value:
            conditions.append(dataframe['dm_delta'] > 0)

        if self.buy_mfi_enabled.value:
            conditions.append(dataframe['mfi'] > self.buy_mfi.value)

        # only buy if close is below SAR
        if self.buy_sar_enabled.value:
            conditions.append(dataframe['close'] < dataframe['sar'])

        if self.buy_macd_enabled.value:
            conditions.append(dataframe['macd'] > dataframe['macdsignal'])

        if self.buy_fisher_enabled.value:
            conditions.append(dataframe['fisher_rsi'] < self.buy_fisher.value)

        # green candle
        conditions.append(dataframe['close'] > dataframe['open'])

        # Potential gain greater than goal
        conditions.append(dataframe['kc_gain'] >= self.buy_kc_gain.value)

        # TRIGGERS
        # candle crossed either BB or KC lower band and ended up above both
        conditions.append(
            (
                    (qtpylib.crossed_above(dataframe['close'], dataframe['bb_lowerband'])) |
                    (qtpylib.crossed_above(dataframe['close'], dataframe['kc_lower']))
            ) &
            (
                    (dataframe['close'] >= dataframe['bb_lowerband']) &
                    (dataframe['close'] >= dataframe['kc_lower'])
            )
        )

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
        if self.sell_hold.value:
            dataframe.loc[(dataframe['close'].notnull() ), 'sell'] = 0
            return dataframe

        # sell if price crosses above BB and KC upper band
        if self.sell_upper_band.value:
            conditions.append(
                (
                    # candle crossed either BB or KC lower band and ended up above both
                        (
                                (qtpylib.crossed_above(dataframe['close'], dataframe['bb_upperband'])) |
                                (qtpylib.crossed_above(dataframe['close'], dataframe['kc_upper']))
                        ) &
                        (
                                (dataframe['close'] >= dataframe['bb_upperband']) &
                                (dataframe['close'] >= dataframe['kc_upper'])
                        )
                )
                # ) |
                # (
                #     # safeguard: if prices drops below both lower bounds, sell
                #         (
                #                 (qtpylib.crossed_below(dataframe['close'], dataframe['bb_lowerband'])) &
                #                 (qtpylib.crossed_below(dataframe['close'], dataframe['kc_lower']))
                #         ) &
                #         (
                #                 (dataframe['close'] <= dataframe['bb_upperband']) &
                #                 (dataframe['close'] <= dataframe['kc_upper'])
                #         )
                # )
            )

            if conditions:
                dataframe.loc[
                    reduce(lambda x, y: x & y, conditions),
                    'sell'] = 1

        return dataframe