
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



class SqueezeMomentum(IStrategy):
    """
    Strategy based on LazyBear Squeeze Momentum Indicator (on TradingView.com)

    How to use it?
    > python3 ./freqtrade/main.py -s SqueezeMomentum
    """

    # Hyperparameters
    # Buy hyperspace params:
    buy_params = {
        "buy_accel_enabled": True,
        "buy_adx": 14.0,
        "buy_adx_enabled": True,
        "buy_bb_enabled": False,
        "buy_bb_gain": 0.03,
        "buy_ema_enabled": True,
        "buy_macd_enabled": True,
        "buy_mfi": 6.0,
        "buy_mfi_enabled": False,
        "buy_period": 11,
        "buy_predict_enabled": True,
        "buy_sqz_band": 0.0112,
    }

    buy_period = IntParameter(3, 20, default=16, space="buy")
    buy_adx = DecimalParameter(1, 99, decimals=0, default=21, space="buy")
    buy_mfi = DecimalParameter(1, 30, decimals=0, default=14, space="buy")
    buy_sqz_band = DecimalParameter(0.003, 0.02, decimals=4, default=0.0165, space="buy")
    buy_bb_gain = DecimalParameter(0.01, 0.10, decimals=2, default=0.08, space="buy")

    buy_bb_enabled = CategoricalParameter([True, False], default=True, space="buy")
    # buy_sqz_enabled = CategoricalParameter([True, False], default=True, space="buy")
    buy_macd_enabled = CategoricalParameter([True, False], default=True, space="buy")
    buy_adx_enabled = CategoricalParameter([True, False], default=False, space="buy")
    buy_mfi_enabled = CategoricalParameter([True, False], default=True, space="buy")
    buy_ema_enabled = CategoricalParameter([True, False], default=True, space="buy")
    # buy_dc_enabled = CategoricalParameter([True, False], default=False, space="buy")
    buy_predict_enabled = CategoricalParameter([True, False], default=True, space="buy")
    buy_accel_enabled = CategoricalParameter([True, False], default=True, space="buy")

    sell_sar_enabled = CategoricalParameter([True, False], default=False, space="sell")
    sell_dc_enabled = CategoricalParameter([True, False], default=False, space="sell")
    sell_fisher = DecimalParameter(-1, 1, decimals=2, default=-0.30, space="sell")
    sell_standard_triggers = CategoricalParameter([True, False], default=False, space="sell")
    sell_hold = CategoricalParameter([True, False], default=True, space="sell")


    # set the startup candles count to the longest average used (EMA, EMA etc)
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
        dataframe['ema10'] = ta.EMA(dataframe, timeperiod=10)

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
        # bollinger = qtpylib.bollinger_bands(qtpylib.typical_price(dataframe), window=self.buy_period.value, stds=2)
        bollinger = qtpylib.weighted_bollinger_bands(qtpylib.typical_price(dataframe), window=20, stds=2)
        dataframe['bb_upperband'] = bollinger['upper']
        dataframe['bb_mid'] = bollinger['mid']
        dataframe['bb_lowerband'] = bollinger['lower']
        dataframe["bb_gain"] = ((dataframe["bb_upperband"] - dataframe["close"]) / dataframe["close"])

        # Keltner Channel
        keltner = qtpylib.keltner_channel(dataframe)
        dataframe["kc_upper"] = keltner["upper"]
        dataframe["kc_lower"] = keltner["lower"]
        dataframe["kc_middle"] = keltner["mid"]

        # Donchian Channels
        dataframe['dc_upper'] = ta.MAX(dataframe['high'], timeperiod=self.buy_period.value)
        dataframe['dc_lower'] = ta.MIN(dataframe['low'], timeperiod=self.buy_period.value)
        dataframe['dc_mid'] = ta.TEMA(((dataframe['dc_upper'] + dataframe['dc_lower']) / 2),
                                     timeperiod=self.buy_period.value)
        # Fibonacci Levels (of Donchian Channel)
        dataframe['dc_dist'] = (dataframe['dc_upper']  - dataframe['dc_lower'])
        dataframe['dc_hf'] = dataframe['dc_upper'] - dataframe['dc_dist'] * 0.236 # Highest Fib
        dataframe['dc_chf'] = dataframe['dc_upper'] - dataframe['dc_dist'] * 0.382 # Centre High Fib
        dataframe['dc_clf'] = dataframe['dc_upper'] - dataframe['dc_dist'] * 0.618 # Centre Low Fib
        dataframe['dc_lf'] = dataframe['dc_upper'] - dataframe['dc_dist'] * 0.764 # Low Fib

        # Squeeze Indicators.
        #   'on'  means Bollinger Band lies completely within the Keltner Channel
        #   'off' means Keltner Channel lies completely within the Bollinger Band
        #   Booleans are funky with dataframes, so just do an intermediate calculation
        dataframe['sqz_upper'] = (dataframe['bb_upperband'] - dataframe["kc_upper"])
        dataframe['sqz_lower'] = (dataframe['bb_lowerband'] - dataframe["kc_lower"])
        dataframe['sqz_on'] = ((dataframe['sqz_upper'] < 0) & (dataframe['sqz_lower'] > 0))
        dataframe['sqz_off'] = ((dataframe['sqz_upper'] > 0) & (dataframe['sqz_lower'] < 0))

        # Momentum
        # value is: Close - Moving Average( (Donchian midline + EMA) / 2 )

        # get momentum value by running linear regression on delta
        dataframe['sqz_ave'] = ta.TEMA(((dataframe['dc_mid'] + dataframe['tema']) / 2),
                                      timeperiod=self.buy_period.value)
        dataframe['sqz_delta'] = ta.TEMA((dataframe['close'] - dataframe['sqz_ave']),
                                      timeperiod=30)
        #                               timeperiod = self.buy_period.value)
        dataframe['sqz_val'] = ta.LINEARREG(dataframe['sqz_delta'], timeperiod=self.buy_period.value)
        # the angle will show turnaround points (at 0). Use just a little averaging to avoid 'wiggles'
        #dataframe['sqz_angle'] = ta.LINEARREG_ANGLE(dataframe['sqz_delta'], timeperiod=3)
        dataframe['sqz_angle'] = ta.LINEARREG_SLOPE(dataframe['sqz_delta'], timeperiod=3)
        dataframe['sqz_a'] = ta.LINEARREG(dataframe['sqz_angle'], timeperiod=3)
        dataframe['sqz_accel'] = ta.LINEARREG_SLOPE(dataframe['sqz_a'], timeperiod=3)

        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        conditions = []
        # GUARDS AND TRENDS
        # check that volume is not 0 (can happen in testing, or if there are issues with exchange data)
        # conditions.append(dataframe['volume'] > 0)

        # ADX with DM+ > DM- indicates uptrend
        if self.buy_adx_enabled.value:
            conditions.append(
                # (dataframe['adx'] > self.buy_adx.value)
                (dataframe['adx'] >= self.buy_adx.value) &
                (dataframe['dm_plus'] >= dataframe['dm_minus'])
            )

        if self.buy_mfi_enabled.value:
            conditions.append(dataframe['mfi'] <= self.buy_mfi.value)

        # only buy if close is below EMA
        if self.buy_ema_enabled.value:
            conditions.append(dataframe['close'] < dataframe['ema'])

        # only buy if close above High Fibonacci (i.e. potential breakout)
        # if self.buy_dc_enabled.value:
        #         conditions.append(
        #             (dataframe['close'] >= dataframe['dc_hf']) |
        #             (dataframe['open'] >= dataframe['dc_hf'])
        #         )

        # MACD -ve (this is the opposite of common sense, because it is a lagging indicator)
        if self.buy_macd_enabled.value:
            conditions.append(dataframe['macd'] <= 0)
            conditions.append(dataframe['macd'] <= dataframe['macdsignal'])

        # potential gain > goal
        if self.buy_bb_enabled.value:
            conditions.append(dataframe['bb_gain'] >= self.buy_bb_gain.value)


        # squeeze is 'on'
        # if self.buy_sqz_enabled.value:
        #     conditions.append(dataframe['sqz_on'])

        # We can (try to) predict an upcoming swing (up) by looking for a reversal during an 'off' period
        if self.buy_predict_enabled.value:

            # check for startup issue
            conditions.append(
                (dataframe['sqz_val'].notnull()) &
                (dataframe['sqz_val'].shift(5).notnull())
            )

            # Green candle
            conditions.append(dataframe['close'] > dataframe['open'])

            # TRIGGERS

            # squeeze values are -ve but turning around
            conditions.append(dataframe['sqz_val'] < -self.buy_sqz_band.value)
            if self.buy_accel_enabled.value:
                conditions.append(qtpylib.crossed_above(dataframe['sqz_accel'], 0))
            else:
                conditions.append(qtpylib.crossed_above(dataframe['sqz_angle'], 0))

        else:
             # during back testing, data can be undefined, so check
            # conditions.append(dataframe['sqz_upper'].notnull())

            # TRIGGERS
            # Momentum goes positive , and is increasing
            conditions.append(qtpylib.crossed_above(dataframe['sqz_val'], 0))

            # current candle is green
            # conditions.append(dataframe['close'] > dataframe['open'])

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

        # if hold flag is set then don't issue any sell signals at all (rely on ROI and stoploss)
        if self.sell_hold.value:
            dataframe.loc[
                (dataframe['close'].notnull()),
                'sell'] = 0
            return dataframe

        conditions = []
        # GUARDS AND TRENDS
        # check that volume is not 0 (can happen in testing, or if there are issues with exchange data)
        #conditions.append(dataframe['volume'] > 0)

        # only sell if close is above SAR
        if self.sell_sar_enabled.value:
            conditions.append(dataframe['close'] > dataframe['sar'])

        # only buy if close below Low Fibonacci (i.e. potential breakdown)
        if self.sell_dc_enabled.value:
            conditions.append(dataframe['close'] <= dataframe['dc_lf'])

        # squeeze is 'off'
        # if self.buy_sqz_enabled.value:
        #     conditions.append(dataframe['sqz_on'] != True)

        # We can (try to) predict an upcoming swing (down) by looking for a reversal during an 'on' period
        if self.buy_predict_enabled:

            # TRIGGERS
            # squeeze values are +ve but turning around
            conditions.append(dataframe['sqz_val'] > self.buy_sqz_band.value)
            #conditions.append(qtpylib.crossed_below(dataframe['sqz_angle'], 0))
            conditions.append(qtpylib.crossed_below(dataframe['sqz_accel'], 0))
            # conditions.append(
            #     (dataframe['sqz_val'].shift(1) > self.buy_sqz_band.value) &
            #     (dataframe['sqz_val'] < dataframe['sqz_val'].shift(1)) &
            #     (dataframe['sqz_val'].shift(1) >= dataframe['sqz_val'].shift(2)) &
            #     (dataframe['sqz_val'].shift(2) >= dataframe['sqz_val'].shift(3)) &
            #     (dataframe['sqz_val'].shift(3) >= dataframe['sqz_val'].shift(4))
            # )

        else:

            # during back testing, data can be undefined, so check
            conditions.append(dataframe['sqz_val'].notnull())

            # TRIGGERS
            # Momentum goes negative
            conditions.append(qtpylib.crossed_below(dataframe['sqz_val'], 0))
            #conditions.append(
            #    (dataframe['sqz_val'] < 0) &
            #    (dataframe['sqz_val'].shift(1) < 0) &
            #    (dataframe['sqz_val'].shift(2) >= 0)
            #)

        # 'standard' sell triggers
        orconditions = []
        if self.sell_standard_triggers.value:
            orconditions.append(
                (dataframe['fisher_rsi'] > self.sell_fisher.value) &
                (dataframe['sar'] > dataframe['close'])
            )

        # build the dataframe using the conditions
        r1 = False
        r2 = False
        if conditions:
             r1 = reduce(lambda x, y: x & y, conditions)

        if orconditions:
            r2 = reduce(lambda x, y: x & y, orconditions)

        dataframe.loc[(r1 | r2), 'sell'] = 1

        return dataframe