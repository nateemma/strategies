
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
import math


class ComboHold(IStrategy):
    """
    Combines several buy strategies, and just holds until ROI kicks in
    This version doesn't issue a sell signal, just holds until ROI or stoploss kicks in

    How to use it?
    > python3 ./freqtrade/main.py -s ComboHold
    """

    # Hyperparameters

    buy_ndrop_enabled = CategoricalParameter([True, False], default=True, space="buy")
    buy_nseq_enabled = CategoricalParameter([True, False], default=True, space="buy")
    buy_emabounce_enabled = CategoricalParameter([True, False], default=True, space="buy")
    buy_strat3_enabled = CategoricalParameter([True, False], default=False, space="buy")

    # NDrop parameters:
    buy_ndrop_num_candles = IntParameter(2, 9, default=2, space="buy")
    buy_ndrop_drop = DecimalParameter(0.01, 0.06, decimals=3, default=0.025, space="buy")
    buy_ndrop_fisher = DecimalParameter(-1, 1, decimals=2, default=-0.02, space="buy")
    buy_ndrop_mfi = DecimalParameter(10, 40, decimals=0, default=11.0, space="buy")
    buy_ndrop_fisher_enabled = CategoricalParameter([True, False], default=False, space="buy")
    buy_ndrop_mfi_enabled = CategoricalParameter([True, False], default=False, space="buy")
    buy_ndrop_bb_enabled = CategoricalParameter([True, False], default=False, space="buy")

    # NSeq parameters:
    buy_nseq_num_candles = IntParameter(3, 9, default=3, space="buy")
    buy_nseq_drop = DecimalParameter(0.005, 0.06, decimals=3, default=0.011, space="buy")
    buy_nseq_fisher = DecimalParameter(-1, 1, decimals=2, default=0.96, space="buy")
    buy_nseq_mfi = DecimalParameter(10, 40, decimals=0, default=15.0, space="buy")
    buy_nseq_fisher_enabled = CategoricalParameter([True, False], default=True, space="buy")
    buy_nseq_mfi_enabled = CategoricalParameter([True, False], default=False, space="buy")
    buy_nseq_bb_enabled = CategoricalParameter([True, False], default=True, space="buy")

    # EMABounce parameters
    buy_emabounce_long_period = IntParameter(20, 100, default=50, space="buy")
    buy_emabounce_short_period = IntParameter(5, 15, default=10, space="buy")
    buy_emabounce_diff = DecimalParameter(0.01, 0.10, decimals=3, default=0.065, space="buy")

    # Strategy003 Parameters

    buy_strat3_rsi = DecimalParameter(0, 50, decimals=0, default=15, space="buy")
    buy_strat3_mfi = DecimalParameter(0, 50, decimals=0, default=24, space="buy")
    buy_strat3_fisher = DecimalParameter(-1, 1, decimals=2, default=-0.28, space="buy")
    buy_strat3_rsi_enabled = CategoricalParameter([True, False], default=True, space="buy")
    buy_strat3_sma_enabled = CategoricalParameter([True, False], default=False, space="buy")
    buy_strat3_ema_enabled = CategoricalParameter([True, False], default=False, space="buy")
    buy_strat3_mfi_enabled = CategoricalParameter([True, False], default=True, space="buy")
    buy_strat3_fastd_enabled = CategoricalParameter([True, False], default=False, space="buy")
    buy_strat3_fisher_enabled = CategoricalParameter([True, False], default=False, space="buy")


    # ROI table:
    minimal_roi = {
        "0": 0.217,
        "40": 0.061,
        "82": 0.039,
        "125": 0
    }

    # Stoploss:
    stoploss = -0.349

    # Trailing stop:
    trailing_stop = True
    trailing_stop_positive = 0.304
    trailing_stop_positive_offset = 0.333
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

        # SMA - Simple Moving Average
        dataframe['sma'] = ta.SMA(dataframe, timeperiod=40)

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
        #bollinger = qtpylib.bollinger_bands(qtpylib.typical_price(dataframe), window=20, stds=2)
        bollinger = qtpylib.weighted_bollinger_bands(qtpylib.typical_price(dataframe), window=20, stds=2)
        dataframe['bb_upperband'] = bollinger['upper']
        dataframe['bb_middleband'] = bollinger['mid']
        dataframe['bb_lowerband'] = bollinger['lower']
        dataframe["bb_gain"] = ((dataframe["bb_upperband"] - dataframe["close"]) / dataframe["close"])

        # EMA - Exponential Moving Average
        dataframe['ema5'] = ta.EMA(dataframe, timeperiod=5)
        dataframe['ema10'] = ta.EMA(dataframe, timeperiod=10)
        dataframe['ema50'] = ta.EMA(dataframe, timeperiod=50)
        dataframe['ema100'] = ta.EMA(dataframe, timeperiod=100)

        dataframe['ema'] = ta.EMA(dataframe, timeperiod=self.buy_emabounce_long_period.value)
        dataframe['ema_short'] = ta.EMA(dataframe, timeperiod=self.buy_emabounce_short_period.value)
        dataframe['ema_angle'] = ta.LINEARREG_SLOPE(dataframe['ema_short'], timeperiod=3) / (2.0 * math.pi)
        dataframe['ema_diff'] = (((dataframe['ema'] - dataframe['close']) /
                                      dataframe['ema'])) \
                                    - self.buy_emabounce_diff.value

        # SAR Parabol
        dataframe['sar'] = ta.SAR(dataframe)

        return dataframe

    def NDrop_conditions(self, dataframe: DataFrame):
        conditions = []

        # GUARDS AND TRENDS
        if self.buy_ndrop_mfi_enabled.value:
            conditions.append(dataframe['mfi'] <= self.buy_ndrop_mfi.value)

        if self.buy_ndrop_fisher_enabled.value:
            conditions.append(dataframe['fisher_rsi'] < self.buy_ndrop_fisher.value)

        if self.buy_ndrop_bb_enabled.value:
            conditions.append(dataframe['close'] <= dataframe['bb_lowerband'])

        # TRIGGERS

        # N red candles
        if self.buy_ndrop_num_candles.value >= 1:
            for i in range(self.buy_ndrop_num_candles.value):
                conditions.append(dataframe['close'].shift(i) <= dataframe['open'].shift(i))

        # big enough drop?
        conditions.append(
            (((dataframe['open'].shift(self.buy_ndrop_num_candles.value - 1) - dataframe['close']) /
              dataframe['open'].shift(self.buy_ndrop_num_candles.value - 1)) >= self.buy_ndrop_drop.value)
        )
        return conditions


    def NSeq_conditions(self, dataframe: DataFrame):
        conditions = []
        # GUARDS AND TRENDS
        if self.buy_nseq_mfi_enabled.value:
            conditions.append(dataframe['mfi'] <= self.buy_nseq_mfi.value)

        if self.buy_nseq_fisher_enabled.value:
            conditions.append(dataframe['fisher_rsi'] < self.buy_nseq_fisher.value)

        if self.buy_nseq_bb_enabled.value:
            conditions.append(dataframe['close'] <= dataframe['bb_lowerband'])

        # TRIGGERS

        # current candle is green
        conditions.append(dataframe['close'] >= dataframe['open'])

        # N red candles
        if self.buy_nseq_num_candles.value >= 1:
            for i in range(self.buy_nseq_num_candles.value):
                conditions.append(dataframe['close'].shift(i+1) <= dataframe['open'].shift(i+1))

        # big enough drop?
        conditions.append(
            (((dataframe['open'].shift(self.buy_nseq_num_candles.value) - dataframe['close']) /
            dataframe['open'].shift(self.buy_nseq_num_candles.value)) >= self.buy_nseq_drop.value)
        )
        return conditions

    def EMABounce_conditions(self, dataframe: DataFrame):
        conditions = []
        # GUARDS AND TRENDS

        # TRIGGERS

        # EMA flattened?
        conditions.append(qtpylib.crossed_above(dataframe['ema_angle'], 0))

        # buy if price is far enough below EMA
        conditions.append(dataframe['ema_diff'] > 0.0)

        return conditions

    def Strat3_conditions(self, dataframe: DataFrame):
        conditions = []

        # GUARDS AND TRENDS
        if self.buy_strat3_rsi_enabled.value:
            conditions.append(
                (dataframe['rsi'] < self.buy_strat3_rsi.value) &
                (dataframe['rsi'] > 0)
            )

        if self.buy_strat3_sma_enabled.value:
            conditions.append(dataframe['close'] < dataframe['sma'])

        if self.buy_strat3_fisher_enabled.value:
            conditions.append(dataframe['fisher_rsi'] < self.buy_strat3_fisher.value)

        if self.buy_strat3_mfi_enabled.value:
            conditions.append(dataframe['mfi'] <= self.buy_strat3_mfi.value)

        if self.buy_strat3_ema_enabled.value:
            conditions.append(
                (dataframe['ema50'] > dataframe['ema100']) |
                (qtpylib.crossed_above(dataframe['ema5'], dataframe['ema10']))
            )

        if self.buy_strat3_fastd_enabled.value:
            conditions.append(
                (dataframe['fastd'] > dataframe['fastk']) &
                (dataframe['fastd'] > 0)
            )

        return conditions

    def populate_buy_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:


        # build the dataframe using the conditions
        # AND together the conditions for each strategy
        conditions = []
        if self.buy_ndrop_enabled.value:
            conditions.append(reduce(lambda x, y: x & y, self.NDrop_conditions(dataframe)))

        if self.buy_nseq_enabled.value:
            conditions.append(reduce(lambda x, y: x & y, self.NSeq_conditions(dataframe)))

        if self.buy_emabounce_enabled.value:
            conditions.append(reduce(lambda x, y: x & y, self.EMABounce_conditions(dataframe)))

        if self.buy_strat3_enabled.value:
            conditions.append(reduce(lambda x, y: x & y, self.Strat3_conditions(dataframe)))

        # OR them together
        if conditions:
            dataframe.loc[reduce(lambda x, y: x | y, conditions), 'buy'] = 1

        return dataframe

    def populate_sell_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Based on TA indicators, populates the sell signal for the given dataframe
        :param dataframe: DataFrame
        :return: DataFrame with buy column

        """
        # Don't sell (have to set something in 'sell' column)
        dataframe.loc[(dataframe['close'] >= 0), 'sell'] = 0
        return dataframe
