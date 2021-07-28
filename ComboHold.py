
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
from freqtrade.optimize.space import Categorical, Dimension, Integer, SKDecimal, Real  # noqa
from freqtrade.strategy.strategy_helper import merge_informative_pair

from user_data.strategies import Config

class ComboHold(IStrategy):
    """
    Combines several buy strategies, and just holds until ROI kicks in
    This version doesn't issue a sell signal, just holds until ROI or stoploss kicks in

    How to use it?
    > python3 ./freqtrade/main.py -s ComboHold
    """

    # Hyperparameters
    # Buy hyperspace params:
    buy_params = {
        "buy_bbbhold_enabled": True,
        "buy_bigdrop_enabled": True,
        "buy_btcjump_enabled": True,
        "buy_btcndrop_enabled": True,
        "buy_btcnseq_enabled": True,
        "buy_emabounce_enabled": False,
        "buy_macdcross_enabled": True,
        "buy_fisherbb_enabled": True,
        "buy_ndrop_enabled": True,
        "buy_nseq_enabled": False,
    }

    buy_bbbhold_enabled = CategoricalParameter([True, False], default=True, space="buy")
    buy_bigdrop_enabled = CategoricalParameter([True, False], default=False, space="buy")
    buy_btcjump_enabled = CategoricalParameter([True, False], default=False, space="buy")
    buy_btcndrop_enabled = CategoricalParameter([True, False], default=False, space="buy")
    buy_btcnseq_enabled = CategoricalParameter([True, False], default=False, space="buy")
    buy_emabounce_enabled = CategoricalParameter([True, False], default=True, space="buy")
    buy_macdcross_enabled = CategoricalParameter([True, False], default=False, space="buy")
    buy_fisherbb_enabled = CategoricalParameter([True, False], default=True, space="buy")
    buy_ndrop_enabled = CategoricalParameter([True, False], default=True, space="buy")
    buy_nseq_enabled = CategoricalParameter([True, False], default=False, space="buy")

    # The following were hyperparameters for each individual strategy, but we are just making them constants here
    # This is so that we can run hyperopt and stand a chance of getting decent results (otherwise the search space
    # is too large)
    # NOTE: the strategies run differently in combination, so sometimes the best settings for the individual
    #       strategy do not work well when used here

    # NDrop parameters:
    buy_ndrop_bb_enabled = False
    buy_ndrop_drop = 0.029
    buy_ndrop_fisher = -0.23
    buy_ndrop_fisher_enabled = True
    buy_ndrop_mfi = 38.0
    buy_ndrop_mfi_enabled = False
    buy_ndrop_num_candles = 3

    # These seem to work best in combination, not necessarily in the individual strategy:
    # buy_ndrop_bb_enabled = False
    # buy_ndrop_drop = 0.023
    # buy_ndrop_fisher = -0.81
    # buy_ndrop_fisher_enabled = False
    # buy_ndrop_mfi = 16.0
    # buy_ndrop_mfi_enabled = True
    # buy_ndrop_num_candles = 7

    # NSeq parameters:
    """
        buy_params = {
        "buy_bb_enabled": False,
        "buy_drop": 0.026,
        "buy_fisher": -0.68,
        "buy_fisher_enabled": False,
        "buy_mfi": 35.0,
        "buy_mfi_enabled": False,
        "buy_num_candles": 3,
    }
    """
    buy_nseq_num_candles = 3
    buy_nseq_drop = 0.021
    buy_nseq_fisher = -0.5
    buy_nseq_mfi = 39.0
    buy_nseq_fisher_enabled = False
    buy_nseq_mfi_enabled = False
    buy_nseq_bb_enabled = False

    # EMABounce parameters
    buy_emabounce_long_period = 50
    buy_emabounce_short_period = 10
    buy_emabounce_diff = 0.065

    # MACDCross Parameters
    """
        buy_params = {
        "buy_adx": 1.0,
        "buy_adx_enabled": False,
        "buy_bb_enabled": True,
        "buy_bb_gain": 0.04,
        "buy_dm_enabled": True,
        "buy_fisher": 0.18,
        "buy_fisher_enabled": True,
        "buy_mfi": 79.0,
        "buy_mfi_enabled": False,
        "buy_neg_macd_enabled": True,
        "buy_period": 16,
        "buy_sar_enabled": False,
    }
    """
    
    buy_macdcross_adx = 1.0
    buy_macdcross_adx_enabled = False
    buy_macdcross_bb_enabled = True
    buy_macdcross_bb_gain = 0.04
    buy_macdcross_dm_enabled = True
    buy_macdcross_fisher = 0.18
    buy_macdcross_fisher_enabled = True
    buy_macdcross_mfi = 79.0
    buy_macdcross_mfi_enabled = False
    buy_macdcross_neg_macd_enabled = True
    buy_macdcross_period = 16
    buy_macdcross_sar_enabled = False

    # BigDrop parameters:
    """
        buy_params = {
        "buy_bb_enabled": False,
        "buy_drop": 0.038,
        "buy_fisher": 0.12,
        "buy_fisher_enabled": True,
        "buy_mfi": 14.0,
        "buy_mfi_enabled": False,
        "buy_num_candles": 8,
    }
    """
    buy_bigdrop_bb_enabled = False
    buy_bigdrop_drop = 0.06
    buy_bigdrop_fisher = -0.23
    buy_bigdrop_fisher_enabled = True
    buy_bigdrop_mfi = 31.0
    buy_bigdrop_mfi_enabled = False
    buy_bigdrop_num_candles = 9

    # FisherBB parameters:

    buy_fisherbb_bb_gain = 0.09
    buy_fisherbb_fisher = -0.01

    # BBBHold parameters:
    """
        buy_params = {
        "buy_bb_gain": 0.06,
        "buy_fisher": 0.52,
        "buy_fisher_enabled": True,
        "buy_mfi": 28.0,
        "buy_mfi_enabled": False,
    }
    """
    buy_bbbhold_bb_gain = 0.06
    buy_bbbhold_fisher = 0.52
    buy_bbbhold_fisher_enabled = True
    buy_bbbhold_mfi = 28.0
    buy_bbbhold_mfi_enabled = False

    # BTCNDrop Parameters
    """
    buy_params = {

    }
    """
    buy_btcndrop_bb_enabled = False
    buy_btcndrop_drop = 0.14
    buy_btcndrop_fisher = -0.02
    buy_btcndrop_fisher_enabled = False
    buy_btcndrop_mfi = 11.0
    buy_btcndrop_mfi_enabled = False
    buy_btcndrop_num_candles = 3

    # BTCNSeq Parameters
    """
    buy_params = {
        "buy_bb_enabled": True,
        "buy_bb_gain": 0.08,
        "buy_drop": 0.016,
        "buy_fisher": -0.06,
        "buy_fisher_enabled": True,
        "buy_num_candles": 3,
    }
    """
    buy_btcnseq_bb_enabled = True
    buy_btcnseq_bb_gain = 0.08
    buy_btcnseq_drop = 0.016
    buy_btcnseq_fisher = -0.06
    buy_btcnseq_fisher_enabled = True
    buy_btcnseq_num_candles = 3


    # BTCJump Parameters
    """
    buy_params = {
        "buy_bb_gain": 0.09,
        "buy_btc_jump": 0.005,
        "buy_fisher": -0.12,
    }
    """
    buy_btcjump_bb_gain = 0.09
    buy_btcjump_btc_jump = 0.005
    buy_btcjump_fisher = -0.12

    # Strategy Configuration

    # set the startup candles count to the longest average used (EMA, EMA etc)
    startup_candle_count = max(buy_emabounce_long_period, 20)

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

    # Define custom ROI ranges
    class HyperOpt:
        # Define a custom ROI space.
        def roi_space() -> List[Dimension]:
            return [
                Integer(10, 240, name='roi_t1'),
                Integer(10, 120, name='roi_t2'),
                Integer(10, 80, name='roi_t3'),
                SKDecimal(0.01, 0.04, decimals=3, name='roi_p1'),
                SKDecimal(0.01, 0.07, decimals=3, name='roi_p2'),
                SKDecimal(0.01, 0.20, decimals=3, name='roi_p3'),
            ]

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

        # BTC data

        # NOTE: we are applying this to the BTC/USD dataframe, not the normal dataframe (or in addition to anyway)
        if not self.dp:
            # Don't do anything if DataProvider is not available.
            return dataframe

        # get BTC dataframe
        inf_tf = '5m'
        btc_dataframe = self.dp.get_pair_dataframe(pair="BTC/USD", timeframe=inf_tf)

        # merge into main dataframe. This will create columns with a "_5m" suffix for the BTC data
        dataframe = merge_informative_pair(dataframe, btc_dataframe, self.timeframe, "5m", ffill=True)

        # BTC gain
        dataframe['btc_gain'] = (dataframe['close_5m'] - dataframe['open_5m']) / dataframe['open_5m']
        dataframe['btc_zgain'] = dataframe['btc_gain'] - self.buy_btcjump_btc_jump

        # ADX
        dataframe['adx'] = ta.ADX(dataframe)

        # Plus Directional Indicator / Movement
        dataframe['dm_plus'] = ta.PLUS_DM(dataframe)
        dataframe['di_plus'] = ta.PLUS_DI(dataframe)

        # Minus Directional Indicator / Movement
        dataframe['dm_minus'] = ta.MINUS_DM(dataframe)
        dataframe['di_minus'] = ta.MINUS_DI(dataframe)
        dataframe['dm_delta'] = dataframe['dm_plus'] - dataframe['dm_minus']
        dataframe['di_delta'] = dataframe['di_plus'] - dataframe['di_minus']

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

        dataframe['ema'] = ta.EMA(dataframe, timeperiod=self.buy_emabounce_long_period)
        dataframe['ema_short'] = ta.EMA(dataframe, timeperiod=self.buy_emabounce_short_period)
        dataframe['ema_angle'] = ta.LINEARREG_SLOPE(dataframe['ema_short'], timeperiod=3) / (2.0 * math.pi)
        dataframe['ema_diff'] = (((dataframe['ema'] - dataframe['close']) /
                                      dataframe['ema'])) \
                                    - self.buy_emabounce_diff

        # SAR Parabol
        dataframe['sar'] = ta.SAR(dataframe)

        return dataframe

    def NDrop_conditions(self, dataframe: DataFrame):
        conditions = []

        # GUARDS AND TRENDS
        if self.buy_ndrop_mfi_enabled:
            conditions.append(dataframe['mfi'] <= self.buy_ndrop_mfi)

        if self.buy_ndrop_fisher_enabled:
            conditions.append(dataframe['fisher_rsi'] < self.buy_ndrop_fisher)

        if self.buy_ndrop_bb_enabled:
            conditions.append(dataframe['close'] <= dataframe['bb_lowerband'])

        # TRIGGERS

        # N red candles
        if self.buy_ndrop_num_candles >= 1:
            for i in range(self.buy_ndrop_num_candles):
                conditions.append(dataframe['close'].shift(i) <= dataframe['open'].shift(i))

        # big enough drop?
        conditions.append(
            (((dataframe['open'].shift(self.buy_ndrop_num_candles - 1) - dataframe['close']) /
              dataframe['open'].shift(self.buy_ndrop_num_candles - 1)) >= self.buy_ndrop_drop)
        )
        return conditions


    def NSeq_conditions(self, dataframe: DataFrame):
        conditions = []
        # GUARDS AND TRENDS
        if self.buy_nseq_mfi_enabled:
            conditions.append(dataframe['mfi'] <= self.buy_nseq_mfi)

        if self.buy_nseq_fisher_enabled:
            conditions.append(dataframe['fisher_rsi'] < self.buy_nseq_fisher)

        if self.buy_nseq_bb_enabled:
            conditions.append(dataframe['close'] <= dataframe['bb_lowerband'])

        # TRIGGERS

        # current candle is green
        conditions.append(dataframe['close'] >= dataframe['open'])

        # N red candles
        if self.buy_nseq_num_candles >= 1:
            for i in range(self.buy_nseq_num_candles):
                conditions.append(dataframe['close'].shift(i+1) <= dataframe['open'].shift(i+1))

        # big enough drop?
        conditions.append(
            (((dataframe['open'].shift(self.buy_nseq_num_candles) - dataframe['close']) /
            dataframe['open'].shift(self.buy_nseq_num_candles)) >= self.buy_nseq_drop)
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

    def MACDCross_conditions(self, dataframe: DataFrame):
        conditions = []

        # GUARDS AND TRENDS

        if self.buy_macdcross_adx_enabled:
            conditions.append(dataframe['adx'] >= self.buy_macdcross_adx)

        if self.buy_macdcross_dm_enabled:
            conditions.append(dataframe['dm_delta'] > 0)

        if self.buy_macdcross_mfi_enabled:
            conditions.append(dataframe['mfi'] > self.buy_macdcross_mfi)

        # only buy if close is below SAR
        if self.buy_macdcross_sar_enabled:
            conditions.append(dataframe['close'] < dataframe['sar'])

        if self.buy_macdcross_fisher_enabled:
            conditions.append(dataframe['fisher_rsi'] < self.buy_macdcross_fisher)

        if self.buy_macdcross_neg_macd_enabled:
            conditions.append(dataframe['macd'] < 0.0)

        # potential gain > goal
        if self.buy_macdcross_bb_enabled:
            conditions.append(dataframe['bb_gain'] >= self.buy_macdcross_bb_gain)

        # Triggers
        conditions.append(qtpylib.crossed_above(dataframe['macd'], dataframe['macdsignal']))

        # check that volume is not 0
        conditions.append(dataframe['volume'] > 0)

        return conditions


    def BigDrop_conditions(self, dataframe: DataFrame):
        conditions = []

        # GUARDS AND TRENDS
        if self.buy_bigdrop_mfi_enabled:
            conditions.append(dataframe['mfi'] <= self.buy_bigdrop_mfi)

        if self.buy_bigdrop_fisher_enabled:
            conditions.append(dataframe['fisher_rsi'] < self.buy_bigdrop_fisher)

        if self.buy_bigdrop_bb_enabled:
            conditions.append(dataframe['close'] <= dataframe['bb_lowerband'])

        # TRIGGERS

        # big enough drop?
        conditions.append(
            (((dataframe['open'].shift(self.buy_bigdrop_num_candles - 1) - dataframe['close']) /
              dataframe['open'].shift(self.buy_bigdrop_num_candles - 1)) >= self.buy_bigdrop_drop)
        )
        return conditions


    def FisherBB_conditions(self, dataframe: DataFrame):
        conditions = []

        # GUARDS AND TRENDS

        conditions.append(dataframe['fisher_rsi'] <= self.buy_fisherbb_fisher)
        conditions.append(dataframe['bb_gain'] >= self.buy_fisherbb_bb_gain)

        # TRIGGERS
        # none for this strategy, just guards

        return conditions


    def BBBHold_conditions(self, dataframe: DataFrame):
        conditions = []

        # GUARDS AND TRENDS
        if self.buy_bbbhold_mfi_enabled:
            conditions.append(dataframe['mfi'] <= self.buy_bbbhold_mfi)

        if self.buy_bbbhold_fisher_enabled:
            conditions.append(dataframe['fisher_rsi'] < self.buy_bbbhold_fisher)

        # check that volume is not 0
        conditions.append(dataframe['volume'] > 0)

        # TRIGGERS
        # potential gain > goal
        conditions.append(dataframe['bb_gain'] >= self.buy_bbbhold_bb_gain)

        # current candle is green
        conditions.append(dataframe['close'] > dataframe['open'])

        # candle crosses lower BB boundary
        conditions.append(
            (dataframe['open'] < dataframe['bb_lowerband']) &
            (dataframe['close'] >= dataframe['bb_lowerband'])
        )

        return conditions


    def BTCNDrop_conditions(self, dataframe: DataFrame):
        conditions = []

        # GUARDS AND TRENDS
        if self.buy_btcndrop_mfi_enabled:
            conditions.append(dataframe['mfi'] <= self.buy_btcndrop_mfi)

        if self.buy_btcndrop_fisher_enabled:
            conditions.append(dataframe['fisher_rsi'] <= self.buy_btcndrop_fisher)

        if self.buy_btcndrop_bb_enabled:
            conditions.append(dataframe['close'] <= dataframe['bb_lowerband'])

        # TRIGGERS

        # N red candles in BTC (not the current pair)
        if self.buy_btcndrop_num_candles >= 1:
            for i in range(self.buy_btcndrop_num_candles):
                conditions.append(dataframe['close_5m'].shift(i) <= dataframe['open_5m'].shift(i))

        # big enough drop?
        conditions.append(
            (((dataframe['open_5m'].shift(self.buy_btcndrop_num_candles-1) - dataframe['close_5m']) /
            dataframe['open_5m'].shift(self.buy_btcndrop_num_candles-1)) >= self.buy_btcndrop_drop)
        )

        return conditions


    def BTCNSeq_conditions(self, dataframe: DataFrame):
        conditions = []

        # GUARDS AND TRENDS

        if self.buy_btcnseq_fisher_enabled:
            conditions.append(dataframe['fisher_rsi'] <= self.buy_btcnseq_fisher)

        if self.buy_btcnseq_bb_enabled:
            conditions.append(dataframe['bb_gain'] >= self.buy_btcnseq_bb_gain)

        # TRIGGERS


        # Green candle preceeded by N red candles in BTC (not the current pair)
        conditions.append(dataframe['close_5m'] > dataframe['open_5m'])
        if self.buy_btcnseq_num_candles >= 1:
            for i in range(self.buy_btcnseq_num_candles):
                conditions.append(dataframe['close_5m'].shift(i+1) <= dataframe['open_5m'].shift(i+1))

        # big enough drop?
        conditions.append(
            (((dataframe['open_5m'].shift(self.buy_btcnseq_num_candles-1) - dataframe['close_5m']) /
            dataframe['open_5m'].shift(self.buy_btcnseq_num_candles-1)) >= self.buy_btcnseq_drop)
        )

        return conditions


    def BTCJump_conditions(self, dataframe: DataFrame):
        conditions = []

        # GUARDS AND TRENDS
        conditions.append(dataframe['fisher_rsi'] <= self.buy_btcjump_fisher)
        conditions.append(dataframe['bb_gain'] >= self.buy_btcjump_bb_gain)

        # TRIGGERS

        # did BTC gain exceed target?
        conditions.append(qtpylib.crossed_above(dataframe['btc_zgain'], 0))

        return conditions

    def populate_buy_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:

        # build the dataframe using the conditions
        # AND together the conditions for each strategy
        conditions = []
        c = []
        if self.buy_ndrop_enabled.value:
            c = self.NDrop_conditions(dataframe)
            if c:
                conditions.append(reduce(lambda x, y: x & y, c))

        if self.buy_nseq_enabled.value:
            c = self.NSeq_conditions(dataframe)
            if c:
                conditions.append(reduce(lambda x, y: x & y, c))

        if self.buy_emabounce_enabled.value:
            c = self.EMABounce_conditions(dataframe)
            if c:
                conditions.append(reduce(lambda x, y: x & y, c))

        if self.buy_macdcross_enabled.value:
            c = self.MACDCross_conditions(dataframe)
            if c:
                conditions.append(reduce(lambda x, y: x & y, c))

        if self.buy_bigdrop_enabled.value:
            c = self.BigDrop_conditions(dataframe)
            if c:
                conditions.append(reduce(lambda x, y: x & y, c))

        if self.buy_fisherbb_enabled.value:
            c = self.FisherBB_conditions(dataframe)
            if c:
                conditions.append(reduce(lambda x, y: x & y, c))

        if self.buy_bbbhold_enabled.value:
            c = self.BBBHold_conditions(dataframe)
            if c:
                conditions.append(reduce(lambda x, y: x & y, c))

        if self.buy_btcndrop_enabled.value:
            c = self.BTCNDrop_conditions(dataframe)
            if c:
                conditions.append(reduce(lambda x, y: x & y, c))

        if self.buy_btcnseq_enabled.value:
            c = self.BTCNSeq_conditions(dataframe)
            if c:
                conditions.append(reduce(lambda x, y: x & y, c))

        if self.buy_btcjump_enabled.value:
            c = self.BTCJump_conditions(dataframe)
            if c:
                conditions.append(reduce(lambda x, y: x & y, c))


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
