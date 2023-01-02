
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

#from user_data.strategies import Config
import Config

class ComboHold(IStrategy):
    """
    Combines several buy strategies, and just holds until ROI kicks in
    This version doesn't issue a sell signal, just holds until ROI or stoploss kicks in

    How to use it?
    > python3 ./freqtrade/main.py -s ComboHold
    """

    # Hyperparameters
    # Buy hyperspace params:
    buy_params = Config.strategyParameters["ComboHold"]
    print("Exchange: ", Config.exchange_name)

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
    # Organise alphabetically, so that it's easier to update based on hyperopt runs
    # parameters are defined in ComboHoldParams for easier update


    # BBBHold
    buy_bbbhold_bb_gain = Config.strategyParameters["BBBHold"]["buy_bb_gain"]
    buy_bbbhold_fisher = Config.strategyParameters["BBBHold"]["buy_fisher"]
    buy_bbbhold_fisher_enabled = Config.strategyParameters["BBBHold"]["buy_fisher_enabled"]
    buy_bbbhold_mfi = Config.strategyParameters["BBBHold"]["buy_mfi"]
    buy_bbbhold_mfi_enabled = Config.strategyParameters["BBBHold"]["buy_mfi_enabled"]

    # BigDrop
    buy_bigdrop_bb_enabled = Config.strategyParameters["BigDrop"]["buy_bb_enabled"]
    buy_bigdrop_drop = Config.strategyParameters["BigDrop"]["buy_drop"]
    buy_bigdrop_fisher = Config.strategyParameters["BigDrop"]["buy_fisher"]
    buy_bigdrop_fisher_enabled = Config.strategyParameters["BigDrop"]["buy_fisher_enabled"]
    buy_bigdrop_mfi = Config.strategyParameters["BigDrop"]["buy_mfi"]
    buy_bigdrop_mfi_enabled = Config.strategyParameters["BigDrop"]["buy_mfi_enabled"]
    buy_bigdrop_num_candles = Config.strategyParameters["BigDrop"]["buy_num_candles"]

    # BTCJump
    buy_btcjump_bb_gain = Config.strategyParameters["BTCJump"]["buy_bb_gain"]
    buy_btcjump_btc_jump = Config.strategyParameters["BTCJump"]["buy_btc_jump"]
    buy_btcjump_fisher = Config.strategyParameters["BTCJump"]["buy_fisher"]

    # BTCNDrop
    buy_btcndrop_bb_enabled = Config.strategyParameters["BTCNDrop"]["buy_bb_enabled"]
    buy_btcndrop_drop = Config.strategyParameters["BTCNDrop"]["buy_drop"]
    buy_btcndrop_fisher = Config.strategyParameters["BTCNDrop"]["buy_fisher"]
    buy_btcndrop_fisher_enabled = Config.strategyParameters["BTCNDrop"]["buy_fisher_enabled"]
    buy_btcndrop_mfi = Config.strategyParameters["BTCNDrop"]["buy_mfi"]
    buy_btcndrop_mfi_enabled = Config.strategyParameters["BTCNDrop"]["buy_mfi_enabled"]
    buy_btcndrop_num_candles = Config.strategyParameters["BTCNDrop"]["buy_num_candles"]

    # BTCNSeq
    buy_btcnseq_bb_enabled = Config.strategyParameters["BTCNSeq"]["buy_bb_enabled"]
    buy_btcnseq_bb_gain = Config.strategyParameters["BTCNSeq"]["buy_bb_gain"]
    buy_btcnseq_drop = Config.strategyParameters["BTCNSeq"]["buy_drop"]
    buy_btcnseq_fisher = Config.strategyParameters["BTCNSeq"]["buy_fisher"]
    buy_btcnseq_fisher_enabled = Config.strategyParameters["BTCNSeq"]["buy_fisher_enabled"]
    buy_btcnseq_num_candles = Config.strategyParameters["BTCNSeq"]["buy_num_candles"]

    # EMABounce
    buy_emabounce_long_period = Config.strategyParameters["EMABounce"]["buy_long_period"]
    buy_emabounce_short_period = Config.strategyParameters["EMABounce"]["buy_short_period"]
    buy_emabounce_diff = Config.strategyParameters["EMABounce"]["buy_diff"]

    # FisherBB
    buy_fisherbb_bb_gain = Config.strategyParameters["FisherBB"]["buy_bb_gain"]
    buy_fisherbb_fisher = Config.strategyParameters["FisherBB"]["buy_fisher"]

    # MACDCross
    buy_macdcross_adx = Config.strategyParameters["MACDCross"]["buy_adx"]
    buy_macdcross_adx_enabled = Config.strategyParameters["MACDCross"]["buy_adx_enabled"]
    buy_macdcross_bb_enabled = Config.strategyParameters["MACDCross"]["buy_bb_enabled"]
    buy_macdcross_bb_gain = Config.strategyParameters["MACDCross"]["buy_bb_gain"]
    buy_macdcross_dm_enabled = Config.strategyParameters["MACDCross"]["buy_dm_enabled"]
    buy_macdcross_fisher = Config.strategyParameters["MACDCross"]["buy_fisher"]
    buy_macdcross_fisher_enabled = Config.strategyParameters["MACDCross"]["buy_fisher_enabled"]
    buy_macdcross_mfi = Config.strategyParameters["MACDCross"]["buy_mfi"]
    buy_macdcross_mfi_enabled = Config.strategyParameters["MACDCross"]["buy_mfi_enabled"]
    buy_macdcross_neg_macd_enabled = Config.strategyParameters["MACDCross"]["buy_neg_macd_enabled"]
    buy_macdcross_period = Config.strategyParameters["MACDCross"]["buy_period"]
    buy_macdcross_sar_enabled = Config.strategyParameters["MACDCross"]["buy_sar_enabled"]

    # NDrop
    buy_ndrop_bb_enabled = Config.strategyParameters["NDrop"]["buy_bb_enabled"]
    buy_ndrop_drop = Config.strategyParameters["NDrop"]["buy_drop"]
    buy_ndrop_fisher = Config.strategyParameters["NDrop"]["buy_fisher"]
    buy_ndrop_fisher_enabled = Config.strategyParameters["NDrop"]["buy_fisher_enabled"]
    buy_ndrop_mfi = Config.strategyParameters["NDrop"]["buy_mfi"]
    buy_ndrop_mfi_enabled = Config.strategyParameters["NDrop"]["buy_mfi_enabled"]
    buy_ndrop_num_candles = Config.strategyParameters["NDrop"]["buy_num_candles"]

    # NSeq
    buy_nseq_bb_enabled = Config.strategyParameters["NSeq"]["buy_bb_enabled"]
    buy_nseq_drop = Config.strategyParameters["NSeq"]["buy_drop"]
    buy_nseq_fisher = Config.strategyParameters["NSeq"]["buy_fisher"]
    buy_nseq_fisher_enabled = Config.strategyParameters["NSeq"]["buy_fisher_enabled"]
    buy_nseq_mfi = Config.strategyParameters["NSeq"]["buy_mfi"]
    buy_nseq_mfi_enabled = Config.strategyParameters["NSeq"]["buy_mfi_enabled"]
    buy_nseq_num_candles = Config.strategyParameters["NSeq"]["buy_num_candles"]



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
        btc_dataframe = self.dp.get_pair_dataframe(pair=Config.informative_pair, timeframe=inf_tf)

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

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:

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

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Based on TA indicators, populates the sell signal for the given dataframe
        :param dataframe: DataFrame
        :return: DataFrame with buy column

        """
        # Don't sell (have to set something in 'sell' column)
        dataframe.loc[(dataframe['close'] >= 0), 'sell'] = 0
        return dataframe
