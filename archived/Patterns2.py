
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

import Config


class Patterns2(IStrategy):
    """
    Trades based on detection of the 3 White Soldiers candlestick pattern

    How to use it?
    > python3 ./freqtrade/main.py -s Patterns2
    """
    # Buy hyperspace params:
    buy_params = {
        "buy_bb_enabled": False,
        "buy_bb_gain": 0.01,
        "buy_mfi": 19.0,
        "buy_mfi_enabled": True,
        "buy_rsi": 4.0,
        "buy_rsi_enabled": False,
        "buy_sma_enabled": True,
    }
    pattern_strength = 90
    buy_rsi = DecimalParameter(1, 50, decimals=0, default=31, space="buy")
    buy_mfi = DecimalParameter(1, 50, decimals=0, default=50, space="buy")
    buy_bb_gain = DecimalParameter(0.01, 0.10, decimals=2, default=0.02, space="buy")

    buy_bb_enabled = CategoricalParameter([True, False], default=True, space="buy")
    buy_sma_enabled = CategoricalParameter([True, False], default=True, space="buy")
    buy_rsi_enabled = CategoricalParameter([True, False], default=True, space="buy")
    buy_mfi_enabled = CategoricalParameter([True, False], default=True, space="buy")

    # # Buy hyperspace params:
    # buy_params = {
    #     "buy_CDL3INSIDE_enabled": True,
    #     "buy_CDL3LINESTRIKE_enabled": False,
    #     "buy_CDL3OUTSIDE_enabled": False,
    #     "buy_CDL3WHITESOLDIERS_enabled": False,
    #     "buy_CDLDRAGONFLYDOJI_enabled": True,
    #     "buy_CDLENGULFING_enabled": False,
    #     "buy_CDLHAMMER_enabled": True,
    #     "buy_CDLHARAMI_enabled": False,
    #     "buy_CDLINVERTEDHAMMER_enabled": True,
    #     "buy_CDLMORNINGSTAR_enabled": False,
    #     "buy_CDLPIERCING_enabled": True,
    #     "buy_CDLSPINNINGTOP_enabled": False,
    # }
    #
    buy_CDLHAMMER_enabled = True
    buy_CDLINVERTEDHAMMER_enabled = True
    buy_CDLDRAGONFLYDOJI_enabled = True
    buy_CDLPIERCING_enabled = True
    buy_CDLMORNINGSTAR_enabled = False
    buy_CDL3WHITESOLDIERS_enabled = False

    buy_CDL3LINESTRIKE_enabled = False
    buy_CDLSPINNINGTOP_enabled = False
    buy_CDLENGULFING_enabled = False
    buy_CDLHARAMI_enabled = False
    buy_CDL3OUTSIDE_enabled = False
    buy_CDL3INSIDE_enabled = True

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

        # # Stoch fast
        # stoch_fast = ta.STOCHF(dataframe)
        # dataframe['fastd'] = stoch_fast['fastd']
        # dataframe['fastk'] = stoch_fast['fastk']

        # RSI
        dataframe['rsi'] = ta.RSI(dataframe)

        # # Inverse Fisher transform on RSI, values [-1.0, 1.0] (https://goo.gl/2JGGoy)
        # rsi = 0.1 * (dataframe['rsi'] - 50)
        # dataframe['fisher_rsi'] = (numpy.exp(2 * rsi) - 1) / (numpy.exp(2 * rsi) + 1)

        # Bollinger bands
        bollinger = qtpylib.bollinger_bands(qtpylib.typical_price(dataframe), window=20, stds=2)
        dataframe['bb_lowerband'] = bollinger['lower']
        dataframe['bb_upperband'] = bollinger['lower']
        dataframe["bb_gain"] = ((dataframe["bb_upperband"] - dataframe["close"]) / dataframe["close"])

        # # EMA - Exponential Moving Average
        # dataframe['ema5'] = ta.EMA(dataframe, timeperiod=5)
        # dataframe['ema10'] = ta.EMA(dataframe, timeperiod=10)
        # dataframe['ema50'] = ta.EMA(dataframe, timeperiod=50)
        # dataframe['ema100'] = ta.EMA(dataframe, timeperiod=100)
        #
        # # SAR Parabol
        # dataframe['sar'] = ta.SAR(dataframe)

        # SMA - Simple Moving Average
        dataframe['sma'] = ta.SMA(dataframe, timeperiod=40)

        # Pattern Recognition - Bullish candlestick patterns
        # ------------------------------------
        # Hammer: values [0, 100]
        dataframe['CDLHAMMER'] = ta.CDLHAMMER(dataframe)
        # Inverted Hammer: values [0, 100]
        dataframe['CDLINVERTEDHAMMER'] = ta.CDLINVERTEDHAMMER(dataframe)
        # Dragonfly Doji: values [0, 100]
        dataframe['CDLDRAGONFLYDOJI'] = ta.CDLDRAGONFLYDOJI(dataframe)
        # Piercing Line: values [0, 100]
        dataframe['CDLPIERCING'] = ta.CDLPIERCING(dataframe) # values [0, 100]
        # Morningstar: values [0, 100]
        dataframe['CDLMORNINGSTAR'] = ta.CDLMORNINGSTAR(dataframe) # values [0, 100]
        # Three White Soldiers: values [0, 100]
        dataframe['CDL3WHITESOLDIERS'] = ta.CDL3WHITESOLDIERS(dataframe) # values [0, 100]

        # Pattern Recognition - Bearish candlestick patterns
        # ------------------------------------
        # Hanging Man: values [0, 100]
        dataframe['CDLHANGINGMAN'] = ta.CDLHANGINGMAN(dataframe)
        # Shooting Star: values [0, 100]
        dataframe['CDLSHOOTINGSTAR'] = ta.CDLSHOOTINGSTAR(dataframe)
        # Gravestone Doji: values [0, 100]
        dataframe['CDLGRAVESTONEDOJI'] = ta.CDLGRAVESTONEDOJI(dataframe)
        # Dark Cloud Cover: values [0, 100]
        dataframe['CDLDARKCLOUDCOVER'] = ta.CDLDARKCLOUDCOVER(dataframe)
        # Evening Doji Star: values [0, 100]
        dataframe['CDLEVENINGDOJISTAR'] = ta.CDLEVENINGDOJISTAR(dataframe)
        # Evening Star: values [0, 100]
        dataframe['CDLEVENINGSTAR'] = ta.CDLEVENINGSTAR(dataframe)
        #
        # # Pattern Recognition - Bullish/Bearish candlestick patterns
        # # ------------------------------------
        # Three Line Strike: values [0, -100, 100]
        dataframe['CDL3LINESTRIKE'] = ta.CDL3LINESTRIKE(dataframe)
        # Spinning Top: values [0, -100, 100]
        dataframe['CDLSPINNINGTOP'] = ta.CDLSPINNINGTOP(dataframe) # values [0, -100, 100]
        # Engulfing: values [0, -100, 100]
        dataframe['CDLENGULFING'] = ta.CDLENGULFING(dataframe) # values [0, -100, 100]
        # Harami: values [0, -100, 100]
        dataframe['CDLHARAMI'] = ta.CDLHARAMI(dataframe) # values [0, -100, 100]
        # Three Outside Up/Down: values [0, -100, 100]
        dataframe['CDL3OUTSIDE'] = ta.CDL3OUTSIDE(dataframe) # values [0, -100, 100]
        # Three Inside Up/Down: values [0, -100, 100]
        dataframe['CDL3INSIDE'] = ta.CDL3INSIDE(dataframe) # values [0, -100, 100]


        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Based on TA indicators, populates the buy signal for the given dataframe
        :param dataframe: DataFrame
        :return: DataFrame with buy column
        """

        gconditions = []

        # GUARDS AND TRENDS
        if self.buy_rsi_enabled.value:
            gconditions.append(
                (dataframe['rsi'] <= self.buy_rsi.value) &
                (dataframe['rsi'] > 0)
            )

        if self.buy_sma_enabled.value:
            gconditions.append(dataframe['close'] < dataframe['sma'])

        if self.buy_mfi_enabled.value:
            gconditions.append(dataframe['mfi'] <= self.buy_mfi.value)

        # potential gain > goal
        if self.buy_bb_enabled.value:
            gconditions.append(dataframe['bb_gain'] >= self.buy_bb_gain.value)

        tconditions = []

        # look for 3 White Soldiers pattern
        if self.buy_CDL3WHITESOLDIERS_enabled:
            tconditions.append(dataframe['CDL3WHITESOLDIERS'] >= self.pattern_strength)
        if self.buy_CDLMORNINGSTAR_enabled:
            tconditions.append(dataframe['CDLMORNINGSTAR'] >= self.pattern_strength)
        if self.buy_CDL3LINESTRIKE_enabled:
            tconditions.append(dataframe['CDL3LINESTRIKE'] >= self.pattern_strength)
        if self.buy_CDL3OUTSIDE_enabled:
            tconditions.append(dataframe['CDL3OUTSIDE'] >= self.pattern_strength)

        if self.buy_CDLHAMMER_enabled:
            tconditions.append(dataframe['CDLHAMMER'] >= self.pattern_strength)
        if self.buy_CDLINVERTEDHAMMER_enabled:
            tconditions.append(dataframe['CDLINVERTEDHAMMER'] >= self.pattern_strength)
        if self.buy_CDLDRAGONFLYDOJI_enabled:
            tconditions.append(dataframe['CDLDRAGONFLYDOJI'] >= self.pattern_strength)
        if self.buy_CDLPIERCING_enabled:
            tconditions.append(dataframe['CDLPIERCING'] >= self.pattern_strength)

        if self.buy_CDLSPINNINGTOP_enabled:
            tconditions.append(dataframe['CDLSPINNINGTOP'] >= self.pattern_strength)
        if self.buy_CDLENGULFING_enabled:
            tconditions.append(dataframe['CDLENGULFING'] >= self.pattern_strength)
        if self.buy_CDLHARAMI_enabled:
            tconditions.append(dataframe['CDLHARAMI'] >= self.pattern_strength)
        if self.buy_CDL3INSIDE_enabled:
            tconditions.append(dataframe['CDL3INSIDE'] >= self.pattern_strength)

        # build the dataframe using the guard and pattern results
        gr = False
        pr = False
        if gconditions:
            gr = reduce(lambda x, y: x & y, gconditions)
        if tconditions:
            pr = reduce(lambda x, y: x | y, tconditions)

        dataframe.loc[(gr & pr), 'buy'] = 1

        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Based on TA indicators, populates the sell signal for the given dataframe
        :param dataframe: DataFrame
        :return: DataFrame with buy column
        """

        conditions = []

        # don't set a sell signal (just use ROI)
        dataframe.loc[(dataframe['close'].notnull() ), 'sell'] = 0

        return dataframe