
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


class Patterns(IStrategy):
    """
    Trades based on detection of candlestick patterns

    How to use it?
    > python3 ./freqtrade/main.py -s Patterns
    """


    pattern_strength = 90
    rsi_limit = 20
    mfi_limit = 25

    # flags to enable/disable each pattern
    buy_CDLHAMMER_enabled = CategoricalParameter([True, False], default=False, space="buy")
    buy_CDLINVERTEDHAMMER_enabled = CategoricalParameter([True, False], default=False, space="buy")
    buy_CDLDRAGONFLYDOJI_enabled = CategoricalParameter([True, False], default=False, space="buy")
    buy_CDLPIERCING_enabled = CategoricalParameter([True, False], default=False, space="buy")
    buy_CDLMORNINGSTAR_enabled = CategoricalParameter([True, False], default=False, space="buy")
    buy_CDL3WHITESOLDIERS_enabled = CategoricalParameter([True, False], default=True, space="buy")

    buy_CDL3LINESTRIKE_enabled = CategoricalParameter([True, False], default=False, space="buy")
    buy_CDLSPINNINGTOP_enabled = CategoricalParameter([True, False], default=False, space="buy")
    buy_CDLENGULFING_enabled = CategoricalParameter([True, False], default=False, space="buy")
    buy_CDLHARAMI_enabled = CategoricalParameter([True, False], default=False, space="buy")
    buy_CDL3OUTSIDE_enabled = CategoricalParameter([True, False], default=False, space="buy")
    buy_CDL3INSIDE_enabled = CategoricalParameter([True, False], default=False, space="buy")

    # Sell hyperspace params:
    sell_params = {
        "sell_CDL3INSIDE_enabled": True,
        "sell_CDL3LINESTRIKE_enabled": True,
        "sell_CDL3OUTSIDE_enabled": True,
        "sell_CDLDARKCLOUDCOVER_enabled": False,
        "sell_CDLENGULFING_enabled": False,
        "sell_CDLEVENINGDOJISTAR_enabled": True,
        "sell_CDLEVENINGSTAR_enabled": True,
        "sell_CDLGRAVESTONEDOJI_enabled": True,
        "sell_CDLHANGINGMAN_enabled": True,
        "sell_CDLHARAMI_enabled": True,
        "sell_CDLSHOOTINGSTAR_enabled": True,
        "sell_CDLSPINNINGTOP_enabled": True,
        "sell_hold_enabled": False,
    }
    sell_CDL3LINESTRIKE_enabled = CategoricalParameter([True, False], default=True, space="sell")
    sell_CDLSPINNINGTOP_enabled = CategoricalParameter([True, False], default=True, space="sell")
    sell_CDLENGULFING_enabled = CategoricalParameter([True, False], default=True, space="sell")
    sell_CDLHARAMI_enabled = CategoricalParameter([True, False], default=True, space="sell")
    sell_CDL3OUTSIDE_enabled = CategoricalParameter([True, False], default=True, space="sell")
    sell_CDL3INSIDE_enabled = CategoricalParameter([True, False], default=True, space="sell")

    sell_CDLHANGINGMAN_enabled = CategoricalParameter([True, False], default=True, space="sell")
    sell_CDLSHOOTINGSTAR_enabled = CategoricalParameter([True, False], default=True, space="sell")
    sell_CDLGRAVESTONEDOJI_enabled = CategoricalParameter([True, False], default=True, space="sell")
    sell_CDLDARKCLOUDCOVER_enabled = CategoricalParameter([True, False], default=False, space="sell")
    sell_CDLEVENINGDOJISTAR_enabled = CategoricalParameter([True, False], default=True, space="sell")
    sell_CDLEVENINGSTAR_enabled = CategoricalParameter([True, False], default=True, space="sell")

    sell_hold_enabled = CategoricalParameter([True, False], default=True, space="sell")

    if sell_hold_enabled.value:
        # ROI table:
        minimal_roi = {
            "0": 0.278,
            "39": 0.087,
            "124": 0.038,
            "135": 0
        }

        # Trailing stop:
        trailing_stop = True
        trailing_stop_positive = 0.172
        trailing_stop_positive_offset = 0.212
        trailing_only_offset_is_reached = False

        # Stoploss:
        stoploss = -0.333
    else:
        # ROI table:
        minimal_roi = {
            "0": 0.296,
            "26": 0.104,
            "36": 0.037,
            "65": 0
        }

        # Stoploss:
        stoploss = -0.284

        # Trailing stop:
        trailing_stop = True
        trailing_stop_positive = 0.183
        trailing_stop_positive_offset = 0.274
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

        # EMA - Exponential Moving Average
        dataframe['ema5'] = ta.EMA(dataframe, timeperiod=5)
        dataframe['ema10'] = ta.EMA(dataframe, timeperiod=10)
        dataframe['ema50'] = ta.EMA(dataframe, timeperiod=50)
        dataframe['ema100'] = ta.EMA(dataframe, timeperiod=100)

        # SAR Parabol
        dataframe['sar'] = ta.SAR(dataframe)

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

        # Pattern Recognition - Bullish/Bearish candlestick patterns
        # ------------------------------------
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

        # GUARDS AND TRENDS
        gconditions = []
        gconditions.append(
            (dataframe['rsi'] < self.rsi_limit) &
            (dataframe['rsi'] > 0)
        )

        gconditions.append(dataframe['close'] < dataframe['sma'])

        gconditions.append(dataframe['mfi'] <= self.mfi_limit)

        # reset array for pattern conditions
        conditions = []

        # Bullish candlestick patterns
        # ordered by strength
        if self.buy_CDL3WHITESOLDIERS_enabled.value:
            conditions.append(dataframe['CDL3WHITESOLDIERS'] >= self.pattern_strength)
        if self.buy_CDLMORNINGSTAR_enabled.value:
            conditions.append(dataframe['CDLMORNINGSTAR'] >= self.pattern_strength)
        if self.buy_CDL3LINESTRIKE_enabled.value:
            conditions.append(dataframe['CDL3LINESTRIKE'] >= self.pattern_strength)
        if self.buy_CDL3OUTSIDE_enabled.value:
            conditions.append(dataframe['CDL3OUTSIDE'] >= self.pattern_strength)

        if self.buy_CDLHAMMER_enabled.value:
            conditions.append(dataframe['CDLHAMMER'] >= self.pattern_strength)
        if self.buy_CDLINVERTEDHAMMER_enabled.value:
            conditions.append(dataframe['CDLINVERTEDHAMMER'] >= self.pattern_strength)
        if self.buy_CDLDRAGONFLYDOJI_enabled.value:
            conditions.append(dataframe['CDLDRAGONFLYDOJI'] >= self.pattern_strength)
        if self.buy_CDLPIERCING_enabled.value:
            conditions.append(dataframe['CDLPIERCING'] >= self.pattern_strength)

        if self.buy_CDLSPINNINGTOP_enabled.value:
            conditions.append(dataframe['CDLSPINNINGTOP'] >= self.pattern_strength)
        if self.buy_CDLENGULFING_enabled.value:
            conditions.append(dataframe['CDLENGULFING'] >= self.pattern_strength)
        if self.buy_CDLHARAMI_enabled.value:
            conditions.append(dataframe['CDLHARAMI'] >= self.pattern_strength)
        if self.buy_CDL3INSIDE_enabled.value:
            conditions.append(dataframe['CDL3INSIDE'] >= self.pattern_strength)

        # build the dataframe using the guard and pattern results

        # calculate intermediate result from guard condittions
        if gconditions:
            gr = reduce(lambda x, y: x & y, gconditions)
            pr = False
            if conditions:
                pr = reduce(lambda x, y: x | y, conditions)
            # dataframe.loc[(gr | pr), 'buy'] = 1
            dataframe.loc[(gr & pr), 'buy'] = 1

        else:
            if conditions:
                dataframe.loc[reduce(lambda x, y: x | y, conditions), 'buy'] = 1

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
            # Pattern Recognition - Bearish candlestick patterns
            if self.sell_CDLEVENINGSTAR_enabled.value:
                conditions.append(dataframe['CDLEVENINGSTAR'] >= self.pattern_strength)
            if self.sell_CDL3LINESTRIKE_enabled.value:
                conditions.append(dataframe['CDL3LINESTRIKE'] >= self.pattern_strength)
            if self.sell_CDLEVENINGDOJISTAR_enabled.value:
                conditions.append(dataframe['CDLEVENINGDOJISTAR'] >= self.pattern_strength)
            if self.sell_CDL3LINESTRIKE_enabled.value:
                conditions.append(dataframe['CDL3LINESTRIKE'] <= -self.pattern_strength)
            if self.sell_CDL3OUTSIDE_enabled.value:
                conditions.append(dataframe['CDL3OUTSIDE'] <= -self.pattern_strength)

            if self.sell_CDLHANGINGMAN_enabled.value:
                conditions.append(dataframe['CDLHANGINGMAN'] >= self.pattern_strength)
            if self.sell_CDLSHOOTINGSTAR_enabled.value:
                conditions.append(dataframe['CDLSHOOTINGSTAR'] >= self.pattern_strength)
            if self.sell_CDLGRAVESTONEDOJI_enabled.value:
                conditions.append(dataframe['CDLGRAVESTONEDOJI'] >= self.pattern_strength)
            if self.sell_CDLDARKCLOUDCOVER_enabled.value:
                conditions.append(dataframe['CDLDARKCLOUDCOVER'] >= self.pattern_strength)

            if self.sell_CDLSPINNINGTOP_enabled.value:
                conditions.append(dataframe['CDLSPINNINGTOP'] <= -self.pattern_strength)
            if self.sell_CDLENGULFING_enabled.value:
                conditions.append(dataframe['CDLENGULFING'] <= -self.pattern_strength)
            if self.sell_CDLHARAMI_enabled.value:
                conditions.append(dataframe['CDLHARAMI'] <= -self.pattern_strength)
            if self.sell_CDL3INSIDE_enabled.value:
                conditions.append(dataframe['CDL3INSIDE'] <= -self.pattern_strength)

            dataframe.loc[reduce(lambda x, y: x | y, conditions), 'sell'] = 1


        return dataframe