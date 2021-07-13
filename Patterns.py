
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

    buy_pattern_strength = IntParameter(0, 100, default=90, space="buy")
    buy_rsi = DecimalParameter(0, 50, decimals=0, default=15, space="buy")
    buy_mfi = DecimalParameter(0, 50, decimals=0, default=24, space="buy")
    buy_fisher = DecimalParameter(-1, 1, decimals=2, default=-0.28, space="buy")

    buy_rsi_enabled = CategoricalParameter([True, False], default=True, space="buy")
    buy_sma_enabled = CategoricalParameter([True, False], default=False, space="buy")
    buy_ema_enabled = CategoricalParameter([True, False], default=False, space="buy")
    buy_mfi_enabled = CategoricalParameter([True, False], default=True, space="buy")
    buy_fastd_enabled = CategoricalParameter([True, False], default=False, space="buy")
    buy_fisher_enabled = CategoricalParameter([True, False], default=False, space="buy")

    sell_hold_enabled = CategoricalParameter([True, False], default=False, space="sell")

    # ROI table:
    minimal_roi = {
        "0": 0.263,
        "39": 0.079,
        "55": 0.025,
        "166": 0
    }


    # Stoploss:
    stoploss = -0.266

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

    def populate_buy_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Based on TA indicators, populates the buy signal for the given dataframe
        :param dataframe: DataFrame
        :return: DataFrame with buy column
        """

        # GUARDS AND TRENDS
        gconditions = []
        if self.buy_rsi_enabled.value:
            gconditions.append(
                (dataframe['rsi'] < self.buy_rsi.value) &
                (dataframe['rsi'] > 0)
            )

        if self.buy_sma_enabled.value:
            gconditions.append(dataframe['close'] < dataframe['sma'])

        if self.buy_fisher_enabled.value:
            gconditions.append(dataframe['fisher_rsi'] < self.buy_fisher.value)

        if self.buy_mfi_enabled.value:
            gconditions.append(dataframe['mfi'] <= self.buy_mfi.value)

        if self.buy_ema_enabled.value:
            gconditions.append(
                (dataframe['ema50'] > dataframe['ema100']) |
                (qtpylib.crossed_above(dataframe['ema5'], dataframe['ema10']))
            )

        if self.buy_fastd_enabled.value:
            gconditions.append(
                (dataframe['fastd'] > dataframe['fastk']) &
                (dataframe['fastd'] > 0)
            )

        # reset array for pattern conditions
        conditions = []

        # Bullish candlestick patterns
        # ordered by strength
        conditions.append(dataframe['CDL3WHITESOLDIERS'] >= self.buy_pattern_strength.value)
        #conditions.append(dataframe['CDLMORNINGSTAR'] >= self.buy_pattern_strength.value)
        #conditions.append(dataframe['CDL3LINESTRIKE'] >= self.buy_pattern_strength.value)
        #conditions.append(dataframe['CDL3OUTSIDE'] >= self.buy_pattern_strength.value)

        # conditions.append(dataframe['CDLHAMMER'] >= self.buy_pattern_strength.value)
        # conditions.append(dataframe['CDLINVERTEDHAMMER'] >= self.buy_pattern_strength.value)
        # conditions.append(dataframe['CDLDRAGONFLYDOJI'] >= self.buy_pattern_strength.value)
        # conditions.append(dataframe['CDLPIERCING'] >= self.buy_pattern_strength.value)
        #
        # conditions.append(dataframe['CDLSPINNINGTOP'] >= self.buy_pattern_strength.value)
        # conditions.append(dataframe['CDLENGULFING'] >= self.buy_pattern_strength.value)
        # conditions.append(dataframe['CDLHARAMI'] >= self.buy_pattern_strength.value)
        # conditions.append(dataframe['CDL3INSIDE'] >= self.buy_pattern_strength.value)

        # build the dataframe using the guard and pattern results

        # calculate intermediate result from guard condittions
        if gconditions:
            gr = reduce(lambda x, y: x & y, gconditions)
            pr = False
            if conditions:
                pr = reduce(lambda x, y: x | y, conditions)
            dataframe.loc[(gr | pr), 'buy'] = 1
        else:
            if conditions:
                dataframe.loc[reduce(lambda x, y: x | y, conditions), 'buy'] = 1

        return dataframe

    def populate_sell_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
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
            # conditions.append(dataframe['CDLEVENINGSTAR'] >= self.buy_pattern_strength.value)
            conditions.append(dataframe['CDL3LINESTRIKE'] >= self.buy_pattern_strength.value)
            # conditions.append(dataframe['CDLEVENINGDOJISTAR'] >= self.buy_pattern_strength.value)
            # conditions.append(dataframe['CDL3LINESTRIKE'] <= -self.buy_pattern_strength.value)
            # conditions.append(dataframe['CDL3OUTSIDE'] <= -self.buy_pattern_strength.value)
 
            # conditions.append(dataframe['CDLHANGINGMAN'] >= self.buy_pattern_strength.value)
            # conditions.append(dataframe['CDLSHOOTINGSTAR'] >= self.buy_pattern_strength.value)
            conditions.append(dataframe['CDLGRAVESTONEDOJI'] >= self.buy_pattern_strength.value)
            # conditions.append(dataframe['CDLDARKCLOUDCOVER'] >= self.buy_pattern_strength.value)
            #
            conditions.append(dataframe['CDLSPINNINGTOP'] <= -self.buy_pattern_strength.value)
            conditions.append(dataframe['CDLENGULFING'] <= -self.buy_pattern_strength.value)
            # conditions.append(dataframe['CDLHARAMI'] <= -self.buy_pattern_strength.value)
            # conditions.append(dataframe['CDL3INSIDE'] <= -self.buy_pattern_strength.value)

            dataframe.loc[reduce(lambda x, y: x | y, conditions), 'sell'] = 1


        return dataframe