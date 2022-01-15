
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



class FBB_(IStrategy):
    """
    Simple strategy based on Inverse Fisher Transform and Bollinger Bands
    Note that this is a base class, i.e. the concrete class must derive from this

    How to use it?
    > python3 ./freqtrade/main.py -s FBB_WtdProfit
    """

    """
    Derived class must declare the following:

    buy_bb_gain = DecimalParameter(0.01, 0.20, decimals=3, default=0.070, space="buy")
    buy_fisher = DecimalParameter(-1.0, 1.0, decimals=3, default=-0.280, space="buy")
    buy_num_candles = IntParameter(0, 12, default=2, space="buy")

    Cannot declare them here because of python scoping rules (the base class vars will be used instead of the child class)
    """

    startup_candle_count = 20

    # Optimal timeframe for the strategy
    timeframe = '5m'

    # run "populate_indicators" only for new candle
    process_only_new_candles = False



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

        # # ADX
        # dataframe['adx'] = ta.ADX(dataframe)
        #
        # # Plus Directional Indicator / Movement
        # dataframe['dm_plus'] = ta.PLUS_DM(dataframe)
        # dataframe['di_plus'] = ta.PLUS_DI(dataframe)
        #
        # # Minus Directional Indicator / Movement
        # dataframe['dm_minus'] = ta.MINUS_DM(dataframe)
        # dataframe['di_minus'] = ta.MINUS_DI(dataframe)
        # dataframe['dm_delta'] = dataframe['dm_plus'] - dataframe['dm_minus']
        # dataframe['di_delta'] = dataframe['di_plus'] - dataframe['di_minus']
        #
        # # MFI
        # dataframe['mfi'] = ta.MFI(dataframe)

        # MACD
        macd = ta.MACD(dataframe)
        dataframe['macd'] = macd['macd']
        dataframe['macdsignal'] = macd['macdsignal']
        dataframe['macdhist'] = macd['macdhist']

        # # Stoch fast
        # stoch_fast = ta.STOCHF(dataframe)
        # dataframe['fastd'] = stoch_fast['fastd']
        # dataframe['fastk'] = stoch_fast['fastk']

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

        return dataframe

    def populate_buy_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Based on TA indicators, populates the buy signal for the given dataframe
        :param dataframe: DataFrame
        :return: DataFrame with buy column
        """
        conditions = []

        # GUARDS AND TRENDS

        # FBB_ way (lots of triggers)
        conditions.append(dataframe['fisher_rsi'] <= self.buy_fisher.value)
        conditions.append(dataframe['bb_gain'] >= self.buy_bb_gain.value)

        # check for sequence of triggers
        if self.buy_num_candles.value > 0:
            for i in range(self.buy_num_candles.value):
                conditions.append(
                    (dataframe['fisher_rsi'].shift(i) <= self.buy_fisher.value) &
                    (dataframe['bb_gain'].shift(i) >= self.buy_bb_gain.value)
                )


        if conditions:
            dataframe.loc[reduce(lambda x, y: x & y, conditions), 'buy'] = 1

        return dataframe

    def populate_sell_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Based on TA indicators, populates the sell signal for the given dataframe
        :param dataframe: DataFrame
        :return: DataFrame with buy column
        """
        dataframe.loc[(dataframe['close'].notnull() ), 'sell'] = 0

        return dataframe