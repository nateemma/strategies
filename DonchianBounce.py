
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



class DonchianBounce(IStrategy):
    """
    Simple strategy based on Contrarian Donchian Channel Bounce from the bottom band

    How to use it?
    > python3 ./freqtrade/main.py -s DonchianBounce.py
    """

    # Hyperparameters
    buy_dc_period = IntParameter(1, 50, default=20, space="buy")
    buy_adx = DecimalParameter(1, 99, decimals=0, default=25, space="buy")
    buy_sma_enabled = CategoricalParameter([True, False], default=False, space="buy")
    buy_ema_enabled = CategoricalParameter([True, False], default=False, space="buy")
    buy_adx_enabled = CategoricalParameter([True, False], default=False, space="buy")
    buy_sar_enabled = CategoricalParameter([True, False], default=False, space="buy")

    sell_sar_enabled = CategoricalParameter([True, False], default=False, space="sell")

    # set the startup candles count to the longest average used (SMA, EMA etc)
    startup_candle_count = 200

    # The ROI, Stoploss and Trailing Stop values are typically found using hyperopt

    # ROI table:
    minimal_roi = {
        "0": 0.272,
        "37": 0.099,
        "94": 0.035,
        "126": 0
    }

    # Stoploss:
    stoploss = -0.081

    # Trailing stop:
    trailing_stop = True
    trailing_stop_positive = 0.337
    trailing_stop_positive_offset = 0.383
    trailing_only_offset_is_reached = False


    # Optimal timeframe for the strategy
    timeframe = '5m'


    # run "populate_indicators" only for new candle
    process_only_new_candles = False

    # Experimental settings (configuration will overide these if set)
    use_sell_signal = True
    sell_profit_only = True
    ignore_roi_if_buy_signal = True

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

        # Donchian Channels
        dataframe['dc_upper'] = ta.MAX(dataframe['high'], timeperiod=self.buy_dc_period.value)
        dataframe['dc_lower'] = ta.MIN(dataframe['low'], timeperiod=self.buy_dc_period.value)
        dataframe['dc_mid'] = ((dataframe['dc_upper'] + dataframe['dc_lower']) / 2)

        # Fibonacci Levels (of Donchian Channel)
        dataframe['dc_dist'] = (dataframe['dc_upper']  - dataframe['dc_lower'])
        dataframe['dc_hf'] = dataframe['dc_upper'] - dataframe['dc_dist'] * 0.236 # Highest Fib
        dataframe['dc_chf'] = dataframe['dc_upper'] - dataframe['dc_dist'] * 0.382 # Centre High Fib
        dataframe['dc_clf'] = dataframe['dc_upper'] - dataframe['dc_dist'] * 0.618 # Centre Low Fib
        dataframe['dc_lf'] = dataframe['dc_upper'] - dataframe['dc_dist'] * 0.764 # Low Fib

        #print("\nupper: ", dataframe['dc_upper'])
        #print("\nlower: ", dataframe['dc_lower'])
        #print("\nmid: ", dataframe['dc_mid'])

        # ADX
        dataframe['adx'] = ta.ADX(dataframe)
        dataframe['dm_plus'] = ta.PLUS_DM(dataframe)
        dataframe['dm_minus'] = ta.MINUS_DM(dataframe)

        # MFI
        dataframe['mfi'] = ta.MFI(dataframe)

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

        # EMA - Exponential Moving Average
        dataframe['ema5'] = ta.EMA(dataframe, timeperiod=5)
        dataframe['ema10'] = ta.EMA(dataframe, timeperiod=10)
        dataframe['ema50'] = ta.EMA(dataframe, timeperiod=50)
        dataframe['ema100'] = ta.EMA(dataframe, timeperiod=100)

        # SAR Parabolic
        dataframe['sar'] = ta.SAR(dataframe)

        # SMA - Simple Moving Average
        dataframe['sma'] = ta.SMA(dataframe, timeperiod=200)
        #print("\nSMA: ", dataframe['sma'])

        return dataframe

    def populate_buy_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        conditions = []
        # GUARDS AND TRENDS

        # check that volume is not 0 (can happen in testing, or if there are issues with exchange data)
        # conditions.append(dataframe['volume'] > 0)

        # during back testing, data can be undefined, so check
        conditions.append(dataframe['dc_hf'].notnull())

        if self.buy_sar_enabled.value:
            conditions.append(dataframe['sar'].notnull())
            conditions.append(dataframe['close'] < dataframe['sar'])

        if self.buy_sma_enabled.value:
            conditions.append(dataframe['sma'].notnull())
            conditions.append(dataframe['close'] > dataframe['sma'])

        if self.buy_ema_enabled.value:
            conditions.append(dataframe['ema50'].notnull())
            conditions.append(dataframe['close'] > dataframe['ema50'])

        # ADX with DM+ > DM- indicates uptrend
        if self.buy_adx_enabled.value:
            conditions.append(
                (dataframe['adx'] > self.buy_adx.value) &
                (dataframe['dm_plus'] >= dataframe['dm_minus'])
            )

        # TRIGGERS
        # closing price above SAR
        #conditions.append(dataframe['sar'] < dataframe['close'])

        # price crosses or jumps above lower band, green candle
        conditions.append(
            (dataframe['dc_lf'].notnull()) &
            (dataframe['close'] >= dataframe['open']) &
            (
                (qtpylib.crossed_above(dataframe['close'], dataframe['dc_lf'])) |
                (
                        (dataframe['close'] >= dataframe['dc_lf']) &
                        (dataframe['close'].shift(1) < dataframe['dc_lf'].shift(1))
                )
            )

        )

        # build the dataframe using the conditions
        if conditions:
            dataframe.loc[
                reduce(lambda x, y: x & y, conditions),
                'buy'] = 1

        return dataframe

    def populate_sell_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Based on TA indicators, populates the sell signal for the given dataframe
        :param dataframe: DataFrame
        :return: DataFrame with buy column

        """

        conditions = []

        # price crosses or jumps below high band, red candle
        conditions.append(
            (dataframe['dc_hf'].notnull()) &
            (dataframe['close'] < dataframe['open']) &
            (
                    (qtpylib.crossed_below(dataframe['close'], dataframe['dc_hf'])) |
                    (
                            (dataframe['close'] <= dataframe['dc_hf']) &
                            (dataframe['close'].shift(1) > dataframe['dc_hf'].shift(1))
                    )
            )
        )

        if self.sell_sar_enabled.value:
            conditions.append(dataframe['sar'].notnull())
            conditions.append(dataframe['close'] < dataframe['sar'])

        # build the dataframe using the conditions
        if conditions:
            dataframe.loc[
                reduce(lambda x, y: x & y, conditions),
                'sell'] = 1

        return dataframe