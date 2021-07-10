
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



class ADXDM(IStrategy):
    """
    Simple strategy based on ADX value and DM+/DM- crossing

    How to use it?
    > python3 ./freqtrade/main.py -s ADXDM
    """

    # Hyperparameters
    buy_adx = DecimalParameter(1, 99, decimals=0, default=25, space="buy")

    # set the startup candles count to the longest average used (SMA, EMA etc)
    startup_candle_count = 50

    # The ROI, Stoploss and Trailing Stop values are typically found using hyperopt

    # ROI table:
    minimal_roi = {
        "0": 0.25,
        "11": 0.053,
        "61": 0.04,
        "145": 0
    }

    # Stoploss:
    stoploss = -0.213

    # Trailing stop:
    trailing_stop = True
    trailing_stop_positive = 0.104
    trailing_stop_positive_offset = 0.163
    trailing_only_offset_is_reached = True


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


        # ADX
        dataframe['adx'] = ta.ADX(dataframe)
        dataframe['dm_plus'] = ta.PLUS_DM(dataframe)
        dataframe['dm_minus'] = ta.MINUS_DM(dataframe)

        # SAR Parabolic
        dataframe['sar'] = ta.SAR(dataframe)

        return dataframe

    def populate_buy_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        conditions = []
        # GUARDS AND TRENDS

        # check that volume is not 0 (can happen in testing, or if there are issues with exchange data)
        #conditions.append(dataframe['volume'] > 0)

        # during back testing, data can be undefined, so check
        conditions.append(dataframe['adx'].notnull())

        # SAR check
        #conditions.append(dataframe['sar'] < dataframe['close'])

        # TRIGGERS
        # ADX with DM+ > DM- indicates uptrend
        conditions.append(
            (dataframe['adx'] > self.buy_adx.value) &
            (qtpylib.crossed_above(dataframe['dm_plus'], dataframe['dm_minus']))
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
        # Exit long position if price crosses below lower band
        dataframe.loc[
            (
                    (dataframe['adx'].notnull()) &
                    (dataframe['adx'] > self.buy_adx.value) &
                    (qtpylib.crossed_below(dataframe['dm_plus'], dataframe['dm_minus']))
            ),
            'sell'] = 1
        return dataframe