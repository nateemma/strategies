import numpy as np
import talib.abstract as ta
import freqtrade.vendor.qtpylib.indicators as qtpylib
import arrow

from freqtrade.strategy import (IStrategy, merge_informative_pair, stoploss_from_open,
                                IntParameter, DecimalParameter, CategoricalParameter)

from typing import Dict, List, Optional, Tuple, Union
from pandas import DataFrame, Series
from functools import reduce
from datetime import datetime, timedelta
from freqtrade.persistence import Trade

# Get rid of pandas warnings during backtesting
import pandas as pd
from freqtrade.optimize.space import Categorical, Dimension, Integer, SKDecimal, Real  # noqa


class FBB_ROI(IStrategy):
    """
    Simple strategy based on Inverse Fisher Transform and Bollinger Bands (and Williams %R)

    Note that there are no sell parameters, it just relies on the ROI/stoploss mechanism to sell

    How to use it?
    > freqtrade backtest -c <config file> --strategy-path <path to strategy> -s FBB_ROI
    """

    # NOTE: hyperspace parameters are in the associated .json file (<clasname>.json)
    #       Values in that file will override the default values in the variable definitions below
    #       If the .json file does not exist, you will need to run hyperopt to generate it

    # FBB_ hyperparams
    buy_bb_gain = DecimalParameter(0.01, 0.25, decimals=2, default=0.09, space='buy', load=True, optimize=True)
    buy_fisher_wr = DecimalParameter(-0.99, 0.0, decimals=2, default=-0.75, space='buy', load=True, optimize=True)
    # buy_force_fisher_wr = DecimalParameter(-0.99, -0.75, decimals=2, default=-0.99, space='buy', load=True,
    #                                        optimize=True)

    ## Trailing params

    # hard stoploss profit
    pHSL = DecimalParameter(-0.200, -0.040, default=-0.08, decimals=3, space='sell', load=True)
    # profit threshold 1, trigger point, SL_1 is used
    pPF_1 = DecimalParameter(0.008, 0.020, default=0.016, decimals=3, space='sell', load=True)
    pSL_1 = DecimalParameter(0.008, 0.020, default=0.011, decimals=3, space='sell', load=True)

    # profit threshold 2, SL_2 is used
    pPF_2 = DecimalParameter(0.040, 0.100, default=0.080, decimals=3, space='sell', load=True)
    pSL_2 = DecimalParameter(0.020, 0.070, default=0.040, decimals=3, space='sell', load=True)

    # stoploss
    use_custom_stoploss = True

    # Recommended
    use_sell_signal = True
    sell_profit_only = True
    ignore_roi_if_buy_signal = True

    # Required
    process_only_new_candles = False
    startup_candle_count = 20


    ############################################################################

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

    ############################################################################

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Adds several different TA indicators to the given DataFrame

        Performance Note: For the best performance be frugal on the number of indicators
        you are using. Let uncomment only the indicator you are using in your strategies
        or your hyperopt configuration, otherwise you will waste your memory and CPU usage.
        """
        # FBB_ indicators
        # RSI
        dataframe['rsi'] = ta.RSI(dataframe, timeperiod=14)

        # Inverse Fisher transform on RSI, values [-1.0, 1.0] (https://goo.gl/2JGGoy)
        rsi = 0.1 * (dataframe['rsi'] - 50)
        dataframe['fisher_rsi'] = (np.exp(2 * rsi) - 1) / (np.exp(2 * rsi) + 1)

        # Bollinger bands
        bollinger = qtpylib.bollinger_bands(qtpylib.typical_price(dataframe), window=20, stds=2)
        dataframe['bb_lowerband'] = bollinger['lower']
        dataframe['bb_midband'] = bollinger['mid']
        dataframe['bb_upperband'] = bollinger['upper']
        dataframe['bb_gain'] = ((dataframe['bb_upperband'] - dataframe['close']) / dataframe['close'])
        dataframe['bb_width'] = ((dataframe['bb_upperband'] - dataframe['bb_lowerband']) / dataframe['bb_midband'])

        #
        # Williams %R (scaled to match fisher_rsi)
        dataframe['wr'] = 0.02 * (williams_r(dataframe, period=14) + 50.0)

        # Combined Fisher RSI and Williams %R
        dataframe['fisher_wr'] = (dataframe['wr'] + dataframe['fisher_rsi']) / 2.0

        # the following are mostly for display/debugging
        dataframe['trigger_bb'] = np.where((dataframe['bb_gain'] >= self.buy_bb_gain.value), 1, 0)
        dataframe['trigger_fwr'] = np.where((dataframe['fisher_wr'] >= self.buy_fisher_wr.value), 1, 0)
        # dataframe['trigger_force'] = np.where((dataframe['fisher_wr'] >= self.buy_force_fisher_wr.value), 1, 0)

        return dataframe

    ############################################################################

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:

        conditions = []
        dataframe.loc[:, 'buy_tag'] = ''

        # GUARDS AND TRENDS

        fbb_cond = (
                (dataframe['fisher_wr'] <= self.buy_fisher_wr.value) &
                (qtpylib.crossed_above(dataframe['bb_gain'], self.buy_bb_gain.value))
        )

        # strong_buy_cond = (
        #     # (
        #     #     # qtpylib.crossed_above(dataframe['bb_gain'], 1.5 * self.buy_bb_gain.value) |
        #     #     qtpylib.crossed_below(dataframe['fisher_wr'], self.buy_force_fisher_wr.value)
        #     # ) &
        #     # (
        #     #     (dataframe['bb_gain'] > 0.03)  # make sure there is some potential gain
        #     # )
        #         (qtpylib.crossed_below(dataframe['close'], dataframe['bb_lowerband'])) &
        #         (dataframe['fisher_wr'] <= self.buy_force_fisher_wr.value)
        # )

        # conditions.append(fbb_cond | strong_buy_cond)
        conditions.append(fbb_cond)

        # set buy tags
        dataframe.loc[fbb_cond, 'buy_tag'] += 'fisher_bb '
        # dataframe.loc[strong_buy_cond, 'buy_tag'] += 'strong_buy '

        if conditions:
            dataframe.loc[reduce(lambda x, y: x & y, conditions), 'buy'] = 1

        return dataframe

    ############################################################################

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Based on TA indicators, populates the sell signal for the given dataframe
        :param dataframe: DataFrame
        :return: DataFrame with buy column
        """
        dataframe.loc[(dataframe['close'].notnull()), 'sell'] = 0

        return dataframe

    ############################################################################

    ## Custom Trailing stoploss ( credit to Perkmeister for this custom stoploss to help the strategy ride a green candle )
    def custom_stoploss(self, pair: str, trade: 'Trade', current_time: datetime,
                        current_rate: float, current_profit: float, **kwargs) -> float:

        # hard stoploss profit
        HSL = self.pHSL.value
        PF_1 = self.pPF_1.value
        SL_1 = self.pSL_1.value
        PF_2 = self.pPF_2.value
        SL_2 = self.pSL_2.value

        # For profits between PF_1 and PF_2 the stoploss (sl_profit) used is linearly interpolated
        # between the values of SL_1 and SL_2. For all profits above PL_2 the sl_profit value
        # rises linearly with current profit, for profits below PF_1 the hard stoploss profit is used.

        if (current_profit > PF_2):
            sl_profit = SL_2 + (current_profit - PF_2)
        elif (current_profit > PF_1):
            sl_profit = SL_1 + ((current_profit - PF_1) * (SL_2 - SL_1) / (PF_2 - PF_1))
        else:
            sl_profit = HSL

        # Only for hyperopt invalid return
        if (sl_profit >= current_profit):
            return -0.99

        return min(-0.01, max(stoploss_from_open(sl_profit, current_profit), -0.99))


############################################################################


# Williams %R
def williams_r(dataframe: DataFrame, period: int = 14) -> Series:
    '''Williams %R, or just %R, is a technical analysis oscillator showing the current closing price in relation to the high and low
        of the past N days (for a given N). It was developed by a publisher and promoter of trading materials, Larry Williams.
        Its purpose is to tell whether a stock or commodity market is trading near the high or the low, or somewhere in between,
        of its recent trading range.
        The oscillator is on a negative scale, from −100 (lowest) up to 0 (highest).
    '''

    highest_high = dataframe['high'].rolling(center=False, window=period).max()
    lowest_low = dataframe['low'].rolling(center=False, window=period).min()

    WR = Series(
        (highest_high - dataframe['close']) / (highest_high - lowest_low),
        name=f'{period} Williams %R',
    )

    return WR * -100
