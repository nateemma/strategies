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

from technical.indicators import hull_moving_average

pd.options.mode.chained_assignment = None  # default='warn'

# Strategy specific imports, files must reside in same folder as strategy
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent))

import custom_indicators as cta

'''

FBB_2  combined with Squeeze Momentum (adapted from the CrazyBear script on TradingView.com

'''


class FBB_2Sqz(IStrategy):

    # NOTE: hyperspace parameters are in the associated .json file (<clasname>.json)
    #       Values in that file will override the default values in the variable definitions below
    #       If the .json file does not exist, you will need to run hyperopt to generate it



    ## Buy Space Hyperopt Variables

    # FBB_ hyperparams
    buy_bb_gain = DecimalParameter(0.01, 0.20, decimals=2, default=0.09, space='buy', load=True, optimize=True)
    buy_fisher_wr = DecimalParameter(-0.99, 0.99, decimals=2, default=-0.75, space='buy', load=True, optimize=True)
    buy_force_fisher_wr = DecimalParameter(-0.99, -0.75, decimals=2, default=-0.99, space='buy', load=True, optimize=True)

    ## Sell Space Hyperopt Variables

    sell_bb_gain = DecimalParameter(0.7, 1.3, decimals=2, default=0.8, space='sell', load=True, optimize=True)
    sell_fisher_wr = DecimalParameter(0.75, 0.99, decimals=2, default=0.75, space='sell', load=True, optimize=True)
    sell_force_fisher_wr = DecimalParameter(-0.99, 0.99, decimals=2, default=0.99, space='sell', load=True, optimize=True)
    ## Trailing params

    # hard stoploss profit
    pHSL = DecimalParameter(-0.200, -0.040, default=-0.08, decimals=3, space='sell', load=True)
    # profit threshold 1, trigger point, SL_1 is used
    pPF_1 = DecimalParameter(0.008, 0.020, default=0.016, decimals=3, space='sell', load=True)
    pSL_1 = DecimalParameter(0.008, 0.020, default=0.011, decimals=3, space='sell', load=True)

    # profit threshold 2, SL_2 is used
    pPF_2 = DecimalParameter(0.040, 0.100, default=0.080, decimals=3, space='sell', load=True)
    pSL_2 = DecimalParameter(0.020, 0.070, default=0.040, decimals=3, space='sell', load=True)

    timeframe = '5m'
    inf_timeframe = '1h'

    use_custom_stoploss = True

    # Recommended
    use_sell_signal = True
    sell_profit_only = False
    ignore_roi_if_buy_signal = True

    # Required
    startup_candle_count: int = 24
    process_only_new_candles = False

    # Strategy Specific Variable Storage
    custom_trade_info = {}
    custom_fiat = 'USDT'  # Only relevant if stake is BTC or ETH
    custom_btc_inf = False  # Don't change this.

    ############################################################################

    '''
    Informative Pair Definitions
    '''

    def informative_pairs(self):
        return {}

    ############################################################################

    '''
    Indicator Definitions
    '''

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        # FBB_ indicators
        # RSI
        dataframe['rsi'] = ta.RSI(dataframe, timeperiod=14)

        # Inverse Fisher transform on RSI, values [-1.0, 1.0] (https://goo.gl/2JGGoy)
        rsi = 0.1 * (dataframe['rsi'] - 50)
        dataframe['fisher_rsi'] = (np.exp(2 * rsi) - 1) / (np.exp(2 * rsi) + 1)

        # Bollinger bands
        bollinger = qtpylib.bollinger_bands(qtpylib.typical_price(dataframe), window=20, stds=2)
        dataframe['bb_lowerband'] = bollinger['lower']
        dataframe['bb_upperband'] = bollinger['upper']
        dataframe['bb_gain'] = ((dataframe['bb_upperband'] - dataframe['close']) / dataframe['close'])
        #
        # Williams %R (scaled to match fisher_rsi)
        dataframe['wr'] = 0.02 * (williams_r(dataframe, period=14) + 50.0)

        # Combined Fisher RSI and Williams %R
        dataframe['fisher_wr'] = (dataframe['wr'] + dataframe['fisher_rsi']) / 2.0

        # for debug:
        dataframe['bb_g'] = dataframe['bb_gain'] * 10.0

        # Keltner Channel
        keltner = qtpylib.keltner_channel(dataframe)
        dataframe["kc_upper"] = keltner["upper"]
        dataframe["kc_lower"] = keltner["lower"]
        dataframe["kc_middle"] = keltner["mid"]

        # Squeeze Momentum
        dataframe['dc_upper'] = ta.MAX(dataframe['high'], timeperiod=21)
        dataframe['dc_lower'] = ta.MIN(dataframe['low'], timeperiod=21)
        dataframe['dc_mid'] = ta.TEMA(((dataframe['dc_upper'] + dataframe['dc_lower']) / 2), timeperiod=21)
        dataframe['tema'] = ta.TEMA(dataframe['close'], timeperiod=21)

        # Squeeze Indicators.
        #   'on'  means Bollinger Band lies completely within the Keltner Channel
        #   'off' means Keltner Channel lies completely within the Bollinger Band
        #   Booleans are funky with dataframes, so just do an intermediate calculation
        dataframe['sqz_upper'] = (dataframe['bb_upperband'] - dataframe["kc_upper"])
        dataframe['sqz_lower'] = (dataframe['bb_lowerband'] - dataframe["kc_lower"])
        dataframe['sqz_on'] = np.where(((dataframe['sqz_upper'] < 0) & (dataframe['sqz_lower'] > 0)), 1, 0)
        dataframe['sqz_off'] = np.where(((dataframe['sqz_upper'] > 0) & (dataframe['sqz_lower'] < 0)), 1, 0)

        dataframe['sqz_ave'] = ta.TEMA(((dataframe['dc_mid'] + dataframe['tema']) / 2), timeperiod=21)
        dataframe['sqz_delta'] = ta.TEMA((dataframe['close'] - dataframe['sqz_ave']), timeperiod=21)
        dataframe['sqz_val'] = ta.LINEARREG(dataframe['sqz_delta'], timeperiod=21)
        dataframe['sqz_slope'] = (ta.LINEARREG_SLOPE(dataframe['sqz_delta'], timeperiod=3)) * 4.0

        return dataframe

    ############################################################################

    '''
    Buy Signal
    '''

    def populate_buy_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        conditions = []
        dataframe.loc[:, 'buy_tag'] = ''

        conditions.append(dataframe['volume'] > 0)

        # FBB_ triggers
        fbb_cond = (
            # Fisher RSI
                (dataframe['fisher_wr'] <= self.buy_fisher_wr.value) &

                # Bollinger Band
                (dataframe['bb_gain'] >= self.buy_bb_gain.value)
        )

        sqz_cond = (
            # Squeeze Momentum (Trigger)
            (
                # (qtpylib.crossed_above(dataframe['sqz_val'], 0))
                #     (qtpylib.crossed_below(dataframe['sqz_slope'], 0)) &
                    (qtpylib.crossed_above(dataframe['sqz_off'], 0.5)) &
                    (dataframe['sqz_val'] < 0)
            )
        )

        strong_buy_cond = (
                (
                        qtpylib.crossed_above(dataframe['bb_gain'], 1.5 * self.buy_bb_gain.value) |
                        qtpylib.crossed_below(dataframe['fisher_wr'], self.buy_force_fisher_wr.value)
                ) &
                (
                    (dataframe['bb_gain'] > 0.02)  # make sure there is some potential gain
                )
        )

        conditions.append((fbb_cond & sqz_cond) | strong_buy_cond)

        # set buy tags
        dataframe.loc[fbb_cond, 'buy_tag'] += 'fisher_bb '
        dataframe.loc[strong_buy_cond, 'buy_tag'] += 'strong_buy '
        dataframe.loc[sqz_cond, 'buy_tag'] += 'squeeze '

        if conditions:
            dataframe.loc[reduce(lambda x, y: x & y, conditions), 'buy'] = 1

        return dataframe

    ############################################################################

    '''
    Sell Signal
    '''

    def populate_sell_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        conditions = []
        dataframe.loc[:, 'exit_tag'] = ''

        # FBB_ triggers
        fbb_cond = (
            # Fisher RSI
                (dataframe['fisher_wr'] > self.sell_fisher_wr.value) &

                # Bollinger Band
                (dataframe['close'] >= (dataframe['bb_upperband'] * self.sell_bb_gain.value))
        )

        sqz_cond = (
            # Squeeze Momentum (Trigger)
            (
                # qtpylib.crossed_below(dataframe['sqz_val'], 0)
                # (qtpylib.crossed_above(dataframe['sqz_val'], 0))
                #     (qtpylib.crossed_above(dataframe['sqz_slope'], 0)) &
                    (qtpylib.crossed_above(dataframe['sqz_on'], 0.5)) &
                    (dataframe['sqz_val'] > 0)
            )
        )

        strong_sell_cond = (
            qtpylib.crossed_above(dataframe['fisher_wr'], self.sell_force_fisher_wr.value) #&
            # (dataframe['close'] > dataframe['bb_upperband'] * self.sell_bb_gain.value)
        )

        conditions.append((fbb_cond & sqz_cond) | strong_sell_cond)

        # set exit tags
        dataframe.loc[fbb_cond, 'exit_tag'] += 'fisher_bb '
        dataframe.loc[strong_sell_cond, 'exit_tag'] += 'strong_sell '
        dataframe.loc[sqz_cond, 'exit_tag'] += 'squeeze '

        if conditions:
            dataframe.loc[
                reduce(lambda x, y: x & y, conditions),
                'sell'] = 1

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

        return max(stoploss_from_open(sl_profit, current_profit), -1)

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
