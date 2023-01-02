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

"""

FBB_ buy signals combined with Solipsis_V5 informative trend checks and sell logic based on Bollinger Bands

Based on: Solipsis - By @werkkrew

Credits - 
@JimmyNixx for many of the ideas used throughout as well as helping me stay motivated throughout development!
@rk for submitting many PR's that have made this strategy possible! 

I ask for nothing in return except that if you make changes which bring you greater success than what has been provided, you share those ideas back to 
the community. Also, please don't nag me with a million questions and especially don't blame me if you lose a ton of money using this.

I take no responsibility for any success or failure you have using this strategy.

VERSION: 5.2.1

NOTE: it takes a long time for hyperopt to find profitable solutions, you need at least 500 epochs if running for the "buy sell" space.
      Do not run hypropt on the roi, stoploss or trailing spaces (because that would negate the sell logic)
"""


class FBB_2(IStrategy):

    # NOTE: hyperspace parameters are in the associated .json file (<clasname>.json)
    #       Values in that file will override the default values in the variable definitions below
    #       If the .json file does not exist, you will need to run hyperopt to generate it

    # stoploss
    use_custom_stoploss = True

    # # Trailing stop:
    # if use_custom_stoploss:
    #     stoploss = -0.99
    #     trailing_stop = False
    #     trailing_stop_positive = None
    #     trailing_stop_positive_offset = 0.0
    #     trailing_only_offset_is_reached = False
    # else:
    #     stoploss = -0.10
    #     trailing_stop = True
    #     trailing_stop_positive = 0.01
    #     trailing_stop_positive_offset = 0.11
    #     trailing_only_offset_is_reached = True


    ## Buy Space Hyperopt Variables

    # FBB_ hyperparams
    buy_bb_gain = DecimalParameter(0.01, 0.50, decimals=2, default=0.09, space='buy', load=True, optimize=True)
    buy_fisher_wr = DecimalParameter(-0.99, 0.99, decimals=2, default=-0.75, space='buy', load=True, optimize=True)
    # buy_force_fisher_wr = DecimalParameter(-0.99, -0.75, decimals=2, default=-0.99, space='buy', load=True, optimize=True)

    inf_pct_adr = DecimalParameter(0.70, 0.99, default=0.80, space='buy', load=True, optimize=True)

    # BTC Informative
    xbtc_guard = CategoricalParameter(['strict', 'lazy', 'none'], default='lazy', space='buy', optimize=True)
    xbtc_base_rmi = IntParameter(20, 70, default=40, space='buy', load=True, optimize=True)

    # BTC / ETH Stake Parameters
    xtra_base_stake_rmi = IntParameter(10, 50, default=50, space='buy', load=True, optimize=True)
    xtra_base_fiat_rmi = IntParameter(30, 70, default=50, space='buy', load=True, optimize=True)

    ## Sell Space Hyperopt Variables

    sell_bb_gain = DecimalParameter(0.7, 1.3, decimals=2, default=0.8, space='sell', load=True, optimize=True)
    sell_fisher_wr = DecimalParameter(-0.99, 0.99, decimals=2, default=0.75, space='sell', load=True, optimize=True)
    sell_force_fisher_wr = DecimalParameter(0.75, 0.99, decimals=2, default=0.99, space='sell', load=True, optimize=True)

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
    startup_candle_count: int = 233
    process_only_new_candles = False

    # Strategy Specific Variable Storage
    custom_trade_info = {}
    custom_fiat = "USDT"  # Only relevant if stake is BTC or ETH
    custom_btc_inf = False  # Don't change this.

    ############################################################################

    """
    Informative Pair Definitions
    """

    def informative_pairs(self):

        '''

        # add all whitelisted pairs on informative timeframe
        pairs = self.dp.current_whitelist()
        informative_pairs = [(pair, self.inf_timeframe) for pair in pairs]

        # add extra informative pairs if the stake is BTC or ETH
        if self.config['stake_currency'] in ('BTC', 'ETH'):
            for pair in pairs:
                coin, stake = pair.split('/')
                coin_fiat = f"{coin}/{self.custom_fiat}"
                informative_pairs += [(coin_fiat, self.timeframe)]

            stake_fiat = f"{self.config['stake_currency']}/{self.custom_fiat}"
            informative_pairs += [(stake_fiat, self.timeframe)]
        # if BTC/STAKE is not in whitelist, add it as an informative pair on both timeframes
        else:
            btc_stake = f"BTC/{self.config['stake_currency']}"
            if not btc_stake in pairs:
                informative_pairs += [(btc_stake, self.timeframe)]
        '''

        btc_stake = f"BTC/{self.config['stake_currency']}"
        return [(btc_stake, self.timeframe)]

        return informative_pairs

    ############################################################################

    """
    Indicator Definitions
    """

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        if not metadata['pair'] in self.custom_trade_info:
            self.custom_trade_info[metadata['pair']] = {}
            if not 'had-trend' in self.custom_trade_info[metadata["pair"]]:
                self.custom_trade_info[metadata['pair']]['had-trend'] = False

        ## Base Timeframe / Pair

        # Kaufmann Adaptive Moving Average
        dataframe['kama'] = ta.KAMA(dataframe, length=233)

        # RMI: https://www.tradingview.com/script/kwIt9OgQ-Relative-Momentum-Index/
        dataframe['rmi'] = cta.RMI(dataframe, length=24, mom=5)

        # Momentum Pinball: https://www.tradingview.com/script/fBpVB1ez-Momentum-Pinball-Indicator/
        dataframe['roc-mp'] = ta.ROC(dataframe, timeperiod=1)
        dataframe['mp'] = ta.RSI(dataframe['roc-mp'], timeperiod=3)

        # MA Streak: https://www.tradingview.com/script/Yq1z7cIv-MA-Streak-Can-Show-When-a-Run-Is-Getting-Long-in-the-Tooth/
        dataframe['mastreak'] = cta.mastreak(dataframe, period=4)

        # Percent Change Channel: https://www.tradingview.com/script/6wwAWXA1-MA-Streak-Change-Channel/
        upper, mid, lower = cta.pcc(dataframe, period=40, mult=3)
        dataframe['pcc-lowerband'] = lower
        dataframe['pcc-upperband'] = upper

        lookup_idxs = dataframe.index.values - (abs(dataframe['mastreak'].values) + 1)
        valid_lookups = lookup_idxs >= 0
        dataframe['sbc'] = np.nan
        dataframe.loc[valid_lookups, 'sbc'] = dataframe['close'].to_numpy()[lookup_idxs[valid_lookups].astype(int)]

        dataframe['streak-roc'] = 100 * (dataframe['close'] - dataframe['sbc']) / dataframe['sbc']

        # Trends, Peaks and Crosses
        dataframe['candle-up'] = np.where(dataframe['close'] >= dataframe['open'], 1, 0)
        dataframe['candle-up-trend'] = np.where(dataframe['candle-up'].rolling(5).sum() >= 3, 1, 0)

        dataframe['rmi-up'] = np.where(dataframe['rmi'] >= dataframe['rmi'].shift(), 1, 0)
        dataframe['rmi-up-trend'] = np.where(dataframe['rmi-up'].rolling(5).sum() >= 3, 1, 0)

        dataframe['rmi-dn'] = np.where(dataframe['rmi'] <= dataframe['rmi'].shift(), 1, 0)
        dataframe['rmi-dn-count'] = dataframe['rmi-dn'].rolling(8).sum()

        dataframe['streak-bo'] = np.where(dataframe['streak-roc'] < dataframe['pcc-lowerband'], 1, 0)
        dataframe['streak-bo-count'] = dataframe['streak-bo'].rolling(8).sum()

        # Indicators used only for ROI and Custom Stoploss
        ssldown, sslup = cta.SSLChannels_ATR(dataframe, length=21)
        dataframe['sroc'] = cta.SROC(dataframe, roclen=21, emalen=13, smooth=21)
        dataframe['ssl-dir'] = np.where(sslup > ssldown, 'up', 'down')

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
        dataframe["bb_gain"] = ((dataframe["bb_upperband"] - dataframe["close"]) / dataframe["close"])

        # Williams %R
        dataframe['wr'] = 0.02 * (williams_r(dataframe, period=14) + 50.0)

        # Combined Fisher RSI and Williams %R
        dataframe['fisher_wr'] = (dataframe['wr'] + dataframe['fisher_rsi']) / 2.0

        # Base pair informative timeframe indicators
        informative = self.dp.get_pair_dataframe(pair=metadata['pair'], timeframe=self.inf_timeframe)

        # Get the "average day range" between the 1d high and 1d low to set up guards
        informative['1d-high'] = informative['close'].rolling(24).max()
        informative['1d-low'] = informative['close'].rolling(24).min()
        informative['adr'] = informative['1d-high'] - informative['1d-low']

        dataframe = merge_informative_pair(dataframe, informative, self.timeframe, self.inf_timeframe, ffill=True)

        # Other stake specific informative indicators
        # e.g if stake is BTC and current coin is XLM (pair: XLM/BTC)
        if self.config['stake_currency'] in ('BTC', 'ETH'):
            coin, stake = metadata['pair'].split('/')
            fiat = self.custom_fiat
            coin_fiat = f"{coin}/{fiat}"
            stake_fiat = f"{stake}/{fiat}"

            # Informative COIN/FIAT e.g. XLM/USD - Base Timeframe
            coin_fiat_tf = self.dp.get_pair_dataframe(pair=coin_fiat, timeframe=self.timeframe)
            dataframe[f"{fiat}_rmi"] = cta.RMI(coin_fiat_tf, length=55, mom=5)

            # Informative STAKE/FIAT e.g. BTC/USD - Base Timeframe
            stake_fiat_tf = self.dp.get_pair_dataframe(pair=stake_fiat, timeframe=self.timeframe)
            dataframe[f"{stake}_rmi"] = cta.RMI(stake_fiat_tf, length=55, mom=5)

        # Informatives for BTC/STAKE if not in whitelist
        else:
            pairs = self.dp.current_whitelist()
            btc_stake = f"BTC/{self.config['stake_currency']}"
            if not btc_stake in pairs:
                self.custom_btc_inf = True
                # BTC/STAKE - Base Timeframe
                btc_stake_tf = self.dp.get_pair_dataframe(pair=btc_stake, timeframe=self.timeframe)
                dataframe['BTC_rmi'] = cta.RMI(btc_stake_tf, length=55, mom=5)
                dataframe['BTC_close'] = btc_stake_tf['close']
                dataframe['BTC_kama'] = ta.KAMA(btc_stake_tf, length=144)

        df = dataframe.copy() # remove fragmentation
        return df

    ############################################################################

    """
    Buy Signal
    """

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        conditions = []
        dataframe.loc[:, 'buy_tag'] = ''

        conditions.append(dataframe['volume'] > 0)

        # Informative Timeframe Guards
        inf_cond = (
            (dataframe['close'] <= dataframe[f"1d-low_{self.inf_timeframe}"] +
             (self.inf_pct_adr.value * dataframe[f"adr_{self.inf_timeframe}"])
             )
        )
        conditions.append(inf_cond)


        # FBB_ triggers
        fbb_cond = (
                (dataframe['fisher_wr'] <= self.buy_fisher_wr.value) &
                (qtpylib.crossed_above(dataframe['bb_gain'], self.buy_bb_gain.value))
        )

        # strong_buy_cond = (
        #         (
        #                 qtpylib.crossed_above(dataframe['bb_gain'], 1.5 * self.buy_bb_gain.value) |
        #                 qtpylib.crossed_below(dataframe['fisher_wr'], self.buy_force_fisher_wr.value)
        #         ) &
        #         (
        #             (dataframe['bb_gain'] > 0.02)  # make sure there is some potential gain
        #         )
        # )

        conditions.append(fbb_cond)
        # conditions.append(fbb_cond | strong_buy_cond)

        # set buy tags
        dataframe.loc[fbb_cond, 'buy_tag'] += 'fisher_bb '
        # dataframe.loc[strong_buy_cond, 'buy_tag'] += 'strong_buy '
        # dataframe.loc[inf_cond, 'buy_tag'] += 'informative ' # always to true on buy, so omit


        # Additional informative
        if self.custom_btc_inf:
            if self.xbtc_guard.value == 'strict':
                xbtc_strict_cond = (
                        (dataframe['BTC_rmi'] > self.xbtc_base_rmi.value) &
                        (dataframe['BTC_close'] > dataframe['BTC_kama'])
                )
                conditions.append(xbtc_strict_cond)
                dataframe.loc[xbtc_strict_cond, 'buy_tag'] += 'xbtc_strict '

            if self.xbtc_guard.value == 'lazy':
                xbtc_lazy_cond = (
                        (dataframe['close'] > dataframe['kama']) |
                        (
                                (dataframe['BTC_rmi'] > self.xbtc_base_rmi.value) &
                                (dataframe['BTC_close'] > dataframe['BTC_kama'])
                        )
                )
                conditions.append(xbtc_lazy_cond)
                dataframe.loc[xbtc_lazy_cond, 'buy_tag'] += 'xbtc_strict '

        if conditions:
            dataframe.loc[
                reduce(lambda x, y: x & y, conditions),
                'buy'] = 1

        return dataframe


    """
    Sell Signal
    """

    ############################################################################

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:

        conditions = []
        dataframe.loc[:, 'exit_tag'] = ''

        # FBB_ triggers
        fbb_cond = (
            # Fisher RSI
                (dataframe['fisher_wr'] > self.sell_fisher_wr.value) &

                # Bollinger Band
                (dataframe['close'] >= (dataframe['bb_upperband'] * self.sell_bb_gain.value))
        )

        strong_sell_cond = (
            qtpylib.crossed_above(dataframe['fisher_wr'], self.sell_force_fisher_wr.value) #&
            # (dataframe['close'] > dataframe['bb_upperband'] * self.sell_bb_gain.value)
        )

        conditions.append(fbb_cond | strong_sell_cond)

        # set exit tags
        dataframe.loc[fbb_cond, 'exit_tag'] += 'fisher_bb '
        dataframe.loc[strong_sell_cond, 'exit_tag'] += 'strong_sell '

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

        return min(-0.01, max(stoploss_from_open(sl_profit, current_profit), -0.99))

   ############################################################################


# Williams %R
def williams_r(dataframe: DataFrame, period: int = 14) -> Series:
    """Williams %R, or just %R, is a technical analysis oscillator showing the current closing price in relation to the high and low
        of the past N days (for a given N). It was developed by a publisher and promoter of trading materials, Larry Williams.
        Its purpose is to tell whether a stock or commodity market is trading near the high or the low, or somewhere in between,
        of its recent trading range.
        The oscillator is on a negative scale, from −100 (lowest) up to 0 (highest).
    """

    highest_high = dataframe["high"].rolling(center=False, window=period).max()
    lowest_low = dataframe["low"].rolling(center=False, window=period).min()

    WR = Series(
        (highest_high - dataframe["close"]) / (highest_high - lowest_low),
        name=f"{period} Williams %R",
        )

    return WR * -100