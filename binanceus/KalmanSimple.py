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

pd.options.mode.chained_assignment = None  # default='warn'

# Strategy specific imports, files must reside in same folder as strategy
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent))

import logging
import warnings

log = logging.getLogger(__name__)
# log.setLevel(logging.DEBUG)
warnings.simplefilter(action='ignore', category=pd.errors.PerformanceWarning)

import custom_indicators as cta

try:
    from pykalman import KalmanFilter
except ImportError:
    log.error(
        "IMPORTANT - please install the pykalman python module which is needed for this strategy. "
        "pip install pykalman"
    )
else:
    log.info("pykalman successfully imported")

"""

Kalman - use a Kalman Filter to estimate future price movements

This is the 'simple' version, which basically removes all custom sell/stoploss logic and relies on the Kalman filter
sell signal.

Note that this necessarily requires a 'long' timeframe because predicting a short-term swing is pretty useless - by the
time a trade was executed, the estimate would be outdated.
So, I use informative pairs that match the whitelist at 1h intervals to predict movements

Custom sell/stoploss logic shamnelessly copied from Solipsis by @werkkrew (https://github.com/werkkrew/freqtrade-strategies)

"""


class KalmanSimple(IStrategy):
    # Do *not* hyperopt for the roi and stoploss spaces

    # ROI table:
    minimal_roi = {
        "0": 100
    }

    # Stoploss:
    stoploss = -0.10

    # Trailing stop:
    trailing_stop = False
    trailing_stop_positive = None
    trailing_stop_positive_offset = 0.0
    trailing_only_offset_is_reached = False

    timeframe = '5m'
    inf_timeframe = '1h'

    use_custom_stoploss = False

    # Recommended
    use_sell_signal = True
    sell_profit_only = False
    ignore_roi_if_buy_signal = True

    # Required
    startup_candle_count: int = 40
    process_only_new_candles = False

    # Strategy Specific Variable Storage
    custom_trade_info = {}
    custom_fiat = "USD"  # Only relevant if stake is BTC or ETH
    custom_btc_inf = False  # Don't change this.

    ## Buy Space Hyperopt Variables

    # Kalman Filter limits
    buy_kf_gain = DecimalParameter(0.000, 0.050, decimals=3, default=0.015, space='buy', load=True, optimize=True)

    ## Sell Space Params are being used for both custom_stoploss and custom_sell

    sell_kf_loss = DecimalParameter(-0.050, 0.000, decimals=3, default=-0.005, space='sell', load=True, optimize=True)

    # Kalman Filter
    kalman_filter = KalmanFilter(transition_matrices=[1],
                                 observation_matrices=[1],
                                 initial_state_mean=0,
                                 initial_state_covariance=1,
                                 observation_covariance=1,
                                 transition_covariance=0.0001)

    """
    Informative Pair Definitions
    """

    def informative_pairs(self):

        pairs = self.dp.current_whitelist()
        informative_pairs = [(pair, self.inf_timeframe) for pair in pairs]
        return informative_pairs

    """
    Indicator Definitions
    """

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        # if not metadata['pair'] in self.custom_trade_info:
        #     self.custom_trade_info[metadata['pair']] = {}
        #     if not 'had-trend' in self.custom_trade_info[metadata["pair"]]:
        #         self.custom_trade_info[metadata['pair']]['had-trend'] = False
        #
        # ## Base Timeframe / Pair
        #
        # # # Kaufmann Adaptive Moving Average
        # # dataframe['kama'] = ta.KAMA(dataframe, length=233)
        #
        # # RMI: https://www.tradingview.com/script/kwIt9OgQ-Relative-Momentum-Index/
        # dataframe['rmi'] = cta.RMI(dataframe, length=24, mom=5)
        #
        # # # Momentum Pinball: https://www.tradingview.com/script/fBpVB1ez-Momentum-Pinball-Indicator/
        # # dataframe['roc-mp'] = ta.ROC(dataframe, timeperiod=1)
        # # dataframe['mp'] = ta.RSI(dataframe['roc-mp'], timeperiod=3)
        #
        # # MA Streak: https://www.tradingview.com/script/Yq1z7cIv-MA-Streak-Can-Show-When-a-Run-Is-Getting-Long-in-the-Tooth/
        # dataframe['mastreak'] = cta.mastreak(dataframe, period=4)

        # # Percent Change Channel: https://www.tradingview.com/script/6wwAWXA1-MA-Streak-Change-Channel/
        # upper, mid, lower = cta.pcc(dataframe, period=40, mult=3)
        # dataframe['pcc-lowerband'] = lower
        # dataframe['pcc-upperband'] = upper

        # lookup_idxs = dataframe.index.values - (abs(dataframe['mastreak'].values) + 1)
        # valid_lookups = lookup_idxs >= 0
        # dataframe['sbc'] = np.nan
        # dataframe.loc[valid_lookups, 'sbc'] = dataframe['close'].to_numpy()[lookup_idxs[valid_lookups].astype(int)]

        # dataframe['streak-roc'] = 100 * (dataframe['close'] - dataframe['sbc']) / dataframe['sbc']

        # # Trends, Peaks and Crosses
        # dataframe['candle-up'] = np.where(dataframe['close'] >= dataframe['open'], 1, 0)
        # dataframe['candle-up-trend'] = np.where(dataframe['candle-up'].rolling(5).sum() >= 3, 1, 0)
        #
        # dataframe['rmi-up'] = np.where(dataframe['rmi'] >= dataframe['rmi'].shift(), 1, 0)
        # dataframe['rmi-up-trend'] = np.where(dataframe['rmi-up'].rolling(5).sum() >= 3, 1, 0)
        #
        # dataframe['rmi-dn'] = np.where(dataframe['rmi'] <= dataframe['rmi'].shift(), 1, 0)
        # dataframe['rmi-dn-count'] = dataframe['rmi-dn'].rolling(8).sum()
        #
        # # dataframe['streak-bo'] = np.where(dataframe['streak-roc'] < dataframe['pcc-lowerband'], 1, 0)
        # # dataframe['streak-bo-count'] = dataframe['streak-bo'].rolling(8).sum()
        #
        # # Indicators used only for ROI and Custom Stoploss
        # ssldown, sslup = cta.SSLChannels_ATR(dataframe, length=21)
        # dataframe['sroc'] = cta.SROC(dataframe, roclen=21, emalen=13, smooth=21)
        # dataframe['ssl-dir'] = np.where(sslup > ssldown, 'up', 'down')

        # Base pair informative timeframe indicators
        informative = self.dp.get_pair_dataframe(pair=metadata['pair'], timeframe=self.inf_timeframe)

        # # Get the "average day range" between the 1d high and 1d low to set up guards
        # informative['1d-high'] = informative['close'].rolling(24).max()
        # informative['1d-low'] = informative['close'].rolling(24).min()
        # informative['adr'] = informative['1d-high'] - informative['1d-low']

        # Kalman filter

        # update filter (note: this is slow)
        self.kalman_filter = self.kalman_filter.em(informative['close'], n_iter=6)

        # current trend
        # mean, cov = self.kalman_filter.filter(informative['close'])
        # informative['kf_mean'] = mean.squeeze()
        # # informative['kf_std'] = np.std(cov.squeeze())
        # informative['kf_diff'] = (informative['kf_mean'] - informative['close']) / informative['close']

        # predict next close
        pr_mean, pr_cov = self.kalman_filter.smooth(informative['close'])
        informative['kf_predict'] = pr_mean.squeeze()
        # informative['kf_predict_cov'] = np.std(pr_cov.squeeze())
        informative['kf_predict_diff'] = (informative['kf_predict'] - informative['close']) / informative['close']
        # informative['kf_err'] = (informative['kf_predict'].shift(1) - informative['close']) / informative['close'].shift(1)

        dataframe = merge_informative_pair(dataframe, informative, self.timeframe, self.inf_timeframe, ffill=True)

        return dataframe

    """
    Buy Signal
    """

    def populate_buy_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        conditions = []
        dataframe.loc[:, 'buy_tag'] = ''

        conditions.append(dataframe['volume'] > 0)

        # Kalman triggers
        kalman_cond = (
            qtpylib.crossed_above(dataframe[f"kf_predict_diff_{self.inf_timeframe}"], self.buy_kf_gain.value)
        )

        conditions.append(kalman_cond)

        # set buy tags
        dataframe.loc[kalman_cond, 'buy_tag'] += 'kf '

        if conditions:
            dataframe.loc[reduce(lambda x, y: x & y, conditions), 'buy'] = 1

        return dataframe

    """
    Sell Signal
    """

    def populate_sell_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:

        conditions = []
        dataframe.loc[:, 'exit_tag'] = ''

        # Kalman triggers
        kalman_cond = (
            qtpylib.crossed_below(dataframe[f"kf_predict_diff_{self.inf_timeframe}"], self.sell_kf_loss.value)
        )

        conditions.append(kalman_cond)

        # set buy tags
        dataframe.loc[kalman_cond, 'exit_tag'] += 'kf '

        if conditions:
            dataframe.loc[reduce(lambda x, y: x & y, conditions), 'sell'] = 1

        return dataframe
