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
####################################################################################
KalmanSimple - use a Kalman Filter to estimate future price movements

This is the 'simple' version, which basically removes all custom sell/stoploss logic and relies on the Kalman filter
sell signal.

Note that this necessarily requires a 'long' timeframe because predicting a short-term swing is pretty useless - by the
time a trade was executed, the estimate would be outdated. Also, updating the Kalman Filter is expensive and runs
too slowly if you update every candle.

So, I use informative pairs that match the whitelist at 1h intervals to predict movements. The downside of this
is that the strategy can only trade once every hour 
Results actually seem to be better with the longer timeframe anyway

####################################################################################
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
    process_only_new_candles = True

    ###################################

    # Strategy Specific Variable Storage

    ## Hyperopt Variables

    # Kalman Filter limits
    buy_kf_gain = DecimalParameter(0.000, 0.050, decimals=3, default=0.015, space='buy', load=True, optimize=True)
    sell_kf_loss = DecimalParameter(-0.050, 0.000, decimals=3, default=-0.005, space='sell', load=True, optimize=True)

    # Kalman Filter
    kalman_filter = KalmanFilter(transition_matrices=[1],
                                 observation_matrices=[1],
                                 initial_state_mean=0,
                                 initial_state_covariance=1,
                                 observation_covariance=1,
                                 transition_covariance=0.01)

    ###################################

    """
    Informative Pair Definitions
    """

    def informative_pairs(self):
        pairs = self.dp.current_whitelist()
        informative_pairs = [(pair, self.inf_timeframe) for pair in pairs]
        return informative_pairs

    ###################################

    """
    Indicator Definitions
    """

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        # Base pair informative timeframe indicators
        informative = self.dp.get_pair_dataframe(pair=metadata['pair'], timeframe=self.inf_timeframe)

        # Kalman filter

        # update filter (note: this is slow, which is why we run it on the slower timeframe)
        lookback_len = 6
        data = informative['close']
        self.kalman_filter = self.kalman_filter.em(data, n_iter=4)

        if 'kf_predict' not in informative:
            informative['kf_predict'] = informative['close']
        if 'kf_mean' not in informative:
            informative['kf_mean'] = informative['close']

        # predict next close
        pr_mean, pr_cov = self.kalman_filter.smooth(data)
        informative['kf_predict'] = pd.Series(pr_mean.squeeze())
        # informative['kf_predict_cov'] = np.std(pr_cov.squeeze())

        mean, cov = self.kalman_filter.filter(data)
        informative['kf_mean'] = pd.Series(mean.squeeze())
        # # informative['kf_std'] = np.std(cov.squeeze())

        '''
         data = informative['close'].tail(lookback_len)
         self.kalman_filter = self.kalman_filter.em(data, n_iter=4)
    
         if 'kf_predict' not in informative:
             informative['kf_predict'] = informative['close']
         if 'kf_mean' not in informative:
             informative['kf_mean'] = informative['close']
    
         # predict next close
         pr_mean, pr_cov = self.kalman_filter.smooth(data)
         pr_means = pd.Series(pr_mean.squeeze())
         informative['kf_predict'].iloc[-len(pr_means):] = np.array(pr_means)
    
         mean, cov = self.kalman_filter.filter(data)
         means = pd.Series(mean.squeeze())
         informative['kf_mean'].iloc[-len(means):] = np.array(means)
         '''

        # %change prediction relative to current close
        informative['kf_predict_diff'] = (informative['kf_predict'] - informative['close']) / informative['close']

        # % Difference between prediction and smoothed model
        informative['kf_delta'] = (informative['kf_predict'] - informative['kf_mean']) / informative['kf_mean']

        dataframe = merge_informative_pair(dataframe, informative, self.timeframe, self.inf_timeframe, ffill=True)

        # copy informative into main timeframe, just to make accessing easier (and to allow further manipulation)
        dataframe['kf_delta'] = dataframe[f"kf_delta_{self.inf_timeframe}"]
        dataframe['kf_predict'] = dataframe[f"kf_predict_{self.inf_timeframe}"]
        dataframe['kf_predict_diff'] = dataframe[f"kf_predict_diff_{self.inf_timeframe}"]

        # NOTE: I played with 'upscaling' the predicted data, but the results were much worse for some reason

        return dataframe


    ###################################

    """
    Buy Signal
    """


    def populate_buy_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        conditions = []
        dataframe.loc[:, 'buy_tag'] = ''

        # conditions.append(dataframe['volume'] > 0)

        # Kalman triggers
        kalman_cond = (
            qtpylib.crossed_above(dataframe['kf_predict_diff'], self.buy_kf_gain.value)
        )

        # 'latch' the result (it sometimes gets missed in live run, probably takes too long)
        latch_cond = (
                (dataframe['kf_predict_diff'].shift(1) >= self.buy_kf_gain.value) &
                (dataframe['kf_predict_diff'].shift(2) < self.buy_kf_gain.value)
        )

        delta_cond = (
            (dataframe['kf_delta'] > 0)
        )

        conditions.append(delta_cond)
        conditions.append(kalman_cond | latch_cond)

        # set buy tags
        dataframe.loc[kalman_cond, 'buy_tag'] += 'kf_buy '
        dataframe.loc[latch_cond, 'buy_tag'] += 'kf_buy2 '
        dataframe.loc[delta_cond, 'buy_tag'] += 'kf_delta '

        if conditions:
            dataframe.loc[reduce(lambda x, y: x & y, conditions), 'buy'] = 1

        return dataframe


    ###################################

    """
    Sell Signal
    """


    def populate_sell_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        conditions = []
        dataframe.loc[:, 'exit_tag'] = ''

        # Kalman triggers
        kalman_cond = (
            qtpylib.crossed_below(dataframe['kf_predict_diff'], self.sell_kf_loss.value)
        )

        # 'latch' the result (it sometimes gets missed in live run, probably takes too long)
        latch_cond = (
                (dataframe['kf_predict_diff'].shift(1) < self.sell_kf_loss.value) &
                (dataframe['kf_predict_diff'].shift(2) > self.sell_kf_loss.value)
        )

        # delta_cond = (
        #     (dataframe['kf_delta'] < 0)
        # )
        #
        # conditions.append(delta_cond)

        conditions.append(kalman_cond | latch_cond)

        # set buy tags
        dataframe.loc[kalman_cond, 'exit_tag'] += 'kf_sell '
        dataframe.loc[latch_cond, 'exit_tag'] += 'kf_sell2 '

        if conditions:
            dataframe.loc[reduce(lambda x, y: x & y, conditions), 'sell'] = 1

        return dataframe
