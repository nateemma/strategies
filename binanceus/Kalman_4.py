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
Kalman_4 - use a Kalman Filter to estimate future price movements


####################################################################################
"""


class Kalman_4(IStrategy):
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
    startup_candle_count: int = 48
    process_only_new_candles = True

    ###################################

    # Strategy Specific Variable Storage

    ## Hyperopt Variables

    # Kalman Filter limits
    buy_kf_gain = DecimalParameter(0.000, 0.050, decimals=3, default=0.001, space='buy', load=True, optimize=True)
    sell_kf_loss = DecimalParameter(-0.050, 0.000, decimals=3, default=-0.002, space='sell', load=True, optimize=True)

    # Kalman Filter
    # kalman_filter = KalmanFilter(transition_matrices=[1],
    #                              observation_matrices=[1],
    #                              initial_state_mean=0,
    #                              initial_state_covariance=1,
    #                              observation_covariance=1,
    #                              transition_covariance=0.01)
    kalman_filter = KalmanFilter(transition_matrices=1.0,
                                 observation_matrices=1.0,
                                 initial_state_mean=0.0,
                                 initial_state_covariance=1.0,
                                 observation_covariance=1.0,
                                 transition_covariance=0.1)

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
        informative['kf_predict'] = informative['close'].rolling(window=16).apply(self.runKalman)

        # merge informative frame
        dataframe = merge_informative_pair(dataframe, informative, self.timeframe, self.inf_timeframe, ffill=True)

        # calculate predictive indicators in shorter timeframe (not informative)

        dataframe['kf_predict'] = ta.LINEARREG(dataframe[f"kf_predict_{self.inf_timeframe}"], timeperiod=48)

        dataframe['kf_predict_diff'] = (dataframe['kf_predict'] - dataframe['close']) / dataframe['close']

        return dataframe

    def runKalman(self, data: np.ndarray) -> np.float:
        #must return scalar, so just calculate prediction and take last value

        # update filter
        self.kalman_filter = self.kalman_filter.em(np.array(data), n_iter=6)

        # get prediction
        pr_mean, pr_cov = self.kalman_filter.smooth(data)

        # extract last entry
        length = len(pr_mean)
        return pr_mean[length-1]

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

        conditions.append(kalman_cond )

        # set buy tags
        dataframe.loc[kalman_cond, 'buy_tag'] += 'kf_buy_1 '

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

        conditions.append(kalman_cond )

        # set buy tags
        dataframe.loc[kalman_cond, 'exit_tag'] += 'kf_sell_1 '

        if conditions:
            dataframe.loc[reduce(lambda x, y: x & y, conditions), 'sell'] = 1

        return dataframe
