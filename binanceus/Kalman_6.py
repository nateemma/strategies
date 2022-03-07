import numpy as np
import talib.abstract as ta
import freqtrade.vendor.qtpylib.indicators as qtpylib
import arrow

from freqtrade.exchange import timeframe_to_minutes
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
import scipy
from pykalman import KalmanFilter

'''
####################################################################################
Kalman_6 - use a Kalman Filter to estimate future price movements

####################################################################################
'''


class Kalman_6(IStrategy):
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
    inf_timeframe = '1h' # slow strat, don't use faster timeframe

    use_custom_stoploss = False

    # Recommended
    use_sell_signal = True
    sell_profit_only = False
    ignore_roi_if_buy_signal = True

    # Required
    startup_candle_count: int = 128
    process_only_new_candles = True

    ###################################

    # Strategy Specific Variable Storage

    ## Hyperopt Variables

    # Kalman Filter limits
    buy_kf_diff = DecimalParameter(0.000, 0.050, decimals=3, default=0.017, space='buy', load=True, optimize=True)
    # buy_kf_window = IntParameter(8, 164, default=32, space='buy', load=True, optimize=True)
    # buy_kf_lookahead = IntParameter(0, 64, default=8, space='buy', load=True, optimize=True)
    kf_window = 128
    kf_lookahead = 1

    sell_kf_diff = DecimalParameter(-0.050, 0.000, decimals=3, default=-0.01, space='sell', load=True, optimize=True)

    # Kalman Filter

    lookback_len = 8

    kalman_filter = KalmanFilter(transition_matrices=1.0,
                                 observation_matrices=1.0,
                                 initial_state_mean=0.0,
                                 initial_state_covariance=1.0,
                                 observation_covariance=1.0,
                                 transition_covariance=0.1)

    filter_list = {}

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
        curr_pair = metadata['pair']
        informative = self.dp.get_pair_dataframe(pair=curr_pair, timeframe=self.inf_timeframe)

        # Kalman filter

        # get filter for current pair

        # create if not already done
        if not curr_pair in self.filter_list:
            self.filter_list[curr_pair] = kalman_filter = KalmanFilter(
                transition_matrices=1.0,
                observation_matrices=1.0,
                initial_state_mean=0.0,
                initial_state_covariance=1.0,
                observation_covariance=1.0,
                transition_covariance=0.1
            )

        # set current filter (can't pass parameter to apply())
        self.kalman_filter = self.filter_list[curr_pair]

        # run the filter on a rolling window
        # informative['kf_model'] = informative['close'].rolling(window=self.buy_kf_window.value).apply(self.model)
        informative['kf_predict'] = informative['close'].rolling(window=self.kf_window).apply(self.predict)

        # merge into normal timeframe
        dataframe = merge_informative_pair(dataframe, informative, self.timeframe, self.inf_timeframe, ffill=True)

        # calculate predictive indicators in shorter timeframe (not informative)

        # dataframe['kf_predict'] = ta.LINEARREG(dataframe[f"kf_predict_{self.inf_timeframe}"], timeperiod=12)
        dataframe['kf_predict'] = dataframe[f"kf_predict_{self.inf_timeframe}"]
        # dataframe['kf_model'] = dataframe[f"kf_model_{self.inf_timeframe}"]
        # dataframe['kf_predict_diff'] = (dataframe['kf_predict'] - dataframe['kf_model']) / dataframe['kf_model']
        dataframe['kf_predict_diff'] = (dataframe['kf_predict'] - dataframe['close']) / dataframe['close']
        # dataframe['kf_angle'] = ta.LINEARREG_SLOPE(dataframe['kf_predict'], timeperiod=3)

        return dataframe

    def model(self, a: np.ndarray) -> np.float:
        # must return scalar, so just calculate prediction and take last value
        model = self.kalmanModel(np.array(a))
        length = len(model)
        return model[length - 1]

    def kalmanModel(self, data):
        # update filter
        self.kalman_filter = self.kalman_filter.em(np.array(data), n_iter=6)

        # run filter to get model
        mean, cov = self.kalman_filter.filter(data)
        model = mean.squeeze()

        return model

    def predict(self, a: np.ndarray) -> np.float:
        # must return scalar, so just calculate prediction and take last value
        npredict = self.kf_lookahead
        prediction = self.kalmanExtrapolation(np.array(a), npredict)
        return prediction

    def kalmanExtrapolation(self, data, n_predict):
        # update filter
        # Note: this very slow, but it does increase accuracy
        # self.kalman_filter = self.kalman_filter.em(np.array(data), n_iter=6)

        mean, cov = self.kalman_filter.smooth(data)
        predict = mean.squeeze()

        # print (predict)

        return predict[len(predict)-1]

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
            qtpylib.crossed_above(dataframe['kf_predict_diff'], self.buy_kf_diff.value)
        )

        conditions.append(kalman_cond)

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
            qtpylib.crossed_below(dataframe['kf_predict_diff'], self.sell_kf_diff.value)
        )

        conditions.append(kalman_cond)

        # set buy tags
        dataframe.loc[kalman_cond, 'exit_tag'] += 'kf_sell_1 '

        if conditions:
            dataframe.loc[reduce(lambda x, y: x & y, conditions), 'sell'] = 1

        return dataframe
