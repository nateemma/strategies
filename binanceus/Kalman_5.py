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


try:
    import  simdkalman
except ImportError:
    log.error(
        "IMPORTANT - please install the import simdkalman python module which is needed for this strategy. "
        "pip install pykalman"
    )
else:
    log.info("import simdkalman successfully imported")

'''
####################################################################################
Kalman_5 - use a Kalman Filter to estimate future price movements

Version using simdkalman instead of pykalman 

This is the 'simple' version, which basically removes all custom sell/stoploss logic and relies on the Kalman filter
sell signal.

Note that this necessarily requires a 'long' timeframe because predicting a short-term swing is pretty useless - by the
time a trade was executed, the estimate would be outdated. Also, updating the Kalman Filter is expensive and runs
too slowly if you update every candle.

So, I use informative pairs that match the whitelist at 1h intervals to predict movements. The downside of this
is that the strategy can only trade once every hour 
Results actually seem to be better with the longer timeframe anyway

####################################################################################
'''


class Kalman_5(IStrategy):
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
    inf_timeframe = '5m'

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
    buy_kf_window = IntParameter(8, 164, default=32, space='buy', load=True, optimize=True)
    buy_kf_lookahead = IntParameter(0, 64, default=8, space='buy', load=True, optimize=True)

    sell_kf_diff = DecimalParameter(-0.050, 0.000, decimals=3, default=-0.01, space='sell', load=True, optimize=True)

    # Kalman Filter

    lookback_len = 8

    kalman_filter = simdkalman.KalmanFilter(
        state_transition=1.0,
        process_noise=0.1,
        observation_model=1.0,
        observation_noise=1.0
    )

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
            self.filter_list[curr_pair] = kalman_filter = simdkalman.KalmanFilter(
                state_transition=1.0,
                process_noise=0.1,
                observation_model=1.0,
                observation_noise=1.0
            )

        # set current filter (can't pass parameter to apply())
        self.kalman_filter = self.filter_list[curr_pair]

        # run the filter on a rolling window
        informative['kf_predict'] = informative['close'].rolling(window=self.buy_kf_window.value).apply(self.runKalman)

        # merge into normal timeframe
        dataframe = merge_informative_pair(dataframe, informative, self.timeframe, self.inf_timeframe, ffill=True)

        # calculate predictive indicators in shorter timeframe (not informative)

        # dataframe['kf_predict'] = ta.LINEARREG(dataframe[f"kf_predict_{self.inf_timeframe}"], timeperiod=12)
        dataframe['kf_predict'] = dataframe[f"kf_predict_{self.inf_timeframe}"]
        dataframe['kf_predict_diff'] = (dataframe['kf_predict'] - dataframe['close']) / dataframe['kf_predict']
        dataframe['kf_angle'] = ta.LINEARREG_SLOPE(dataframe['kf_predict'], timeperiod=3)

        return dataframe

    def runKalman(self, data: np.ndarray) -> np.float:

        # fit noise parameters to data with the EM algorithm (optional)
        # kalman_filter = self.kalman_filter.em(data, n_iter=self.lookback_len)
        # self.kalman_filter = kalman_filter

        # smooth and explain existing data
        smoothed = self.kalman_filter.smooth(data)
        # mean = pd.Series(smoothed.states.mean[:,0])

        # computed = self.kalman_filter.compute(data, self.buy_kf_lookahead.value)
        # print ("computed: ", computed)
        #
        # predicted = self.kalman_filter.predict(data, self.buy_kf_lookahead.value)
        # print ("predicted: ", predicted.observations.mean)

        # predict the next value 'n' steps away
        # NOTE: currently not working well for n > 1
        nsteps = self.buy_kf_lookahead.value
        length = len(data)

        x = np.linspace(0, length-1, num=length)
        y = smoothed.states.mean[:,0]

        # print("x: ", x)
        # print("y: ", y)

        f = scipy.interpolate.interp1d(x, y, fill_value='extrapolate')

        predict = f(length-1+nsteps)

        # print("Predict: ", predict)

        return predict

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
            qtpylib.crossed_below(dataframe['kf_predict_diff'], self.buy_kf_diff.value)
        )

        angle_cond = (
            (dataframe['kf_angle'] >= 0.0) &
            (dataframe['kf_angle'].shift(1) < 0.0)
        )

        conditions.append(kalman_cond)
        # conditions.append(kalman_cond & angle_cond)
        # conditions.append(kalman_cond | angle_cond)
        # conditions.append(angle_cond)

        # set buy tags
        dataframe.loc[kalman_cond, 'buy_tag'] += 'kf_buy_1 '
        dataframe.loc[angle_cond, 'buy_tag'] += 'kf_bottom '

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
            qtpylib.crossed_above(dataframe['kf_predict_diff'], self.sell_kf_diff.value)
        )

        angle_cond = (
            (dataframe['kf_angle'] <= 0.0) &
            (dataframe['kf_angle'].shift(1) > 0.0)
        )

        conditions.append(kalman_cond)
        # conditions.append(kalman_cond & angle_cond)
        # conditions.append(kalman_cond | angle_cond)
        # conditions.append(angle_cond)

        # set buy tags
        dataframe.loc[kalman_cond, 'exit_tag'] += 'kf_sell_1 '
        dataframe.loc[angle_cond, 'exit_tag'] += 'kf_top '

        if conditions:
            dataframe.loc[reduce(lambda x, y: x & y, conditions), 'sell'] = 1

        return dataframe
