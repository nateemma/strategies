import numpy as np
from enum import Enum

import pywt
import talib.abstract as ta
from scipy.ndimage import gaussian_filter1d
from statsmodels.discrete.discrete_model import Probit

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
import pandas_ta as pta

pd.options.mode.chained_assignment = None  # default='warn'

# Strategy specific imports, files must reside in same folder as strategy
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent))
sys.path.append(str(Path(__file__)))

import logging
import warnings

log = logging.getLogger(__name__)
# log.setLevel(logging.DEBUG)
warnings.simplefilter(action='ignore', category=pd.errors.PerformanceWarning)

from PCA import PCA

"""
####################################################################################
PCA_minmax:
    This is a subclass of PCA, which provides a framework for deriving a dimensionally-reduced model
    This class trains the model based on detecting points which are either the max (sell) or min (buy)
    of both the past window and the future window
    Note: this strat should benefit from wider windows, otherwise it'll tag evry local peak/valley

####################################################################################
"""


class PCA_minmax(PCA):


    # Do *not* hyperopt for the roi and stoploss spaces

    # Have to re-declare any globals that we need to modify

    # These parameters control much of the behaviour because they control the generation of the training data
    # Unfortunately, these cannot be hyperopt params because they are used in populate_indicators, which is only run
    # once during hyperopt
    lookahead_hours = 8.0
    n_profit_stddevs = 0.0
    n_loss_stddevs = 0.0
    min_f1_score = 0.70

    custom_trade_info = {}

    dbg_scan_classifiers = False  # if True, scan all viable classifiers and choose the best. Very slow!
    dbg_test_classifier = False  # test classifiers after fitting
    dbg_analyse_pca = False  # analyze PCA weights
    dbg_verbose = True  # controls debug output
    dbg_curr_df: DataFrame = None  # for debugging of current dataframe

    ###################################

    # Strategy Specific Variable Storage

    ## Hyperopt Variables

    # PCA hyperparams
    # buy_pca_gain = IntParameter(1, 50, default=4, space='buy', load=True, optimize=True)
    #
    # sell_pca_gain = IntParameter(-1, -15, default=-4, space='sell', load=True, optimize=True)

    # Custom Sell Profit (formerly Dynamic ROI)
    cexit_roi_type = CategoricalParameter(['static', 'decay', 'step'], default='step', space='sell', load=True,
                                          optimize=True)
    cexit_roi_time = IntParameter(720, 1440, default=720, space='sell', load=True, optimize=True)
    cexit_roi_start = DecimalParameter(0.01, 0.05, default=0.01, space='sell', load=True, optimize=True)
    cexit_roi_end = DecimalParameter(0.0, 0.01, default=0, space='sell', load=True, optimize=True)
    cexit_trend_type = CategoricalParameter(['rmi', 'ssl', 'candle', 'any', 'none'], default='any', space='sell',
                                            load=True, optimize=True)
    cexit_pullback = CategoricalParameter([True, False], default=True, space='sell', load=True, optimize=True)
    cexit_pullback_amount = DecimalParameter(0.005, 0.03, default=0.01, space='sell', load=True, optimize=True)
    cexit_pullback_respect_roi = CategoricalParameter([True, False], default=False, space='sell', load=True,
                                                      optimize=True)
    cexit_endtrend_respect_roi = CategoricalParameter([True, False], default=False, space='sell', load=True,
                                                      optimize=True)

    # Custom Stoploss
    cstop_loss_threshold = DecimalParameter(-0.05, -0.01, default=-0.03, space='sell', load=True, optimize=True)
    cstop_bail_how = CategoricalParameter(['roc', 'time', 'any', 'none'], default='none', space='sell', load=True,
                                          optimize=True)
    cstop_bail_roc = DecimalParameter(-5.0, -1.0, default=-3.0, space='sell', load=True, optimize=True)
    cstop_bail_time = IntParameter(60, 1440, default=720, space='sell', load=True, optimize=True)
    cstop_bail_time_trend = CategoricalParameter([True, False], default=True, space='sell', load=True, optimize=True)
    cstop_max_stoploss = DecimalParameter(-0.30, -0.01, default=-0.10, space='sell', load=True, optimize=True)

    ###################################

    # Override the training signals

    # find local min/max within past & future window
    # This is pretty cool because it doesn't care about 'jitter' within the window, or any measure of profit/loss
    # Note that this will find a lot of results, may want to add a few more guards

    def get_train_buy_signals(self, future_df: DataFrame):

        series = np.where(
            (
                    (future_df['mfi'] < 30) & # loose guard
                    # (future_df['dwt_at_min'] > 0) & # at min of previous window
                    # (future_df['dwt_bottom'] > 0) &  # at min of previous window

                    (future_df['full_dwt'] <= future_df['dwt_recent_min']) &  # at min of past window
                    (future_df['full_dwt'] <= future_df['future_min']) &  # at min of future window

                    (future_df['future_gain'] >= future_df['future_profit_threshold']) # profit in next window exceeds threshold
            ), 1.0, 0.0)

        return series

    def get_train_sell_signals(self, future_df: DataFrame):

        series = np.where(
            (
                    (future_df['mfi'] > 70) &  # loose guard
                    # (future_df['dwt_at_max'] > 0) & # at max of previous window
                    # (future_df['dwt_top'] > 0) & # at max of previous window

                    (future_df['full_dwt'] >= future_df['dwt_recent_min']) &  # at max of past window
                    (future_df['full_dwt'] >= future_df['future_max']) &  # at max of future window

                    (future_df['future_gain'] <= future_df['future_loss_threshold']) # loss in next window exceeds threshold
            ), 1.0, 0.0)

        return series

    def get_strategy_entry_guard_conditions(self, dataframe: DataFrame):
        cond = np.where(
            (
                # sell signal is not active
                (dataframe['predict_sell'] < 0.1)
            ), 1.0, 0.0)
        return cond


    # save the indicators used here so that we can see them in plots (prefixed by '%')
    def save_debug_indicators(self, future_df: DataFrame):
        # self.add_debug_indicator(future_df, 'future_min')
        # # self.add_debug_indicator(future_df, 'future_max')
        # self.add_debug_indicator(future_df, 'dwt_at_min')
        # self.add_debug_indicator(future_df, 'dwt_at_max')
        self.add_debug_indicator(future_df, 'train_buy')
        self.add_debug_indicator(future_df, 'train_sell')

        return

    ###################################
