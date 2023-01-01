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

from NNBC import NNBC

"""
####################################################################################
NNBC_profit:
    This is a subclass of NNBC, which provides a framework for deriving a dimensionally-reduced model
    This class trains the model based on projecting that the future price will exceed the previous window
    and profit will exceed historical average

####################################################################################
"""


class NNBC_profit(NNBC):
    plot_config = {
        'main_plot': {
            'dwt_smooth': {'color': 'salmon'},
        },
        'subplots': {
            "Diff": {
                '%train_buy': {'color': 'green'},
                'predict_buy': {'color': 'blue'},
                '%train_sell': {'color': 'red'},
                'predict_sell': {'color': 'orange'},
                '%profit_max': {'color': 'purple'},
                '%loss_min': {'color': 'tomato'},
            },
        }

}


    # Do *not* hyperopt for the roi and stoploss spaces

    # Have to re-declare any globals that we need to modify

    # These parameters control much of the behaviour because they control the generation of the training data
    # Unfortunately, these cannot be hyperopt params because they are used in populate_indicators, which is only run
    # once during hyperopt
    lookahead_hours = 1.0
    n_profit_stddevs = 1.0
    n_loss_stddevs = 1.0
    min_f1_score = 0.60

    custom_trade_info = {}

    dbg_scan_classifiers = False  # if True, scan all viable classifiers and choose the best. Very slow!
    dbg_test_classifier = True  # test clasifiers after fitting
    dbg_analyse_pca = False  # analyze NNBC weights
    dbg_verbose = True  # controls debug output
    dbg_curr_df: DataFrame = None  # for debugging of current dataframe

    ###################################

    # Strategy Specific Variable Storage

    ## Hyperopt Variables

    # NNBC hyperparams
    # buy_pca_gain = IntParameter(1, 50, default=4, space='buy', load=True, optimize=True)
    #
    # sell_pca_gain = IntParameter(-1, -15, default=-4, space='sell', load=True, optimize=True)

    # Custom Sell Profit (formerly Dynamic ROI)
    csell_roi_type = CategoricalParameter(['static', 'decay', 'step'], default='step', space='sell', load=True,
                                          optimize=True)
    csell_roi_time = IntParameter(720, 1440, default=720, space='sell', load=True, optimize=True)
    csell_roi_start = DecimalParameter(0.01, 0.05, default=0.01, space='sell', load=True, optimize=True)
    csell_roi_end = DecimalParameter(0.0, 0.01, default=0, space='sell', load=True, optimize=True)
    csell_trend_type = CategoricalParameter(['rmi', 'ssl', 'candle', 'any', 'none'], default='any', space='sell',
                                            load=True, optimize=True)
    csell_pullback = CategoricalParameter([True, False], default=True, space='sell', load=True, optimize=True)
    csell_pullback_amount = DecimalParameter(0.005, 0.03, default=0.01, space='sell', load=True, optimize=True)
    csell_pullback_respect_roi = CategoricalParameter([True, False], default=False, space='sell', load=True,
                                                      optimize=True)
    csell_endtrend_respect_roi = CategoricalParameter([True, False], default=False, space='sell', load=True,
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

    # find where future price is higher/lower than previous window max/min and exceeds threshold

    def get_train_buy_signals(self, future_df: DataFrame):
        series = np.where(
            (
                    (future_df['dwt_gain'] <= self.loss_threshold) & # loss exceeds threshold

                    (future_df['profit_max'] >= self.profit_threshold) & # future profit exceeds threshold
                    (future_df['future_max'] > future_df['dwt_recent_max']) # future window max exceeds prior window max
            ), 1.0, 0.0)

        return series

    def get_train_sell_signals(self, future_df: DataFrame):
        series = np.where(
            (
                    (future_df['dwt_gain'] >= self.profit_threshold) & # profit exceeds threshold

                    (future_df['loss_min'] <= self.loss_threshold) & # future loss exceeds threshold
                    (future_df['future_min'] < future_df['dwt_recent_min']) # future window max exceeds prior window max
            ), 1.0, 0.0)

        return series

        ###################################

    # provide additional buy/sell signals
    # These usually repeat the (backward-looking) guards from the training signal criteria. This helps filter out
    # bad predictions (Machine Learning is not perfect)


    def get_train_buy_signals(self, dataframe: DataFrame):
        buys = np.where(
            (
                    (dataframe['dwt_gain'] <= self.loss_threshold)   #  loss exceeds threshold
            ), 1.0, 0.0)

        return buys

    def get_train_sell_signals(self, dataframe: DataFrame):
        sells = np.where(
            (
                    (dataframe['dwt_gain'] >= self.profit_threshold)  # profit exceeds threshold
            ), 1.0, 0.0)

        return sells

    ###################################

    # save the indicators used here so that we can see them in plots (prefixed by '%')
    def save_debug_indicators(self, future_df: DataFrame):

        self.add_debug_indicator(future_df, 'profit_max')
        self.add_debug_indicator(future_df, 'profit_threshold')
        self.add_debug_indicator(future_df, 'profit_max')
        self.add_debug_indicator(future_df, 'future_max')
        self.add_debug_indicator(future_df, 'loss_min')
        self.add_debug_indicator(future_df, 'loss_threshold')
        self.add_debug_indicator(future_df, 'future_min')

        return

    ###################################
