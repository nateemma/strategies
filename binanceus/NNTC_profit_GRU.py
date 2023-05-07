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

from NNTC import NNTC

"""
####################################################################################
NNTC_profit:
    This is a subclass of NNTC, which provides a framework for deriving a dimensionally-reduced model
    This class trains the model based on future profit/loss conditions

####################################################################################
"""


class NNTC_profit_GRU(NNTC):

    plot_config = {
        'main_plot': {
            # 'dwt': {'color': 'darkcyan'},
            # '%future_min': {'color': 'salmon'},
            # '%future_max': {'color': 'cadetblue'},
        },
        'subplots': {
            "Diff": {
                '%train_buy': {'color': 'mediumaquamarine'},
                'predict_buy': {'color': 'cornflowerblue'},
                '%train_sell': {'color': 'salmon'},
                'predict_sell': {'color': 'orange'},
            },
        }
    }

    # Do *not* hyperopt for the roi and stoploss spaces

    # Have to re-declare any globals that we need to modify

    # These parameters control much of the behaviour because they control the generation of the training data
    # Unfortunately, these cannot be hyperopt params because they are used in populate_indicators, which is only run
    # once during hyperopt
    lookahead_hours = 1.0
    n_profit_stddevs = 2.0
    n_loss_stddevs = 2.0
    min_f1_score = 0.70

    custom_trade_info = {}

    refit_model = False  # only set to True when training. If False, then existing model is used, if present


    dbg_scan_classifiers = False  # if True, scan all viable classifiers and choose the best. Very slow!
    dbg_test_classifier = False  # test clasifiers after fitting
    dbg_analyse_pca = False  # analyze PCA weights
    dbg_verbose = False  # controls debug output
    dbg_curr_df: DataFrame = None  # for debugging of current dataframe

    classifier_name = 'GRU'

    ###################################

    # Strategy Specific Variable Storage

    ## Hyperopt Variables

    # buy/sell hyperparams
    buy_nseq_dn = IntParameter(2, 10, default=4, space='buy', load=True, optimize=True)
    sell_nseq_up = IntParameter(2, 10, default=8, space='sell', load=True, optimize=True)

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

    # override the default training signal generation

    # find where future price is higher/lower than previous window max/min and exceeds threshold

    def get_train_buy_signals(self, future_df: DataFrame):
        series = np.where(
            (
                    (future_df['future_profit_max'] >= future_df['fwd_profit_threshold']) & # future profit exceeds threshold
                    (future_df['future_max'] > future_df['dwt_recent_max']) # future window max exceeds prior window max
            ), 1.0, 0.0)

        return series

    def get_train_sell_signals(self, future_df: DataFrame):
        series = np.where(
            (
                    (future_df['future_loss_min'] <= future_df['fwd_loss_threshold']) & # future loss exceeds threshold
                    (future_df['future_min'] < future_df['dwt_recent_min']) # future window max exceeds prior window max
            ), 1.0, 0.0)

        return series

    # save the indicators used here so that we can see them in plots (prefixed by '%')
    def save_debug_indicators(self, future_df: DataFrame):

        self.add_debug_indicator(future_df, 'future_profit_max')
        # self.add_debug_indicator(future_df, 'profit_threshold')
        self.add_debug_indicator(future_df, 'future_max')
        self.add_debug_indicator(future_df, 'future_loss_min')
        # self.add_debug_indicator(future_df, 'loss_threshold')
        self.add_debug_indicator(future_df, 'future_min')

        return

    ###################################


    # callbacks to add conditions to main buy/sell decision (rather than trainng)

    def get_strategy_buy_conditions(self, dataframe: DataFrame):
        cond = np.where(
            (
                # N down sequences
                (dataframe['dwt_nseq_dn'] >= self.buy_nseq_dn.value)
            ), 1.0, 0.0)
        return cond

    def get_strategy_sell_conditions(self, dataframe: DataFrame):
        cond = np.where(
            (
                # N up sequences
                ( dataframe['dwt_nseq_up'] >= self.sell_nseq_up.value)
            ), 1.0, 0.0)
        return cond

    ###################################

