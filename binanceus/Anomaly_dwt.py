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

from Anomaly import Anomaly

"""
####################################################################################
Anomaly_dwt:
    This is a subclass of Anomaly, which provides a framework for deriving a dimensionally-reduced model
    This class trains the model based on comparing the forward-looking DWT model to the backward-looking model

####################################################################################
"""


class Anomaly_dwt(Anomaly):


    plot_config = {
        'main_plot': {
            'close': {'color': 'cornflowerblue'},
            'dwt': {'color': 'salmon'},
        },
        'subplots': {
            "Diff": {
                '%train_buy': {'color': 'darkseagreen'},
                'predict_buy': {'color': 'dodgerblue'},
                '%train_sell': {'color': 'lightcoral'},
                'predict_sell': {'color': 'mediumvioletred'},
            },
        }
    }

    # Do *not* hyperopt for the roi and stoploss spaces

    # Have to re-declare globals, so that we can change them without affecting (or having to change) the base class,
    # and also avoiding affecting other subclasses of Anomaly

    use_simpler_custom_stoploss = False

    # These parameters control much of the behaviour because they control the generation of the training data
    # Unfortunately, these cannot be hyperopt params because they are used in populate_indicators, which is only run
    # once during hyperopt
    lookahead_hours = 1.0
    n_profit_stddevs = 0.0
    n_loss_stddevs = 0.0
    min_f1_score = 0.48

    custom_trade_info = {}

    dbg_scan_classifiers = False  # if True, scan all viable classifiers and choose the best. Very slow!
    dbg_test_classifier = True  # test clasifiers after fitting
    dbg_analyse_pca = False  # analyze PCA weights
    dbg_verbose = False  # controls debug output
    dbg_curr_df: DataFrame = None  # for debugging of current dataframe

    ###################################

    # Strategy Specific Variable Storage

    ## Hyperopt Variables

    # Anomaly hyperparams
    # buy_pca_gain = IntParameter(1, 50, default=4, space='buy', load=True, optimize=True)
    #
    # sell_pca_gain = IntParameter(-1, -15, default=-4, space='sell', load=True, optimize=True)

    # Custom Sell Profit (formerly Dynamic ROI)

    if use_simpler_custom_stoploss:
        sell_params = {
            "pHSL": -0.068,
            "pPF_1": 0.008,
            "pPF_2": 0.098,
            "pSL_1": 0.02,
            "pSL_2": 0.065,
        }

        # hard stoploss profit
        pHSL = DecimalParameter(-0.200, -0.010, default=-0.08, decimals=3, space='sell', load=True)

        # profit threshold 1, trigger point, SL_1 is used
        pPF_1 = DecimalParameter(0.008, 0.020, default=0.016, decimals=3, space='sell', load=True)
        pSL_1 = DecimalParameter(0.008, 0.020, default=0.011, decimals=3, space='sell', load=True)

        # profit threshold 2, SL_2 is used
        pPF_2 = DecimalParameter(0.040, 0.100, default=0.080, decimals=3, space='sell', load=True)
        pSL_2 = DecimalParameter(0.020, 0.070, default=0.040, decimals=3, space='sell', load=True)

    else:

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
        cstop_bail_time_trend = CategoricalParameter([True, False], default=True, space='sell', load=True,
                                                     optimize=True)
        cstop_max_stoploss = DecimalParameter(-0.30, -0.01, default=-0.10, space='sell', load=True, optimize=True)


    ###################################

    # Override the training signals

    # find where future price is higher/lower than previous window max/min and exceeds threshold

    def get_train_buy_signals(self, future_df: DataFrame):
        series = np.where(
            (
                # forward model above backward model
                    (future_df['dwt_diff'] < 0) &
                    # current loss below threshold
                    (future_df['dwt_diff'] <= future_df['future_loss_threshold']) &
                    # forward model below backward model at lookahead
                    (future_df['dwt_diff'].shift(-self.curr_lookahead) > 0)
            ), 1.0, 0.0)

        return series

    def get_train_sell_signals(self, future_df: DataFrame):
        series = np.where(
            (
                # forward model above backward model
                    (future_df['dwt_diff'] > 0) &
                    # current profit above threshold
                    (future_df['dwt_diff'] >= future_df['future_profit_threshold']) &
                    # forward model below backward model at lookahead
                    (future_df['dwt_diff'].shift(-self.curr_lookahead) < 0)
            ), 1.0, 0.0)

        return series

    # save the indicators used here so that we can see them in plots (prefixed by '%')
    def save_debug_indicators(self, future_df: DataFrame):

        self.add_debug_indicator(future_df, 'future_max')
        self.add_debug_indicator(future_df, 'future_min')

        return

    ###################################
