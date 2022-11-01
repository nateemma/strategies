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
PCA_swing:
    This is a subclass of PCA, which provides a framework for deriving a dimensionally-reduced model
    This class trains the model based on detecting swings in smoothed/average price, followed
    by a profit (for buys) or loss (for sells)

####################################################################################
"""


class PCA_swing(PCA):

    # Do *not* hyperopt for the roi and stoploss spaces


    # Have to re-decalre globals, so that we can change them without affecting (or having to change) the base class
    # ROI table:
    minimal_roi = {
        "0": 0.1
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

    use_custom_stoploss = True

    # Recommended
    use_sell_signal = True
    sell_profit_only = False
    ignore_roi_if_buy_signal = True

    # Required
    startup_candle_count: int = 128  # must be power of 2
    process_only_new_candles = True

    # Strategy-specific global vars

    inf_mins = timeframe_to_minutes(inf_timeframe)
    data_mins = timeframe_to_minutes(timeframe)
    inf_ratio = int(inf_mins / data_mins)

    # These parameters control much of the behaviour because they control the generation of the training data
    # Unfortunately, these cannot be hyperopt params because they are used in populate_indicators, which is only run
    # once during hyperopt
    lookahead_hours = 0.5
    n_profit_stddevs = 0.0
    n_loss_stddevs = 0.0
    min_f1_score = 0.51

    indicator_list = []  # list of parameters to use (technical indicators)

    inf_lookahead = int((12 / inf_ratio) * lookahead_hours)
    curr_lookahead = inf_lookahead

    curr_pair = ""
    custom_trade_info = {}

    # profit/loss thresholds used for assessing buy/sell signals. Keep these realistic!
    # Note: if self.dynamic_gain_thresholds is True, these will be adjusted for each pair, based on historical mean
    default_profit_threshold = 0.3
    default_loss_threshold = -0.3
    profit_threshold = default_profit_threshold
    loss_threshold = default_loss_threshold
    dynamic_gain_thresholds = True  # dynamically adjust gain thresholds based on actual mean (beware, training data could be bad)

    dwt_window = startup_candle_count

    num_pairs = 0
    pair_model_info = {}  # holds model-related info for each pair

    # debug flags
    first_time = True  # mostly for debug
    first_run = True  # used to identify first time through buy/sell populate funcs

    dbg_scan_classifiers = True  # if True, scan all viable classifiers and choose the best. Very slow!
    dbg_test_classifier = True  # test clasifiers after fitting
    dbg_analyse_pca = False  # analyze PCA weights
    dbg_verbose = False  # controls debug output
    dbg_curr_df: DataFrame = None  # for debugging of current dataframe

    ###################################

    # Strategy Specific Variable Storage

    ## Hyperopt Variables

    # PCA hyperparams
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

    # override the default training signal generation

    # detect points where smoothed estimate is at a top/bottom and there is a direction change

    def get_train_buy_signals(self, future_df: DataFrame):

        buys = np.where(
            (
                    # (future_df['volume'] > 0) & # volume check
                    (future_df['future_gain'] > self.profit_threshold) &  # future gain
                    (future_df['dwt_bottom'] > 0)   # bottom of trend
                    # (qtpylib.crossed_above(future_df['dwt_deriv'], 0.0)) # start of upswing

            ), 1.0, 0.0)

        return buys

    def get_train_sell_signals(self, future_df: DataFrame):

        sells = np.where(
            (
                    # (future_df['volume'] > 0) & # volume check
                    (future_df['future_gain'] < self.loss_threshold) & # future loss
                    (future_df['dwt_top'] > 0)  # top of trend
                    # (qtpylib.crossed_below(future_df['dwt_deriv'], 0.0)) # start of downswing
            ), 1.0, 0.0)

        return sells

    # save the indicators used here so that we can see them in plots (prefixed by '%')
    def save_debug_indicators(self, future_df: DataFrame):
        self.add_debug_indicator(future_df, 'future_gain')

        return

