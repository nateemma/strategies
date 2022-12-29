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
NNBC_jump:
    This is a subclass of NNBC, which provides a framework for deriving a neural network model
    This class trains the model based on consecutive sequences of up/down candles, followed by big profit/loss

####################################################################################
"""


class NNBC_nseq(NNBC):

    plot_config = {
        'main_plot': {
            'tema': {'color': 'darkcyan'},
            'dwt_smooth': {'color': 'salmon'},
        },
        'subplots': {
            "Diff": {
                '%future_nseq_up': {'color': 'salmon'},
                'dwt_nseq_dn': {'color': 'mediumslateblue'},
                '%train_buy': {'color': 'darkseagreen'},
                'predict_buy': {'color': 'dodgerblue'},
                '%train_sell': {'color': 'lightcoral'},
                'predict_sell': {'color': 'mediumvioletred'},
            },
        }
    }

    # Do *not* hyperopt for the roi and stoploss spaces

    # Have to re-declare any globals that we need to modify

    # These parameters control much of the behaviour because they control the generation of the training data
    # Unfortunately, these cannot be hyperopt params because they are used in populate_indicators, which is only run
    # once during hyperopt
    lookahead_hours = 1.0
    n_profit_stddevs = 0.0
    n_loss_stddevs = 0.0
    min_f1_score = 0.4

    cherrypick_data = False
    preload_model = True # don't set to true if you are changing buy/sell conditions or tweaking models


    custom_trade_info = {}

    dbg_scan_classifiers = False  # if True, scan all viable classifiers and choose the best. Very slow!
    dbg_test_classifier = True  # test classifiers after fitting
    dbg_verbose = True  # controls debug output
    dbg_curr_df: DataFrame = None  # for debugging of current dataframe

    ###################################

    # Strategy Specific Variable Storage

    ## Hyperopt Variables

    # buy/sell hyperparams
    buy_nseq_dn = IntParameter(0, 10, default=4, space='buy', load=True, optimize=True)
    sell_nseq_up = IntParameter(0, 10, default=8, space='sell', load=True, optimize=True)

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

    # find local min/max within past & future window
    # This is pretty cool because it doesn't care about 'jitter' within the window, or any measure of profit/loss
    # Note that this will find a lot of results, may want to add a few more guards

    def get_train_buy_signals(self, future_df: DataFrame):
        series = np.where(
            (
                    # (future_df['mfi'] < 30) & # loose guard

                    # down...
                    (future_df['dwt_nseq_dn'] >= 4) &
                    (future_df['dwt_win_gain'] <= self.loss_threshold) &

                    # then up...
                    (future_df['future_nseq_up'] >= 8) &
                    (future_df['future_win_gain'] >= self.profit_threshold) #&  # future gain
                    # (future_df['profit_max'] >= future_df['profit_threshold'])   # future profit exceeds threshold
            ), 1.0, 0.0)

        return series

    def get_train_sell_signals(self, future_df: DataFrame):

        series = np.where(
            (
                    # (future_df['mfi'] > 60) & # loose guard

                    # up...
                    (future_df['dwt_nseq_up'] >= 4) &
                    (future_df['dwt_win_gain'] >= self.profit_threshold) &

                    # then down...
                    (future_df['future_nseq_dn'] >= 8) &
                    # (future_df['future_win_gain'] <= self.loss_threshold) #&
                    (future_df['future_gain'] <= self.loss_threshold) #&
                    # (future_df['loss_min'] <= future_df['loss_threshold'])   # future loss exceeds threshold
            ), 1.0, 0.0)

        return series


    # save the indicators used here so that we can see them in plots (prefixed by '%')
    def save_debug_indicators(self, future_df: DataFrame):
        self.add_debug_indicator(future_df, 'future_nseq_dn')
        self.add_debug_indicator(future_df, 'future_nseq_up')

        return

    ###################################

    # callbacks to add conditions to main buy/sell decision (rather than trainng)

    def get_strategy_buy_conditions(self, dataframe: DataFrame):
        cond = np.where(
            (
                # N down sequences
                (dataframe['dwt_nseq_dn'] >= self.buy_nseq_dn.value) #&
                # loss above threshold
                # (dataframe['dwt_win_gain'] <= self.loss_threshold)
            ), 1.0, 0.0)
        return cond

    def get_strategy_sell_conditions(self, dataframe: DataFrame):
        cond = np.where(
            (
                # N up sequences
                (dataframe['dwt_nseq_up'] >= self.sell_nseq_up.value) #&
                # profit above threshold
                # (dataframe['dwt_win_gain'] >= self.profit_threshold)
            ), 1.0, 0.0)
        return cond

    ###################################