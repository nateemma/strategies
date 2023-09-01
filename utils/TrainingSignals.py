# pylint: disable=C0103,C0114,C0301,C0116,C0413,C0115,W0611,W0613

# This is a set of utilities for getting training signal (buy/sell) events
# The signals are collected here so that the different approaches to generating events can be more easily used across
# multiple strategies
# Notes:
#   - Any hard-coded numbers have been derived by staring at charts for hours to figure out appropriate values
#   - These functions all look ahead in the data, so they can *ONLY* be used for training
#   - WARNING: any behavioural changes to get_entry_training_signals() or get_exit_training_signals() require
#              that you re-train the associated models (because the training data will change)


import sys
from pathlib import Path

import freqtrade.vendor.qtpylib.indicators as qtpylib

import numpy as np
import pandas as pd
import scipy

sys.path.append(str(Path(__file__).parent))

import logging
import warnings
from enum import Enum

log = logging.getLogger(__name__)
# log.setLevel(logging.DEBUG)
warnings.simplefilter(action='ignore', category=pd.errors.PerformanceWarning)

from pandas import DataFrame

pd.options.mode.chained_assignment = None  # default='warn'

# -----------------------------------
# define a xxx_signals class for each type of training signal
# This allows us to deal with the different types in a  generic fashion
# Each class must contain (at least) the follow methods :
#   get_entry_training_signals()
#   get_exit_training_signals()
#   get_entry_guard_conditions()
#   get_exit_guard_conditions()
#   get_debug_indicators()

# To add a new type of training signal, define the corresponding class and add it to the SignalType enum

# -----------------------------------

# base class - to allow generic treatment of all signal types

from abc import ABC, abstractmethod


class base_signals(ABC):
    lookahead_hours = 1.0
    lookahead = 12 # candles
    n_profit_stddevs = 1.0
    n_loss_stddevs = 1.0

    def __init__(self, lookahead):
        super().__init__()

        self.lookahead = lookahead

    def get_lookahead(self):
        return self.lookahead

    def get_n_profit_stddevs(self):
        return self.n_profit_stddevs

    def get_n_loss_stddevs(self):
        return self.n_loss_stddevs

    # returns the 'abbreviated' name used for this signal. Intended for naming models
    # Note that this relies on the naming convention xxx_signals, where xxx will be the abbreviated name
    def get_signal_name(self):
        name = self.__class__.__name__.replace("_signals", "")
        return name

    # check that the supplied indicators are present in the dataframe
    def indicators_present(self, ind_list, dataframe) -> bool:
        if len(ind_list) > 0:
            result = True
            for ind in ind_list:
                if ind not in dataframe.columns:
                    result = False
                    print(f"    ERROR: indicator not in dataframe: {ind}")
        else:
            print("    WARNOING: empty indicator list provided")
            result = False

        return result

    @abstractmethod
    def check_indicators(self, future_df: DataFrame) -> bool:
        return False

    @abstractmethod
    def get_entry_training_signals(self, future_df: DataFrame):
        return np.zeros(future_df.shape[0], dtype=float)

    # function to get sell signals

    @abstractmethod
    def get_exit_training_signals(self, future_df: DataFrame):
        return np.zeros(future_df.shape[0], dtype=float)

    # function to get entry/buy guard conditions

    @abstractmethod
    def get_entry_guard_conditions(self, dataframe: DataFrame):
        return None

    # function to get entry/buy guard conditions

    @abstractmethod
    def get_exit_guard_conditions(self, dataframe: DataFrame):
        return None

    # function to get list of debug indicators to make visible (e.g. for plotting)

    @abstractmethod
    def get_debug_indicators(self):
        return []


# -----------------------------------

# training signals based on ADX and DI indicators

class adx_signals(base_signals):

    def check_indicators(self, dataframe) -> bool:
        # check that needed indicators are present
        ind_list = [
            'adx', 'dm_delta', 'di_delta',
            'future_loss_min', 'future_loss_threshold',
            'future_profit_max', 'future_profit_threshold'
        ]
        return self.indicators_present(ind_list, dataframe)

    # function to get buy signals
    def get_entry_training_signals(self, future_df: DataFrame):

        # classic ADX:
        # ADX above 25 and DI+ above DI-: That's an uptrend.
        # ADX above 25 and DI- above DI+: That's a downtrend.

        signals = np.where(
            (
                (future_df['fisher_wr'] < -0.5) & # guard

                (future_df['adx'] > 25) &
                (future_df['di_delta'] < 0) & # downtrend

                # future profit exceeds threshold
                (future_df['future_profit_max'] >= future_df['future_profit_threshold'])
            ), 1.0, 0.0)
        return signals

    # function to get sell signals

    def get_exit_training_signals(self, future_df: DataFrame):

        signals = np.where(
            (
                (future_df['fisher_wr'] > 0.5) & # guard

                (future_df['adx'] > 25) &
                (future_df['di_delta'] > 0) & # uptrend

                # future loss exceeds threshold
                (future_df['future_loss_min'] <= future_df['future_loss_threshold'])
            ), 1.0, 0.0)
        return signals

    # function to get entry/buy guard conditions

    def get_entry_guard_conditions(self, dataframe: DataFrame):
        conditions = np.where(
            (
                (dataframe['fisher_wr'] < -0.5)
            ), 1.0, 0.0)
        return conditions

    # function to get entry/buy guard conditions

    def get_exit_guard_conditions(self, dataframe: DataFrame):
        conditions = np.where(
            (
                (dataframe['fisher_wr'] > 0.5)
            ), 1.0, 0.0)
        return conditions

    # function to get list of debug indicators to make visible (e.g. for plotting)

    def get_debug_indicators(self):
        dbg_list = []
        return dbg_list


# -----------------------------------

# training signals based on ADX and DI indicators
# This variant looks for DI+ and DI- crossings

class adx2_signals(base_signals):

    def check_indicators(self, dataframe) -> bool:
        # check that needed indicators are present
        ind_list = [
            'adx', 'di_plus', 'di_minus',
            'future_loss_min', 'future_loss_threshold',
            'future_profit_max', 'future_profit_threshold'
        ]
        return self.indicators_present(ind_list, dataframe)

    # function to get buy signals
    def get_entry_training_signals(self, future_df: DataFrame):

        # classic ADX crossing:
        # ADX above 20 and DI+ crosses above DI-
        # ADX above 20 and DI+ crosses below DI+

        # just the crossing points don't generate enough signals, so look for range around the crossing instead

        signals = np.where(
            (
                # (future_df['adx'] > 20) &
                (future_df['di_delta'] >= 0) &
                (future_df['di_delta'] <= 5) &

                # future profit exceeds threshold
                (future_df['future_profit_max'] >= future_df['future_profit_threshold'])
            ), 1.0, 0.0)
        return signals

    # function to get sell signals

    def get_exit_training_signals(self, future_df: DataFrame):

        signals = np.where(
            (
                # (future_df['adx'] > 20) &
                (future_df['di_delta'] <= 0) &
                (future_df['di_delta'] >= -5) &

                # future loss exceeds threshold
                (future_df['future_loss_min'] <= future_df['future_loss_threshold'])
            ), 1.0, 0.0)
        return signals

    # function to get entry/buy guard conditions

    def get_entry_guard_conditions(self, dataframe: DataFrame):
        conditions = np.where(
            (
                (dataframe['fisher_wr'] < -0.5)
            ), 1.0, 0.0)
        return conditions

    # function to get entry/buy guard conditions

    def get_exit_guard_conditions(self, dataframe: DataFrame):
        conditions = np.where(
            (
                (dataframe['fisher_wr'] > 0.5)
            ), 1.0, 0.0)
        return conditions

    # function to get list of debug indicators to make visible (e.g. for plotting)

    def get_debug_indicators(self):
        dbg_list = []
        return dbg_list


# -----------------------------------

# training signals based on ADX and DI indicators
# This variant looks for ADX > 25 and di_diff > 25 or < -25

class adx3_signals(base_signals):

    def check_indicators(self, dataframe) -> bool:
        # check that needed indicators are present
        ind_list = [
            'adx', 'di_plus', 'di_minus',
            'future_loss_min', 'future_loss_threshold',
            'future_profit_max', 'future_profit_threshold'
        ]
        return self.indicators_present(ind_list, dataframe)

    # function to get buy signals
    def get_entry_training_signals(self, future_df: DataFrame):

        # classic ADX crossing:
        # ADX above 20 and DI+ crosses above DI-
        # ADX above 20 and DI+ crosses below DI+

        # just the crossing points don't generate enough signals, so look for range around the crossing instead

        signals = np.where(
            (
                (future_df['adx'] > 20) &
                (future_df['di_delta'] <= -10) &

                # future profit exceeds threshold
                (future_df['future_profit_max'] >= future_df['future_profit_threshold'])
            ), 1.0, 0.0)
        return signals

    # function to get sell signals

    def get_exit_training_signals(self, future_df: DataFrame):

        signals = np.where(
            (
                (future_df['adx'] > 20) &
                (future_df['di_delta'] >= 10) &

                # future loss exceeds threshold
                (future_df['future_loss_min'] <= future_df['future_loss_threshold'])
            ), 1.0, 0.0)
        return signals

    # function to get entry/buy guard conditions

    def get_entry_guard_conditions(self, dataframe: DataFrame):
        conditions = np.where(
            (
                (dataframe['fisher_wr'] < -0.5)
            ), 1.0, 0.0)
        return conditions

    # function to get entry/buy guard conditions

    def get_exit_guard_conditions(self, dataframe: DataFrame):
        conditions = np.where(
            (
                (dataframe['fisher_wr'] > 0.5)
            ), 1.0, 0.0)
        return conditions

    # function to get list of debug indicators to make visible (e.g. for plotting)

    def get_debug_indicators(self):
        dbg_list = []
        return dbg_list


# -----------------------------------

# training signals based on teh Aroon Oscillator

        # classic Arron:
        # Aroon Up > Aroon Down and (Aroon Up near 100, Aroon Down near 0): That's an uptrend.
        # Aroon Up < Aroon Down and (Aroon Down near 100, Aroon Up near 0): That's a downtrend.

class aroon_signals(base_signals):

    def check_indicators(self, dataframe) -> bool:
        # check that needed indicators are present
        ind_list = [
            'aroonup', 'aroondown',
            'future_loss_min', 'future_loss_threshold',
            'future_profit_max', 'future_profit_threshold'
        ]
        return self.indicators_present(ind_list, dataframe)

    # function to get buy signals
    def get_entry_training_signals(self, future_df: DataFrame):
        signals = np.where(
            (
                # uptrend
                    (future_df['aroonup'] > future_df['aroondown']) &
                    (
                            (future_df['aroonup'] > 90) &
                            (future_df['aroondown'] < 10)
                    ) &

                    # future profit exceeds threshold
                    (future_df['future_profit_max'] >= future_df['future_profit_threshold'])
            ), 1.0, 0.0)
        return signals

    # function to get sell signals

    def get_exit_training_signals(self, future_df: DataFrame):

        signals = np.where(
            (
                # downtrend
                    (future_df['aroonup'] < future_df['aroondown']) &
                    (
                            (future_df['aroonup'] < 10) &
                            (future_df['aroondown'] > 90)
                    ) &

                # future loss exceeds threshold
                (future_df['future_loss_min'] <= future_df['future_loss_threshold'])
            ), 1.0, 0.0)
        return signals

    # function to get entry/buy guard conditions

    def get_entry_guard_conditions(self, dataframe: DataFrame):
        conditions = None
        return conditions

    # function to get entry/buy guard conditions

    def get_exit_guard_conditions(self, dataframe: DataFrame):
        conditions = None
        return conditions

    # function to get list of debug indicators to make visible (e.g. for plotting)

    def get_debug_indicators(self):
        dbg_list = []
        return dbg_list


# -----------------------------------

# training signals based on the width of the Bollinger Bands

class bbw_signals(base_signals):

    def check_indicators(self, dataframe) -> bool:
        # check that needed indicators are present
        ind_list = [
            'bb_width',
            'future_loss_min', 'future_loss_threshold',
            'future_profit_max', 'future_profit_threshold'
        ]
        return self.indicators_present(ind_list, dataframe)

    # function to get buy signals
    def get_entry_training_signals(self, future_df: DataFrame):

        peaks = np.zeros(future_df.shape[0], dtype=float)
        order = 4

        p_idx = scipy.signal.argrelextrema(future_df['bb_width'].to_numpy(), np.greater_equal, order=order)[0]
        peaks[p_idx] = 1.0

        signals = np.where(
            (
                # (future_df['fisher_wr'] > 0.1) &  # guard

                # peak detected
                    (peaks > 0) &

                    # future profit exceeds threshold
                    (future_df['future_profit_max'] >= future_df['future_profit_threshold'])

            ), 1.0, 0.0)
        return signals

    # function to get sell signals

    def get_exit_training_signals(self, future_df: DataFrame):
        valleys = np.zeros(future_df.shape[0], dtype=float)
        order = 4

        v_idx = scipy.signal.argrelextrema(future_df['bb_width'].to_numpy(), np.less_equal, order=order)[0]
        valleys[v_idx] = 1.0

        signals = np.where(
            (
                # (future_df['fisher_wr'] < -0.1) &  # guard

                # valley detected
                    (valleys > 0.0) &

                    # future loss exceeds threshold
                    (future_df['future_loss_min'] <= future_df['future_loss_threshold'])
            ), 1.0, 0.0)
        return signals

    # function to get entry/buy guard conditions

    def get_entry_guard_conditions(self, dataframe: DataFrame):
        conditions = np.where(
            (
                (dataframe['fisher_wr'] < -0.5)
                # (dataframe['close'] <= dataframe['recent_min'])  # local low
            ), 1.0, 0.0)
        return conditions

    # function to get entry/buy guard conditions

    def get_exit_guard_conditions(self, dataframe: DataFrame):
        conditions = np.where(
            (
                (dataframe['fisher_wr'] > 0.5)
                # (dataframe['close'] >= dataframe['recent_max'])  # local high
            ), 1.0, 0.0)
        return conditions

    # function to get list of debug indicators to make visible (e.g. for plotting)

    def get_debug_indicators(self):
        dbg_list = []
        return dbg_list


# -----------------------------------

# training signals based on the discreet Wavelet Transform (DWT) of the closing price.
# 'dwt' is a rolling calculation, i.e. does not look forward
# 'full_dwt' looks ahead, and so can be used for finding buy/sell points in the future
# 'dwt_diff' is the difference between the forward looking model and the rolling model

class dwt_signals(base_signals):

    def check_indicators(self, dataframe) -> bool:
        # check that needed indicators are present
        ind_list = [
            'dwt_diff',
            'future_loss_min', 'future_loss_threshold',
            'future_profit_max', 'future_profit_threshold',
            'recent_max'
        ]
        return self.indicators_present(ind_list, dataframe)

    def get_entry_training_signals(self, future_df: DataFrame):
        global lookahead
        signals = np.where(
            (
                (future_df['fisher_wr'] < -0.5) &
                # (future_df['close'] <= future_df['recent_min']) &  # local low

                # forward model below backward model
                    (future_df['dwt_diff'] < 0) &

                    # # current loss below threshold
                    # (future_df['dwt_diff'] <= future_df['future_loss_threshold']) &

                    # forward model above backward model at lookahead
                    (future_df['dwt_diff'].shift(-self.lookahead) > 0) &

                    # future profit exceeds threshold
                    (future_df['future_profit_max'] >= future_df['future_profit_threshold'])
            ), 1.0, 0.0)
        return signals

    def get_exit_training_signals(self, future_df: DataFrame):
        global lookahead
        signals = np.where(
            (
                (future_df['fisher_wr'] > 0.5) &
                    # (future_df['close'] >= future_df['recent_max']) &  # local high

                    # forward model above backward model
                    (future_df['dwt_diff'] > 0) &

                    # # current profit above threshold
                    # (future_df['dwt_diff'] >= future_df['future_profit_threshold']) &

                    # forward model below backward model at lookahead
                    (future_df['dwt_diff'].shift(-self.lookahead) < 0) &

                    # future loss exceeds threshold
                    (future_df['future_loss_min'] <= future_df['future_loss_threshold'])
            ), 1.0, 0.0)
        return signals

    def get_entry_guard_conditions(self, dataframe: DataFrame):
        condition = np.where(
            (
                (dataframe['fisher_wr'] < -0.5) &
                (dataframe['close'] <= dataframe['recent_min'])  # local low
            ), 1.0, 0.0)
        return condition

        # return None

    def get_exit_guard_conditions(self, dataframe: DataFrame):
        condition = np.where(
            (
                (dataframe['fisher_wr'] > 0.5) &
                (dataframe['close'] >= dataframe['recent_max'])  # local high
            ), 1.0, 0.0)
        return condition

        # return None

    def get_debug_indicators(self):
        return [
            'full_dwt', 'future_max', 'future_min'
        ]


# -----------------------------------

# this version does peak/valley detection on the forward-looking DWT estimate

class dwt2_signals(base_signals):

    def check_indicators(self, dataframe) -> bool:
        # check that needed indicators are present
        ind_list = [
            'full_dwt', 'fisher_wr',
            'future_loss_min', 'future_loss_threshold',
            'future_profit_max', 'future_profit_threshold'
        ]
        return self.indicators_present(ind_list, dataframe)

    def get_entry_training_signals(self, future_df: DataFrame):
        global lookahead
        # detect valleys
        valleys = np.zeros(future_df.shape[0], dtype=float)
        order = 4

        v_idx = scipy.signal.argrelextrema(future_df['full_dwt'].to_numpy(), np.less_equal, order=order)[0]
        valleys[v_idx] = 1.0

        signals = np.where(
            (
                    (future_df['fisher_wr'] < -0.1) &  # guard

                    # valley detected
                    (valleys > 0.0) &

                    # future profit exceeds threshold
                    (future_df['future_profit_max'] >= future_df['future_profit_threshold'])
            ), 1.0, 0.0)
        return signals

    def get_exit_training_signals(self, future_df: DataFrame):
        global lookahead

        peaks = np.zeros(future_df.shape[0], dtype=float)
        order = 4

        p_idx = scipy.signal.argrelextrema(future_df['full_dwt'].to_numpy(), np.greater_equal, order=order)[0]
        peaks[p_idx] = 1.0

        signals = np.where(
            (
                    (future_df['fisher_wr'] > 0.1) &  # guard

                    # peak detected
                    (peaks > 0) &

                    # future loss exceeds threshold
                    (future_df['future_loss_min'] <= future_df['future_loss_threshold'])
            ), 1.0, 0.0)
        return signals

    def get_entry_guard_conditions(self, dataframe: DataFrame):
        condition = np.where(
            (
                (dataframe['fisher_wr'] < -0.1)
            ), 1.0, 0.0)
        return condition

    def get_exit_guard_conditions(self, dataframe: DataFrame):
        condition = np.where(
            (
                (dataframe['fisher_wr'] > 0.1)
            ), 1.0, 0.0)
        return condition

    def get_debug_indicators(self):
        return [
            'full_dwt', 'future_max', 'future_min'
        ]


# -----------------------------------

# fbb = Fisher/Williams and Bollinger Band
# fisher_wr is a combination of the Fisher ratio and Williams Ratio, scaled to [-1, 1]
#   -ve values indicate an oversold region, +ve overbought
# bb_gain and bb_loss represent the potential gain/loss if the price jumps to top/bottom of the current
#    Bollinger Band

# This gives really good signals, but it appears that it is not easy for ML models to correlate this to inputs

class fbb_signals(base_signals):

    def check_indicators(self, dataframe) -> bool:
        # check that needed indicators are present
        ind_list = [
            'mfi', 'fisher_wr', 'bb_gain',
            'profit_threshold', 'loss_threshold',
            'future_loss_min', 'future_loss_threshold',
            'future_profit_max', 'future_profit_threshold'
        ]
        return self.indicators_present(ind_list, dataframe)

    def get_entry_training_signals(self, future_df: DataFrame):
        signals = np.where(
            (
                # oversold condition with high potential profit
                    (future_df['mfi'] < 50) &  # MFI in buy range
                    (future_df['fisher_wr'] < -0.8) &
                    (future_df['bb_gain'] >= future_df['profit_threshold'] / 100.0) &

                    # future profit
                    (future_df['future_profit_max'] >= future_df['future_profit_threshold']) &
                    (future_df['future_gain'] > 0)
            ), 1.0, 0.0)
        return signals

    def get_exit_training_signals(self, future_df: DataFrame):
        signals = np.where(
            (
                # overbought condition with high potential loss
                    (future_df['mfi'] > 50) &  # MFI in sell range
                    (future_df['fisher_wr'] > 0.8) &
                    (future_df['bb_loss'] <= future_df['loss_threshold'] / 100.0) &

                    # future loss
                    (future_df['future_profit_min'] <= future_df['future_loss_threshold'])
            ), 1.0, 0.0)
        return signals

    def get_entry_guard_conditions(self, dataframe: DataFrame):
        condition = np.where(
            (
                # buy region
                    (dataframe['fisher_wr'] < -0.5) &

                    # N down sequences
                    (dataframe['dwt_nseq_dn'] >= 2)
            ), 1.0, 0.0)
        return condition

    def get_exit_guard_conditions(self, dataframe: DataFrame):
        condition = np.where(
            (
                # sell region
                    (dataframe['fisher_wr'] > 0.5) &

                    # N up sequences
                    (dataframe['dwt_nseq_up'] >= 2)
            ), 1.0, 0.0)
        return condition

    def get_debug_indicators(self):
        return [
            'future_gain',
            'future_profit_max',
            'future_loss_min'
        ]


# -----------------------------------

# fwr = Fisher/WIlliams ratio
#   fairly simple oversold/overbought signals (with future profi/loss)

class fwr_signals(base_signals):

    def check_indicators(self, dataframe) -> bool:
        # check that needed indicators are present
        ind_list = [
            'fisher_wr',
            'future_loss_min', 'future_loss_threshold',
            'future_profit_max', 'future_profit_threshold'
        ]
        return self.indicators_present(ind_list, dataframe)

    def get_entry_training_signals(self, future_df: DataFrame):
        signals = np.where(
            (
                # oversold condition
                    (future_df['fisher_wr'] <= -0.8) &

                    # future profit
                    (future_df['future_profit_max'] >= future_df['future_profit_threshold'])
            ), 1.0, 0.0)
        return signals

    def get_exit_training_signals(self, future_df: DataFrame):
        signals = np.where(
            (
                # overbought condition
                    (future_df['fisher_wr'] >= 0.8) &

                    # future loss
                    (future_df['future_loss_min'] <= future_df['future_loss_threshold'])
            ), 1.0, 0.0)
        return signals

    def get_entry_guard_conditions(self, dataframe: DataFrame):
        condition = np.where(
            (
                (dataframe['fisher_wr'] < -0.5)
            ), 1.0, 0.0)
        return condition

    def get_exit_guard_conditions(self, dataframe: DataFrame):
        condition = np.where(
            (
                (dataframe['fisher_wr'] > 0.5)
            ), 1.0, 0.0)
        return condition

    def get_debug_indicators(self):
        return [
        ]


# -----------------------------------

# highlow - detects highs and lows within the lookahead window

class highlow_signals(base_signals):

    def check_indicators(self, dataframe) -> bool:
        # check that needed indicators are present
        ind_list = [
            'dwt_at_low', 'dwt_at_high', 'fisher_wr',
            'future_loss_min', 'future_loss_threshold',
            'future_profit_max', 'future_profit_threshold'
        ]
        return self.indicators_present(ind_list, dataframe)

    def get_entry_training_signals(self, future_df: DataFrame):
        signals = np.where(
            (
                    (future_df['dwt_at_low'] > 0) &  # at low of full window
                    # (future_df['full_dwt'] <= future_df['future_min'])  # at min of future window
                    (future_df['future_profit_max'] >= future_df['future_profit_threshold'])
            ), 1.0, 0.0)
        return signals

    def get_exit_training_signals(self, future_df: DataFrame):
        signals = np.where(
            (
                    (future_df['dwt_at_high'] > 0) &  # at high of full window
                    # (future_df['full_dwt'] >= future_df['future_max'])  # at max of future window
                    (future_df['future_loss_min'] <= future_df['future_loss_threshold'])
            ), 1.0, 0.0)
        return signals

    def get_entry_guard_conditions(self, dataframe: DataFrame):
        condition = np.where(
            (
                (dataframe['fisher_wr'] < -0.5)
                # (dataframe['close'] <= dataframe['recent_min'])  # local low
            ), 1.0, 0.0)
        return condition

    def get_exit_guard_conditions(self, dataframe: DataFrame):
        condition = np.where(
            (
                (dataframe['fisher_wr'] > 0.5)
                # (dataframe['close'] >= dataframe['recent_max'])  # local high
            ), 1.0, 0.0)
        return condition

    def get_debug_indicators(self):
        return [
            'dwt_at_low',
            'dwt_at_high',
            'future_min',
            'future_max'
        ]


# -----------------------------------

# jump - looks for 'big' jumps in price, with future profit/loss

class jump_signals(base_signals):

    def check_indicators(self, dataframe) -> bool:
        # check that needed indicators are present
        ind_list = [
            'dwt_delta_min', 'dwt_delta_max',
            'future_loss_min', 'future_loss_threshold',
            'future_profit_max', 'future_profit_threshold'
        ]
        return self.indicators_present(ind_list, dataframe)

    def get_entry_training_signals(self, future_df: DataFrame):
        signals = np.where(
            (
                # previous candle dropped more than 2 std dev
                    (
                            future_df['gain'].shift() <=
                            (future_df['future_loss_mean'] - 2.0 * abs(future_df['future_loss_std']))
                    ) &

                    # big drop somewhere in previous window
                    (future_df['dwt_delta_min'] <= 1.0) &

                    # upcoming window exceeds profit threshold
                    (future_df['future_profit_max'] >= future_df['future_profit_threshold'])
            ), 1.0, 0.0)
        return signals

    def get_exit_training_signals(self, future_df: DataFrame):
        signals = np.where(
            (
                # previous candle gained more than 2 std dev
                    (
                            future_df['gain'].shift() >=
                            (future_df['future_profit_mean'] + 2.0 * abs(future_df['future_profit_std']))
                    ) &

                    # big gain somewhere in previous window
                    (future_df['dwt_delta_max'] >= 1.0) &

                    # upcoming window exceeds loss threshold
                    (future_df['future_loss_min'] <= future_df['future_loss_threshold'])
            ), 1.0, 0.0)
        return signals

    def get_entry_guard_conditions(self, dataframe: DataFrame):
        condition = np.where(
            (
                # N down sequences
                (dataframe['dwt_nseq_dn'] >= 2)
            ), 1.0, 0.0)
        return condition

    def get_exit_guard_conditions(self, dataframe: DataFrame):
        condition = np.where(
            (
                # N up sequences
                (dataframe['dwt_nseq_up'] >= 2)
            ), 1.0, 0.0)
        return condition

    def get_debug_indicators(self):
        return [
            'dwt_delta_min',
            'dwt_delta_max'
        ]


# -----------------------------------

# macd - classic MACD crossing events

class macd_signals(base_signals):

    def check_indicators(self, dataframe) -> bool:
        # check that needed indicators are present
        ind_list = [
            'macdhist',
            'future_loss_min', 'future_loss_threshold',
            'future_profit_max', 'future_profit_threshold'
        ]
        return self.indicators_present(ind_list, dataframe)

    def get_entry_training_signals(self, future_df: DataFrame):
        signals = np.where(
            (
                # MACD turns around -ve to +ve
                    (future_df['macdhist'].shift() < 0) &
                    (future_df['macdhist'] >= 0) &

                    # future gain exceeds threshold
                    (future_df['future_profit_max'] >= future_df['future_profit_threshold'])
            ), 1.0, 0.0)
        return signals

    def get_exit_training_signals(self, future_df: DataFrame):
        signals = np.where(
            (
                # MACD turns around +ve to -ve
                    (future_df['macdhist'].shift() > 0) &
                    (future_df['macdhist'] <= 0) &

                    # future loss exceeds threshold
                    (future_df['future_loss_min'] <= future_df['future_loss_threshold'])
            ), 1.0, 0.0)
        return signals

    def get_entry_guard_conditions(self, dataframe: DataFrame):
        condition = np.where(
            (
                (dataframe['macdhist'] < 0)
            ), 1.0, 0.0)
        return condition

    def get_exit_guard_conditions(self, dataframe: DataFrame):
        condition = np.where(
            (
                (dataframe['macdhist'] > 0)
            ), 1.0, 0.0)
        return condition

    def get_debug_indicators(self):
        return [
        ]


# -----------------------------------

# macd2 - modified MACD, trigger when macdhistory is at a low/high

class macd2_signals(base_signals):

    def check_indicators(self, dataframe) -> bool:
        # check that needed indicators are present
        ind_list = [
            'macdhist',
            'future_loss_min', 'future_loss_threshold',
            'future_profit_max', 'future_profit_threshold'
        ]
        return self.indicators_present(ind_list, dataframe)

    def get_entry_training_signals(self, future_df: DataFrame):
        # detect valleys
        valleys = np.zeros(future_df.shape[0], dtype=float)
        order = 4

        v_idx = scipy.signal.argrelextrema(future_df['macdhist'].to_numpy(), np.less_equal, order=order)[0]
        valleys[v_idx] = 1.0

        signals = np.where(
            (
                # valley detected
                    (valleys > 0.0) &

                    # future profit exceeds threshold
                    (future_df['future_profit_max'] >= future_df['future_profit_threshold'])
            ), 1.0, 0.0)
        return signals

    def get_exit_training_signals(self, future_df: DataFrame):
        peaks = np.zeros(future_df.shape[0], dtype=float)
        order = 2

        p_idx = scipy.signal.argrelextrema(future_df['macdhist'].to_numpy(), np.greater_equal, order=order)[0]
        peaks[p_idx] = 1.0

        signals = np.where(
            (
                # peak detected
                    (peaks > 0) &

                    # future loss exceeds threshold
                    (future_df['future_loss_min'] <= future_df['future_loss_threshold'])
            ), 1.0, 0.0)
        return signals

    def get_entry_guard_conditions(self, dataframe: DataFrame):
        condition = np.where(
            (
                (dataframe['macdhist'] < 0)
            ), 1.0, 0.0)
        return condition

    def get_exit_guard_conditions(self, dataframe: DataFrame):
        condition = np.where(
            (
                (dataframe['macdhist'] > 0)
            ), 1.0, 0.0)
        return condition

    def get_debug_indicators(self):
        return [
        ]


# -----------------------------------

# macd3 - modified MACD, trigger when macdhistory is in a low/high region

class macd3_signals(base_signals):

    def check_indicators(self, dataframe) -> bool:
        # check that needed indicators are present
        ind_list = [
            'macdhist', 'mfi',
            'future_loss_min', 'future_loss_threshold',
            'future_profit_max', 'future_profit_threshold'
        ]
        return self.indicators_present(ind_list, dataframe)

    def get_entry_training_signals(self, future_df: DataFrame):
        # MACD is related to price, so need to figure out scale
        macd_neg = future_df['macdhist'].clip(upper=0.0)
        # threshold = macd_neg.mean() - abs(macd_neg.std())
        threshold = macd_neg.mean()

        # print(f"DBG: buy threshold: {threshold}")

        signals = np.where(
            (
                # macdhist in low region
                    (future_df['macdhist'] < threshold) &

                    # buy region
                    (future_df['mfi'] < 50) &

                    # future profit exceeds threshold
                    (future_df['future_profit_max'] >= future_df['future_profit_threshold'])
            ), 1.0, 0.0)
        return signals

    def get_exit_training_signals(self, future_df: DataFrame):
        macd_pos = future_df['macdhist'].clip(lower=0.0)
        # threshold = macd_pos.mean() + abs(macd_pos.std())
        threshold = macd_pos.mean()

        # print(f"DBG: sell threshold: {threshold}")

        signals = np.where(
            (
                # macdhist in high region
                    (future_df['macdhist'] > threshold) &

                    # sell region
                    (future_df['mfi'] > 50) &

                    # future loss exceeds threshold
                    (future_df['future_loss_min'] <= future_df['future_loss_threshold'])
            ), 1.0, 0.0)
        return signals

    def get_entry_guard_conditions(self, dataframe: DataFrame):
        condition = np.where(
            (
                (dataframe['macdhist'] < 0)
            ), 1.0, 0.0)
        return condition

    def get_exit_guard_conditions(self, dataframe: DataFrame):
        condition = np.where(
            (
                (dataframe['macdhist'] > 0)
            ), 1.0, 0.0)
        return condition

    def get_debug_indicators(self):
        return [
        ]


# -----------------------------------

# MFI - Chaikin Money Flow Indicator. Simple oversold/overbought strategy

class mfi_signals(base_signals):

    def check_indicators(self, dataframe) -> bool:
        # check that needed indicators are present
        ind_list = [
            'mfi',
            'future_loss_min', 'future_loss_threshold',
            'future_profit_max', 'future_profit_threshold'
        ]
        return self.indicators_present(ind_list, dataframe)

    def get_entry_training_signals(self, future_df: DataFrame):
        signals = np.where(
            (
                # oversold condition
                    (future_df['mfi'] <= 15) &

                    # future profit
                    (future_df['future_profit_max'] >= future_df['future_profit_threshold'])
            ), 1.0, 0.0)
        return signals

    def get_exit_training_signals(self, future_df: DataFrame):
        signals = np.where(
            (
                # overbought condition
                    (future_df['mfi'] >= 85) &

                    # future loss
                    (future_df['future_loss_min'] <= future_df['future_loss_threshold'])
            ), 1.0, 0.0)
        return signals

    def get_entry_guard_conditions(self, dataframe: DataFrame):
        condition = np.where(
            (
                (dataframe['mfi'] <= 30)
            ), 1.0, 0.0)
        return condition

    def get_exit_guard_conditions(self, dataframe: DataFrame):
        condition = np.where(
            (
                (dataframe['mfi'] >= 60)
            ), 1.0, 0.0)
        return condition

    def get_debug_indicators(self):
        return [
        ]


# -----------------------------------

# detect the max (sell) or min (buy) of both the past window and the future window

class minmax_signals(base_signals):

    def check_indicators(self, dataframe) -> bool:
        # check that needed indicators are present
        ind_list = [
            'full_dwt', 'dwt_recent_min', 'dwt_recent_max', 'fisher_wr',
            'future_loss_min', 'future_loss_threshold',
            'future_profit_max', 'future_profit_threshold'
        ]
        return self.indicators_present(ind_list, dataframe)

    def get_entry_training_signals(self, future_df: DataFrame):
        signals = np.where(
            (
                # at min of past window
                    (future_df['full_dwt'] <= future_df['dwt_recent_min']) &

                    # at min of future window
                    (future_df['full_dwt'] <= future_df['future_min']) &

                    # future profit exceeds threshold
                    (future_df['future_profit_max'] >= future_df['future_profit_threshold'])
            ), 1.0, 0.0)
        return signals

    def get_exit_training_signals(self, future_df: DataFrame):
        signals = np.where(
            (
                # at max of past window
                    (future_df['full_dwt'] >= future_df['dwt_recent_max']) &

                    # at max of future window
                    (future_df['full_dwt'] >= future_df['future_max']) &

                    # loss in next window exceeds threshold
                    (future_df['future_loss_min'] <= future_df['future_loss_threshold'])
            ), 1.0, 0.0)
        return signals

    def get_entry_guard_conditions(self, dataframe: DataFrame):
        condition = np.where(
            (
                (dataframe['fisher_wr'] < -0.5)
                # (dataframe['close'] <= dataframe['recent_min'])  # local low
            ), 1.0, 0.0)
        return condition

    def get_exit_guard_conditions(self, dataframe: DataFrame):
        condition = np.where(
            (
                (dataframe['fisher_wr'] > 0.5)
                # (dataframe['close'] >= dataframe['recent_max'])  # local high
            ), 1.0, 0.0)
        return condition

    def get_debug_indicators(self):
        return [
            'future_loss_min',
            'future_profit_max'
        ]


# -----------------------------------

# NSeq - continuous sequences of up/down movements

class nseq_signals(base_signals):

    def check_indicators(self, dataframe) -> bool:
        # check that needed indicators are present
        ind_list = [
            'dwt_nseq_dn', 'dwt_nseq_up',
            'full_dwt_nseq_dn', 'full_dwt_nseq_up',
            'future_loss_min', 'future_loss_threshold',
            'future_profit_max', 'future_profit_threshold'
        ]
        return self.indicators_present(ind_list, dataframe)

    def get_entry_training_signals(self, future_df: DataFrame):
        signals = np.where(
            (
                # long down run just happened, or a long up run is about to happen
                    (
                            (future_df['full_dwt_nseq_dn'] >= 10) |

                            (future_df['future_nseq_up'] >= 15)
                    ) &

                    # future profit exceeds threshold
                    (future_df['future_profit_max'] >= future_df['future_profit_threshold'])
            ), 1.0, 0.0)
        return signals

    def get_exit_training_signals(self, future_df: DataFrame):
        signals = np.where(
            (
                # long up run just happened, or a long down run is about to happen
                    (
                            (future_df['full_dwt_nseq_up'] >= 10) |

                            (future_df['future_nseq_dn'] >= 15)
                    ) &

                    # future profit exceeds threshold
                    (future_df['future_loss_min'] <= future_df['future_loss_threshold'])
            ), 1.0, 0.0)
        return signals

    def get_entry_guard_conditions(self, dataframe: DataFrame):
        cond = np.where(
            (
                # N down sequences
                (dataframe['dwt_nseq_dn'] >= 2)
            ), 1.0, 0.0)
        return cond

    def get_exit_guard_conditions(self, dataframe: DataFrame):
        cond = np.where(
            (
                # N up sequences
                (dataframe['dwt_nseq_up'] >= 2)
            ), 1.0, 0.0)
        return cond

    def get_debug_indicators(self):
        return [
            'full_dwt',
            'full_dwt_nseq_up',
            'full_dwt_nseq_dn',
            'future_nseq_up',
            'future_nseq_dn'
        ]


# -----------------------------------

# over - combination of various overbought/oversold indicators

# Note: cannot be too restrictive or there will not be enough signals to train

class over_signals(base_signals):

    def check_indicators(self, dataframe) -> bool:
        # check that needed indicators are present
        ind_list = [
            'rsi', 'mfi', 'fisher_wr',
            'future_loss_min', 'future_loss_threshold',
            'future_profit_max', 'future_profit_threshold'
        ]
        return self.indicators_present(ind_list, dataframe)

    def get_entry_training_signals(self, future_df: DataFrame):
        signals = np.where(
            (
                # various overbought condition (can't be too strict or there will be no matches)
                    (future_df['rsi'] < 40) &
                    (future_df['mfi'] < 40) &
                    (future_df['fisher_wr'] < -0.4) &

                    # future profit
                    (future_df['future_profit_max'] >= future_df['future_profit_threshold'])
            ), 1.0, 0.0)
        return signals

    def get_exit_training_signals(self, future_df: DataFrame):
        signals = np.where(
            (
                    (future_df['rsi'] > 60) &
                    (future_df['mfi'] > 60) &
                    (future_df['fisher_wr'] > 0.6) &

                    # future loss
                    (future_df['future_loss_min'] <= future_df['future_loss_threshold'])
            ), 1.0, 0.0)
        return signals

    def get_entry_guard_conditions(self, dataframe: DataFrame):
        condition = np.where(
            (
                (dataframe['fisher_wr'] < -0.5)
            ), 1.0, 0.0)
        return condition

    def get_exit_guard_conditions(self, dataframe: DataFrame):
        condition = np.where(
            (
                (dataframe['fisher_wr'] > 0.5)
            ), 1.0, 0.0)
        return condition

    def get_debug_indicators(self):
        return [
            'future_loss_min',
            'future_profit_max'
        ]


# -----------------------------------

# Profit

class profit_signals(base_signals):

    n_profit_stddevs = 2.0
    n_loss_stddevs = 2.0

    def check_indicators(self, dataframe) -> bool:
        # check that needed indicators are present
        ind_list = [
            'fisher_wr',
            'future_loss_min', 'future_loss_threshold',
            'future_profit_max', 'future_profit_threshold'
        ]
        return self.indicators_present(ind_list, dataframe)

    def get_entry_training_signals(self, future_df: DataFrame):
        signals = np.where(
            (
                    (future_df['fisher_wr'] < -0.5) &

                    # qtpylib.crossed_above(future_df['future_gain'], 2.0 * future_df['future_profit_threshold'])
                    (future_df['future_profit_max'] >= 2.0 * future_df['future_profit_threshold'])
            ), 1.0, 0.0)
        return signals

    def get_exit_training_signals(self, future_df: DataFrame):
        signals = np.where(
            (
                    (future_df['fisher_wr'] > 0.5) &

                    # qtpylib.crossed_below(future_df['future_gain'], 2.0 * future_df['future_loss_threshold'])
                    (future_df['future_loss_min'] <= 2.0 * future_df['future_loss_threshold'])
            ), 1.0, 0.0)
        return signals

    def get_entry_guard_conditions(self, dataframe: DataFrame):
        condition = np.where(
            (
                (dataframe['fisher_wr'] < -0.5)
            ), 1.0, 0.0)
        return condition

    def get_exit_guard_conditions(self, dataframe: DataFrame):
        condition = np.where(
            (
                (dataframe['fisher_wr'] > 0.5)
            ), 1.0, 0.0)
        return condition

    def get_debug_indicators(self):
        return [
            'future_profit_threshold',
            'future_loss_threshold',
            'future_gain',
            'future_slope'
        ]


# -----------------------------------

# pv - peak/valley detection

class pv_signals(base_signals):

    def check_indicators(self, dataframe) -> bool:
        # check that needed indicators are present
        ind_list = [
            'full_dwt',
            'recent_min', 'recent_max',
            'future_loss_min', 'future_loss_threshold',
            'future_profit_max', 'future_profit_threshold'
        ]
        return self.indicators_present(ind_list, dataframe)

    def get_entry_training_signals(self, future_df: DataFrame):
        global lookahead
        valleys = np.zeros(future_df.shape[0], dtype=float)
        order = 4

        v_idx = scipy.signal.argrelextrema(future_df['full_dwt'].to_numpy(), np.less_equal, order=order)[0]
        valleys[v_idx] = 1.0

        # print(f'future_df: {future_df.shape} valleys: {np.shape(valleys)}')
        signals = np.where(
            (
                # overbought condition with high potential profit
                    (valleys > 0.0) &

                    # future profit
                    (future_df['future_profit_max'] >= future_df['future_profit_threshold']) &
                    (future_df['future_gain'] > 0)
            ), 1.0, 0.0)
        return signals

    def get_exit_training_signals(self, future_df: DataFrame):
        global lookahead
        peaks = np.zeros(future_df.shape[0], dtype=float)
        order = 4

        p_idx = scipy.signal.argrelextrema(future_df['full_dwt'].to_numpy(), np.greater_equal, order=order)[0]
        peaks[p_idx] = 1.0

        signals = np.where(
            (
                    (peaks > 0) &

                    # future loss
                    (future_df['future_loss_min'] <= future_df['future_loss_threshold']) &
                    (future_df['future_gain'] < 0)
            ), 1.0, 0.0)
        return signals

    def get_entry_guard_conditions(self, dataframe: DataFrame):
        condition = np.where(
            (
                (dataframe['close'] <= dataframe['recent_min'])  # local low
            ), 1.0, 0.0)
        return condition

    def get_exit_guard_conditions(self, dataframe: DataFrame):
        condition = np.where(
            (
                (dataframe['close'] >= dataframe['recent_max'])  # local high
            ), 1.0, 0.0)
        return condition

    def get_debug_indicators(self):
        return [
        ]


# -----------------------------------

# slope - examines the average slope, past & future

class slope_signals(base_signals):

    def check_indicators(self, dataframe) -> bool:
        # check that needed indicators are present
        ind_list = [
            'dwt_slope', 'fisher_wr', 'future_slope',
            'future_loss_min', 'future_loss_threshold',
            'future_profit_max', 'future_profit_threshold'
        ]
        return self.indicators_present(ind_list, dataframe)

    def get_entry_training_signals(self, future_df: DataFrame):
        global lookahead
        signals = np.where(
            (
                # in a downtrend
                    (future_df['dwt_slope'] < 0) &

                    # future up trend
                    (future_df['future_slope'] > 0) &

                    (future_df['fisher_wr'] < -0.5) &

                    # future profit
                    (future_df['future_profit_max'] >= future_df['future_profit_threshold']) &
                    (future_df['future_gain'] > 0)
            ), 1.0, 0.0)
        return signals

    def get_exit_training_signals(self, future_df: DataFrame):
        global lookahead
        signals = np.where(
            (
                # in an uptrend
                    (future_df['dwt_slope'] > 0) &

                    # future down trend
                    (future_df['future_slope'] < 0) &

                    (future_df['fisher_wr'] > 0.5) &

                    # future loss
                    (future_df['future_loss_min'] <= future_df['future_loss_threshold']) &
                    (future_df['future_gain'] < 0)

            ), 1.0, 0.0)
        return signals

    def get_entry_guard_conditions(self, dataframe: DataFrame):
        condition = np.where(
            (
                (dataframe['fisher_wr'] < -0.5)
            ), 1.0, 0.0)
        return condition
        # return None

    def get_exit_guard_conditions(self, dataframe: DataFrame):
        condition = np.where(
            (
                (dataframe['fisher_wr'] > 0.5)
            ), 1.0, 0.0)
        return condition
        # return None

    def get_debug_indicators(self):
        return [
            'future_slope'
        ]


# -----------------------------------

# smooth - find peaks & valleys on smoothed version of price
# the theory is that this should avoid the smaller peaks & valleys

class smooth_signals(base_signals):

    def check_indicators(self, dataframe) -> bool:
        # check that needed indicators are present
        ind_list = [
            'mid', 'fisher_wr',
            'future_loss_min', 'future_loss_threshold',
            'future_profit_max', 'future_profit_threshold'
        ]
        return self.indicators_present(ind_list, dataframe)

    def get_entry_training_signals(self, future_df: DataFrame):
        global lookahead

        # get the smoothed version
        smoothed = future_df['mid'].ewm(span=7).mean().to_numpy()

        # detect valleys
        valleys = np.zeros(future_df.shape[0], dtype=float)
        order = 4

        v_idx = scipy.signal.argrelextrema(smoothed, np.less_equal, order=order)[0]
        valleys[v_idx] = 1.0

        signals = np.where(
            (
                    (future_df['fisher_wr'] < -0.1) &  # guard

                    # valley detected
                    (valleys > 0.0) &

                    # future profit exceeds threshold
                    (future_df['future_profit_max'] >= future_df['future_profit_threshold'])
            ), 1.0, 0.0)
        return signals

    def get_exit_training_signals(self, future_df: DataFrame):
        global lookahead

        # get the smoothed version
        smoothed = future_df['mid'].ewm(span=7).mean().to_numpy()

        peaks = np.zeros(future_df.shape[0], dtype=float)
        order = 4

        p_idx = scipy.signal.argrelextrema(smoothed, np.greater_equal, order=order)[0]
        peaks[p_idx] = 1.0

        signals = np.where(
            (
                    (future_df['fisher_wr'] > 0.1) &  # guard

                    # peak detected
                    (peaks > 0) &

                    # future loss exceeds threshold
                    (future_df['future_loss_min'] <= future_df['future_loss_threshold'])
            ), 1.0, 0.0)
        return signals

    def get_entry_guard_conditions(self, dataframe: DataFrame):
        condition = np.where(
            (
                (dataframe['fisher_wr'] < -0.5)
            ), 1.0, 0.0)
        return condition
        # return None

    def get_exit_guard_conditions(self, dataframe: DataFrame):
        condition = np.where(
            (
                (dataframe['fisher_wr'] > 0.5)
            ), 1.0, 0.0)
        return condition
        # return None

    def get_debug_indicators(self):
        return [
            'future_slope'
        ]


# -----------------------------------

# stochastic - detect points where fast stochastic (%K) changes direction
# above 80 implies sell, below 20 implies buy

class stochastic_signals(base_signals):

    def check_indicators(self, dataframe) -> bool:
        # check that needed indicators are present
        ind_list = [
            'fast_diff',
            'future_loss_min', 'future_loss_threshold',
            'future_profit_max', 'future_profit_threshold'
        ]
        return self.indicators_present(ind_list, dataframe)

    def get_entry_training_signals(self, future_df: DataFrame):
        global lookahead
        signals = np.where(
            (
                # stochastics show overbought condition
                    ((future_df['fast_diff'] > 0) & (future_df['fast_diff'].shift(-self.lookahead) <= 0)) &

                    # future profit
                    (future_df['future_profit_max'] >= future_df['future_profit_threshold']) &
                    (future_df['future_gain'] > 0)
            ), 1.0, 0.0)
        return signals

    def get_exit_training_signals(self, future_df: DataFrame):
        global lookahead
        signals = np.where(
            (
                # stochastics show oversold condition
                    ((future_df['fast_diff'] < 0) & (future_df['fast_diff'].shift(-self.lookahead) >= 0)) &

                    # future loss
                    (future_df['future_loss_min'] <= future_df['future_loss_threshold']) &
                    (future_df['future_gain'] < 0)

            ), 1.0, 0.0)
        return signals

    def get_entry_guard_conditions(self, dataframe: DataFrame):
        condition = np.where(
            (
                (dataframe['fast_diff'] > 0)
            ), 1.0, 0.0)
        return condition

    def get_exit_guard_conditions(self, dataframe: DataFrame):
        condition = np.where(
            (
                (dataframe['fast_diff'] < 0)
            ), 1.0, 0.0)
        return condition

    def get_debug_indicators(self):
        return [
        ]


# -----------------------------------

# 'large' swings (down/up or up/down)

class swing_signals(base_signals):

    def check_indicators(self, dataframe) -> bool:
        # check that needed indicators are present
        ind_list = [
            'dwt_bottom', 'dwt_top',
            'recent_min', 'recent_max',
            'future_loss_min', 'future_loss_threshold',
            'future_profit_max', 'future_profit_threshold'
        ]
        return self.indicators_present(ind_list, dataframe)

    def get_entry_training_signals(self, future_df: DataFrame):
        signals = np.where(
            (
                # bottom of trend
                    (future_df['dwt_bottom'] > 0) &

                    # future gain
                    (future_df['future_profit_max'] >= future_df['future_profit_threshold']) &
                    (future_df['future_profit'] > 0)
            ), 1.0, 0.0)
        return signals

    def get_exit_training_signals(self, future_df: DataFrame):
        signals = np.where(
            (
                # top of trend
                    (future_df['dwt_top'] > 0) &

                    # future gain
                    (future_df['future_loss_min'] <= future_df['future_loss_threshold']) &
                    (future_df['future_loss'] < 0)
            ), 1.0, 0.0)
        return signals

    def get_entry_guard_conditions(self, dataframe: DataFrame):
        condition = np.where(
            (
                (dataframe['close'] <= dataframe['recent_min'])  # local low
            ), 1.0, 0.0)
        return condition

    def get_exit_guard_conditions(self, dataframe: DataFrame):
        condition = np.where(
            (
                (dataframe['close'] >= dataframe['recent_max'])  # local high
            ), 1.0, 0.0)
        return condition

    def get_debug_indicators(self):
        return [
            'future_loss_min',
            'future_profit_max'
        ]

# -----------------------------------

# combines all signals together (in an OR fashion, not AND)

class all_signals(base_signals):

    n_profit_stddevs = 2.0
    n_loss_stddevs = 2.0

    def check_indicators(self, dataframe) -> bool:
        # check that needed indicators are present
        ind_list = [
            'fisher_wr'
        ]
        return self.indicators_present(ind_list, dataframe)

    def get_entry_training_signals(self, future_df: DataFrame):

        signals = None
        count = None # for tracking stats

        # loop through all signal types (except this one)
        for stype in SignalType:
            if stype != SignalType.ALL:
                tsig = create_training_signals(stype, 12)
                sigs = tsig.get_entry_training_signals(future_df)
                if signals is None:
                    signals = sigs
                else:
                    signals = pd.Series(np.where(signals == 1, 1, np.where(sigs == 1, 1, 0)))

                if count is None:
                    count = np.array(sigs)
                else:
                    count = count + np.array(sigs)

        # create filter where number of signals is greater than the mean + stddev (rounded down)
        threshold = int(count.mean()) + int(count.std())
        filter = np.where(count>threshold, 1, 0)

        trend_filter = np.where(
            (
                 (future_df['fisher_wr'] < 0)
            ), 1.0, 0.0)
        
        # filter out the signals, otherwise there are far too many
        signals = signals.astype(int) & filter.astype(int) & trend_filter.astype(int)

        return signals


    def get_exit_training_signals(self, future_df: DataFrame):

        signals = None
        count = None # for tracking stats

        # loop through all signal types (except this one)
        for stype in SignalType:
            if stype != SignalType.ALL:
                tsig = create_training_signals(stype, 12)
                sigs = tsig.get_exit_training_signals(future_df)
                if signals is None:
                    signals = sigs
                else:
                    signals = pd.Series(np.where(signals == 1, 1, np.where(sigs == 1, 1, 0)))
                    # signals = signals | sigs

                if count is None:
                    count = np.array(sigs)
                else:
                    count = count + np.array(sigs)


        # create filter where number of signals is greater than the mean + stddev (rounded down)
        threshold = int(count.mean()) + int(count.std())
        filter = np.where(count>threshold, 1, 0)

        trend_filter = np.where(
            (
                 (future_df['fisher_wr'] > 0)
            ), 1.0, 0.0)
                
        # filter out the signals, otherwise there are far too many
        signals = signals.astype(int) & filter.astype(int) & trend_filter.astype(int)

        return signals

    def get_entry_guard_conditions(self, dataframe: DataFrame):
        condition = np.where(
            (
                 (dataframe['fisher_wr'] <= -0.5)
            ), 1.0, 0.0)
        return condition

    def get_exit_guard_conditions(self, dataframe: DataFrame):
        condition = np.where(
            (
                 (dataframe['fisher_wr'] >= 0.5)
            ), 1.0, 0.0)
        return condition

    def get_debug_indicators(self):
        return [
            'future_loss_min',
            'future_profit_max'
        ]


# -----------------------------------


# enum of all available signal types

class SignalType(Enum):
    ADX = adx_signals
    ADX2 = adx2_signals
    ADX3 = adx3_signals
    ALL = all_signals
    Aroon = aroon_signals
    Bollinger_Width = bbw_signals
    DWT = dwt_signals
    DWT2 = dwt2_signals
    Fisher_Bollinger = fbb_signals
    Fisher_Williams = fwr_signals
    High_Low = highlow_signals
    Jump = jump_signals
    MACD = macd_signals
    MACD2 = macd2_signals
    MACD3 = macd3_signals
    Money_Flow = mfi_signals
    Min_Max = minmax_signals
    N_Sequence = nseq_signals
    Oversold = over_signals
    Profit = profit_signals
    Peaks_Valleys = pv_signals
    Smooth = smooth_signals
    Slope = slope_signals
    Stochastic = stochastic_signals
    Swing = swing_signals


def create_training_signals(signal_type: SignalType, lookahead):
    return signal_type.value(lookahead)

# -----------------------------------
