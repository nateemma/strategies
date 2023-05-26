# pylint: disable=C0103,C0114,C0301,C0116,C0413,C0115,W0611,W0613

# This is a set of utilities for getting training signal (buy/sell) events
# The signals are collected here so that the different approaches to generating events can be more easily used across
# multiple strategies
# Notes:
#   - This not a class, just a collection of functions and data types
#   - Any hard-coded numbers have been derived by staring at charts for hours to figure out appropriate values
#   - These functions all look ahead in the data, so they can *ONLY* be used for training


import numpy as np
import pandas as pd

import scipy

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent))

import logging
import warnings
from enum import Enum

log = logging.getLogger(__name__)
# log.setLevel(logging.DEBUG)
warnings.simplefilter(action='ignore', category=pd.errors.PerformanceWarning)

from pandas import DataFrame, Series

pd.options.mode.chained_assignment = None  # default='warn'


# -----------------------------------
# define a xxx_signals class for each type of training signal
# This allows us to deal with the different types in a fairly generic fashion
# Each class must contain the follow *static* functions (use@staticmethod):
#   get_entry_training_signals()
#   get_exit_training_signals()
#   get_entry_guard_conditions()
#   get_exit_guard_conditions()
#   get_debug_indicators()

# To add a new type of training signal, define the corresponding class and add it to the SignalType enum

# -----------------------------------

# base clas
class base_signals:
    @staticmethod
    def entry_training_signals(future_df: DataFrame):
        return np.zeros(future_df.shape[0], dtype=float)

    # function to get sell signals
    @staticmethod
    def exit_training_signals(future_df: DataFrame):
        return np.zeros(future_df.shape[0], dtype=float)

    # function to get entry/buy guard conditions
    @staticmethod
    def entry_guard_conditions(dataframe: DataFrame):
        return None

    # function to get entry/buy guard conditions
    @staticmethod
    def exit_guard_conditions(dataframe: DataFrame):
        return None

    # function to get list of debug indicators to make visible (e.g. for plotting)
    @staticmethod
    def debug_indicators():
        return []


# -----------------------------------

# training signals based on the width of the Bollinger Bands

class bbw_signals(base_signals):
    # function to get buy signals
    @staticmethod
    def entry_training_signals(future_df: DataFrame):
        peaks = np.zeros(future_df.shape[0], dtype=float)
        order = 4

        p_idx = scipy.signal.argrelextrema(future_df['bb_width'].to_numpy(), np.greater_equal, order=order)[0]
        peaks[p_idx] = 1.0

        signals = np.where(
            (
                # (future_df['fisher_wr'] > 0.1) &  # guard

                # peak detected
                    (peaks > 0) &

                    # future loss exceeds threshold
                    (future_df['future_loss_min'] <= future_df['future_loss_threshold'])
            ), 1.0, 0.0)
        return signals

    # function to get sell signals
    @staticmethod
    def exit_training_signals(future_df: DataFrame):
        valleys = np.zeros(future_df.shape[0], dtype=float)
        order = 4

        v_idx = scipy.signal.argrelextrema(future_df['bb_width'].to_numpy(), np.less_equal, order=order)[0]
        valleys[v_idx] = 1.0

        signals = np.where(
            (
                # (future_df['fisher_wr'] < -0.1) &  # guard

                # valley detected
                    (valleys > 0.0) &

                    # future profit exceeds threshold
                    (future_df['future_profit_max'] >= future_df['future_profit_threshold'])
            ), 1.0, 0.0)
        return signals

    # function to get entry/buy guard conditions
    @staticmethod
    def entry_guard_conditions(dataframe: DataFrame):
        conditions = None
        return conditions

    # function to get entry/buy guard conditions
    @staticmethod
    def exit_guard_conditions(dataframe: DataFrame):
        conditions = None
        return conditions

    # function to get list of debug indicators to make visible (e.g. for plotting)
    @staticmethod
    def debug_indicators():
        dbg_list = []
        return dbg_list


# -----------------------------------

# training signals based on the discreet Wavelet Transform (DWT) of the closing price.
# 'dwt' is a rolling calculation, i.e. does not look forward
# 'full_dwt' looks ahead, and so can be used for finding buy/sell points in the future
# 'dwt_diff' is the difference between the forward looking model and the rolling model

class dwt_signals(base_signals):

    @staticmethod
    def entry_training_signals(future_df: DataFrame):
        global curr_lookahead
        signals = np.where(
            (
                # forward model below backward model
                    (future_df['dwt_diff'] < 0) &

                    # # current loss below threshold
                    # (future_df['dwt_diff'] <= future_df['future_loss_threshold']) &

                    # forward model above backward model at lookahead
                    (future_df['dwt_diff'].shift(-curr_lookahead) > 0) &

                    # future profit exceeds threshold
                    (future_df['future_profit'] >= future_df['future_profit_threshold'])
            ), 1.0, 0.0)
        return signals

    @staticmethod
    def exit_training_signals(future_df: DataFrame):
        global curr_lookahead
        signals = np.where(
            (
                # forward model above backward model
                    (future_df['dwt_diff'] > 0) &

                    # # current profit above threshold
                    # (future_df['dwt_diff'] >= future_df['future_profit_threshold']) &

                    # forward model below backward model at lookahead
                    (future_df['dwt_diff'].shift(-curr_lookahead) < 0) &

                    # future loss exceeds threshold
                    (future_df['future_loss'] <= future_df['future_loss_threshold'])
            ), 1.0, 0.0)
        return signals

    @staticmethod
    def entry_guard_conditions(dataframe: DataFrame):
        return None

    @staticmethod
    def exit_guard_conditions(dataframe: DataFrame):
        return None

    @staticmethod
    def debug_indicators():
        return [
            'full_dwt', 'future_max', 'future_min'
        ]


# -----------------------------------

# this version does peak/valley detection on the forward-looking DWT estimate

class dwt2_signals(base_signals):

    @staticmethod
    def entry_training_signals(future_df: DataFrame):
        global curr_lookahead
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
                    (future_df['future_profit'] >= future_df['future_profit_threshold'])
            ), 1.0, 0.0)
        return signals

    @staticmethod
    def exit_training_signals(future_df: DataFrame):
        global curr_lookahead

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
                    (future_df['future_loss'] <= future_df['future_loss_threshold'])
            ), 1.0, 0.0)
        return signals

    @staticmethod
    def entry_guard_conditions(dataframe: DataFrame):
        condition = np.where(
            (
                (dataframe['fisher_wr'] < -0.1)
            ), 1.0, 0.0)
        return condition

    @staticmethod
    def exit_guard_conditions(dataframe: DataFrame):
        condition = np.where(
            (
                (dataframe['fisher_wr'] > 0.1)
            ), 1.0, 0.0)
        return condition

    @staticmethod
    def debug_indicators():
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

    @staticmethod
    def entry_training_signals(future_df: DataFrame):
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

    @staticmethod
    def exit_training_signals(future_df: DataFrame):
        signals = np.where(
            (
                # overbought condition with high potential loss
                    (future_df['mfi'] > 50) &  # MFI in sell range
                    (future_df['fisher_wr'] > 0.8) &
                    (future_df['bb_loss'] <= future_df['loss_threshold'] / 100.0) &

                    # future loss
                    (future_df['future_gain'] <= future_df['future_loss_threshold'])
            ), 1.0, 0.0)
        return signals

    @staticmethod
    def entry_guard_conditions(dataframe: DataFrame):
        condition = np.where(
            (
                # buy region
                    (dataframe['fisher_wr'] < -0.5) &

                    # N down sequences
                    (dataframe['dwt_nseq_dn'] >= 2)
            ), 1.0, 0.0)
        return condition

    @staticmethod
    def exit_guard_conditions(dataframe: DataFrame):
        condition = np.where(
            (
                # sell region
                    (dataframe['fisher_wr'] > 0.5) &

                    # N up sequences
                    (dataframe['dwt_nseq_up'] >= 2)
            ), 1.0, 0.0)
        return condition

    @staticmethod
    def debug_indicators():
        return [
            'future_gain',
            'future_profit_max',
            'future_loss_min'
        ]


# -----------------------------------

# fwr = Fisher/WIlliams ratio
#   fairly simple oversold/overbought signals (with future profi/loss)

class fwr_signals(base_signals):

    @staticmethod
    def entry_training_signals(future_df: DataFrame):
        signals = np.where(
            (
                # oversold condition
                    (future_df['fisher_wr'] <= -0.8) &

                    # future profit
                    (future_df['future_gain'] >= future_df['future_profit_threshold'])
            ), 1.0, 0.0)
        return signals

    @staticmethod
    def exit_training_signals(future_df: DataFrame):
        signals = np.where(
            (
                # overbought condition
                    (future_df['fisher_wr'] >= 0.8) &

                    # future loss
                    (future_df['future_gain'] <= future_df['future_loss_threshold'])
            ), 1.0, 0.0)
        return signals

    @staticmethod
    def entry_guard_conditions(dataframe: DataFrame):
        condition = np.where(
            (
                (dataframe['fisher_wr'] < -0.5)
            ), 1.0, 0.0)
        return condition

    @staticmethod
    def exit_guard_conditions(dataframe: DataFrame):
        condition = np.where(
            (
                (dataframe['fisher_wr'] > 0.5)
            ), 1.0, 0.0)
        return condition

    @staticmethod
    def debug_indicators():
        return [
        ]


# -----------------------------------

# highlow - detects highs and lows within the lookahead window

class highlow_signals(base_signals):

    @staticmethod
    def entry_training_signals(future_df: DataFrame):
        signals = np.where(
            (
                    (future_df['dwt_at_low'] > 0) &  # at low of full window
                    # (future_df['full_dwt'] <= future_df['future_min'])  # at min of future window
                    (future_df['future_profit_max'] >= future_df['future_profit_threshold'])
            ), 1.0, 0.0)
        return signals

    @staticmethod
    def exit_training_signals(future_df: DataFrame):
        signals = np.where(
            (
                    (future_df['dwt_at_high'] > 0) &  # at high of full window
                    # (future_df['full_dwt'] >= future_df['future_max'])  # at max of future window
                    (future_df['future_loss_min'] <= future_df['future_loss_threshold'])
            ), 1.0, 0.0)
        return signals

    @staticmethod
    def entry_guard_conditions(dataframe: DataFrame):
        condition = np.where(
            (
                (dataframe['fisher_wr'] < -0.5)
                # (dataframe['close'] <= dataframe['recent_min'])  # local low
            ), 1.0, 0.0)
        return condition

    @staticmethod
    def exit_guard_conditions(dataframe: DataFrame):
        condition = np.where(
            (
                (dataframe['fisher_wr'] > 0.5)
                # (dataframe['close'] >= dataframe['recent_max'])  # local high
            ), 1.0, 0.0)
        return condition

    @staticmethod
    def debug_indicators():
        return [
            'dwt_at_low',
            'dwt_at_high',
            'future_min',
            'future_max'
        ]


# -----------------------------------

# jump - looks for 'big' jumps in price, with future profit/loss

class jump_signals(base_signals):

    @staticmethod
    def entry_training_signals(future_df: DataFrame):
        signals = np.where(
            (
                # previous candle dropped more than 2 std dev
                    (
                            future_df['gain'].shift() <=
                            (future_df['future_loss_mean'] - 2.0 * abs(future_df['future_loss_std']))
                    ) &

                    # upcoming window exceeds profit threshold
                    (future_df['future_profit_max'] >= future_df['future_profit_threshold'])
            ), 1.0, 0.0)
        return signals

    @staticmethod
    def exit_training_signals(future_df: DataFrame):
        signals = np.where(
            (
                # previous candle gained more than 2 std dev
                    (
                            future_df['gain'].shift() >=
                            (future_df['future_profit_mean'] + 2.0 * abs(future_df['future_profit_std']))
                    ) &

                    # upcoming window exceeds loss threshold
                    (future_df['future_loss_min'] <= future_df['future_loss_threshold'])
            ), 1.0, 0.0)
        return signals

    @staticmethod
    def entry_guard_conditions(dataframe: DataFrame):
        condition = np.where(
            (
                # N down sequences
                (dataframe['dwt_nseq_dn'] >= 2)
            ), 1.0, 0.0)
        return condition

    @staticmethod
    def exit_guard_conditions(dataframe: DataFrame):
        condition = np.where(
            (
                # N up sequences
                (dataframe['dwt_nseq_up'] >= 2)
            ), 1.0, 0.0)
        return condition

    @staticmethod
    def debug_indicators():
        return [
        ]


# -----------------------------------

# macd - classic MACD crossing events

class macd_signals(base_signals):

    @staticmethod
    def entry_training_signals(future_df: DataFrame):
        signals = np.where(
            (
                # MACD turns around -ve to +ve
                    (future_df['macdhist'].shift() < 0) &
                    (future_df['macdhist'] >= 0) &

                    # future gain exceeds threshold
                    (future_df['future_profit_max'] >= future_df['future_profit_threshold'])
            ), 1.0, 0.0)
        return signals

    @staticmethod
    def exit_training_signals(future_df: DataFrame):
        signals = np.where(
            (
                # MACD turns around +ve to -ve
                    (future_df['macdhist'].shift() > 0) &
                    (future_df['macdhist'] <= 0) &

                    # future loss exceeds threshold
                    (future_df['future_loss_min'] <= future_df['future_loss_threshold'])
            ), 1.0, 0.0)
        return signals

    @staticmethod
    def entry_guard_conditions(dataframe: DataFrame):
        condition = np.where(
            (
                (dataframe['macdhist'] < 0)
            ), 1.0, 0.0)
        return condition

    @staticmethod
    def exit_guard_conditions(dataframe: DataFrame):
        condition = np.where(
            (
                (dataframe['macdhist'] > 0)
            ), 1.0, 0.0)
        return condition

    @staticmethod
    def debug_indicators():
        return [
        ]


# -----------------------------------

# macd2 - modified MACD, trigger when macdhistory is at a low/high

class macd2_signals(base_signals):

    @staticmethod
    def entry_training_signals(future_df: DataFrame):
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

    @staticmethod
    def exit_training_signals(future_df: DataFrame):
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

    @staticmethod
    def entry_guard_conditions(dataframe: DataFrame):
        condition = np.where(
            (
                (dataframe['macdhist'] < 0)
            ), 1.0, 0.0)
        return condition

    @staticmethod
    def exit_guard_conditions(dataframe: DataFrame):
        condition = np.where(
            (
                (dataframe['macdhist'] > 0)
            ), 1.0, 0.0)
        return condition

    @staticmethod
    def debug_indicators():
        return [
        ]


# -----------------------------------

# macd3 - modified MACD, trigger when macdhistory is in a low/high region

class macd3_signals(base_signals):

    @staticmethod
    def entry_training_signals(future_df: DataFrame):
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

    @staticmethod
    def exit_training_signals(future_df: DataFrame):
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

    @staticmethod
    def entry_guard_conditions(dataframe: DataFrame):
        condition = np.where(
            (
                (dataframe['macdhist'] < 0)
            ), 1.0, 0.0)
        return condition

    @staticmethod
    def exit_guard_conditions(dataframe: DataFrame):
        condition = np.where(
            (
                (dataframe['macdhist'] > 0)
            ), 1.0, 0.0)
        return condition

    @staticmethod
    def debug_indicators():
        return [
        ]


# -----------------------------------

# MFI - Chaikin Money Flow Indicator. Simple oversold/overbought strategy

class mfi_signals(base_signals):

    @staticmethod
    def entry_training_signals(future_df: DataFrame):
        signals = np.where(
            (
                # oversold condition
                    (future_df['mfi'] <= 15) &

                    # future profit
                    (future_df['future_profit_max'] >= future_df['future_profit_threshold'])
            ), 1.0, 0.0)
        return signals

    @staticmethod
    def exit_training_signals(future_df: DataFrame):
        signals = np.where(
            (
                # overbought condition
                    (future_df['mfi'] >= 85) &

                    # future loss
                    (future_df['future_loss_min'] <= future_df['future_loss_threshold'])
            ), 1.0, 0.0)
        return signals

    @staticmethod
    def entry_guard_conditions(dataframe: DataFrame):
        condition = np.where(
            (
                (dataframe['mfi'] <= 30)
            ), 1.0, 0.0)
        return condition

    @staticmethod
    def exit_guard_conditions(dataframe: DataFrame):
        condition = np.where(
            (
                (dataframe['mfi'] >= 60)
            ), 1.0, 0.0)
        return condition

    @staticmethod
    def debug_indicators():
        return [
        ]


# -----------------------------------

# detect the max (sell) or min (buy) of both the past window and the future window

class minmax_signals(base_signals):

    @staticmethod
    def entry_training_signals(future_df: DataFrame):
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

    @staticmethod
    def exit_training_signals(future_df: DataFrame):
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

    @staticmethod
    def entry_guard_conditions(dataframe: DataFrame):
        condition = np.where(
            (
                (dataframe['fisher_wr'] < -0.5)
                # (dataframe['close'] <= dataframe['recent_min'])  # local low
            ), 1.0, 0.0)
        return condition

    @staticmethod
    def exit_guard_conditions(dataframe: DataFrame):
        condition = np.where(
            (
                (dataframe['fisher_wr'] > 0.5)
                # (dataframe['close'] >= dataframe['recent_max'])  # local high
            ), 1.0, 0.0)
        return condition

    @staticmethod
    def debug_indicators():
        return [
            'future_loss_min',
            'future_profit_max'
        ]


# -----------------------------------

# NSeq - continuous sequences of up/down movements

class nseq_signals(base_signals):

    @staticmethod
    def entry_training_signals(future_df: DataFrame):
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

    @staticmethod
    def exit_training_signals(future_df: DataFrame):
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

    @staticmethod
    def entry_guard_conditions(dataframe: DataFrame):
        cond = np.where(
            (
                # N down sequences
                (dataframe['dwt_nseq_dn'] >= 2)
            ), 1.0, 0.0)
        return cond

    @staticmethod
    def exit_guard_conditions(dataframe: DataFrame):
        cond = np.where(
            (
                # N up sequences
                (dataframe['dwt_nseq_up'] >= 2)
            ), 1.0, 0.0)
        return cond

    @staticmethod
    def debug_indicators():
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

class over_signals:

    @staticmethod
    def entry_training_signals(future_df: DataFrame):
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

    @staticmethod
    def exit_training_signals(future_df: DataFrame):
        signals = np.where(
            (
                    (future_df['rsi'] > 60) &
                    (future_df['mfi'] > 60) &
                    (future_df['fisher_wr'] > 0.6) &

                    # future loss
                    (future_df['future_loss_min'] <= future_df['future_loss_threshold'])
            ), 1.0, 0.0)
        return signals

    @staticmethod
    def entry_guard_conditions(dataframe: DataFrame):
        condition = np.where(
            (
                (dataframe['fisher_wr'] < -0.5)
            ), 1.0, 0.0)
        return condition

    @staticmethod
    def exit_guard_conditions(dataframe: DataFrame):
        condition = np.where(
            (
                (dataframe['fisher_wr'] > 0.5)
            ), 1.0, 0.0)
        return condition

    @staticmethod
    def debug_indicators():
        return [
            'future_loss_min',
            'future_profit_max'
        ]


# -----------------------------------

# Profit

class profit_signals(base_signals):

    @staticmethod
    def entry_training_signals(future_df: DataFrame):
        signals = np.where(
            (
                    (future_df['fisher_wr'] < -0.1) &

                    # qtpylib.crossed_above(future_df['future_gain'], 2.0 * future_df['future_profit_threshold'])
                    (future_df['future_profit_max'] >= 2.0 * future_df['future_profit_threshold'])
            ), 1.0, 0.0)
        return signals

    @staticmethod
    def exit_training_signals(future_df: DataFrame):
        signals = np.where(
            (
                    (future_df['fisher_wr'] > 0.1) &

                    # qtpylib.crossed_below(future_df['future_gain'], 2.0 * future_df['future_loss_threshold'])
                    (future_df['future_loss_min'] <= future_df['future_loss_threshold'])
            ), 1.0, 0.0)
        return signals

    @staticmethod
    def entry_guard_conditions(dataframe: DataFrame):
        condition = np.where(
            (
                (dataframe['fisher_wr'] < -0.5)
            ), 1.0, 0.0)
        return condition

    @staticmethod
    def exit_guard_conditions(dataframe: DataFrame):
        condition = np.where(
            (
                (dataframe['fisher_wr'] > 0.5)
            ), 1.0, 0.0)
        return condition

    @staticmethod
    def debug_indicators():
        return [
            'future_profit_threshold',
            'future_loss_threshold',
            'future_gain'
        ]


# -----------------------------------

# pv - peak/valley detection

class pv_signals(base_signals):

    @staticmethod
    def entry_training_signals(future_df: DataFrame):
        global curr_lookahead
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

    @staticmethod
    def exit_training_signals(future_df: DataFrame):
        global curr_lookahead
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

    @staticmethod
    def entry_guard_conditions(dataframe: DataFrame):
        condition = np.where(
            (
                (dataframe['close'] <= dataframe['recent_min'])  # local low
            ), 1.0, 0.0)
        return condition

    @staticmethod
    def exit_guard_conditions(dataframe: DataFrame):
        condition = np.where(
            (
                (dataframe['close'] >= dataframe['recent_max'])  # local high
            ), 1.0, 0.0)
        return condition

    @staticmethod
    def debug_indicators():
        return [
        ]


# -----------------------------------

# stochastic - detect points where fast stochastic (%K) changes direction
# above 80 implies sell, below 20 implies buy

class stochastic_signals(base_signals):

    @staticmethod
    def entry_training_signals(future_df: DataFrame):
        global curr_lookahead
        signals = np.where(
            (
                # stochastics show overbought condition
                    ((future_df['fast_diff'] > 0) & (future_df['fast_diff'].shift(-curr_lookahead) <= 0)) &

                    # future profit
                    (future_df['future_profit_max'] >= future_df['future_profit_threshold']) &
                    (future_df['future_gain'] > 0)
            ), 1.0, 0.0)
        return signals

    @staticmethod
    def exit_training_signals(future_df: DataFrame):
        global curr_lookahead
        signals = np.where(
            (
                # stochastics show oversold condition
                    ((future_df['fast_diff'] < 0) & (future_df['fast_diff'].shift(-curr_lookahead) >= 0)) &

                    # future loss
                    (future_df['future_loss_min'] <= future_df['future_loss_threshold']) &
                    (future_df['future_gain'] < 0)

            ), 1.0, 0.0)
        return signals

    @staticmethod
    def entry_guard_conditions(dataframe: DataFrame):
        condition = np.where(
            (
                (dataframe['fast_diff'] > 0)
            ), 1.0, 0.0)
        return condition

    @staticmethod
    def exit_guard_conditions(dataframe: DataFrame):
        condition = np.where(
            (
                (dataframe['fast_diff'] < 0)
            ), 1.0, 0.0)
        return condition

    @staticmethod
    def debug_indicators():
        return [
        ]


# -----------------------------------

# 'large' swings (down/up or up/down)

class swing_signals(base_signals):

    @staticmethod
    def entry_training_signals(future_df: DataFrame):
        signals = np.where(
            (
                # bottom of trend
                    (future_df['dwt_bottom'] > 0) &

                    # future gain
                    (future_df['future_profit_max'] >= future_df['future_profit_threshold']) &
                    (future_df['future_profit'] > 0)
            ), 1.0, 0.0)
        return signals

    @staticmethod
    def exit_training_signals(future_df: DataFrame):
        signals = np.where(
            (
                # top of trend
                    (future_df['dwt_top'] > 0) &

                    # future gain
                    (future_df['future_loss_min'] <= future_df['future_loss_threshold']) &
                    (future_df['future_loss'] < 0)
            ), 1.0, 0.0)
        return signals

    @staticmethod
    def entry_guard_conditions(dataframe: DataFrame):
        condition = np.where(
            (
                (dataframe['close'] <= dataframe['recent_min'])  # local low
            ), 1.0, 0.0)
        return condition

    @staticmethod
    def exit_guard_conditions(dataframe: DataFrame):
        condition = np.where(
            (
                (dataframe['close'] >= dataframe['recent_max'])  # local high
            ), 1.0, 0.0)
        return condition

    @staticmethod
    def debug_indicators():
        return [
            'future_loss_min',
            'future_profit_max'
        ]


# -----------------------------------

# class TrainingSignals2:
# -----------------------------------
curr_lookahead = 12
n_profit_stddevs = 1.0
n_loss_stddevs = 1.0


# function to set parameters used by many signal algorithms
@staticmethod
def set_strategy_parameters(lookahead, n_profit_std, n_loss_std):
    global curr_lookahead
    global n_profit_stddevs
    global n_loss_stddevs

    curr_lookahead = lookahead
    n_profit_stddevs = n_profit_std
    n_loss_stddevs = n_loss_std


# enum of all available signal types

class SignalType(Enum):
    Bollinger_Width = bbw_signals()
    DWT = dwt_signals()
    DWT2 = dwt2_signals()
    Fisher_Bollinger = fbb_signals()
    Fisher_Williams = fwr_signals()
    High_Low = highlow_signals()
    Jump = jump_signals()
    MACD = macd_signals()
    MACD2 = macd2_signals()
    MACD3 = macd3_signals()
    Money_Flow = mfi_signals()
    Min_Max = minmax_signals()
    N_Sequence = nseq_signals()
    Oversold = over_signals()
    Profit = profit_signals()
    Peaks_Valleys = pv_signals()
    Stochastic = stochastic_signals()
    Swing = swing_signals()

    # # @classmethod
    # def get_entry_training_signals(self, future_df: DataFrame):
    #     st = self.value
    #     print(f"self:{self}")
    #     return self.value().entry_training_signals(future_df)


# function to get buy signals
@staticmethod
def get_entry_training_signals(signal_type: SignalType, future_df: DataFrame):
    return signal_type.value.entry_training_signals(future_df)


# function to get sell signals
@staticmethod
def get_exit_training_signals(signal_type: SignalType, future_df: DataFrame):
    return signal_type.value.exit_training_signals(future_df)


# function to get entry/buy guard conditions
@staticmethod
def get_entry_guard_conditions(signal_type: SignalType, dataframe: DataFrame):
    return signal_type.value.entry_guard_conditions(dataframe)


# function to get entry/buy guard conditions
@staticmethod
def get_exit_guard_conditions(signal_type: SignalType, dataframe: DataFrame):
    return signal_type.value.exit_guard_conditions(dataframe)


# function to get list of debug indicators to make visible (e.g. for plotting)
@staticmethod
def get_debug_indicators(signal_type: SignalType):
    return signal_type.value.debug_indicators()

# -----------------------------------
