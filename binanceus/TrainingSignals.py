# This is a set of utilities for getting training signal (buy/sell) events
# The signals are collected here so that the different approaches to generating events can be more easily used across
# multiple strategies
# Notes:
#   - This not a class, just a collection of functions and data types
#   - Any hard-coded numbers have been derived by staring at charts to figure out appropriate values
#   - These functions all look ahead in the data, so they can *ONLY* be used for training


import numpy as np
import pandas as pd

import scipy

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent))

import logging
import warnings
from enum import Enum, auto
import freqtrade.vendor.qtpylib.indicators as qtpylib

log = logging.getLogger(__name__)
# log.setLevel(logging.DEBUG)
warnings.simplefilter(action='ignore', category=pd.errors.PerformanceWarning)

from pandas import DataFrame, Series
from datetime import datetime, timedelta, timezone

pd.options.mode.chained_assignment = None  # default='warn'


class SignalType(Enum):
    Undefined = auto()
    Bollinger_Width = auto()
    DWT = auto()
    DWT2 = auto()
    Fisher_Bollinger = auto()
    Fisher_Williams = auto()
    High_Low = auto()
    Jump = auto()
    MACD = auto()
    MACD2 = auto()
    MACD3 = auto()
    Money_Flow = auto()
    Min_Max = auto()
    N_Sequence = auto()
    Oversold = auto()
    Profit = auto()
    Peaks_Valleys = auto()
    Stochastic = auto()
    Swing = auto()


# -----------------------------------
curr_lookahead = 12
n_profit_stddevs = 1.0
n_loss_stddevs = 1.0


# function to set parameters used by many signal algorithms
def set_strategy_parameters(lookahead, n_profit_std, n_loss_std):
    global curr_lookahead
    global n_profit_stddevs
    global n_loss_stddevs

    curr_lookahead = lookahead
    n_profit_stddevs = n_profit_std
    n_loss_stddevs = n_loss_std


# -----------------------------------

# training signals based on the width of the Bollinger Bands

def bbw_buy(future_df):
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


def bbw_sell(future_df):
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


def bbw_entry_guard(dataframe):
    return None


def bbw_exit_guard(dataframe):
    return None


def bbw_dbg_indicators():
    return [
    ]


# -----------------------------------

# training signals based on the discreet Wavelet Transform (DWT) of the closing price.
# 'dwt' is a rolling calculation, i.e. does not look forward
# 'full_dwt' looks ahead, and so can be used for finding buy/sell points in the future
# 'dwt_diff' is the difference between the forward looking model and the rolling model

def dwt_buy(future_df):
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


def dwt_sell(future_df):
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


def dwt_entry_guard(dataframe):
    return None


def dwt_exit_guard(dataframe):
    return None


def dwt_dbg_indicators():
    return [
        'full_dwt', 'future_max', 'future_min'
    ]


# -----------------------------------

# this version does peak/valley detection on the forward-looking DWT estimate
def dwt2_buy(future_df):
    global curr_lookahead
    # detect valleys
    valleys = np.zeros(future_df.shape[0], dtype=float)
    order = 4
    # order = curr_lookahead

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


def dwt2_sell(future_df):
    global curr_lookahead

    peaks = np.zeros(future_df.shape[0], dtype=float)
    order = 4
    # order = curr_lookahead

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


def dwt2_entry_guard(dataframe):
    condition = np.where(
        (
            (dataframe['fisher_wr'] < -0.1)
        ), 1.0, 0.0)
    return condition


def dwt2_exit_guard(dataframe):
    condition = np.where(
        (
            (dataframe['fisher_wr'] > 0.1)
        ), 1.0, 0.0)
    return condition


def dwt2_dbg_indicators():
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

def fbb_buy(future_df):
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


def fbb_sell(future_df):
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


def fbb_entry_guard(dataframe):
    condition = np.where(
        (
            # buy region
            (dataframe['fisher_wr'] < -0.5) &

            # N down sequences
            (dataframe['dwt_nseq_dn'] >= 2)
        ), 1.0, 0.0)
    return condition


def fbb_exit_guard(dataframe):
    condition = np.where(
        (
            # sell region
            (dataframe['fisher_wr'] > 0.5) &

            # N up sequences
            (dataframe['dwt_nseq_up'] >= 2)
        ), 1.0, 0.0)
    return condition


def fbb_dbg_indicators():
    return [
        'future_gain',
        'future_profit_max',
        'future_loss_min'
    ]


# -----------------------------------

# fwr = Fisher/WIlliams ratio
#   fairly simple oversold/overbought signals (with future profi/loss)

def fwr_buy(future_df):
    signals = np.where(
        (
            # oversold condition
                (future_df['fisher_wr'] <= -0.8) &

                # future profit
                (future_df['future_gain'] >= future_df['future_profit_threshold'])
        ), 1.0, 0.0)
    return signals


def fwr_sell(future_df):
    signals = np.where(
        (
            # overbought condition
                (future_df['fisher_wr'] >= 0.8) &

                # future loss
                (future_df['future_gain'] <= future_df['future_loss_threshold'])
        ), 1.0, 0.0)
    return signals


def fwr_entry_guard(dataframe):
    condition = np.where(
        (
            (dataframe['fisher_wr'] < -0.5)
        ), 1.0, 0.0)
    return condition


def fwr_exit_guard(dataframe):
    condition = np.where(
        (
            (dataframe['fisher_wr'] > 0.5)
        ), 1.0, 0.0)
    return condition


def fwr_dbg_indicators():
    return [
    ]


# -----------------------------------

# highlow - detects highs and lows within the lookahead window

def highlow_buy(future_df):
    signals = np.where(
        (
                (future_df['dwt_at_low'] > 0) &  # at low of full window
                # (future_df['full_dwt'] <= future_df['future_min'])  # at min of future window
                (future_df['future_profit_max'] >= future_df['future_profit_threshold'])
        ), 1.0, 0.0)
    return signals


def highlow_sell(future_df):
    signals = np.where(
        (
                (future_df['dwt_at_high'] > 0) &  # at high of full window
                # (future_df['full_dwt'] >= future_df['future_max'])  # at max of future window
                (future_df['future_loss_min'] <= future_df['future_loss_threshold'])
        ), 1.0, 0.0)
    return signals


def highlow_entry_guard(dataframe):
    condition = np.where(
        (
            (dataframe['fisher_wr'] < -0.5)
            # (dataframe['close'] <= dataframe['recent_min'])  # local low
        ), 1.0, 0.0)
    return condition


def highlow_exit_guard(dataframe):
    condition = np.where(
        (
            (dataframe['fisher_wr'] > 0.5)
            # (dataframe['close'] >= dataframe['recent_max'])  # local high
        ), 1.0, 0.0)
    return condition


def highlow_dbg_indicators():
    return [
        'dwt_at_low',
        'dwt_at_high',
        'future_min',
        'future_max'
    ]


# -----------------------------------

# jump - looks for 'big' jumps in price, with future profit/loss

def jump_buy(future_df):
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


def jump_sell(future_df):
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


def jump_entry_guard(dataframe):
    condition = np.where(
        (
            # N down sequences
            (dataframe['dwt_nseq_dn'] >= 2)
        ), 1.0, 0.0)
    return condition


def jump_exit_guard(dataframe):
    condition = np.where(
        (
            # N up sequences
            (dataframe['dwt_nseq_up'] >= 2)
        ), 1.0, 0.0)
    return condition


def jump_dbg_indicators():
    return [
    ]


# -----------------------------------

# macd - classic MACD crossing events
def macd_buy(future_df):
    signals = np.where(
        (
            # MACD turns around -ve to +ve
                (future_df['macdhist'].shift() < 0) &
                (future_df['macdhist'] >= 0) &

                # future gain exceeds threshold
                (future_df['future_profit_max'] >= future_df['future_profit_threshold'])
        ), 1.0, 0.0)
    return signals


def macd_sell(future_df):
    signals = np.where(
        (
            # MACD turns around +ve to -ve
                (future_df['macdhist'].shift() > 0) &
                (future_df['macdhist'] <= 0) &

                # future loss exceeds threshold
                (future_df['future_loss_min'] <= future_df['future_loss_threshold'])
        ), 1.0, 0.0)
    return signals


def macd_entry_guard(dataframe):
    condition = np.where(
        (
            (dataframe['macdhist'] < 0)
        ), 1.0, 0.0)
    return condition


def macd_exit_guard(dataframe):
    condition = np.where(
        (
            (dataframe['macdhist'] > 0)
        ), 1.0, 0.0)
    return condition


def macd_dbg_indicators():
    return [
    ]


# -----------------------------------

# macd2 - modified MACD, trigger when macdhistory is at a low/high
def macd2_buy(future_df):
    # detect valleys
    valleys = np.zeros(future_df.shape[0], dtype=float)
    order = 4
    # order = curr_lookahead

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


def macd2_sell(future_df):
    peaks = np.zeros(future_df.shape[0], dtype=float)
    order = 2
    # order = curr_lookahead

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


def macd2_entry_guard(dataframe):
    condition = np.where(
        (
            (dataframe['macdhist'] < 0)
        ), 1.0, 0.0)
    return condition


def macd2_exit_guard(dataframe):
    condition = np.where(
        (
            (dataframe['macdhist'] > 0)
        ), 1.0, 0.0)
    return condition


def macd2_dbg_indicators():
    return [
    ]


# -----------------------------------

# macd3 - modified MACD, trigger when macdhistory is in a low/high region
def macd3_buy(future_df):
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


def macd3_sell(future_df):
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


def macd3_entry_guard(dataframe):
    condition = np.where(
        (
            (dataframe['macdhist'] < 0)
        ), 1.0, 0.0)
    return condition


def macd3_exit_guard(dataframe):
    condition = np.where(
        (
            (dataframe['macdhist'] > 0)
        ), 1.0, 0.0)
    return condition


def macd3_dbg_indicators():
    return [
    ]


# -----------------------------------

# MFI - Chaikin Money Flow Indicator. Simple oversold/overbought strategy
def mfi_buy(future_df):
    signals = np.where(
        (
            # oversold condition
                (future_df['mfi'] <= 15) &

                # future profit
                (future_df['future_profit_max'] >= future_df['future_profit_threshold'])
        ), 1.0, 0.0)
    return signals


def mfi_sell(future_df):
    signals = np.where(
        (
            # overbought condition
                (future_df['mfi'] >= 85) &

                # future loss
                (future_df['future_loss_min'] <= future_df['future_loss_threshold'])
        ), 1.0, 0.0)
    return signals


def mfi_entry_guard(dataframe):
    condition = np.where(
        (
            (dataframe['mfi'] <= 30)
        ), 1.0, 0.0)
    return condition


def mfi_exit_guard(dataframe):
    condition = np.where(
        (
            (dataframe['mfi'] >= 60)
        ), 1.0, 0.0)
    return condition


def mfi_dbg_indicators():
    return [
    ]


# -----------------------------------

# detect the max (sell) or min (buy) of both the past window and the future window

def minmax_buy(future_df):
    signals = np.where(
        (
            # at min of past window
                (future_df['full_dwt'] <= future_df['dwt_recent_min']) &

                # at min of future window
                (future_df['full_dwt'] <= future_df['future_min'])  &

            # future profit exceeds threshold
            (future_df['future_profit_max'] >= future_df['future_profit_threshold'])
        ), 1.0, 0.0)
    return signals


def minmax_sell(future_df):
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


def minmax_entry_guard(dataframe):
    condition = np.where(
        (
            (dataframe['fisher_wr'] < -0.5)
            # (dataframe['close'] <= dataframe['recent_min'])  # local low
        ), 1.0, 0.0)
    return condition


def minmax_exit_guard(dataframe):
    condition = np.where(
        (
            (dataframe['fisher_wr'] > 0.5)
            # (dataframe['close'] >= dataframe['recent_max'])  # local high
        ), 1.0, 0.0)
    return condition


def minmax_dbg_indicators():
    return [
        'future_loss_min',
        'future_profit_max'
    ]


# -----------------------------------

# NSeq - continuous sequences of up/down movements
def nseq_buy(future_df):
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


def nseq_sell(future_df):
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


def nseq_entry_guard(dataframe):
    cond = np.where(
        (
            # N down sequences
            (dataframe['dwt_nseq_dn'] >= 2)
        ), 1.0, 0.0)
    return cond



def nseq_exit_guard(dataframe):
    cond = np.where(
        (
            # N up sequences
            (dataframe['dwt_nseq_up'] >= 2)
        ), 1.0, 0.0)
    return cond



def nseq_dbg_indicators():
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

def over_buy(future_df):
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


def over_sell(future_df):
    signals = np.where(
        (
                (future_df['rsi'] > 60) &
                (future_df['mfi'] > 60) &
                (future_df['fisher_wr'] > 0.6) &

                # future loss
                (future_df['future_loss_min'] <= future_df['future_loss_threshold'])
        ), 1.0, 0.0)
    return signals


def over_entry_guard(dataframe):
    condition = np.where(
        (
            (dataframe['fisher_wr'] < -0.5)
        ), 1.0, 0.0)
    return condition


def over_exit_guard(dataframe):
    condition = np.where(
        (
            (dataframe['fisher_wr'] > 0.5)
        ), 1.0, 0.0)
    return condition


def over_dbg_indicators():
    return [
        'future_loss_min',
        'future_profit_max'
    ]


# -----------------------------------

# Profit

def profit_buy(future_df):
    signals = np.where(
        (
                (future_df['fisher_wr'] < -0.1) &

                # qtpylib.crossed_above(future_df['future_gain'], 2.0 * future_df['future_profit_threshold'])
                (future_df['future_profit_max'] >= 2.0 * future_df['future_profit_threshold'])
        ), 1.0, 0.0)
    return signals


def profit_sell(future_df):
    signals = np.where(
        (
                (future_df['fisher_wr'] > 0.1) &

                # qtpylib.crossed_below(future_df['future_gain'], 2.0 * future_df['future_loss_threshold'])
                (future_df['future_loss_min'] <= future_df['future_loss_threshold'])
        ), 1.0, 0.0)
    return signals


def profit_entry_guard(dataframe):
    condition = np.where(
        (
            (dataframe['fisher_wr'] < -0.5)
        ), 1.0, 0.0)
    return condition


def profit_exit_guard(dataframe):
    condition = np.where(
        (
            (dataframe['fisher_wr'] > 0.5)
        ), 1.0, 0.0)
    return condition


def profit_dbg_indicators():
    return [
        'future_profit_threshold',
        'future_loss_threshold',
        'future_gain'
    ]


# -----------------------------------

# pv - peak/valley detection

def pv_buy(future_df):
    global curr_lookahead
    valleys = np.zeros(future_df.shape[0], dtype=float)
    order = 4
    # order = curr_lookahead
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


def pv_sell(future_df):
    global curr_lookahead
    peaks = np.zeros(future_df.shape[0], dtype=float)
    order = 4
    # order = curr_lookahead
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


def pv_entry_guard(dataframe):
    condition = np.where(
        (
            (dataframe['close'] <= dataframe['recent_min'])  # local low
        ), 1.0, 0.0)
    return condition


def pv_exit_guard(dataframe):
    condition = np.where(
        (
            (dataframe['close'] >= dataframe['recent_max'])  # local high
        ), 1.0, 0.0)
    return condition


def pv_dbg_indicators():
    return [
    ]


# -----------------------------------

# stochastic - detect points where fast stochastic (%K) changes direction
# above 80 implies sell, below 20 implies buy

def stochastic_buy(future_df):
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


def stochastic_sell(future_df):
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


def stochastic_entry_guard(dataframe):
    condition = np.where(
        (
            (dataframe['fast_diff'] > 0)
        ), 1.0, 0.0)
    return condition


def stochastic_exit_guard(dataframe):
    condition = np.where(
        (
            (dataframe['fast_diff'] < 0)
        ), 1.0, 0.0)
    return condition


def stochastic_dbg_indicators():
    return [
    ]


# -----------------------------------

def swing_buy(future_df):
    signals = np.where(
        (
            # bottom of trend
                (future_df['dwt_bottom'] > 0) &

                # future gain
                (future_df['future_gain_max'] >= future_df['future_profit_threshold']) &
                (future_df['future_gain'] > 0)
        ), 1.0, 0.0)
    return signals


def swing_sell(future_df):
    signals = np.where(
        (
            # top of trend
                (future_df['dwt_top'] > 0) &

                # future gain
                (future_df['future_loss_min'] <= future_df['future_loss_threshold']) &
                (future_df['future_gain'] < 0)
        ), 1.0, 0.0)
    return signals


def swing_entry_guard(dataframe):
    condition = np.where(
        (
            (dataframe['close'] <= dataframe['recent_min'])  # local low
        ), 1.0, 0.0)
    return condition


def swing_exit_guard(dataframe):
    condition = np.where(
        (
            (dataframe['close'] >= dataframe['recent_max'])  # local high
        ), 1.0, 0.0)
    return condition


def swing_dbg_indicators():
    return [
    ]


# -----------------------------------

# table to map straining signal type onto corresponding buy/sell function
# for some reason, I can't just declare this statically, so add each item individually

signal_table = {}


# utility function to add an entry to the signal table
def add_signal_entry(signal_type: SignalType, entry_train, exit_train, entry_guard, exit_guard, dbg_indicators):
    global signal_table
    signal_table[signal_type] = {}
    signal_table[signal_type]['entry_train'] = entry_train
    signal_table[signal_type]['exit_train'] = exit_train
    signal_table[signal_type]['entry_guard'] = entry_guard
    signal_table[signal_type]['exit_guard'] = exit_guard
    signal_table[signal_type]['dbg_ind'] = dbg_indicators


# add entries for each supported signal type
add_signal_entry(SignalType.Bollinger_Width, bbw_buy, bbw_sell, bbw_entry_guard, bbw_exit_guard, bbw_dbg_indicators)
add_signal_entry(SignalType.DWT, dwt_buy, dwt_sell, dwt_entry_guard, dwt_exit_guard, dwt_dbg_indicators)
add_signal_entry(SignalType.DWT2, dwt2_buy, dwt2_sell, dwt2_entry_guard, dwt2_exit_guard, dwt2_dbg_indicators)
add_signal_entry(SignalType.Fisher_Bollinger, fbb_buy, fbb_sell, fbb_entry_guard, fbb_exit_guard, fbb_dbg_indicators)
add_signal_entry(SignalType.Fisher_Williams, fwr_buy, fwr_sell, fwr_entry_guard, fwr_exit_guard, fwr_dbg_indicators)
add_signal_entry(SignalType.High_Low, highlow_buy, highlow_sell, highlow_entry_guard, highlow_exit_guard,
                 highlow_dbg_indicators)
add_signal_entry(SignalType.Jump, jump_buy, jump_sell, jump_entry_guard, jump_exit_guard, jump_dbg_indicators)
add_signal_entry(SignalType.MACD, macd_buy, macd_sell, macd_entry_guard, macd_exit_guard, macd_dbg_indicators)
add_signal_entry(SignalType.MACD2, macd2_buy, macd2_sell, macd2_entry_guard, macd2_exit_guard, macd2_dbg_indicators)
add_signal_entry(SignalType.MACD3, macd3_buy, macd3_sell, macd3_entry_guard, macd3_exit_guard, macd3_dbg_indicators)
add_signal_entry(SignalType.Money_Flow, mfi_buy, mfi_sell, mfi_entry_guard, mfi_exit_guard, mfi_dbg_indicators)
add_signal_entry(SignalType.Min_Max, minmax_buy, minmax_sell, minmax_entry_guard, minmax_exit_guard,
                 minmax_dbg_indicators)
add_signal_entry(SignalType.N_Sequence, nseq_buy, nseq_sell, nseq_entry_guard, nseq_exit_guard, nseq_dbg_indicators)
add_signal_entry(SignalType.Oversold, over_buy, over_sell, over_entry_guard, over_exit_guard, over_dbg_indicators)
add_signal_entry(SignalType.Profit, profit_buy, profit_sell, profit_entry_guard, profit_exit_guard,
                 profit_dbg_indicators)
add_signal_entry(SignalType.Peaks_Valleys, pv_buy, pv_sell, pv_entry_guard, pv_exit_guard, pv_dbg_indicators)
add_signal_entry(SignalType.Stochastic, stochastic_buy, stochastic_sell, stochastic_entry_guard, stochastic_exit_guard,
                 stochastic_dbg_indicators)
add_signal_entry(SignalType.Swing, swing_buy, swing_sell, swing_entry_guard, swing_exit_guard, swing_dbg_indicators)


# get signals (used by a few different signals)
def get_signals(key, signal_type, future_df):
    global signal_table

    signals = None
    if signal_type in signal_table:
        try:
            if key in signal_table[signal_type]:
                signals = signal_table[signal_type][key](future_df)
            else:
                print(f"    ** ERR: Unknown key type: {key}")

        except NameError:
            print(f"    ** Error calling {key} function for type: {signal_type}")
    else:
        print(f"    ** ERR: Unknown signal type: {signal_type}")

    return signals


# ---------------------------------
# the following should be used to access the desired information for a given SignalType

# function to get buy signals
def get_entry_training_signals(signal_type, future_df):
    return get_signals('entry_train', signal_type, future_df)


# function to get sell signals
def get_exit_training_signals(signal_type, future_df):
    return get_signals('exit_train', signal_type, future_df)


# function to get entry/buy guard conditions
def get_entry_guard_conditions(signal_type, future_df):
    return get_signals('entry_guard', signal_type, future_df)


# function to get entry/buy guard conditions
def get_exit_guard_conditions(signal_type, future_df):
    return get_signals('exit_guard', signal_type, future_df)


# function to get list of debvug indicators to make visible (e.g. for plotting)
def get_debug_indicators(signal_type):
    global signal_table

    indicators = []
    if signal_type in signal_table:
        try:
            key = 'dbg_ind'
            if key in signal_table[signal_type]:
                indicators = signal_table[signal_type][key]()
            else:
                print(f"    ** ERR: Unknown key type: {key}")

        except NameError:
            print(f"    ** Error calling debug indicator function for type: {signal_type}")
    else:
        print(f"    ** ERR: Unknown signal type: {signal_type}")

    return indicators
