import operator

import numpy as np
from enum import Enum

import pywt
import talib.abstract as ta
from scipy.ndimage import gaussian_filter1d
from sklearn.preprocessing import RobustScaler, MinMaxScaler, StandardScaler

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
import os
from pathlib import Path

sys.path.append(str(Path(__file__).parent))

import logging
import warnings

log = logging.getLogger(__name__)
# log.setLevel(logging.DEBUG)
warnings.simplefilter(action='ignore', category=pd.errors.PerformanceWarning)

import custom_indicators as cta
from finta import TA as fta

import keras
from keras import layers
from tqdm import tqdm
from tqdm.keras import TqdmCallback

import random

from DataframeUtils import DataframeUtils, ScalerType

from NNPredict import NNPredict
from NNPredictor_MLP import NNPredictor_MLP
from NNPredictor_LSTM import NNPredictor_LSTM

"""
####################################################################################
NNPredict_stripped - uses an LSTM neural network to try and predict the future stock price. This version uses minimal
                     indicators, and is mainly intended as a baseline to assess performance of the NNPredict family of
                     strategies. Do not expect this to be profitable (unless the market is going up).
      
      This works by creating a  model that we train on the historical data, then use that model to predict 
      future values
      
      Note that this is very slow because we are training and running a neural network. 
      This strategy is likely not viable on a configuration of more than a few pairs, and even then needs
      a fast computer, preferably with a GPU
      
      In addition to the normal freqtrade packages, these strategies also require the installation of:
        finta
        keras
        tensorflow
        tqdm

####################################################################################
"""

# this inherits from NNPredict and just replaces the model used for predictions

class NNPredict_stripped(NNPredict):

    plot_config = {
        'main_plot': {
            'mid': {'color': 'cornflowerblue'},
            # 'smooth': {'color': 'teal'},
            'predict': {'color': 'lightpink'},
        },
        'subplots': {
            "Diff": {
                'predict_diff': {'color': 'blue'},
            },
        }
    }


    # These parameters control much of the behaviour because they control the generation of the training data
    # Unfortunately, these cannot be hyperopt params because they are used in populate_indicators, which is only run
    # once during hyperopt
    # lookahead_hours = 1.0
    lookahead_hours = 0.4
    n_profit_stddevs = 0.0
    n_loss_stddevs = 0.0
    min_f1_score = 0.70
    max_train_loss = 0.15

    curr_lookahead = int(12 * lookahead_hours)

    curr_pair = ""
    custom_trade_info = {}

    refit_model = True
    training_only = False

    target_column = 'mid'

    dbg_enable_tracing = False

    ###################################

    # Strategy Specific Variable Storage

    ## Hyperopt Variables

    # LSTM hyperparams

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

    startup_win = 128  # should be a power of 2
    win_size = 14
    runmode = ""  # set this to self.dp.runmode.value


    ################################
    # we override the add_indicators func
    def add_indicators(self, dataframe: DataFrame) -> DataFrame:

        # source data includes open/close/high/low price, volume
        # month, day, hour minute etc. added automatically

        # needed as placeholders for scaling/results later
        dataframe['temp'] = 0.0
        dataframe['predict'] = 0.0

        dataframe['mid'] = (dataframe['open'] + dataframe['close']) / 2.0

        # moving averages
        dataframe['sma'] = ta.SMA(dataframe, timeperiod=self.win_size)
        dataframe['ema'] = ta.EMA(dataframe, timeperiod=self.win_size)
        dataframe['tema'] = ta.TEMA(dataframe, timeperiod=self.win_size)
        # dataframe['tema_stddev'] = dataframe['tema'].rolling(self.win_size).std()

        # RSI
        dataframe['rsi'] = ta.RSI(dataframe, timeperiod=self.win_size)

        # # fast/slow stochastic
        # period = 14
        # smoothD = 3
        # SmoothK = 3
        # stochrsi = (dataframe['rsi'] - dataframe['rsi'].rolling(period).min()) / (
        #         dataframe['rsi'].rolling(period).max() - dataframe['rsi'].rolling(period).min())
        # dataframe['srsi_k'] = stochrsi.rolling(SmoothK).mean() * 100
        # dataframe['srsi_d'] = dataframe['srsi_k'].rolling(smoothD).mean()

        # Bollinger Bands (must include these)
        bollinger = qtpylib.bollinger_bands(dataframe['close'], window=20, stds=2)
        dataframe['bb_lowerband'] = bollinger['lower']
        dataframe['bb_middleband'] = bollinger['mid']
        dataframe['bb_upperband'] = bollinger['upper']
        dataframe['bb_width'] = ((dataframe['bb_upperband'] - dataframe['bb_lowerband']) / dataframe['bb_middleband'])
        dataframe["bb_gain"] = ((dataframe["bb_upperband"] - dataframe["close"]) / dataframe["close"])
        dataframe["bb_loss"] = ((dataframe["bb_lowerband"] - dataframe["close"]) / dataframe["close"])

        # Donchian Channels
        dataframe['dc_upper'] = ta.MAX(dataframe['high'], timeperiod=self.win_size)
        dataframe['dc_lower'] = ta.MIN(dataframe['low'], timeperiod=self.win_size)
        dataframe['dc_mid'] = ta.TEMA(((dataframe['dc_upper'] + dataframe['dc_lower']) / 2), timeperiod=self.win_size)

        # Keltner Channels (these can sometimes produce inf results)
        keltner = qtpylib.keltner_channel(dataframe)
        dataframe["kc_upper"] = keltner["upper"]
        dataframe["kc_lower"] = keltner["lower"]

        # Keltner Channels (these can sometimes produce inf results)
        keltner = qtpylib.keltner_channel(dataframe)
        dataframe["kc_upper"] = keltner["upper"]
        dataframe["kc_lower"] = keltner["lower"]
        dataframe["kc_mid"] = keltner["mid"]

        # # Williams %R
        # dataframe['wr'] = 0.02 * (self.dataframePopulator.williams_r(dataframe, period=14) + 50.0)
        #
        # # Fisher RSI
        # rsi = 0.1 * (dataframe['rsi'] - 50)
        # dataframe['fisher_rsi'] = (np.exp(2 * rsi) - 1) / (np.exp(2 * rsi) + 1)
        #
        # # Combined Fisher RSI and Williams %R
        # dataframe['fisher_wr'] = (dataframe['wr'] + dataframe['fisher_rsi']) / 2.0

        # ADX
        dataframe['adx'] = ta.ADX(dataframe)

        # Plus Directional Indicator / Movement
        dataframe['dm_plus'] = ta.PLUS_DM(dataframe)
        dataframe['di_plus'] = ta.PLUS_DI(dataframe)

        # Minus Directional Indicator / Movement
        dataframe['dm_minus'] = ta.MINUS_DM(dataframe)
        dataframe['di_minus'] = ta.MINUS_DI(dataframe)
        dataframe['dm_delta'] = dataframe['dm_plus'] - dataframe['dm_minus']
        dataframe['di_delta'] = dataframe['di_plus'] - dataframe['di_minus']

        # MACD
        macd = ta.MACD(dataframe)
        dataframe['macd'] = macd['macd']
        dataframe['macdsignal'] = macd['macdsignal']
        dataframe['macdhist'] = macd['macdhist']

        # Stoch fast
        stoch_fast = ta.STOCHF(dataframe)
        dataframe['fastd'] = stoch_fast['fastd']
        dataframe['fastk'] = stoch_fast['fastk']
        dataframe['fast_diff'] = dataframe['fastd'] - dataframe['fastk']

        # mfi is used as a guard
        dataframe['mfi'] = ta.MFI(dataframe)

        # # DWT model
        # # if in backtest, hyperopt or plot, then we have to do rolling calculations
        # if self.runmode in ('hyperopt', 'backtest', 'plot'):
        #     dataframe['dwt'] = dataframe['close'].rolling(window=self.startup_win).apply(self.dataframePopulator.roll_get_dwt)
        #     dataframe['smooth'] = dataframe['close'].rolling(window=self.startup_win).apply(self.dataframePopulator.roll_smooth)
        #     dataframe['dwt'] = dataframe['dwt'].rolling(window=self.startup_win).apply(self.dataframePopulator.roll_smooth)
        # else:
        #     dataframe['dwt'] = self.dataframePopulator.get_dwt(dataframe['close'])
        #     dataframe['smooth'] = gaussian_filter1d(dataframe['close'], 2)
        #     dataframe['dwt'] = gaussian_filter1d(dataframe['dwt'], 2)

        # TODO: remove/fix any columns that contain 'inf'
        self.dataframeUtils.check_inf(dataframe)

        # TODO: fix NaNs
        dataframe.fillna(0.0, inplace=True)

        return dataframe
    ################################

    def get_classifier(self, pair, seq_len: int, num_features: int):
        return NNPredictor_LSTM(pair, seq_len, num_features)

    ################################