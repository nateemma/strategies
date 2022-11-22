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
import Time2Vector
import Transformer
import Attention

"""
####################################################################################
Predict_LSTM - uses a Long-Short Term Memory neural network to try and predict the future stock price
      
      This works by creating a LSTM model that we train on the historical data, then use that model to predict 
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


class Predict_LSTM(IStrategy):
    plot_config = {
        'main_plot': {
            'close': {'color': 'green'},
            'smooth': {'color': 'teal'},
            'predict': {'color': 'lightpink'},
        },
        'subplots': {
            "Diff": {
                'predict_diff': {'color': 'blue'},
            },
        }
    }

    # Do *not* hyperopt for the roi and stoploss spaces (unless you turn off custom stoploss)

    # ROI table:
    minimal_roi = {
        "0": 0.1
    }

    # Stoploss:
    stoploss = -0.05

    # Trailing stop:
    trailing_stop = False
    trailing_stop_positive = None
    trailing_stop_positive_offset = 0.0
    trailing_only_offset_is_reached = False

    timeframe = '5m'

    inf_timeframe = '5m'

    use_custom_stoploss = True

    # Recommended
    use_entry_signal = True
    entry_profit_only = False
    ignore_roi_if_entry_signal = True

    # Required
    startup_candle_count: int = 128  # must be power of 2
    process_only_new_candles = True # this strat is very resource intensive, do not set to False

    # Strategy-specific global vars

    inf_mins = timeframe_to_minutes(inf_timeframe)
    data_mins = timeframe_to_minutes(timeframe)
    inf_ratio = int(inf_mins / data_mins)

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

    # profit/loss thresholds used for assessing buy/sell signals. Keep these realistic!
    # Note: if self.dynamic_gain_thresholds is True, these will be adjusted for each pair, based on historical mean
    default_profit_threshold = 0.3
    default_loss_threshold = -0.3
    profit_threshold = default_profit_threshold
    loss_threshold = default_loss_threshold
    dynamic_gain_thresholds = True  # dynamically adjust gain thresholds based on actual mean (beware, training data could be bad)

    num_pairs = 0
    pair_model_info = {}  # holds model-related info for each pair
    curr_dataframe: DataFrame = None
    normalise_data = True

    # the following affect training of the model. Bigger numbers give better model, but take longer and use more memory
    seq_len = 8 # 'depth' of training sequence
    num_epochs = 64 # number of iterations for training
    batch_size = 512 # batch size for training
    predict_batch_size = 256

    # debug flags
    first_time = True  # mostly for debug
    first_run = True  # used to identify first time through buy/sell populate funcs

    dbg_verbose = True  # controls debug output
    dbg_curr_df: DataFrame = None  # for debugging of current dataframe

    # variables to track state
    class State(Enum):
        INIT = 1
        POPULATE = 2
        STOPLOSS = 3
        RUNNING = 4

    ###################################

    # Strategy Specific Variable Storage

    ## Hyperopt Variables

    # LSTM hyperparams

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

    ################################

    """
    inf Pair Definitions
    """

    def inf_pairs(self):
        # # all pairs in the whitelist are also in the informative list
        # pairs = self.dp.current_whitelist()
        # inf_pairs = [(pair, self.inf_timeframe) for pair in pairs]
        # return inf_pairs
        return []

    ###################################

    """
    Indicator Definitions
    """

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:

        # TODO: add in classifiers from statsmodels
        # TODO; do grid search for MLP hyperparams
        # TODO: keeps stats on classifier performance and selection

        # Base pair inf timeframe indicators
        curr_pair = metadata['pair']
        self.curr_pair = curr_pair
        self.curr_dataframe = dataframe

        self.curr_lookahead = int(12 * self.lookahead_hours)
        self.dbg_curr_df = dataframe

        # reset profit/loss thresholds
        self.profit_threshold = self.default_profit_threshold
        self.loss_threshold = self.default_loss_threshold

        if Predict_LSTM.first_time:
            Predict_LSTM.first_time = False
            print("")
            print("***************************************")
            print("** Warning: startup can be very slow **")
            print("***************************************")

            print("    Lookahead: ", self.curr_lookahead, " candles (", self.lookahead_hours, " hours)")
            print("    Thresholds - Profit:{:.2f}% Loss:{:.2f}%".format(self.profit_threshold,
                                                                        self.loss_threshold))

        print("")
        print(self.curr_pair)

        # populate the training indicators
        dataframe = self.add_training_indicators(dataframe)

        # train the model
        if self.dbg_verbose:
            print("    training model...")

        dataframe = self.train_model(dataframe, self.curr_pair)

        # add predictions
        if self.dbg_verbose:
            print("    running predictions...")

        dataframe = self.add_predictions(dataframe, self.curr_pair)

        # Custom Stoploss
        if self.dbg_verbose:
            print("    updating stoploss data...")
        dataframe = self.add_indicators(dataframe)
        dataframe = self.add_stoploss_indicators(dataframe,self.curr_pair)

        return dataframe

    ###################################

    # add in any indicators to be used for training
    def add_training_indicators(self, dataframe: DataFrame) -> DataFrame:

        # don't add too many indicators, it just muddles the prediction

        # price, high, low, volume automatically included

        win_size = max(self.curr_lookahead, 14)

        dataframe['mid'] = (dataframe['open'] + dataframe['close']) / 2.0

        # % gain relative to previous candle
        dataframe['gain'] = (dataframe['close'] - dataframe['close'].shift(1)) / dataframe['close'].shift(1)

        # smoothed version, for trends
        # dataframe['smooth'] = dataframe['close'].rolling(window=win_size).apply(self.roll_smooth)
        # dataframe['smooth'] = dataframe['mid'].rolling(window=win_size).apply(self.roll_smooth)
        dataframe['smooth'] = dataframe['mid'].rolling(window=win_size).apply(self.roll_strong_smooth)

        # dataframe['tema'] = ta.TEMA(dataframe, timeperiod=win_size)

        # RSI
        # dataframe['rsi'] = ta.RSI(dataframe, timeperiod=win_size)

        # MFI
        dataframe['mfi'] = ta.MFI(dataframe)
        # #
        # # # VFI
        # # dataframe['vfi'] = fta.VFI(dataframe, period=win_size)
        # #
        # # ATR
        # dataframe['atr'] = ta.ATR(dataframe, timeperiod=win_size)
        #
        # # Hilbert Transform Indicator - SineWave
        # hilbert = ta.HT_SINE(dataframe)
        # dataframe['htsine'] = hilbert['sine']
        # dataframe['htleadsine'] = hilbert['leadsine']

        # # Stoch fast
        # stoch_fast = ta.STOCHF(dataframe)
        # dataframe['fastd'] = stoch_fast['fastd']
        # dataframe['fastk'] = stoch_fast['fastk']
        # dataframe['fast_diff'] = dataframe['fastd'] - dataframe['fastk']

        # # # ADX
        dataframe['adx'] = ta.ADX(dataframe)

        # Plus Directional Indicator / Movement
        dataframe['dm_plus'] = ta.PLUS_DM(dataframe)
        dataframe['di_plus'] = ta.PLUS_DI(dataframe)

        # Minus Directional Indicator / Movement
        dataframe['dm_minus'] = ta.MINUS_DM(dataframe)
        dataframe['di_minus'] = ta.MINUS_DI(dataframe)

        dataframe['dm_delta'] = dataframe['dm_plus'] - dataframe['dm_minus']
        dataframe['di_delta'] = dataframe['di_plus'] - dataframe['di_minus']

        # longer term high/low
        dataframe['low_trend'] = dataframe['close'].rolling(window=self.startup_candle_count).min()
        dataframe['high_trend'] = dataframe['close'].rolling(window=self.startup_candle_count).max()

        dataframe['price_dir'] = np.where(dataframe['smooth'].diff() >= 0, 1.0, -1.0)
        dataframe['nseq'] = dataframe['price_dir'].rolling(window=win_size, min_periods=1).sum()

        return dataframe

    # populate dataframe with desired technical indicators
    # NOTE: OK to throw (almost) anything in here, just add it to the parameter list
    # The whole idea is to create a dimension-reduced mapping anyway
    # Warning: do not use indicators that might produce 'inf' results, it messes up the scaling
    def add_indicators(self, dataframe: DataFrame) -> DataFrame:

        win_size = max(self.curr_lookahead, 14)

        # these averages are used internally, do not remove!
        dataframe['sma'] = ta.SMA(dataframe, timeperiod=win_size)
        dataframe['ema'] = ta.EMA(dataframe, timeperiod=win_size)
        # dataframe['tema'] = ta.TEMA(dataframe, timeperiod=win_size)
        # dataframe['tema_stddev'] = dataframe['tema'].rolling(win_size).std()
        # dataframe['rsi'] = ta.RSI(dataframe, timeperiod=win_size)

        # these are here for reference. Uncomment anything you want to use

        # # MACD
        # macd = ta.MACD(dataframe)
        # dataframe['macd'] = macd['macd']
        # dataframe['macdsignal'] = macd['macdsignal']
        # dataframe['macdhist'] = macd['macdhist']
        #
        # # Bollinger Bands (must include these)
        # bollinger = qtpylib.bollinger_bands(dataframe['close'], window=20, stds=2)
        # dataframe['bb_lowerband'] = bollinger['lower']
        # dataframe['bb_middleband'] = bollinger['mid']
        # dataframe['bb_upperband'] = bollinger['upper']
        # dataframe['bb_width'] = ((dataframe['bb_upperband'] - dataframe['bb_lowerband']) / dataframe['bb_middleband'])
        # dataframe["bb_gain"] = ((dataframe["bb_upperband"] - dataframe["close"]) / dataframe["close"])
        # dataframe["bb_loss"] = ((dataframe["bb_lowerband"] - dataframe["close"]) / dataframe["close"])
        #
        # # Donchian Channels
        # dataframe['dc_upper'] = ta.MAX(dataframe['high'], timeperiod=win_size)
        # dataframe['dc_lower'] = ta.MIN(dataframe['low'], timeperiod=win_size)
        # dataframe['dc_mid'] = ta.TEMA(((dataframe['dc_upper'] + dataframe['dc_lower']) / 2), timeperiod=win_size)
        #
        # dataframe["dcbb_dist_upper"] = (dataframe["dc_upper"] - dataframe['bb_upperband'])
        # dataframe["dcbb_dist_lower"] = (dataframe["dc_lower"] - dataframe['bb_lowerband'])
        #
        # # Fibonacci Levels (of Donchian Channel)
        # dataframe['dc_dist'] = (dataframe['dc_upper'] - dataframe['dc_lower'])
        # dataframe['dc_hf'] = dataframe['dc_upper'] - dataframe['dc_dist'] * 0.236  # Highest Fib
        # dataframe['dc_chf'] = dataframe['dc_upper'] - dataframe['dc_dist'] * 0.382  # Centre High Fib
        # dataframe['dc_clf'] = dataframe['dc_upper'] - dataframe['dc_dist'] * 0.618  # Centre Low Fib
        # dataframe['dc_lf'] = dataframe['dc_upper'] - dataframe['dc_dist'] * 0.764  # Low Fib
        #
        #  # Keltner Channels
        # keltner = qtpylib.keltner_channel(dataframe)
        # dataframe["kc_upper"] = keltner["upper"]
        # dataframe["kc_lower"] = keltner["lower"]
        # dataframe["kc_mid"] = keltner["mid"]
        #
        # # Williams %R
        # dataframe['wr'] = 0.02 * (williams_r(dataframe, period=14) + 50.0)
        #
        # # Fisher RSI
        # rsi = 0.1 * (dataframe['rsi'] - 50)
        # dataframe['fisher_rsi'] = (np.exp(2 * rsi) - 1) / (np.exp(2 * rsi) + 1)
        #
        # # Combined Fisher RSI and Williams %R
        # dataframe['fisher_wr'] = (dataframe['wr'] + dataframe['fisher_rsi']) / 2.0
        #
        #
        # # MFI
        # dataframe['mfi'] = ta.MFI(dataframe)
        #
        # # ATR
        # dataframe['atr'] = ta.ATR(dataframe, timeperiod=win_size)
        #
        # # Hilbert Transform Indicator - SineWave
        # hilbert = ta.HT_SINE(dataframe)
        # dataframe['htsine'] = hilbert['sine']
        # dataframe['htleadsine'] = hilbert['leadsine']
        #
        # # ADX
        # dataframe['adx'] = ta.ADX(dataframe)
        #
        # # Plus Directional Indicator / Movement
        # dataframe['dm_plus'] = ta.PLUS_DM(dataframe)
        # dataframe['di_plus'] = ta.PLUS_DI(dataframe)
        #
        # # Minus Directional Indicator / Movement
        # dataframe['dm_minus'] = ta.MINUS_DM(dataframe)
        # dataframe['di_minus'] = ta.MINUS_DI(dataframe)
        # dataframe['dm_delta'] = dataframe['dm_plus'] - dataframe['dm_minus']
        # dataframe['di_delta'] = dataframe['di_plus'] - dataframe['di_minus']

        # # Stoch fast
        # stoch_fast = ta.STOCHF(dataframe)
        # dataframe['fastd'] = stoch_fast['fastd']
        # dataframe['fastk'] = stoch_fast['fastk']
        # dataframe['fast_diff'] = dataframe['fastd'] - dataframe['fastk']
        #
        # # SAR Parabol
        # dataframe['sar'] = ta.SAR(dataframe)
        #
        # dataframe['mom'] = ta.MOM(dataframe, timeperiod=14)
        #
        # # priming indicators
        # dataframe['color'] = np.where((dataframe['close'] > dataframe['open']), 1.0, -1.0)
        # dataframe['rsi_7'] = ta.RSI(dataframe, timeperiod=7)
        # dataframe['roc_6'] = ta.ROC(dataframe, timeperiod=6)
        # dataframe['primed'] = np.where(dataframe['color'].rolling(3).sum() == 3.0, 1.0, -1.0)
        # dataframe['in_the_mood'] = np.where(dataframe['rsi_7'] > dataframe['rsi_7'].rolling(12).mean(), 1.0, -1.0)
        # dataframe['moist'] = np.where(qtpylib.crossed_above(dataframe['macd'], dataframe['macdsignal']), 1.0, -1.0)
        # dataframe['throbbing'] = np.where(dataframe['roc_6'] > dataframe['roc_6'].rolling(12).mean(), 1.0, -1.0)
        #
        # # Oscillators
        #
        # # EWO
        # dataframe['ewo'] = ewo(dataframe, 50, 200)
        #
        # # Ultimate Oscillator
        # dataframe['uo'] = ta.ULTOSC(dataframe)
        #
        # # Aroon, Aroon Oscillator
        # aroon = ta.AROON(dataframe)
        # dataframe['aroonup'] = aroon['aroonup']
        # dataframe['aroondown'] = aroon['aroondown']
        # dataframe['aroonosc'] = ta.AROONOSC(dataframe)
        #
        # # Awesome Oscillator
        # dataframe['ao'] = qtpylib.awesome_oscillator(dataframe)
        #
        # # Commodity Channel Index: values [Oversold:-100, Overbought:100]
        # dataframe['cci'] = ta.CCI(dataframe)

        return dataframe

    def add_stoploss_indicators(self, dataframe: DataFrame, pair) -> DataFrame:

        if not pair in self.custom_trade_info:
            self.custom_trade_info[pair] = {}
            if not 'had_trend' in self.custom_trade_info[pair]:
                self.custom_trade_info[pair]['had_trend'] = False

        # Indicators used for ROI and Custom Stoploss

        # RMI: https://www.tradingview.com/script/kwIt9OgQ-Relative-Momentum-Index/
        dataframe['rmi'] = cta.RMI(dataframe, length=24, mom=5)

        # Trends
        dataframe['candle_up'] = np.where(dataframe['close'] >= dataframe['close'].shift(), 1.0, -1.0)
        dataframe['candle_up_trend'] = np.where(dataframe['candle_up'].rolling(5).sum() > 0.0, 1.0, -1.0)
        dataframe['candle_up_seq'] = dataframe['candle_up'].rolling(5).sum()

        dataframe['candle_dn'] = np.where(dataframe['close'] < dataframe['close'].shift(), 1.0, -1.0)
        dataframe['candle_dn_trend'] = np.where(dataframe['candle_up'].rolling(5).sum() > 0.0, 1.0, -1.0)
        dataframe['candle_dn_seq'] = dataframe['candle_up'].rolling(5).sum()

        dataframe['rmi_up'] = np.where(dataframe['rmi'] >= dataframe['rmi'].shift(), 1.0, -1.0)
        dataframe['rmi_up_trend'] = np.where(dataframe['rmi_up'].rolling(5).sum() > 0.0, 1.0, -1.0)

        dataframe['rmi_dn'] = np.where(dataframe['rmi'] <= dataframe['rmi'].shift(), 1.0, -1.0)
        dataframe['rmi_dn_count'] = dataframe['rmi_dn'].rolling(8).sum()

        ssldown, sslup = cta.SSLChannels_ATR(dataframe, length=21)
        dataframe['sroc'] = cta.SROC(dataframe, roclen=21, emalen=13, smooth=21)
        dataframe['ssl_dir'] = 0
        dataframe['ssl_dir'] = np.where(sslup > ssldown, 1.0, -1.0)

        # TODO: remove/fix any columns that contain 'inf'
        self.check_inf(dataframe)

        # TODO: fix NaNs
        dataframe.fillna(0.0, inplace=True)

        return dataframe

    # add columns based on predictions. Do not call until after model has been trained
    def add_predictions(self, dataframe: DataFrame, pair) -> DataFrame:

        win_size = max(self.curr_lookahead, 14)

        dataframe['predict'] = self.batch_predictions(dataframe)
        dataframe['predict_smooth'] = dataframe['predict'].rolling(window=win_size).apply(self.roll_strong_smooth)

        dataframe['predict_diff'] = 100.0 * (dataframe['predict'] - dataframe['close']) / dataframe['close']

        # dataframe['predict_diff'] = 100.0 * (dataframe['predict_smooth'] - dataframe['smooth']) / dataframe['smooth']

        return dataframe

    ################################

    def train_model(self, dataframe: DataFrame, pair) -> DataFrame:



        nfeatures = dataframe.shape[1]

        # if first time through for this pair, add entry to pair_model_info
        if not (pair in self.pair_model_info):
            self.pair_model_info[pair] = {'model': None, 'interval':0, 'score':0.0}

        if self.pair_model_info[pair]['model'] == None:
            print("    Creating model for: ", pair, " seq_len:", nfeatures)
            self.pair_model_info[pair]['model'] = self.get_lstm(nfeatures, self.seq_len)
            self.pair_model_info[pair]['interval'] = 0


        model = self.pair_model_info[pair]['model']

        # if in a run mode, then periodically load weights and just return
        if self.dp.runmode.value not in ('hyperopt', 'backtest', 'plot'):

            # only run if interval reaches 0 (no point retraining every camdle)
            count = self.pair_model_info[pair]['interval']
            if (count > 0):
                self.pair_model_info[pair]['interval'] = count - 1
                print("Skipping re-train for {} candles".format(self.pair_model_info[pair]['interval']))

            else:
                # reset interval to a random number between 1 and the amount of lookahead
                self.pair_model_info[pair]['interval'] = random.randint(2, max(12, self.curr_lookahead))

                # reload the existing weights, if present
                self.pair_model_info[pair]['model'] = self.get_model_weights(model)

            # return without training
            return dataframe

        model = self.get_model_weights(model)

        # set up training and test data

        # get a mormalised version, then extract data
        df = dataframe.fillna(0.0)
        # df = df.shift(-self.startup_candle_count)  # don't use data from startup period
        df = self.convert_date(df)
        tgt_col = df.columns.get_loc("smooth")
        # tgt_col = df.columns.get_loc("close")
        scaler = self.get_scaler()

        df_norm = scaler.fit_transform(df)

        # constrain size to what will be available in run modes
        df_size = df_norm.shape[0]
        # data_size = int(min(975, df_size))
        data_size = df_size # For backtest/hyperopt/plot, this will be big. Normal size for run modes

        pad = self.curr_lookahead # have to allow for future results to be in range
        train_ratio = 0.8
        test_ratio = 1.0 - train_ratio
        train_size = int(train_ratio * (data_size - pad)) - 1
        test_size = int(test_ratio * (data_size - pad)) - 1

        # trying different test options. For some reason, results vary quite dramatically based on the approach

        test_option = 1
        if test_option == 0:
            # take the middle part of the full dataframe
            train_start = int((df_size - (train_size + test_size + self.curr_lookahead)) / 2)
            test_start = train_start + train_size + 1
        elif test_option == 1:
            # take the end for training (better fit for recent data), earlier section for testing
            train_start = int(data_size - (train_size + pad))
            test_start = 0
        elif test_option == 2:
            # use the whole dataset for training, last section for testing (yes, I know this is not good)
            train_start = 0
            train_size = data_size - pad - 1
            test_start = data_size - (test_size + pad)
        else:
            # the 'classic' - first part train, last part test
            train_start = 0
            test_start = data_size - (test_size + pad) - 1

        train_result_start = train_start + self.curr_lookahead
        test_result_start = test_start + self.curr_lookahead

        # just double-check ;-)
        if (train_size + test_size + self.curr_lookahead) > data_size:
            print("ERR: invalid train/test sizes")
            print("     train_size:{} test_size:{} data_size:{}".format(train_size, test_size, data_size))

        if (train_result_start + train_size) > data_size:
            print("ERR: invalid train result config")
            print("     train_result_start:{} train_size:{} data_size:{}".format(train_result_start,
                                                                                 train_size, data_size))

        if (test_result_start + test_size) > data_size:
            print("ERR: invalid test result config")
            print("     test_result_start:{} train_size:{} data_size:{}".format(test_result_start,
                                                                                test_size, data_size))

        print("    data:[{}:{}] train:[{}:{}] train_result:[{}:{}] test:[{}:{}] test_result:[{}:{}] "
              .format(0, data_size-1,
                      train_start, (train_start+train_size),
                      train_result_start, (train_result_start+train_size),
                      test_start, (test_start+test_size),
                      test_result_start, (test_result_start+test_size)
                      ))

        # convert dataframe to tensor before extracting train/test data (avoid edge effects)
        df_tensor = self.df_to_tensor(df_norm, self.seq_len)
        # train_df_norm = df_norm[train_start:train_start + train_size, :]
        # train_results_norm = df_norm[train_result_start:train_result_start + train_size, tgt_col]
        train_df_norm = df_tensor[train_start:train_start + train_size]
        train_results_norm = df_norm[train_result_start:train_result_start + train_size, tgt_col]

        test_df_norm = df_tensor[test_start:test_start + test_size]
        test_results_norm = df_norm[test_result_start:test_result_start + test_size, tgt_col]

        # print(train_df_norm[:, tgt_col])
        # print(train_results_norm)


        # train_tensor = self.df_to_tensor(train_df_norm, self.seq_len)
        # test_tensor = self.df_to_tensor(test_df_norm, self.seq_len)
        train_tensor = train_df_norm
        test_tensor = test_df_norm

        # re-shape into format expected by LSTM model
        # train_df_norm = np.reshape(np.array(train_df_norm), (train_df_norm.shape[0], self.seq_len, train_df_norm.shape[1]))
        # test_df_norm = np.reshape(np.array(test_df_norm), (test_df_norm.shape[0], self.seq_len, test_df_norm.shape[1]))
        train_tensor = np.reshape(train_tensor, (train_size, self.seq_len, nfeatures))
        test_tensor = np.reshape(test_tensor, (test_size, self.seq_len, nfeatures))
        train_results_norm = np.array(train_results_norm).reshape(-1, 1)
        test_results_norm = np.array(test_results_norm).reshape(-1, 1)

        # print("")
        # print("    train data:", np.shape(train_tensor), " train results:", train_results_norm.shape)
        # print("    test data: ", np.shape(test_tensor), " test results: ", test_results_norm.shape)
        # print("")

        # train the model
        print("    fitting model...")
        print("")

        #TODO: save full model to current path?!

        # callback to control early exit on plateau of results
        early_callback = keras.callbacks.EarlyStopping(
            monitor="loss",
            mode="min",
            patience=4,
            verbose=1)

        plateau_callback = keras.callbacks.ReduceLROnPlateau(
            monitor='loss',
            factor=0.1,
            min_delta=0.001,
            patience=4,
            verbose=0)

        model_name = self.get_model_name()

        # callback to control saving of 'best' model
        # Note that we use validation loss as the metric, not training loss
        checkpoint_callback = keras.callbacks.ModelCheckpoint(
            filepath=model_name,
            save_weights_only=True,
            monitor='val_loss',
            mode='min',
            save_best_only=True,
            verbose=0)

        # TqdmCallback displays a progress bar instead of the default keras output

        # Model weights are saved at the end of every epoch, if it's the best seen so far.
        fhis = model.fit(train_tensor, train_results_norm,
                         batch_size=self.batch_size,
                         epochs=self.num_epochs,
                         callbacks=[checkpoint_callback, plateau_callback, early_callback],
                         validation_data=(test_tensor, test_results_norm),
                         verbose=1)

        # The model weights (that are considered the best) are loaded into th model.
        model = self.get_model_weights(model)

        # model.save(model_name)

        # print("")
        # print(fhis.history)
        print("")

        # test the model
        print("    checking model with test data...")
        results = model.evaluate(test_tensor, test_results_norm,
                                 batch_size=self.batch_size, verbose=0)
        # print("results:", results)

        # update the pair info
        self.pair_model_info[pair]['model'] = model
        self.pair_model_info[pair]['score'] = results[0]
        if results[0] > self.max_train_loss:
            print("    WARNING: high loss: {:.3f}".format(results[0]))

        return dataframe

    def get_model_name(self):
        # Note that keras expects it to be called 'checkpoint'
        checkpoint_dir = '/tmp'
        curr_class = self.__class__.__name__
        model_name = checkpoint_dir + "/" + curr_class + "/" + self.curr_pair.replace("/", "_") + "/checkpoint"
        return model_name

    def get_model_weights(self, model):

        model_name = self.get_model_name()

        # if checkpoint already exists, load it as a starting point
        if os.path.exists(model_name):
            print("    Loading existing model ({})...".format(model_name))
            try:
                model.load_weights(model_name)
            except:
                print("Error loading weights from {}. Check whether model format changed".format(model_name))
        else:
            print("    model not found ({})...".format(model_name))
            if self.dp.runmode.value not in ('hyperopt', 'backtest', 'plot'):
                print("*** ERR: no existing model. You should run backtest first!")
        return model

    # get a scaler for scaling/normalising the data (in a func because I change it routinely)
    def get_scaler(self):
        # uncomment the one yu want
        return StandardScaler()
        # return RobustScaler()
        # return MinMaxScaler()

    def df_to_tensor(self, data, seq_len):
        # input format = [nrows, nfeatures] output = [nrows, seq_len, nfeatures]
        nrows = np.shape(data)[0]
        nfeatures = np.shape(data)[1]
        tensor_arr = np.zeros((nrows, seq_len, nfeatures), dtype=float)
        zero_row = np.zeros((nfeatures), dtype=float)
        # tensor_arr = []

        reverse = True

        # fill the first part (0..seqlen rows), which are only sparsely populated
        for row in range(seq_len):
            for seq in range(seq_len):
                if seq >= (seq_len - row - 1):
                    tensor_arr[row][seq] = data[(row + seq) - seq_len + 1]
                else:
                    tensor_arr[row][seq] = zero_row
            if reverse:
                tensor_arr[row] = np.flipud(tensor_arr[row])

        # fill the rest
        # print("Data:{}, len:{}".format(np.shape(data), seq_len))
        for row in range(seq_len, nrows):
            tensor_arr[row] = data[(row - seq_len) + 1:row + 1]
            if reverse:
                tensor_arr[row] = np.flipud(tensor_arr[row])

        # print("data: ", data)
        # print("tensor: ", tensor_arr)
        # print("data:{} tensor:{}".format(np.shape(data), np.shape(tensor_arr)))
        return tensor_arr

    def get_lstm(self, nfeatures: int, seq_len: int):
        model = keras.Sequential()

        # print("Creating model. nfeatures:{} seq_len:{}".format(nfeatures, seq_len))

        # trying different models...
        model_type = 0
        if model_type == 0:
            # simplest possible model:
            model.add(layers.LSTM(64, return_sequences=True, input_shape=(seq_len, nfeatures)))
            # model.add(layers.Dropout(rate=0.2))
            model.add(layers.Dense(1, activation='linear'))

        elif model_type == 1:
            # intermediate model:
            model.add(layers.GRU(64, return_sequences=True, input_shape=(seq_len, nfeatures)))
            model.add(layers.Dropout(rate=0.4))
            # model.add(layers.Bidirectional(layers.LSTM(64, return_sequences=True)))
            # model.add(layers.Dropout(rate=0.4))
            model.add(layers.Bidirectional(layers.LSTM(64)))
            model.add(layers.Dropout(rate=0.4))
            model.add(layers.Dense(1, activation='linear'))

        elif model_type == 2:
            # complex model:
            model.add(layers.GRU(64, return_sequences=True, input_shape=(seq_len, nfeatures)))
            model.add(layers.LSTM(64, return_sequences=True))
            model.add(layers.Dense(32))
            model.add(layers.Dropout(rate=0.5))
            model.add(layers.LSTM(32, return_sequences=True))
            model.add(layers.Dropout(rate=0.5))
            model.add(layers.LSTM(32, return_sequences=True))
            model.add(layers.Dropout(rate=0.5))
            model.add(layers.LSTM(32, return_sequences=True))
            model.add(layers.Dropout(rate=0.5))
            model.add(layers.LSTM(32, return_sequences=False))
            model.add(layers.Dropout(rate=0.4))
            model.add(layers.Dense(8))
            model.add(layers.Dense(1, activation='linear'))

        elif model_type == 3:
            # Attention (Single Head)
            model.add(layers.LSTM(128, return_sequences=True, input_shape=(seq_len, nfeatures)))
            model.add(layers.Dropout(0.2))
            model.add(layers.BatchNormalization())
            model.add(Attention.Attention(seq_len))
            model.add(layers.Dense(32, activation="relu"))
            model.add(layers.Dropout(0.2))
            model.add(layers.Dense(1, activation='linear'))

        else:
            # simplest possible model:
            model.add(layers.LSTM(64, return_sequences=True, input_shape=(seq_len, nfeatures)))
            model.add(layers.Dense(1, activation='linear'))

        model.summary()  # helps keep track of which model is running, while making changes
        model.compile(optimizer='adam',
                      loss='mean_squared_error',
                      metrics=[keras.metrics.MeanAbsoluteError()])
        return model

    ################################

    # get predictions. Note that the input must already be in tensor format
    def get_predictions(self, df_chunk: np.array):

        # get the model
        model = self.pair_model_info[self.curr_pair]['model']

        # run the prediction
        preds_notrend = model.predict(df_chunk, verbose=0)
        preds_notrend = np.array(preds_notrend[:, 0]).reshape(-1, 1)

        return preds_notrend[:, 0]

    # run prediction in batches over the entire history
    def batch_predictions(self, dataframe: DataFrame):

        # check that model exists
        if self.pair_model_info[self.curr_pair]['model'] == None:
            print("*** No model for pair ", self.curr_pair)
            predictions = dataframe['close']
            return predictions

        # check that model score is good enough
        # check that model exists
        if self.pair_model_info[self.curr_pair]['score'] > self.max_train_loss:
            print("*** Model loss above threshold. Not predicting for: ", self.curr_pair)
            predictions = dataframe['close']
            return predictions

        # scale/normalise
        df = self.convert_date(dataframe)
        # tgt_col = df.columns.get_loc("close")
        tgt_col = df.columns.get_loc("smooth")
        scaler = self.get_scaler()

        scaler = scaler.fit(df)
        df_norm = scaler.transform(df)

        # df_norm3 = np.reshape(df_chunk, (np.shape(df_chunk)[0], self.seq_len, np.shape(df_chunk)[1]))

        # convert dataframe to tensor
        df_tensor = self.df_to_tensor(df_norm, self.seq_len)

        # prediction does not work well when run over a large dataset, so divide into chunks and predict for each one
        # then concatenate the results and return
        preds_notrend: np.array = []
        batch_size = self.predict_batch_size
        nruns = int(dataframe.shape[0] / batch_size)
        for i in tqdm(range(nruns), desc="    Predicting…", ascii=True, ncols=75):
            # for i in range(nruns):
            start = i * batch_size
            end = start + batch_size
            # print("start:{} end:{}".format(start, end))
            chunk = df_tensor[start:end]
            # print(chunk)
            preds = self.get_predictions(chunk)
            # print(preds)
            preds_notrend = np.concatenate((preds_notrend, preds))

        # copy whatever is leftover
        start = nruns * batch_size
        end = dataframe.shape[0]
        # print("start:{} end:{}".format(start, end))
        if end > start:
            chunk = df_tensor[start:end]
            preds = self.get_predictions(chunk)
            preds_notrend = np.concatenate((preds_notrend, preds))

        # re-scale the predictions
        # slight cheat - replace 'gain' column with predictions, then inverse scale
        cl_col = df_norm[:, tgt_col]

        df_norm[:, tgt_col] = preds_notrend
        inv_y = scaler.inverse_transform(df_norm)
        # print("preds_notrend:", preds_notrend.shape, " df_norm:", df_norm.shape, " inv_y:", inv_y.shape)
        predictions = inv_y[:, tgt_col]

        if tgt_col == 'gain':
            # using gain rather than price, so add gain to current price
            predictions = (1.0 + predictions) * dataframe['close']

        print("runs:{} predictions:{}".format(nruns, len(predictions)))
        return predictions

    # returns (rolling) smoothed version of input column
    def roll_smooth(self, col) -> np.float:
        # must return scalar, so just calculate prediction and take last value

        smooth = gaussian_filter1d(col, 4)
        # smooth = gaussian_filter1d(col, 2)

        length = len(smooth)
        if length > 0:
            return smooth[length - 1]
        else:
            return col[len(col) - 1]

    def roll_strong_smooth(self, col) -> np.float:
        # must return scalar, so just calculate prediction and take last value

        smooth = gaussian_filter1d(col, 24)

        length = len(smooth)
        if length > 0:
            return smooth[length - 1]
        else:
            return col[len(col) - 1]

    ################################

    clip_outliers = False

    # convert date column to number
    def convert_date(self, dataframe: DataFrame) -> DataFrame:
        df = dataframe.copy()
        if 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date']).astype('int64')
            df.set_index('date')
            df.reindex()
        return df

    def check_inf(self, dataframe: DataFrame):
        col_name = dataframe.columns.to_series()[np.isinf(dataframe).any()]
        if len(col_name) > 0:
            print("***")
            print("*** Infinity in cols: ", col_name)
            print("***")

    ################################

    """
    Buy Signal
    """

    def populate_buy_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        conditions = []
        dataframe.loc[:, 'enter_tag'] = ''
        curr_pair = metadata['pair']

        conditions.append(dataframe['volume'] > 0)

        # add some fairly loose guards, to help prevent 'bad' predictions

        # # ATR in buy range
        # conditions.append(dataframe['atr_signal'] > 0.0)

        # some trading volume
        conditions.append(dataframe['volume'] > 0)

        # # Fisher RSI + Williams combo
        # conditions.append(dataframe['fisher_wr'] < -0.7)
        #
        # # below Bollinger mid-point
        # conditions.append(dataframe['close'] < dataframe['bb_middleband'])

        # LSTM/Classifier triggers
        lstm_cond = (
            (qtpylib.crossed_above(dataframe['predict_diff'], 2.0))
        )
        conditions.append(lstm_cond)

        # set entry tags
        dataframe.loc[lstm_cond, 'enter_tag'] += 'lstm_entry '

        if conditions:
            dataframe.loc[reduce(lambda x, y: x & y, conditions), 'buy'] = 1
        else:
            dataframe['buy'] = 0

        # set first (startup) period to 0
        dataframe.loc[dataframe.index[:self.startup_candle_count], 'buy'] = 0

        return dataframe

    ###################################

    """
    Sell Signal
    """

    def populate_sell_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        conditions = []
        dataframe.loc[:, 'exit_tag'] = ''
        curr_pair = metadata['pair']

        conditions.append(dataframe['volume'] > 0)

        # # ATR in sell range
        # conditions.append(dataframe['atr_signal'] <= 0.0)

        # # above Bollinger mid-point
        # conditions.append(dataframe['close'] > dataframe['bb_middleband'])

        # # Fisher RSI + Williams combo
        # conditions.append(dataframe['fisher_wr'] > 0.5)

        # LSTM triggers
        lstm_cond = (
            qtpylib.crossed_below(dataframe['predict_diff'], -2.0)
        )

        conditions.append(lstm_cond)

        dataframe.loc[lstm_cond, 'exit_tag'] += 'lstm_exit '

        if conditions:
            dataframe.loc[reduce(lambda x, y: x & y, conditions), 'sell'] = 1
        else:
            dataframe['sell'] = 0

        # set first (startup) period to 0
        dataframe.loc[dataframe.index[:self.startup_candle_count], 'sell'] = 0

        return dataframe

    ###################################

    """
    Custom Stoploss
    """

    def custom_stoploss(self, pair: str, trade: 'Trade', current_time: datetime, current_rate: float,
                        current_profit: float, **kwargs) -> float:

        dataframe, last_updated = self.dp.get_analyzed_dataframe(pair=pair, timeframe=self.timeframe)
        last_candle = dataframe.iloc[-1].squeeze()
        trade_dur = int((current_time.timestamp() - trade.open_date_utc.timestamp()) // 60)
        in_trend = self.custom_trade_info[trade.pair]['had_trend']

        # limit stoploss
        if current_profit < self.cstop_max_stoploss.value:
            return 0.01

        # Determine how we sell when we are in a loss
        if current_profit < self.cstop_loss_threshold.value:
            if self.cstop_bail_how.value == 'roc' or self.cstop_bail_how.value == 'any':
                # Dynamic bailout based on rate of change
                if last_candle['sroc'] <= self.cstop_bail_roc.value:
                    return 0.01
            if self.cstop_bail_how.value == 'time' or self.cstop_bail_how.value == 'any':
                # Dynamic bailout based on time, unless time_trend is true and there is a potential reversal
                if trade_dur > self.cstop_bail_time.value:
                    if self.cstop_bail_time_trend.value == True and in_trend == True:
                        return 1
                    else:
                        return 0.01
        return 1

    ###################################

    """
    Custom Sell
    """

    def custom_sell(self, pair: str, trade: 'Trade', current_time: 'datetime', current_rate: float,
                    current_profit: float, **kwargs):

        dataframe, _ = self.dp.get_analyzed_dataframe(pair=pair, timeframe=self.timeframe)
        last_candle = dataframe.iloc[-1].squeeze()

        trade_dur = int((current_time.timestamp() - trade.open_date_utc.timestamp()) // 60)
        max_profit = max(0, trade.calc_profit_ratio(trade.max_rate))
        pullback_value = max(0, (max_profit - self.csell_pullback_amount.value))
        in_trend = False

        # Determine our current ROI point based on the defined type
        if self.csell_roi_type.value == 'static':
            min_roi = self.csell_roi_start.value
        elif self.csell_roi_type.value == 'decay':
            min_roi = cta.linear_decay(self.csell_roi_start.value, self.csell_roi_end.value, 0,
                                       self.csell_roi_time.value, trade_dur)
        elif self.csell_roi_type.value == 'step':
            if trade_dur < self.csell_roi_time.value:
                min_roi = self.csell_roi_start.value
            else:
                min_roi = self.csell_roi_end.value

        # Determine if there is a trend
        if self.csell_trend_type.value == 'rmi' or self.csell_trend_type.value == 'any':
            if last_candle['rmi_up_trend'] == 1:
                in_trend = True
        if self.csell_trend_type.value == 'ssl' or self.csell_trend_type.value == 'any':
            if last_candle['ssl_dir'] == 1:
                in_trend = True
        if self.csell_trend_type.value == 'candle' or self.csell_trend_type.value == 'any':
            if last_candle['candle_up_trend'] == 1:
                in_trend = True

        # Don't sell if we are in a trend unless the pullback threshold is met
        if in_trend == True and current_profit > 0:
            # Record that we were in a trend for this trade/pair for a more useful sell message later
            self.custom_trade_info[trade.pair]['had_trend'] = True
            # If pullback is enabled and profit has pulled back allow a sell, maybe
            if self.csell_pullback.value == True and (current_profit <= pullback_value):
                if self.csell_pullback_respect_roi.value == True and current_profit > min_roi:
                    return 'intrend_pullback_roi'
                elif self.csell_pullback_respect_roi.value == False:
                    if current_profit > min_roi:
                        return 'intrend_pullback_roi'
                    else:
                        return 'intrend_pullback_noroi'
            # We are in a trend and pullback is disabled or has not happened or various criteria were not met, hold
            return None
        # If we are not in a trend, just use the roi value
        elif in_trend == False:
            if self.custom_trade_info[trade.pair]['had_trend']:
                if current_profit > min_roi:
                    self.custom_trade_info[trade.pair]['had_trend'] = False
                    return 'trend_roi'
                elif self.csell_endtrend_respect_roi.value == False:
                    self.custom_trade_info[trade.pair]['had_trend'] = False
                    return 'trend_noroi'
            elif current_profit > min_roi:
                return 'notrend_roi'
        else:
            return None


#######################

# Utility functions


# Elliot Wave Oscillator
def ewo(dataframe, sma1_length=5, sma2_length=35):
    sma1 = ta.EMA(dataframe, timeperiod=sma1_length)
    sma2 = ta.EMA(dataframe, timeperiod=sma2_length)
    smadif = (sma1 - sma2) / dataframe['close'] * 100
    return smadif


# Chaikin Money Flow
def chaikin_money_flow(dataframe, n=20, fillna=False) -> Series:
    """Chaikin Money Flow (CMF)
    It measures the amount of Money Flow Volume over a specific period.
    http://stockcharts.com/school/doku.php?id=chart_school:technical_indicators:chaikin_money_flow_cmf
    Args:
        dataframe(pandas.Dataframe): dataframe containing ohlcv
        n(int): n period.
        fillna(bool): if True, fill nan values.
    Returns:
        pandas.Series: New feature generated.
    """
    mfv = ((dataframe['close'] - dataframe['low']) - (dataframe['high'] - dataframe['close'])) / (
            dataframe['high'] - dataframe['low'])
    mfv = mfv.fillna(0.0)  # float division by zero
    mfv *= dataframe['volume']
    cmf = (mfv.rolling(n, min_periods=0).sum()
           / dataframe['volume'].rolling(n, min_periods=0).sum())
    if fillna:
        cmf = cmf.replace([np.inf, -np.inf], np.nan).fillna(0)
    return Series(cmf, name='cmf')


# Williams %R
def williams_r(dataframe: DataFrame, period: int = 14) -> Series:
    """Williams %R, or just %R, is a technical analysis oscillator showing the current closing price in relation to the high and low
        of the past N days (for a given N). It was developed by a publisher and promoter of trading materials, Larry Williams.
        Its purpose is to tell whether a stock or commodity market is trading near the high or the low, or somewhere in between,
        of its recent trading range.
        The oscillator is on a negative scale, from −100 (lowest) up to 0 (highest).
    """

    highest_high = dataframe["high"].rolling(center=False, window=period).max()
    lowest_low = dataframe["low"].rolling(center=False, window=period).min()

    WR = Series(
        (highest_high - dataframe["close"]) / (highest_high - lowest_low),
        name=f"{period} Williams %R",
    )

    return WR * -100


# Volume Weighted Moving Average
def vwma(dataframe: DataFrame, length: int = 10):
    """Indicator: Volume Weighted Moving Average (VWMA)"""
    # Calculate Result
    pv = dataframe['close'] * dataframe['volume']
    vwma = Series(ta.SMA(pv, timeperiod=length) / ta.SMA(dataframe['volume'], timeperiod=length))
    vwma = vwma.fillna(0, inplace=True)
    return vwma


# Exponential moving average of a volume weighted simple moving average
def ema_vwma_osc(dataframe, len_slow_ma):
    slow_ema = Series(ta.EMA(vwma(dataframe, len_slow_ma), len_slow_ma))
    return ((slow_ema - slow_ema.shift(1)) / slow_ema.shift(1)) * 100


def t3_average(dataframe, length=5):
    """
    T3 Average by HPotter on Tradingview
    https://www.tradingview.com/script/qzoC9H1I-T3-Average/
    """
    df = dataframe.copy()

    df['xe1'] = ta.EMA(df['close'], timeperiod=length)
    df['xe1'].fillna(0, inplace=True)
    df['xe2'] = ta.EMA(df['xe1'], timeperiod=length)
    df['xe2'].fillna(0, inplace=True)
    df['xe3'] = ta.EMA(df['xe2'], timeperiod=length)
    df['xe3'].fillna(0, inplace=True)
    df['xe4'] = ta.EMA(df['xe3'], timeperiod=length)
    df['xe4'].fillna(0, inplace=True)
    df['xe5'] = ta.EMA(df['xe4'], timeperiod=length)
    df['xe5'].fillna(0, inplace=True)
    df['xe6'] = ta.EMA(df['xe5'], timeperiod=length)
    df['xe6'].fillna(0, inplace=True)
    b = 0.7
    c1 = -b * b * b
    c2 = 3 * b * b + 3 * b * b * b
    c3 = -6 * b * b - 3 * b - 3 * b * b * b
    c4 = 1 + 3 * b + b * b * b + 3 * b * b
    df['T3Average'] = c1 * df['xe6'] + c2 * df['xe5'] + c3 * df['xe4'] + c4 * df['xe3']

    return df['T3Average']


def pivot_points(dataframe: DataFrame, mode='fibonacci') -> Series:
    if mode == 'simple':
        hlc3_pivot = (dataframe['high'] + dataframe['low'] + dataframe['close']).shift(1) / 3
        res1 = hlc3_pivot * 2 - dataframe['low'].shift(1)
        sup1 = hlc3_pivot * 2 - dataframe['high'].shift(1)
        res2 = hlc3_pivot + (dataframe['high'] - dataframe['low']).shift()
        sup2 = hlc3_pivot - (dataframe['high'] - dataframe['low']).shift()
        res3 = hlc3_pivot * 2 + (dataframe['high'] - 2 * dataframe['low']).shift()
        sup3 = hlc3_pivot * 2 - (2 * dataframe['high'] - dataframe['low']).shift()
        return hlc3_pivot, res1, res2, res3, sup1, sup2, sup3
    elif mode == 'fibonacci':
        hlc3_pivot = (dataframe['high'] + dataframe['low'] + dataframe['close']).shift(1) / 3
        hl_range = (dataframe['high'] - dataframe['low']).shift(1)
        res1 = hlc3_pivot + 0.382 * hl_range
        sup1 = hlc3_pivot - 0.382 * hl_range
        res2 = hlc3_pivot + 0.618 * hl_range
        sup2 = hlc3_pivot - 0.618 * hl_range
        res3 = hlc3_pivot + 1 * hl_range
        sup3 = hlc3_pivot - 1 * hl_range
        return hlc3_pivot, res1, res2, res3, sup1, sup2, sup3
    elif mode == 'DeMark':
        demark_pivot_lt = (dataframe['low'] * 2 + dataframe['high'] + dataframe['close'])
        demark_pivot_eq = (dataframe['close'] * 2 + dataframe['low'] + dataframe['high'])
        demark_pivot_gt = (dataframe['high'] * 2 + dataframe['low'] + dataframe['close'])
        demark_pivot = np.where((dataframe['close'] < dataframe['open']), demark_pivot_lt,
                                np.where((dataframe['close'] > dataframe['open']), demark_pivot_gt, demark_pivot_eq))
        dm_pivot = demark_pivot / 4
        dm_res = demark_pivot / 2 - dataframe['low']
        dm_sup = demark_pivot / 2 - dataframe['high']
        return dm_pivot, dm_res, dm_sup


def heikin_ashi(dataframe, smooth_inputs=False, smooth_outputs=False, length=10):
    df = dataframe[['open', 'close', 'high', 'low']].copy().fillna(0)
    if smooth_inputs:
        df['open_s'] = ta.EMA(df['open'], timeframe=length)
        df['high_s'] = ta.EMA(df['high'], timeframe=length)
        df['low_s'] = ta.EMA(df['low'], timeframe=length)
        df['close_s'] = ta.EMA(df['close'], timeframe=length)

        open_ha = (df['open_s'].shift(1) + df['close_s'].shift(1)) / 2
        high_ha = df.loc[:, ['high_s', 'open_s', 'close_s']].max(axis=1)
        low_ha = df.loc[:, ['low_s', 'open_s', 'close_s']].min(axis=1)
        close_ha = (df['open_s'] + df['high_s'] + df['low_s'] + df['close_s']) / 4
    else:
        open_ha = (df['open'].shift(1) + df['close'].shift(1)) / 2
        high_ha = df.loc[:, ['high', 'open', 'close']].max(axis=1)
        low_ha = df.loc[:, ['low', 'open', 'close']].min(axis=1)
        close_ha = (df['open'] + df['high'] + df['low'] + df['close']) / 4

    open_ha = open_ha.fillna(0)
    high_ha = high_ha.fillna(0)
    low_ha = low_ha.fillna(0)
    close_ha = close_ha.fillna(0)

    if smooth_outputs:
        open_sha = ta.EMA(open_ha, timeframe=length)
        high_sha = ta.EMA(high_ha, timeframe=length)
        low_sha = ta.EMA(low_ha, timeframe=length)
        close_sha = ta.EMA(close_ha, timeframe=length)

        return open_sha, close_sha, low_sha
    else:
        return open_ha, close_ha, low_ha


# Range midpoint acts as Support
def is_support(row_data) -> bool:
    conditions = []
    for row in range(len(row_data) - 1):
        if row < len(row_data) // 2:
            conditions.append(row_data[row] > row_data[row + 1])
        else:
            conditions.append(row_data[row] < row_data[row + 1])
    result = reduce(lambda x, y: x & y, conditions)
    return result


# Range midpoint acts as Resistance
def is_resistance(row_data) -> bool:
    conditions = []
    for row in range(len(row_data) - 1):
        if row < len(row_data) // 2:
            conditions.append(row_data[row] < row_data[row + 1])
        else:
            conditions.append(row_data[row] > row_data[row + 1])
    result = reduce(lambda x, y: x & y, conditions)
    return result


def range_percent_change(dataframe: DataFrame, method, length: int) -> float:
    """
    Rolling Percentage Change Maximum across interval.

    :param dataframe: DataFrame The original OHLC dataframe
    :param method: High to Low / Open to Close
    :param length: int The length to look back
    """
    if method == 'HL':
        return (dataframe['high'].rolling(length).max() - dataframe['low'].rolling(length).min()) / dataframe[
            'low'].rolling(length).min()
    elif method == 'OC':
        return (dataframe['open'].rolling(length).max() - dataframe['close'].rolling(length).min()) / dataframe[
            'close'].rolling(length).min()
    else:
        raise ValueError(f"Method {method} not defined!")
