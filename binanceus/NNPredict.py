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
from datetime import datetime, timedelta, timezone
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
import tensorflow as tf
from keras import layers
from tqdm import tqdm
from tqdm.keras import TqdmCallback
import sklearn.decomposition as skd

import random
import Time2Vector
import Transformer
import Attention

from DataframeUtils import DataframeUtils
from DataframePopulator import DataframePopulator
from NNPredictor_LSTM import NNPredictor_LSTM

"""
####################################################################################
NNPredict - uses a Long-Short Term Memory neural network to try and predict the future stock price
      
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


class NNPredict(IStrategy):
    plot_config = {
        'main_plot': {
            'close': {'color': 'green'},
            # 'temp': {'color': 'teal'},
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
        "0": 0.06
    }

    # Stoploss:
    stoploss = -0.99

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
    process_only_new_candles = True  # this strat is very resource intensive, do not set to False

    # Strategy-specific global vars

    inf_mins = timeframe_to_minutes(inf_timeframe)
    data_mins = timeframe_to_minutes(timeframe)
    inf_ratio = int(inf_mins / data_mins)

    # These parameters control much of the behaviour because they control the generation of the training data
    # Unfortunately, these cannot be hyperopt params because they are used in populate_indicators, which is only run
    # once during hyperopt
    # lookahead_hours = 1.0
    lookahead_hours = 1.0
    n_profit_stddevs = 0.0
    n_loss_stddevs = 0.0
    min_f1_score = 0.49
    max_train_loss = 0.15

    curr_lookahead = int(12 * lookahead_hours)

    curr_pair = ""
    custom_trade_info = {}

    num_pairs = 0
    # pair_model_info = {}  # holds model-related info for each pair
    curr_dataframe: DataFrame = None
    normalise_data = True

    # the following affect training of the model. Bigger numbers give better model, but take longer and use more memory
    seq_len = 12  # 'depth' of training sequence
    num_epochs = 128  # number of iterations for training
    batch_size = 1024  # batch size for training
    predict_batch_size = 512

    classifier_list = {}
    compressor = None
    compress_data = False  # currently not working
    refit_model = False  # set to True if you want to refit an existing model (e.g. single model across all pairs)

    dataframeUtils = None
    dataframePopulator = None

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

    ################################
    # class to create custom Keras layer that decompresses and denormalises predictions from the model
    class RestorePredictions(tf.keras.layers.Layer):
        def call(self, preds):
            inputs = keras.Input(preds)
            x = layers.Dense(1)(inputs)  # output shape should be the same as the original input shape
            x = self.compressor.inverse_transform(x)
            x = self.dataframeUtils.get_scaler().inverse_transform(x)
            restored = keras.Model(inputs, x)
            return restored

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

        # Base pair inf timeframe indicators
        curr_pair = metadata['pair']
        self.curr_pair = curr_pair
        self.curr_dataframe = dataframe

        self.curr_lookahead = int(12 * self.lookahead_hours)
        self.dbg_curr_df = dataframe

        if self.dataframeUtils is None:
            self.dataframeUtils = DataframeUtils()

        if self.dataframePopulator is None:
            self.dataframePopulator = DataframePopulator()

            self.dataframePopulator.runmode = self.dp.runmode.value
            self.dataframePopulator.win_size = min(14, self.curr_lookahead)
            self.dataframePopulator.startup_win = self.startup_candle_count
            self.dataframePopulator.n_loss_stddevs = self.n_loss_stddevs
            self.dataframePopulator.n_profit_stddevs = self.n_profit_stddevs

        if NNPredict.first_time:
            NNPredict.first_time = False
            print("")
            print("***************************************")
            print("** Warning: startup can be very slow **")
            print("***************************************")

            print("    Lookahead: ", self.curr_lookahead, " candles (", self.lookahead_hours, " hours)")

        print("")
        print(self.curr_pair)

        # make sure we only retrain in backtest modes
        if self.dp.runmode.value not in ('backtest'):
            self.refit_model = False

        # populate the standard indicators
        dataframe = self.dataframePopulator.add_indicators(dataframe)

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
        dataframe = self.add_stoploss_indicators(dataframe, self.curr_pair)

        return dataframe

    ###################################

    # add in any indicators to be used for training
    def add_training_indicators(self, dataframe: DataFrame) -> DataFrame:

        # placeholders, just need the columns to be there (with some realistic values)
        dataframe['predict'] = dataframe['close']
        dataframe['temp'] = dataframe['close']

        # future_df = self.dataframePopulator.add_hidden_indicators(dataframe.copy())
        # future_df = self.dataframePopulator.add_future_data(future_df, self.curr_lookahead)
        return dataframe

    def add_stoploss_indicators(self, dataframe: DataFrame, pair) -> DataFrame:

        if not pair in self.custom_trade_info:
            self.custom_trade_info[pair] = {}
            if not 'had_trend' in self.custom_trade_info[pair]:
                self.custom_trade_info[pair]['had_trend'] = False

        # Indicators used for ROI and Custom Stoploss
        dataframe = self.dataframePopulator.add_stoploss_indicators(dataframe)
        return dataframe

    # add columns based on predictions. Do not call until after model has been trained
    def add_predictions(self, dataframe: DataFrame, pair) -> DataFrame:

        win_size = max(self.curr_lookahead, 14)

        dataframe = self.add_model_predictions(dataframe)
        dataframe['predict_smooth'] = dataframe['predict'].rolling(window=win_size).apply(self.roll_strong_smooth)

        dataframe['predict_diff'] = 100.0 * (dataframe['predict'] - dataframe['close']) / dataframe['close']

        # dataframe['predict_diff'] = 100.0 * (dataframe['predict_smooth'] - dataframe['smooth']) / dataframe['smooth']

        return dataframe

    ################################

    def train_model(self, dataframe: DataFrame, pair) -> DataFrame:

        df_norm = self.dataframeUtils.norm_dataframe(dataframe)

        # save closing prices for later
        prices = np.array(df_norm['close'])

        # compress data
        if self.compress_data:
            old_size = df_norm.shape[1]
            df_norm = self.compress_dataframe(df_norm)
            print("    Compressed data {} -> {} (features)".format(old_size, df_norm.shape[1]))

        # constrain size to what will be available in run modes
        df_size = df_norm.shape[0]
        # data_size = int(min(975, df_size))
        data_size = df_size  # For backtest/hyperopt/plot, this will be big. Normal size for run modes

        pad = self.curr_lookahead  # have to allow for future results to be in range
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

        # # just double-check ;-)
        # if (train_size + test_size + self.curr_lookahead) > data_size:
        #     print("ERR: invalid train/test sizes")
        #     print("     train_size:{} test_size:{} data_size:{}".format(train_size, test_size, data_size))
        #
        # if (train_result_start + train_size) > data_size:
        #     print("ERR: invalid train result config")
        #     print("     train_result_start:{} train_size:{} data_size:{}".format(train_result_start,
        #                                                                          train_size, data_size))
        #
        # if (test_result_start + test_size) > data_size:
        #     print("ERR: invalid test result config")
        #     print("     test_result_start:{} train_size:{} data_size:{}".format(test_result_start,
        #                                                                         test_size, data_size))
        #
        # print("    data:[{}:{}] train:[{}:{}] train_result:[{}:{}] test:[{}:{}] test_result:[{}:{}] "
        #       .format(0, data_size-1,
        #               train_start, (train_start+train_size),
        #               train_result_start, (train_result_start+train_size),
        #               test_start, (test_start+test_size),
        #               test_result_start, (test_result_start+test_size)
        #               ))

        # convert dataframe to tensor before extracting train/test data (avoid edge effects)
        df_tensor = self.dataframeUtils.df_to_tensor(df_norm, self.seq_len)
        train_df_norm = df_tensor[train_start:train_start + train_size]
        test_df_norm = df_tensor[test_start:test_start + test_size]

        # extract prices from dataframe and convert to tensors
        train_results = prices[train_result_start:train_result_start + train_size]
        test_results = prices[test_result_start:test_result_start + test_size]
        train_results_norm = self.dataframeUtils.df_to_tensor(train_results.reshape(-1, 1), self.seq_len)
        test_results_norm = self.dataframeUtils.df_to_tensor(test_results.reshape(-1, 1), self.seq_len)

        train_tensor = train_df_norm
        test_tensor = test_df_norm

        # print("prices:", np.shape(prices), " train results:", np.shape(train_results))
        # print("train data:", np.shape(train_df_norm), " train results norm:", np.shape(train_results_norm))

        # print("")
        # print("    train data:", np.shape(train_tensor), " train results:", train_results_norm.shape)
        # print("    test data: ", np.shape(test_tensor), " test results: ", test_results_norm.shape)
        # print("")

        # create/retrieve the model
        nfeatures = np.shape(train_tensor)[2]

        if self.curr_pair not in self.classifier_list:
            self.classifier_list[self.curr_pair] = self.get_classifier(self.curr_pair, nfeatures, self.seq_len)

        # train the model
        print("    fitting model...")
        print("")
        force_train = False if (not self.dp.runmode.value in ('backtest')) else self.refit_model
        self.classifier_list[self.curr_pair].train(train_tensor, test_tensor,
                                                   train_results_norm, test_results_norm,
                                                   force_train)

        return dataframe

    # returns the classifier model. Override this function to change the type of classifier
    def get_classifier(self, pair, num_features: int, seq_len: int):
        return NNPredictor_LSTM(pair, seq_len, num_features)

    ################################

    # get predictions. Note that the input must already be in tensor format
    def get_predictions(self, df_chunk: np.array):

        if self.curr_pair not in self.classifier_list:
            print("    ERR: no classifier")
            preds_notrend = np.zeros(np.shape(df_chunk)[0], dtype=float)
            return preds_notrend
        else:
            # run the prediction
            preds_notrend = self.classifier_list[self.curr_pair].predict(df_chunk)
            preds_notrend = np.array(preds_notrend[:, 0]).reshape(-1, 1)
            preds_notrend = preds_notrend[:, 0]

            return preds_notrend

    # run prediction in batches over the entire history
    def add_model_predictions(self, dataframe: DataFrame) -> DataFrame:

        # check that model exists
        if self.curr_pair not in self.classifier_list:
            print("*** No model for pair ", self.curr_pair)
            predictions = dataframe['close']
            return predictions

        # scale/normalise
        # save mean & std of close column, need this to re-scale predictions later
        cl_mean = dataframe['close'].mean()
        cl_std = dataframe['close'].std()

        df_norm = self.dataframeUtils.norm_dataframe(dataframe)

        # compress
        if self.compress_data:
            old_dim = np.shape(df_norm)[1]
            df_norm = self.compress_dataframe(df_norm)
            print(f"    Compressed dataframe {old_dim} -> {np.shape(df_norm)[1]}")

        # convert dataframe to tensor
        df_tensor = self.dataframeUtils.df_to_tensor(df_norm, self.seq_len)

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
        # slight cheat - replace 'temp' column with predictions, then inverse scale

        # decompress
        if self.compress_data:
            # TODO: this is not working - can't add a column because PCA inverse_transform will coomplain
            #      can't really replace a column because the scaling is wrong

            predictions = preds_notrend * cl_std + cl_mean

        else:
            # replace the 'temp' column and de-normalise (this is why we added the 'temp' column earlier, to match dimensions)
            df_norm["temp"] = preds_notrend
            inv_y = self.dataframeUtils.denorm_dataframe(df_norm)
            predictions = inv_y["temp"]

        # print("runs:{} predictions:{}".format(nruns, len(predictions)))

        dataframe['predict'] = predictions
        return dataframe

    #######################################

    def get_compressor(self, df_norm: DataFrame):
        # just use fixed size PCA (easier for classifiers to deal with)
        ncols = 64
        compressor = skd.PCA(n_components=ncols, whiten=True, svd_solver='full').fit(df_norm)
        return compressor

    # compress the supplied dataframe
    def compress_dataframe(self, dataframe: DataFrame) -> DataFrame:
        if not self.compressor:
            self.compressor = self.get_compressor(dataframe)
        return pd.DataFrame(self.compressor.transform(dataframe))

    # decompress the supplied dataframe
    def decompress_dataframe(self, dataframe: DataFrame) -> DataFrame:
        if not self.compressor:
            print("    WARN: dataframe was not compressed")
            return dataframe
        return pd.DataFrame(self.compressor.inverse_transform(dataframe))

    #######################################

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

    ################################

    """
    Buy Signal
    """

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        conditions = []
        dataframe.loc[:, 'enter_tag'] = ''
        curr_pair = metadata['pair']

        # conditions.append(dataframe['volume'] > 0)

        # some trading volume
        conditions.append(dataframe['volume'] > 0)

        # loose guard
        conditions.append(dataframe['mfi'] < 50.0)

        # Classifier triggers
        predict_cond = (
            (qtpylib.crossed_above(dataframe['predict_diff'], 1.0))
        )
        conditions.append(predict_cond)

        # set entry tags
        dataframe.loc[predict_cond, 'enter_tag'] += 'predict_entry '

        if conditions:
            dataframe.loc[reduce(lambda x, y: x & y, conditions), 'buy'] = 1
        else:
            dataframe['entry'] = 0

        # set first (startup) period to 0
        dataframe.loc[dataframe.index[:self.startup_candle_count], 'buy'] = 0

        return dataframe

    ###################################

    """
    Sell Signal
    """

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        conditions = []
        dataframe.loc[:, 'exit_tag'] = ''
        curr_pair = metadata['pair']

        conditions.append(dataframe['volume'] > 0)

        # loose guard
        conditions.append(dataframe['mfi'] > 50.0)

        # Classifier triggers
        predict_cond = (
            qtpylib.crossed_below(dataframe['predict_diff'], -1.0)
        )

        conditions.append(predict_cond)

        dataframe.loc[predict_cond, 'exit_tag'] += 'predict_exit '

        if conditions:
            dataframe.loc[reduce(lambda x, y: x & y, conditions), 'sell'] = 1
        else:
            dataframe['exit'] = 0

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

    def custom_exit(self, pair: str, trade: 'Trade', current_time: 'datetime', current_rate: float,
                    current_profit: float, **kwargs):

        dataframe, _ = self.dp.get_analyzed_dataframe(pair=pair, timeframe=self.timeframe)
        last_candle = dataframe.iloc[-1].squeeze()

        trade_dur = int((current_time.timestamp() - trade.open_date_utc.timestamp()) // 60)
        max_profit = max(0, trade.calc_profit_ratio(trade.max_rate))
        pullback_value = max(0, (max_profit - self.cexit_pullback_amount.value))
        in_trend = False

        # Mod: just take the profit:
        # Above 3%, sell if MFA > 90
        if current_profit > 0.03:
            if last_candle['mfi'] > 90:
                return 'mfi_90'

        # Sell any positions at a loss if they are held for more than one day.
        if current_profit < 0.0 and (current_time - trade.open_date_utc).days >= 2:
            return 'unclog'

        # Determine our current ROI point based on the defined type
        if self.cexit_roi_type.value == 'static':
            min_roi = self.cexit_roi_start.value
        elif self.cexit_roi_type.value == 'decay':
            min_roi = cta.linear_decay(self.cexit_roi_start.value, self.cexit_roi_end.value, 0,
                                       self.cexit_roi_time.value, trade_dur)
        elif self.cexit_roi_type.value == 'step':
            if trade_dur < self.cexit_roi_time.value:
                min_roi = self.cexit_roi_start.value
            else:
                min_roi = self.cexit_roi_end.value

        # Determine if there is a trend
        if self.cexit_trend_type.value == 'rmi' or self.cexit_trend_type.value == 'any':
            if last_candle['rmi_up_trend'] == 1:
                in_trend = True
        if self.cexit_trend_type.value == 'ssl' or self.cexit_trend_type.value == 'any':
            if last_candle['ssl_dir'] == 1:
                in_trend = True
        if self.cexit_trend_type.value == 'candle' or self.cexit_trend_type.value == 'any':
            if last_candle['candle_up_trend'] == 1:
                in_trend = True

        # Don't sell if we are in a trend unless the pullback threshold is met
        if in_trend == True and current_profit > 0:
            # Record that we were in a trend for this trade/pair for a more useful sell message later
            self.custom_trade_info[trade.pair]['had_trend'] = True
            # If pullback is enabled and profit has pulled back allow a sell, maybe
            if self.cexit_pullback.value == True and (current_profit <= pullback_value):
                if self.cexit_pullback_respect_roi.value == True and current_profit > min_roi:
                    return 'intrend_pullback_roi'
                elif self.cexit_pullback_respect_roi.value == False:
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
                elif self.cexit_endtrend_respect_roi.value == False:
                    self.custom_trade_info[trade.pair]['had_trend'] = False
                    return 'trend_noroi'
            elif current_profit > min_roi:
                return 'notrend_roi'
        else:
            return None

#######################
