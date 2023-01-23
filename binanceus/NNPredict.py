import operator

import numpy as np
from enum import Enum

import pywt
import talib.abstract as ta
from scipy.ndimage import gaussian_filter1d

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

from DataframeUtils import DataframeUtils, ScalerType
from DataframePopulator import DataframePopulator
from NNPredictor_LSTM import NNPredictor_LSTM
import Environment

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
            'close': {'color': 'cornflowerblue'},
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
    lookahead_hours = 0.4
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

    classifier_list = {}  # classifier for each pair
    init_done = {}  # flags whether initialisation has been done for a pair or not

    compressor = None
    compress_data = False  # currently not working
    refit_model = False  # set to True if you want to re-train the model. Usually better to just delete it and restart
    scaler_type = ScalerType.Robust  # scaler type used for normalisation
    # scaler_type = ScalerType.Standard  # scaler type used for normalisation
    model_per_pair = True  # set to True to create pair-specific models (better but only works for pairs in whitelist)
    training_only = False  # set to True to just generate models, no backtesting or prediction

    # target_column = 'close'  # which column should be used for training and prediction
    target_column = 'mid'

    dataframeUtils = None
    dataframePopulator = None

    # flags used for initialisation
    first_time = True  # mostly for debug
    first_run = True  # used to identify first time through buy/sell populate funcs

    dbg_verbose = True  # controls debug output
    dbg_test_classifier = False  # test clasifiers after fitting
    dbg_curr_df: DataFrame = None  # for debugging of current dataframe
    dbg_enable_tracing = False  # set to True in subclass to enable function tracing

    # variables to track state
    class State(Enum):
        INIT = 1
        POPULATE = 2
        STOPLOSS = 3
        RUNNING = 4

    ###################################

    # Strategy Specific Variable Storage

    ## Hyperopt Variables

    # buy/sell hyperparams

    # threshold values (in %)
    entry_threshold = DecimalParameter(0.1, 3.0, default=1.0, decimals=1, space='buy', load=True, optimize=True)
    exit_threshold = DecimalParameter(-3.0, -0.1, default=-1.0, decimals=1, space='sell', load=True, optimize=True)

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
            self.dataframePopulator.win_size = max(14, self.curr_lookahead)
            self.dataframePopulator.startup_win = self.startup_candle_count
            self.dataframePopulator.n_loss_stddevs = self.n_loss_stddevs
            self.dataframePopulator.n_profit_stddevs = self.n_profit_stddevs

        if NNPredict.first_time:
            NNPredict.first_time = False

            print("")
            print("***************************************")
            print("** Warning: startup can be very slow **")
            print("***************************************")

            Environment.print_environment()

            print(f"    Lookahead: {self.curr_lookahead} candles ({self.lookahead_hours} hours)")
            print(f"    Re-train existing models: {self.refit_model}")
            print(f"    Training (only) mode: {self.training_only}")

            # debug tracing
            if self.dbg_enable_tracing:
                self.enable_function_tracing()

        print("")
        print(self.curr_pair)

        # make sure we only retrain in backtest modes
        if self.dp.runmode.value not in ('backtest'):
            self.refit_model = False
            self.training_only = False

        # (re-)set the scaler
        self.dataframeUtils.set_scaler_type(self.scaler_type)

        if self.dbg_verbose:
            print("    Adding technical indicators...")
        dataframe = self.add_indicators(dataframe)

        # train the model
        if self.dbg_verbose:
            print("    training model...")

        # if we are training, then force re-training of an existing model
        if self.training_only:
            self.refit_model = True

        dataframe = self.train_model(dataframe, self.curr_pair)

        # if in training mode then skip further processing.
        # Doesn't make sense without the model anyway, and it can sometimes be very slow

        if self.training_only:
            print("    Training mode. Skipping backtesting and prediction steps")
            print("        freqtrade backtest results will show no trades")
            print("        set training_only=False to re-enable full backtesting")

        else:
            # if first time through, run backtest
            if self.curr_pair not in self.init_done:
                self.init_done[self.curr_pair] = True
                print("    running backtest...")
                dataframe = self.backtest_data(dataframe)

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
    # add the 'standard' indicators. Override this is you want to use something else
    def add_indicators(self, dataframe: DataFrame) -> DataFrame:

        # populate the standard indicators
        dataframe = self.dataframePopulator.add_indicators(dataframe)

        # populate the training indicators
        dataframe = self.add_training_indicators(dataframe)

        return dataframe

    # add in any indicators to be used for training
    def add_training_indicators(self, dataframe: DataFrame) -> DataFrame:

        # placeholders, just need the columns to be there for later (with some realistic values)
        dataframe['predict'] = 0.0
        dataframe['temp'] = 0.0

        # no looking ahead in this approach
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

    ################################

    # prepare data and train the model
    def train_model(self, dataframe: DataFrame, pair) -> DataFrame:

        nfeatures = np.shape(dataframe)[1]

        # create the classifier if it doesn't already exist
        if self.curr_pair not in self.classifier_list:
            self.classifier_list[self.curr_pair] = self.make_classifier(self.curr_pair, self.seq_len, nfeatures)

        if self.classifier_list[self.curr_pair].prescale_data():
            df_norm = self.dataframeUtils.norm_dataframe(dataframe)
        else:
            df_norm = dataframe.copy()

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

        # some classifiers take DataFrames as input, others tensors, so check
        if self.classifier_list[self.curr_pair].needs_dataframes():
            # save closing prices for later
            prices = df_norm[self.target_column]

            train_df = df_norm.iloc[train_start:(train_start + train_size)]
            test_df = df_norm.iloc[test_start:(test_start + test_size)]

            # extract prices from dataframe and convert to tensors
            train_prices_df = prices.iloc[train_result_start:train_result_start + train_size]
            test_prices_df = prices.iloc[test_result_start:test_result_start + test_size]

            # copy to the vars used for training
            train_data = train_df
            test_data = test_df
            train_results = train_prices_df
            test_results = test_prices_df

        else:
            # save closing prices for later
            prices = np.array(df_norm[self.target_column])

            # convert dataframe to tensor before extracting train/test data (avoid edge effects)
            df_tensor = self.dataframeUtils.df_to_tensor(df_norm, self.seq_len)
            train_tensor = df_tensor[train_start:train_start + train_size]
            test_tensor = df_tensor[test_start:test_start + test_size]

            # extract prices from dataframe and convert to tensors
            train_prices = prices[train_result_start:train_result_start + train_size]
            test_prices = prices[test_result_start:test_result_start + test_size]
            train_prices_tensor = self.dataframeUtils.df_to_tensor(train_prices.reshape(-1, 1), self.seq_len)
            test_prices_tensor = self.dataframeUtils.df_to_tensor(test_prices.reshape(-1, 1), self.seq_len)

            # copy to the vars used for training
            train_data = train_tensor
            test_data = test_tensor
            train_results = train_prices_tensor
            test_results = test_prices_tensor

            # print("prices:", np.shape(prices), " train results:", np.shape(train_results))
            # print("train data:", np.shape(train_df_norm), " train results norm:", np.shape(train_results_norm))

            # print("")
            # print("    train data:", np.shape(train_tensor), " train results:", train_results_norm.shape)
            # print("    test data: ", np.shape(test_tensor), " test results: ", test_results_norm.shape)
            # print("")

        # create/retrieve the model

        # train the model
        print("    fitting model...")
        print("")
        force_train = self.refit_model if (self.dp.runmode.value in ('backtest')) else False
        # print(f"self.refit_model:{self.refit_model} self.dp.runmode.value:{self.dp.runmode.value}")
        self.classifier_list[self.curr_pair].train(train_data, test_data,
                                                   train_results, test_results,
                                                   force_train)

        if self.dbg_test_classifier:
            self.classifier_list[self.curr_pair].evaluate(test_data)

        return dataframe

    ################################

    # backtest the data and update the dataframe
    def backtest_data(self, dataframe: DataFrame) -> DataFrame:

        # get the current classifier
        classifier = self.classifier_list[self.curr_pair]
        use_dataframes = classifier.needs_dataframes()
        prescale_data = classifier.prescale_data()
        price_scaler = self.dataframeUtils.make_scaler()

        # print(f'backtest_data() - use_dataframes:{use_dataframes} prescale_data:{prescale_data}')

        # pre-scale if needed
        if prescale_data:
            price_scaler.fit(np.array(dataframe[self.target_column]).reshape(1, -1))
            df_norm = self.dataframeUtils.norm_dataframe(dataframe)
        else:
            df_norm = dataframe

        # convert to tensor, if needed
        if use_dataframes:
            data = df_norm
        else:
            data = self.dataframeUtils.df_to_tensor(df_norm, self.seq_len)

        # run backtest
        preds_notrend = classifier.backtest(data)

        # re-scale, if necessary
        if prescale_data:
            # # replace the 'temp' column and de-normalise (this is why we added the 'temp' column earlier, to match dimensions)
            # df_norm["temp"] = preds_notrend
            # inv_y = self.dataframeUtils.denorm_dataframe(df_norm)
            # predictions = inv_y["temp"]
            predictions = price_scaler.inverse_transform(preds_notrend.reshape(1, -1))[0]

            # print("")
            # print(f"preds_notrend: {preds_notrend}")
            # print("")
            # print(f"predictions: {predictions}")
            # print("")

        else:
            # classifier handles scaling
            predictions = preds_notrend

        dataframe['predict'] = predictions

        return dataframe

    ################################

    # update predictions for the latest part of the dataframe
    def update_predictions(self, dataframe: DataFrame) -> DataFrame:

        # get the current classifier
        classifier = self.classifier_list[self.curr_pair]
        use_dataframes = classifier.needs_dataframes()
        prescale_data = classifier.prescale_data()

        # get a scaler for the price data
        price_scaler = self.dataframeUtils.make_scaler()

        # print(f'backtest_data() - use_dataframes:{use_dataframes} prescale_data:{prescale_data}')

        # pre-scale if needed
        if prescale_data:
            df_norm = self.dataframeUtils.norm_dataframe(dataframe)
        else:
            df_norm = dataframe

        # extract the last part of the data
        window = 128
        end = np.shape(df_norm)[0] - 1
        start = end - window
        if use_dataframes:
            data = df_norm.iloc[start:end]
        else:
            tensor = self.dataframeUtils.df_to_tensor(df_norm, self.seq_len)
            data = tensor[start:end]

        if prescale_data:
            # fit price scaler on subset of cloe column
            price_scaler.fit(np.array(dataframe[self.target_column].iloc[start:end]).reshape(1, -1))

        # predict
        latest_prediction = dataframe[self.target_column].iloc[-1]
        if classifier.returns_single_prediction():
            predictions = classifier.predict(data)
            latest_prediction = predictions[-1]
        else:
            preds_notrend = self.get_predictions(data)

            # re-scale, if necessary
            if prescale_data:
                # # replace the 'temp' column and de-normalise (this is why we added the 'temp' column earlier, to match dimensions)
                # df_norm["temp"].iloc[start:end] = preds_notrend
                # inv_y = self.dataframeUtils.denorm_dataframe(df_norm)
                # predictions = inv_y["temp"]
                predictions = price_scaler.inverse_transform(preds_notrend.reshape(1, -1))[0]
                # latest_prediction = predictions.iloc[-1]
                latest_prediction = predictions[-1]
            else:
                # classifier handles scaling
                predictions = preds_notrend
                latest_prediction = predictions[-1]

        # update just the last entry in the predict column
        dataframe['predict'].iloc[-1] = latest_prediction
        return dataframe

    ################################

    # get predictions from the model. Directly updates the 'predict' column
    def add_model_predictions(self, dataframe: DataFrame) -> DataFrame:

        # check that model exists
        if self.curr_pair not in self.classifier_list:
            print("*** No model for pair ", self.curr_pair)
            return dataframe

        # if the model produces single valued predictions then we have to roll through the dataframe
        # if not, we can use a more efficient batching approach
        if self.classifier_list[self.curr_pair].returns_single_prediction():
            dataframe = self.add_model_rolling_predictions(dataframe)
        else:
            dataframe = self.add_model_batch_predictions(dataframe)

        return dataframe

    # get predictions. Note that the input can be dataframe or tensor
    def get_predictions(self, data_chunk):

        if self.curr_pair not in self.classifier_list:
            print("    ERR: no classifier")
            preds_notrend = np.zeros(np.shape(data_chunk)[0], dtype=float)
            return preds_notrend

        # run the prediction
        preds_notrend = self.classifier_list[self.curr_pair].predict(data_chunk)

        predictions = preds_notrend
        # print(df)
        if self.classifier_list[self.curr_pair].returns_single_prediction():
            predictions = [predictions[-1]]

        return predictions

    # run prediction in batches over the entire history
    def add_model_batch_predictions(self, dataframe: DataFrame) -> DataFrame:

        # scale/normalise
        # save mean & std of close column, need this to re-scale predictions later
        cl_mean = dataframe[self.target_column].mean()
        cl_std = dataframe[self.target_column].std()

        # get the current classifier
        classifier = self.classifier_list[self.curr_pair]
        use_dataframes = classifier.needs_dataframes()
        prescale_data = classifier.prescale_data()

        # pre-scale if needed
        if prescale_data:
            df_norm = self.dataframeUtils.norm_dataframe(dataframe)
        else:
            df_norm = dataframe

        # compress if specified
        if self.compress_data:
            old_dim = np.shape(df_norm)[1]
            df_norm = self.compress_dataframe(df_norm)
            print(f"    Compressed dataframe {old_dim} -> {np.shape(df_norm)[1]}")

        if not use_dataframes:
            # convert dataframe to tensor
            df_tensor = self.dataframeUtils.df_to_tensor(df_norm, self.seq_len)

        # prediction does not work well when run over a large dataset, so divide into chunks and predict for each one
        # then concatenate the results and return
        preds_notrend: np.array = []
        batch_size = self.predict_batch_size

        nruns = int(np.shape(df_norm)[0] / batch_size)
        # print(f'    batch_size:{batch_size} nruns:{nruns}')

        for i in tqdm(range(nruns), desc="    Predicting…", ascii=True, ncols=75):
            # for i in range(nruns):
            start = i * batch_size
            end = start + batch_size
            # print("start:{} end:{}".format(start, end))
            if use_dataframes:
                chunk = df_norm.iloc[start:end]
            else:
                chunk = df_tensor[start:end]
            # print(chunk)
            preds = self.get_predictions(chunk)
            # print(preds)
            preds_notrend = np.concatenate((preds_notrend, preds))

        # copy whatever is leftover
        start = nruns * batch_size
        end = np.shape(df_norm)[0]
        # print("start:{} end:{}".format(start, end))
        if end > start:
            if use_dataframes:
                chunk = df_norm.iloc[start:end]
            else:
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
            if prescale_data:
                # replace the 'temp' column and de-normalise (this is why we added the 'temp' column earlier, to match dimensions)
                df_norm["temp"] = preds_notrend
                inv_y = self.dataframeUtils.denorm_dataframe(df_norm)
                predictions = inv_y["temp"]
            else:
                # classifier handles scaling
                predictions = preds_notrend

        # print("runs:{} predictions:{}".format(nruns, len(predictions)))

        dataframe['predict'] = predictions
        return dataframe

    # run prediction in rolling fashion (one result at a time -> slow) over the entire history
    def add_model_rolling_predictions(self, dataframe: DataFrame) -> DataFrame:

        print("    Adding rolling predictions. Might take a while...")

        # probably don't need to handle tensor cases since those classifiers typically do not return single values

        # get the current clasifier
        classifier = self.classifier_list[self.curr_pair]
        use_dataframes = classifier.needs_dataframes()
        prescale_data = classifier.prescale_data()

        # pre-scale if needed
        if prescale_data:
            df_norm = self.dataframeUtils.norm_dataframe(dataframe)
        else:
            df_norm = dataframe

        if not use_dataframes:
            df_tensor = self.dataframeUtils.df_to_tensor(df_norm, self.seq_len)

        window = 64
        start = window
        end = np.shape(df_norm)[0]

        preds_notrend: np.array = []

        # set values for startup window
        for index in range(0, start):
            preds_notrend = np.concatenate((preds_notrend, [df_norm[self.target_column].iloc[index]]))

        # add predictions
        for i in tqdm(range(start, end), desc="    Predicting…", ascii=True, ncols=75):
            # for index in range(start, end):
            if use_dataframes:
                chunk = df_norm.iloc[start:end]
            else:
                chunk = df_tensor[start:end]
            preds = self.get_predictions(chunk)
            preds_notrend = np.concatenate((preds_notrend, preds))

        # re-scale, if needed
        if prescale_data:
            # replace the 'temp' column and de-normalise (this is why we added the 'temp' column earlier, to match dimensions)
            df_norm["temp"] = preds_notrend
            inv_y = self.dataframeUtils.denorm_dataframe(df_norm)
            predictions = inv_y["temp"]
        else:
            # classifier handles scaling
            predictions = preds_notrend

        # copy to dataframe column
        dataframe['predict'] = predictions

        return dataframe

    ################################

    # add columns based on predictions. Do not call until after model has been trained
    def add_predictions(self, dataframe: DataFrame, pair) -> DataFrame:

        win_size = max(self.curr_lookahead, 14)

        # dataframe = self.add_model_predictions(dataframe)
        dataframe = self.update_predictions(dataframe)
        dataframe['predict_smooth'] = dataframe['predict'].rolling(window=win_size).apply(self.roll_strong_smooth)

        dataframe['predict_diff'] = 100.0 * (dataframe['predict'] - dataframe[self.target_column]) / \
                                    dataframe[self.target_column]

        # dataframe['predict_diff'] = 100.0 * (dataframe['predict_smooth'] - dataframe['smooth']) / dataframe['smooth']

        return dataframe

    #######################################

    # returns the classifier model.
    def make_classifier(self, pair, seq_len: int, num_features: int):
        predictor = self.get_classifier(pair, seq_len, num_features)

        # set the model name parameters (the predictor cannot know what we want to call the model)
        category, model_name = self.get_model_identifiers(pair)
        predictor.set_model_name(category, model_name)
        return predictor

    # returns the classifier model. Override this function to change the type of classifier
    def get_classifier(self, pair, seq_len: int, num_features: int):
        return NNPredictor_LSTM(pair, seq_len, num_features)

    # return IDs that control model naming. Should be OK for all subclasses
    def get_model_identifiers(self, pair):
        category = self.__class__.__name__
        model_name = category
        if self.model_per_pair:
            model_name = model_name + "_" + pair.split("/")[0]
        return category, model_name

    ################################

    def get_compressor(self, df_norm: DataFrame):
        # just use fixed size PCA (easier for classifiers to deal with)
        ncols = int(64)
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
    def roll_smooth(self, col) -> float:
        # must return scalar, so just calculate prediction and take last value

        smooth = gaussian_filter1d(col, 4)
        # smooth = gaussian_filter1d(col, 2)

        length = len(smooth)
        if length > 0:
            return smooth[length - 1]
        else:
            return col[len(col) - 1]

    def roll_strong_smooth(self, col) -> float:
        # must return scalar, so just calculate prediction and take last value

        smooth = gaussian_filter1d(col, 24)

        length = len(smooth)
        if length > 0:
            return smooth[length - 1]
        else:
            return col[len(col) - 1]

    ################################

    # utility to trace function calls
    # Usage: call self.enable_function_tracing() in your code to enable

    file_dir = os.path.dirname(str(Path(__file__)))

    def tracefunc(self, frame, event, arg, indent=[0]):

        EXCLUSIONS = {'<', '__', 'roll_'}

        file_path = frame.f_code.co_filename
        # only trace if file is in the same directory
        if self.file_dir in file_path:
            if not any(x in frame.f_code.co_name for x in EXCLUSIONS):
                if event == "call":
                    indent[0] += 2
                    # print("-" * indent[0] + "> ", frame.f_code.co_name)
                    if 'self' in frame.f_locals:
                        class_name = frame.f_locals['self'].__class__.__name__
                        func_name = class_name + '.' + frame.f_code.co_name
                    else:
                        func_name = frame.f_code.co_name

                    # func_name = '{name:->{ind}s}()'.format(ind=indent[0] * 2, name=func_name)
                    func_name = "-" * 2 * indent[0] + "> " + func_name
                    file_name = os.path.basename(file_path)
                    # txt = func_name # temp
                    txt = '{: <60} # {}, {}'.format(func_name, file_name, frame.f_lineno)
                    print(txt)
                elif event == "return":
                    # print("<" + "-" * indent[0], "exit function", frame.f_code.co_name)
                    indent[0] -= 2

        return self.tracefunc

    def enable_function_tracing(self):
        print("Function tracing enabled. This will slow down processing")
        sys.setprofile(self.tracefunc)

    ################################

    """
    Buy Signal
    """

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        conditions = []
        dataframe.loc[:, 'enter_tag'] = ''
        curr_pair = metadata['pair']

        # if we are training a new model, just return (this helps avoid runtime errors)
        if self.training_only:
            return dataframe

        # conditions.append(dataframe['volume'] > 0)

        # some trading volume
        conditions.append(dataframe['volume'] > 0)

        # # loose guard
        # conditions.append(dataframe['mfi'] < 50.0)

        # Classifier triggers
        predict_cond = (
            (qtpylib.crossed_above(dataframe['predict_diff'], self.entry_threshold.value))
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

        # if we are training a new model, just return (this helps avoid runtime errors)
        if self.training_only:
            return dataframe

        # conditions.append(dataframe['volume'] > 0)

        # # loose guard
        # conditions.append(dataframe['mfi'] > 50.0)

        # Classifier triggers
        predict_cond = (
            qtpylib.crossed_below(dataframe['predict_diff'], self.exit_threshold.value)
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
