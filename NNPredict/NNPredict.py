# import operator
# import tracemalloc
from datetime import datetime
from enum import Enum
from functools import reduce
from typing import Optional

import numpy as np
import pandas as pd
from pandas import DataFrame
from scipy.ndimage import gaussian_filter1d

import freqtrade.vendor.qtpylib.indicators as qtpylib
from freqtrade.exchange import timeframe_to_minutes
from freqtrade.persistence import Trade
from freqtrade.strategy import (
    CategoricalParameter,
    DecimalParameter,
    IntParameter,
    IStrategy,
    merge_informative_pair,
    stoploss_from_open,
)


import os
import sys
from pathlib import Path
import logging
import warnings
# import random

import sklearn.decomposition as skd

#import keras
# import tensorflow as tf


# import utils.custom_indicators as cta
import utils.profiler as profiler

# from NNPredictor_LSTM import NNPredictor_LSTM
from NNPredictor_LSTM0 import NNPredictor_LSTM0
from sklearn.preprocessing import MinMaxScaler, RobustScaler, StandardScaler
from tqdm import tqdm
from utils.DataframePopulator import DataframePopulator, DatasetType
from utils.DataframeUtils import DataframeUtils, ScalerType
from utils.Environment import Environment

# set paths so that we can find imports in parallel directories
group_dir = str(Path(__file__).parent)
strat_dir = str(Path(__file__).parent.parent)
sys.path.append(strat_dir)
sys.path.append(group_dir)

# logging setup
pd.options.mode.chained_assignment = None  # default='warn'

log = logging.getLogger(__name__)
# log.setLevel(logging.DEBUG)
warnings.simplefilter(action='ignore', category=pd.errors.PerformanceWarning)
warnings.simplefilter(action="ignore", category=FutureWarning)







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
        },
        'subplots': {
            "Diff": {
                'predicted_gain': {'color': 'purple'},
                'gain': {'color': 'orange'},
                '%future_gain': {'color': 'lightblue'},
                'target_profit': {'color': 'lightgreen'},
                'target_loss': {'color': 'lightsalmon'}
            },
        }
    }

    # Do *not* hyperopt for the roi and stoploss spaces (unless you turn off custom stoploss)

    # ROI table:
    minimal_roi = {
       "0": 0.04,
       "100": 0.02
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
    lookahead_hours = 0.5
    n_profit_stddevs = 0.0
    n_loss_stddevs = 0.0
    min_f1_score = 0.49
    max_train_loss = 0.15

    lookahead = int(12 * lookahead_hours)

    curr_pair = ""
    custom_trade_info = {}

    num_pairs = 0
    # pair_model_info = {}  # holds model-related info for each pair
    curr_dataframe: DataFrame = None
    normalise_data = True
    ignore_exit_signals = False  # set to True if you don't want to process sell/exit signals (let custom sell do it)

    # the following affect training of the model. Bigger numbers give better model, but take longer and use more memory
    seq_len = 12  # 'depth' of training sequence
    num_epochs = 128  # max number of iterations for training
    batch_size = 1024  # batch size for training
    predict_batch_size = 128

    classifier_list = {}  # classifier for each pair
    curr_classifier = None
    init_done = {}  # flags whether initialisation has been done for a pair or not

    compressor = None
    compress_data = False  # currently not working
    refit_model = False  # set to True if you want to re-train the model. Usually better to just delete it and restart

    # scaler type used for normalisation
    scaler_type = ScalerType.Robust  
    # scaler_type = ScalerType.MinMax
    # scaler_type = ScalerType.Standard
    # scaler_type = ScalerType.NoScaling

    scale_target = False # True means also scale target/prediction data

    model_per_pair = False  # set to True to create pair-specific models (better but only works for pairs in whitelist)
    training_mode = False  # set to True to just generate models, no backtesting or prediction
    combine_models = False  # combine training across all pairs

    # which column should be used for training and prediction   
    # target_column = 'close'  
    # target_column = 'mid'
    target_column = 'gain'
    # target_column = 'tema'

    dataframeUtils = None
    dataframePopulator = None

    # flags used for initialisation
    first_time = True  # mostly for debug
    first_run = True  # used to identify first time through buy/sell populate funcs

    dbg_verbose = True  # controls debug output
    dbg_test_classifier = True  # test clasifiers after fitting
    dbg_curr_df: DataFrame = None  # for debugging of current dataframe
    dbg_enable_tracing = False  # set to True in subclass to enable function tracing
    dbg_trace_memory = True
    dbg_trace_pair = ""

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
        

    # Buy hyperspace params:
    buy_params = {
        "cexit_min_profit_th": 0.5,
        "cexit_profit_nstd": 0.6,
        "entry_guard_fwr": 0.0,
        "enable_entry_guards": True,  # value loaded from strategy
    }

    # Sell hyperspace params:
    sell_params = {
        "cexit_fwr_overbought": 0.99,
        "cexit_fwr_take_profit": 0.99,
        "cexit_loss_nstd": 1.4,
        "cexit_min_loss_th": -0.5,
        "exit_guard_fwr": 0.0,
        "cexit_enable_large_drop": False,  # value loaded from strategy
        "cexit_large_drop": -1.9,  # value loaded from strategy
        "enable_exit_guards": True,  # value loaded from strategy
        "enable_exit_signal": True,  # value loaded from strategy
    }


    # enable entry/exit guards (safety vs profit)
    enable_entry_guards = CategoricalParameter(
        [True, False], default=True, space="buy", load=True, optimize=False
        )
    entry_guard_fwr = DecimalParameter(
        -0.4, 0.0, default=-0.2, decimals=1, space="buy", load=True, optimize=True
        )

    enable_exit_guards = CategoricalParameter(
        [True, False], default=True, space="sell", load=True, optimize=False
        )
    exit_guard_fwr = DecimalParameter(
        0.0, 0.4, default=0.2, decimals=1, space="sell", load=True, optimize=True
        )

    # use exit signal? If disabled, just rely on the custom exit checks (or stoploss) to get out
    enable_exit_signal = CategoricalParameter(
        [True, False], default=True, space="sell", load=True, optimize=False
        )

    # Custom Exit

    # No. Standard Deviations of profit/loss for target, and lower limit
    cexit_min_profit_th = DecimalParameter(0.5, 2.0, default=0.7, decimals=1, space="buy", load=True, optimize=True)
    cexit_profit_nstd = DecimalParameter(0.0, 4.0, default=0.9, decimals=1, space="buy", load=True, optimize=True)

    cexit_min_loss_th = DecimalParameter(-2.0, -0.5, default=-0.4, decimals=1, space="sell", load=True, optimize=True)
    cexit_loss_nstd = DecimalParameter(0.0, 4.0, default=0.7, decimals=1, space="sell", load=True, optimize=True)

    # Fisher/Williams sell limits - used to bail out when in profit
    cexit_fwr_overbought = DecimalParameter(
        0.90, 0.99, default=0.99, decimals=2, space="sell", load=True, optimize=True
        )
    cexit_fwr_take_profit = DecimalParameter(
        0.90, 0.99, default=0.99, decimals=2, space="sell", load=True, optimize=True
        )

    # sell if we see a large drop, and how large?
    cexit_enable_large_drop = CategoricalParameter(
        [True, False], default=False, space="sell", load=True, optimize=False
        )
    cexit_large_drop = DecimalParameter(
        -3.0, -1.00, default=-1.9, decimals=1, space="sell", load=True, optimize=False
        )


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

        self.lookahead = int(12 * self.lookahead_hours)
        self.dbg_curr_df = dataframe

        if self.dataframeUtils is None:
            self.dataframeUtils = DataframeUtils()

        if self.dataframePopulator is None:

            if self.dbg_trace_memory and (self.dbg_trace_pair == self.curr_pair):
                self.dbg_trace_pair = curr_pair  # only act when we see this pair (too much otherwise)
                profiler.start(10)
                profiler.snapshot()

            self.dataframePopulator = DataframePopulator()

            self.dataframePopulator.runmode = self.dp.runmode.value
            self.dataframePopulator.win_size = max(14, self.lookahead)
            self.dataframePopulator.startup_win = self.startup_candle_count
            self.dataframePopulator.n_loss_stddevs = self.n_loss_stddevs
            self.dataframePopulator.n_profit_stddevs = self.n_profit_stddevs

        if NNPredict.first_time:
            NNPredict.first_time = False

            print("")
            print("***************************************")
            print("** Warning: startup can be very slow **")
            print("***************************************")

            Environment().print_environment()

            print(f"    Lookahead: {self.lookahead} candles ({self.lookahead_hours} hours)")
            print(f"    Re-train existing models: {self.refit_model}")
            print(f"    Training (only) mode: {self.training_mode}")

            # debug tracing
            if self.dbg_enable_tracing:
                self.enable_function_tracing()

        print("")
        print(self.curr_pair)

        # make sure we only retrain in backtest modes
        if self.dp.runmode.value not in ('backtest'):
            self.refit_model = False
            self.training_mode = False

        # (re-)set the scaler
        self.dataframeUtils.set_scaler_type(self.scaler_type)

        if self.dbg_verbose:
            print("    Adding technical indicators...")
        dataframe = self.add_indicators(dataframe)

        # train the model
        if self.dbg_verbose:
            print("    training model...")

        # if we are training, then force re-training of an existing model
        if self.training_mode:
            self.refit_model = False

        dataframe = self.train_model(dataframe, self.curr_pair)

        # if in training mode then skip further processing.
        # Doesn't make sense without the model anyway, and it can sometimes be very slow


        # print(f"    Training (only) mode: {self.training_mode}")

        if self.training_mode:
            print("    Training mode. Skipping backtesting and prediction steps")
            # print("        freqtrade backtest results will show no trades")
            # print("        set training_mode=False to re-enable full backtesting")

        else:
            # if first time through, run backtest
            if self.curr_pair not in self.init_done:
                self.init_done[self.curr_pair] = True
                # print("    running backtest...")
                # dataframe = self.backtest_data(dataframe)

            # add predictions
            if self.dbg_verbose:
                print("    running predictions...")

            dataframe = self.add_predictions(dataframe, self.curr_pair)

        # # Custom Stoploss
        # if self.dbg_verbose:
        #     print("    updating stoploss data...")
        # dataframe = self.add_stoploss_indicators(dataframe, self.curr_pair)

        if self.dbg_trace_memory and (self.dbg_trace_pair == self.curr_pair):
            profiler.snapshot()

        return dataframe

    ###################################
    # add the 'standard' indicators. Override this is you want to use something else
    def add_indicators(self, dataframe: DataFrame) -> DataFrame:

        # populate the standard indicators
        dataframe = self.dataframePopulator.add_indicators(dataframe, dataset_type=DatasetType.CUSTOM1)

        # add indicators needed for callbacks
        dataframe = self.update_gain_targets(dataframe)

        # populate the training indicators
        dataframe = self.add_training_indicators(dataframe)

        dataframe.fillna(0.0, inplace=True)

        return dataframe

    # add in any indicators to be used for training
    def add_training_indicators(self, dataframe: DataFrame) -> DataFrame:

        # placeholders, just need the columns to be there for later
        dataframe['predicted_gain'] = 0.0
        dataframe['temp'] = 0.0

        # no looking ahead in this approach
        # future_df = self.dataframePopulator.add_hidden_indicators(dataframe.copy())
        # future_df = self.dataframePopulator.add_future_data(future_df, self.lookahead)
        return dataframe
    
    ################################

    def update_gain_targets(self, dataframe):
        win_size = max(self.lookahead, 6)
        self.profit_nstd = float(self.cexit_profit_nstd.value)
        self.loss_nstd = float(self.cexit_loss_nstd.value)

        # backward looking gain
        dataframe["gain"] = (
            100.0
            * (dataframe["close"] - dataframe["close"].shift(self.lookahead))
            / dataframe["close"].shift(self.lookahead)
        )
        dataframe["gain"].fillna(0.0, inplace=True)

        dataframe["gain"] = self.smooth(dataframe["gain"], 8)

        # need to save the gain data for later scaling
        self.gain_data = dataframe["gain"].to_numpy().copy()

        # target profit/loss thresholds
        dataframe["profit"] = dataframe["gain"].clip(lower=0.0)
        dataframe["loss"] = dataframe["gain"].clip(upper=0.0)

        dataframe["target_profit"] = (
            dataframe["profit"].rolling(window=win_size).mean()
            + self.profit_nstd * dataframe["profit"].rolling(window=win_size).std()
        )

        dataframe["target_loss"] = dataframe["loss"].rolling(window=win_size).mean() - self.loss_nstd * abs(
            dataframe["loss"].rolling(window=win_size).std()
        )

        dataframe["target_profit"] = dataframe["target_profit"].clip(lower=float(self.cexit_min_profit_th.value))
        dataframe["target_loss"] = dataframe["target_loss"].clip(upper=float(self.cexit_min_loss_th.value))

        dataframe["target_profit"] = np.nan_to_num(dataframe["target_profit"])
        dataframe["target_loss"] = np.nan_to_num(dataframe["target_loss"])

        dataframe["local_mean"] = dataframe["close"].rolling(window=win_size).mean()
        dataframe["local_min"] = dataframe["close"].rolling(window=win_size).min()
        dataframe["local_max"] = dataframe["close"].rolling(window=win_size).max()

        return dataframe

    ################################

    def check_lookahead(self, target, train_target):
        # offset = int(np.shape(target)[0] / 2)
        offset = 0
        a1 = target[offset:offset+3*self.lookahead:]
        a2 = train_target[offset:offset+3*self.lookahead:]

        print("")
        print(f"target mean:{np.mean(target)} min:{np.min(target)} max:{np.max(target)}")
        print(f"target data: {a1}")
        print("")
        print(f"train data: {a2}")
        print("")


    def check_tensor(self, train_target, train_target_tensor):

            print(f"train_target: {train_target}")
            print(f"train_target_tensor: {train_target_tensor}")

    ################################

    # prepare data and train the model
    def train_model(self, dataframe: DataFrame, pair) -> DataFrame:

        nfeatures = np.shape(dataframe)[1]

        # create the classifier if it doesn't already exist
        if self.model_per_pair:
            if self.curr_pair not in self.classifier_list:
                self.classifier_list[self.curr_pair] = self.make_classifier(self.curr_pair, self.seq_len, nfeatures)
            self.curr_classifier = self.classifier_list[self.curr_pair]

        else:
            if not self.curr_classifier:
                self.curr_classifier = self.make_classifier(self.curr_pair, self.seq_len, nfeatures)

        # constrain size to what will be available in run modes
        df_size = dataframe.shape[0]
        # data_size = int(min(975, df_size))
        data_size = df_size  # For backtest/hyperopt/plot, this will be big. Normal size for run modes

        pad = self.lookahead  # have to allow for future results to be in range
        train_ratio = 0.8
        test_ratio = 1.0 - train_ratio
        train_size = int(train_ratio * (data_size - pad)) - 1
        test_size = int(test_ratio * (data_size - pad)) - 1

        # trying different test options. For some reason, results vary quite dramatically based on the approach

        test_option = 3
        if test_option == 0:
            # take the middle part of the full dataframe
            train_start = int((df_size - (train_size + test_size + self.lookahead)) / 2)
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

        # offset result starts with lookahead value
        # train_result_start = train_start + self.lookahead
        # test_result_start = test_start + self.lookahead
        train_result_start = train_start
        test_result_start = test_start

        # normalise and save closing target for later
        if self.target_column not in dataframe:
            print(f"    ERR: target column not present ({self.target_column})")
            return dataframe

        # dataframe = self.add_stoploss_indicators(dataframe)

        # scale the dataframe
        if self.curr_classifier.prescale_data():
            df_norm = self.dataframeUtils.norm_dataframe(dataframe)
        else:
            df_norm = dataframe.copy()


        future_gain = df_norm['gain'].shift(-self.lookahead)
        future_gain = np.nan_to_num(future_gain)
        # future_gain = dataframe['gain'].shift(-self.lookahead)

        # dataframe['%future_gain'] = self.smooth(future_gain, self.lookahead)
        dataframe['%future_gain'] = np.nan_to_num(future_gain)
        target = future_gain
        self.curr_classifier.set_target_column('gain')

        # compress data
        if self.compress_data:
            old_size = df_norm.shape[1]
            df_norm = self.compress_dataframe(df_norm)
            print("    Compressed data {} -> {} (features)".format(old_size, df_norm.shape[1]))

        # a1 = dataframe[self.target_column].to_numpy()
        # print(f"target mean:{np.mean(a1)} min:{np.min(a1)} max:{np.max(a1)}")
        # print(f"target mean:{np.mean(target)} min:{np.min(target)} max:{np.max(target)}")

        # some classifiers take DataFrames as input, others tensors, so check
        if self.curr_classifier.needs_dataframes():
            
            train_df = df_norm.iloc[train_start:(train_start + train_size)]
            test_df = df_norm.iloc[test_start:(test_start + test_size)]

            train_target_df = target[train_result_start:train_result_start + train_size]
            test_target_df = target[test_result_start:test_result_start + test_size]

            # copy to the vars used for training
            train_data = train_df
            test_data = test_df
            # train_results = train_target_df.reshape(-1, 1)
            # test_results = test_target_df.reshape(-1, 1)
            train_results = train_target_df
            test_results = test_target_df

        else:

            # convert dataframe to tensor before extracting train/test data (avoid edge effects)
            df_tensor = self.dataframeUtils.df_to_tensor(df_norm, self.seq_len)
            train_tensor = df_tensor[train_start:train_start + train_size]
            test_tensor = df_tensor[test_start:test_start + test_size]

            # extract target from dataframe and convert to tensors
            train_target = target[train_result_start:train_result_start + train_size]
            test_target = target[test_result_start:test_result_start + test_size]
            # train_target_tensor = self.dataframeUtils.df_to_tensor(train_target.reshape(-1, 1), self.seq_len)
            # test_target_tensor = self.dataframeUtils.df_to_tensor(test_target.reshape(-1, 1), self.seq_len)

            # train_target_tensor = train_target.reshape(-1, 1)
            # test_target_tensor = test_target.reshape(-1, 1)
            train_target_tensor = train_target
            test_target_tensor = test_target

            # self.check_lookahead(train_target, train_target_tensor)
            # self.check_tensor(train_target, train_target_tensor)

            # copy to the vars used for training
            train_data = train_tensor
            test_data = test_tensor
            train_results = train_target_tensor
            test_results = test_target_tensor

            # print("target:", np.shape(target), " train results:", np.shape(train_results))
            # # print("train data:", np.shape(train_df_norm), " train results norm:", np.shape(train_results_norm))

            # print("")
            # print("    train data:", np.shape(train_tensor), " train results:", train_results.shape)
            # print("    test data: ", np.shape(test_tensor), " test results: ", test_results.shape)
            # print("")

        # create/retrieve the model

        # train the model
        print("    fitting model...")
        print("")
        force_train = self.refit_model if (self.dp.runmode.value in ('backtest')) else False
        # print(f"self.refit_model:{self.refit_model} self.dp.runmode.value:{self.dp.runmode.value}")
        self.curr_classifier.train(train_data, test_data,
                                   train_results, test_results,
                                   force_train)

        if self.dbg_test_classifier:
            self.curr_classifier.evaluate(test_data, test_results)
        
        # If a new model, set training mode
        # print(f'    New model: {self.curr_classifier.new_model_created()}')
        if self.curr_classifier.new_model_created():
            if not self.training_mode:
                print('    Switching to Training mode (no backtest)')
            self.training_mode = True

        return dataframe

    ################################

    # helper funcs for dealing with normalisation of 1d numpy arrays

    # array_scaler = StandardScaler()
    array_scaler = None

    # scale the price data. 
    def norm_array(self, arr: np.array) -> np.array:

        if (self.scaler_type != ScalerType.NoScaling) and self.scale_target:

            if self.array_scaler is None:
                self.array_scaler = self.dataframeUtils.make_scaler()

            if len(arr) > 0:
                narr = self.array_scaler.fit_transform(arr.reshape(-1, 1))
                return np.squeeze(narr)
            else:
                print("    ERR: empty array")
                return arr
        else:
            return arr

    def denorm_array(self, arr: np.array) -> np.array:
    
        if (self.scaler_type != ScalerType.NoScaling) and self.scale_target:

            if self.array_scaler is None:
                print("    ERR: scaler not assigned")
                return arr

            if len(arr) > 0:
                darr = self.array_scaler.inverse_transform(arr.reshape(-1, 1))
                return np.squeeze(darr)
            else:
                print("    ERR: empty array")
                return arr
        else:
            return arr
    
    # fast curve smoothing utility
    def smooth(self, y, window):
        box = np.ones(window)/window
        y_smooth = np.convolve(np.nan_to_num(y), box, mode='same')
        y_smooth = np.round(y_smooth, decimals=3) #Hack: constrain to 3 decimal places (should be elsewhere, but convenient here)
        return np.nan_to_num(y_smooth)
    
    ################################

    # backtest the data and update the dataframe
    def backtest_data(self, dataframe: DataFrame) -> DataFrame:

        # get the current classifier and relevant flags
        classifier = self.curr_classifier
        use_dataframes = classifier.needs_dataframes()
        prescale_data = classifier.prescale_data()
        # tgt_scaler = self.dataframeUtils.make_scaler()

        tmp = self.norm_array(dataframe[self.target_column].to_numpy())

        # print(f'    backtest_data() - use_dataframes:{use_dataframes} prescale_data:{prescale_data}')

        # pre-scale if needed
        if prescale_data:
            # tgt_scaler.fit(dataframe[self.target_column].to_numpy().reshape(1, -1))
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
        # print(f"preds_notrend: {np.shape(preds_notrend)}")

        # # results are not scaled
        # predictions = preds_notrend

        predictions = self.denorm_array(preds_notrend)

        '''
        # re-scale, if necessary
        if prescale_data:
            # replace the 'temp' column and de-normalise (this is why we added the 'temp' column earlier, to match dimensions)
            df_norm["temp"] = preds_notrend
            inv_y = self.dataframeUtils.denorm_dataframe(df_norm)
            predictions = inv_y["temp"]

            # predictions = self.denorm_array(preds_notrend)

            # predictions = tgt_scaler.inverse_transform(preds_notrend.reshape(1, -1))[0]
            dataframe['temp'] = preds_notrend # Debug

            # print("")
            # print(f"preds_notrend: {preds_notrend}")
            # print("")
            # print(f"predictions: {predictions}")
            # print("")

        else:
            # classifier handles scaling
            predictions = preds_notrend
        '''

        dataframe['predicted_gain'] = predictions

        return dataframe

    ################################

    # update predictions for the latest part of the dataframe
    def update_predictions(self, dataframe: DataFrame) -> DataFrame:

        # get the current classifier
        classifier = self.curr_classifier
        use_dataframes = classifier.needs_dataframes()
        prescale_data = classifier.prescale_data()

        # # get a scaler for the price data
        # tgt_scaler = self.dataframeUtils.make_scaler()

        # print(f'backtest_data() - use_dataframes:{use_dataframes} prescale_data:{prescale_data}')

        # pre-scale if needed
        if prescale_data:
            df_norm = self.dataframeUtils.norm_dataframe(dataframe)
        else:
            df_norm = dataframe

        # extract the last part of the data
        window = 128
        # end = np.shape(df_norm)[0] - 1
        end = np.shape(df_norm)[0]
        start = end - window
        if use_dataframes:
            # data = df_norm.iloc[start:end]
            data = df_norm.iloc[-window:]
        else:
            tensor = self.dataframeUtils.df_to_tensor(df_norm, self.seq_len)
            data = tensor[start:end]

        # if prescale_data:
        #     # fit price scaler on subset of target column (note, not the normalised dataframe)
        #     tgt_scaler.fit(np.array(dataframe[self.target_column].iloc[start:end]).reshape(1, -1))
        #     # tgt_scaler.fit(np.array(dataframe[self.target_column].to_numpy()).reshape(1, -1))


        tmp = self.norm_array(dataframe['gain'].iloc[start:end]) # this just sets the scale

        # predict
        latest_prediction = dataframe[self.target_column].iloc[-1]
        if classifier.returns_single_prediction():
            predictions = classifier.predict(data)
            latest_prediction = predictions[-1]
            # update just the last entry in the predict column
            dataframe['predicted_gain'].iloc[-1] = latest_prediction
        else:
            preds_notrend = self.get_predictions(data)

            predictions = self.denorm_array(preds_notrend)

            dataframe['predicted_gain'].iloc[-len(predictions):] = predictions.copy()

        # if prescale_data:
        #     df = self.dataframeUtils.denorm_dataframe(df_norm)
        #     dataframe['temp'] = df['temp']
        return dataframe

    ################################

    # get predictions from the model. Directly updates the 'predict' column
    def add_model_predictions(self, dataframe: DataFrame) -> DataFrame:

        # # check that model exists
        # if self.curr_pair not in self.classifier_list:
        #     print("*** No model for pair ", self.curr_pair)
        #     return dataframe

        # if the model produces single valued predictions then we have to roll through the dataframe
        # if not, we can use a more efficient batching approach
        # if self.curr_classifier.returns_single_prediction():
        #     dataframe = self.add_model_rolling_predictions(dataframe)
        # else:
        #     dataframe = self.add_model_batch_predictions(dataframe)

        # dataframe = self.backtest_data(dataframe)
        dataframe = self.add_model_batch_predictions(dataframe)
        # dataframe = self.update_predictions(dataframe)

        return dataframe

    # get predictions. Note that the input can be dataframe or tensor
    def get_predictions(self, data_chunk):

        # if self.curr_pair not in self.classifier_list:
        if not self.curr_classifier:
            print("    ERR: no classifier")
            preds_notrend = np.zeros(np.shape(data_chunk)[0], dtype=float)
            return preds_notrend

        # run the prediction
        preds_notrend = self.curr_classifier.predict(data_chunk)

        predictions = preds_notrend
        # print(f"predictions: {np.shape(predictions)} {predictions}")

        # print(df)
        if self.curr_classifier.returns_single_prediction():
            predictions = [predictions[-1]]

        return predictions

    # run prediction in batches over the entire history
    def add_model_batch_predictions(self, dataframe: DataFrame) -> DataFrame:

        # scale/normalise
        # save mean & std of close column, need this to re-scale predictions later
        cl_mean = dataframe[self.target_column].mean()
        cl_std = dataframe[self.target_column].std()

        # get the current classifier
        classifier = self.curr_classifier
        use_dataframes = classifier.needs_dataframes()
        prescale_data = classifier.prescale_data()

        # # # get a scaler for the price data
        # tgt_scaler = self.dataframeUtils.make_scaler()
        # tgt_scaler.fit(np.array(dataframe[self.target_column]).reshape(1, s-1))
        # # tgt_scaler.fit(dataframe[self.target_column])


        # normalise the target data, just to set up the scaler
        tmp = self.norm_array(dataframe[self.target_column].to_numpy())
        
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
        predictions: np.array = np.zeros(np.shape(dataframe['gain']), dtype=float)
        batch_size = self.predict_batch_size

        # loop until we get to/past the end of the buffer
        start = batch_size
        end = start + batch_size
        nrows = np.shape(dataframe)[0]

        while end < nrows:

            if use_dataframes:
                chunk = df_norm.iloc[start:end]
            else:
                chunk = df_tensor[start:end]

            preds = self.get_predictions(chunk)

             # normalise the source gain data, just to set up the scaler
            tmp = self.norm_array(dataframe['gain'].iloc[start:end].to_numpy())
            preds = self.denorm_array(preds)

            # copy the predictions for this window into the main predictions array
            predictions[start:end] = preds.copy()

            # move the window to the next segment
            end = end + batch_size
            start = start + batch_size

        # make sure the last section gets processed (the loop above may not exactly fit the data)
        # Note that we cannot use the last section for training because we don't have forward looking data

        if use_dataframes:
            chunk = df_norm.iloc[-batch_size:]
        else:
            chunk = df_tensor[-batch_size:]

        preds = self.get_predictions(chunk)
        tmp = self.norm_array(dataframe['gain'].iloc[-batch_size:].to_numpy())
        preds = self.denorm_array(preds)
        predictions[-batch_size:] = preds.copy()

        dataframe["predicted_gain"].iloc[-len(predictions):] = predictions.copy()

        return dataframe

    # run prediction in rolling fashion (one result at a time -> slow) over the entire history
    def add_model_rolling_predictions(self, dataframe: DataFrame) -> DataFrame:

        print("    Adding rolling predictions. Might take a while...")

        # probably don't need to handle tensor cases since those classifiers typically do not return single values

        # get the current clasifier
        classifier = self.curr_classifier
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
            # df_norm["temp"] = preds_notrend
            # inv_y = self.dataframeUtils.denorm_dataframe(df_norm)
            # predictions = inv_y["temp"]
            predictions = self.denorm_array(preds_notrend)
        else:
            # classifier handles scaling
            predictions = preds_notrend

        # copy to dataframe column
        dataframe['predicted_gain'] = predictions

        return dataframe

    ################################

    # add columns based on predictions. Do not call until after model has been trained
    def add_predictions(self, dataframe: DataFrame, pair) -> DataFrame:

        win_size = max(self.lookahead, 14)

        dataframe = self.add_model_predictions(dataframe)
        dataframe = self.update_predictions(dataframe)
        # dataframe['predict_smooth'] = dataframe['predicted_gain'].rolling(window=win_size).apply(self.roll_strong_smooth)
        dataframe['predicted_gain'] = dataframe['predicted_gain'].clip(lower=-3.0, upper=3.0)

        return dataframe

    #######################################

    # returns the classifier model.
    def make_classifier(self, pair, seq_len: int, num_features: int):
        predictor = self.get_classifier(pair, seq_len, num_features)

        # set the model name parameters (the predictor cannot know what we want to call the model)
        # dir, category, model_name = self.get_model_identifiers(pair)
        predictor.set_model_path(self.get_model_path(pair))
        predictor.set_combine_models(self.combine_models)
        # predictor.set_model_name(category, model_name)

        if predictor.new_model_created():
            self.training_mode = True

        return predictor

    # returns the classifier model. Override this function to change the type of classifier
    def get_classifier(self, pair, seq_len: int, num_features: int):
        # use the simplest predictor, try and remove model issues for testing the gneral framework
        return NNPredictor_LSTM0(pair, seq_len, num_features)

    # return IDs that control model naming. Should be OK for all subclasses
    def get_model_path(self, pair):
        category = self.__class__.__name__
        root_dir = group_dir + "/models/" + category
        model_name = category
        if self.model_per_pair:
            model_name = model_name + "_" + pair.split("/")[0]
        path = root_dir + "/" + model_name + ".keras"
        return path

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
    entry Signal
    """

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        conditions = []
        dataframe.loc[:, "enter_tag"] = ""

        if self.training_mode:
            dataframe["enter_long"] = 0
            return dataframe

        dataframe = self.update_gain_targets(dataframe)

        # some trading volume (otherwise expect spread problems)
        conditions.append(dataframe["volume"] > 1.0)

        if self.enable_entry_guards.value:
            # Fisher/Williams in oversold region
            conditions.append(dataframe["fisher_wr"] < self.entry_guard_fwr.value)


        fwr_cond = dataframe["fisher_wr"] < -0.98

        # model triggers
        model_cond = (
            # model predicts a rise above the entry threshold
            # qtpylib.crossed_above(dataframe["predicted_gain"], dataframe["target_profit"])
            (dataframe["predicted_gain"] > dataframe["target_profit"])
            &
            # in lower portion of previous window
            (dataframe["close"] < dataframe["local_mean"])
        )

        # conditions.append(fwr_cond)
        conditions.append(model_cond)

        # set entry tags
        dataframe.loc[fwr_cond, "enter_tag"] += "fwr_entry "
        dataframe.loc[model_cond, "enter_tag"] += "model_entry "

        if conditions:
            dataframe.loc[reduce(lambda x, y: x & y, conditions), "enter_long"] = 1

        return dataframe

    ###################################

    """
    exit Signal
    """

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        conditions = []
        dataframe.loc[:, "exit_tag"] = ""

        if self.training_mode or (not self.enable_exit_signal.value):
            dataframe["exit_long"] = 0
            return dataframe

        # if self.enable_entry_guards.value:

        # some trading volume (otherwise expect spread problems)
        conditions.append(dataframe["volume"] > 0)

        if self.enable_exit_guards.value:
            # Fisher/Williams in overbought region
            conditions.append(dataframe["fisher_wr"] > self.exit_guard_fwr.value)


        # model triggers
        model_cond = (
            # prediction crossed target
            # qtpylib.crossed_below(dataframe["predicted_gain"], dataframe["target_loss"])

            # use this version if volume checks are enabled, because we might miss the crossing otherwise
            (dataframe["predicted_gain"] < dataframe["target_loss"])
            &
            # in upper portion of previous window
            (dataframe["close"] > dataframe["local_mean"])
        )

        conditions.append(model_cond)

        # set exit tags
        dataframe.loc[model_cond, "exit_tag"] += "model_exit "

        if conditions:
            dataframe.loc[reduce(lambda x, y: x & y, conditions), "exit_long"] = 1

        return dataframe

    ###################################

    def confirm_trade_entry(
        self,
        pair: str,
        order_type: str,
        amount: float,
        rate: float,
        time_in_force: str,
        current_time: datetime,
        entry_tag: Optional[str],
        side: str,
        **kwargs,
    ) -> bool:
        # this only makes sense in 'live' modes
        if self.dp.runmode.value in ("backtest", "plot", "hyperopt"):
            return True

        # in 'real' systems, there is often a delay between the signal and the trade
        # double-check that predicted gain is still above threshold

        if pair in self.custom_trade_info:
            curr_pred = self.custom_trade_info[pair]["curr_prediction"]

            # check latest prediction against latest target

            curr_target = self.custom_trade_info[pair]["curr_target"]
            if curr_pred < curr_target:
                if self.dp.runmode.value not in ("backtest", "plot", "hyperopt"):
                    print(
                        f"    *** {pair} Trade cancelled. Prediction ({curr_pred:.2f}%) below target ({curr_target:.2f}%) "
                    )
                return False

            """
            # check that prediction is still a profit, rather than a loss
            if curr_pred < 0.0:
                # if self.dp.runmode.value not in ('backtest', 'plot', 'hyperopt'):
                if self.dp.runmode.value not in ('hyperopt'):
                    print(
                        f'    *** {pair} Trade cancelled. Prediction ({curr_pred:.2f}%) is now a loss '
                        )
                return False

            """

        # just debug
        if self.dp.runmode.value not in ("backtest", "plot", "hyperopt"):
            print(f"    Trade Entry: {pair}, rate: {rate:.4f} Predicted gain: {curr_pred:.2f}% Target: {curr_target:.2f}%")

        return True

    def confirm_trade_exit(
        self,
        pair: str,
        trade: Trade,
        order_type: str,
        amount: float,
        rate: float,
        time_in_force: str,
        exit_reason: str,
        current_time: datetime,
        **kwargs,
    ) -> bool:
        
        if self.dp.runmode.value not in ("backtest", "plot", "hyperopt"):
            print(f"    Trade Exit: {pair}, rate: {rate:.4f)}")

        return True

    ###################################

    """
    Custom Stoploss
    """

    # simplified version of custom trailing stoploss
    def custom_stoploss(
        self,
        pair: str,
        trade: Trade,
        current_time: datetime,
        current_rate: float,
        current_profit: float,
        after_fill: bool,
        **kwargs,
    ) -> float:
        # this is just here so that we can use custom_exit
        """
        # if enable, use custom trailing ratio, else use default system
        if self.cstop_enable.value:
            # if current profit is above start value, then set stoploss at fraction of current profit
            if current_profit > self.cstop_start.value:
                return current_profit * self.cstop_ratio.value

        """

        # return min(-0.001, max(stoploss_from_open(0.05, current_profit), -0.99))
        return self.stoploss

    ###################################

    """
    Custom Exit
    (Note that this runs even if use_custom_stoploss is False)
    """

    # simplified version of custom exit

    def custom_exit(
        self, pair: str, trade: Trade, current_time: "datetime", current_rate: float, current_profit: float, **kwargs
    ):
        dataframe, _ = self.dp.get_analyzed_dataframe(pair=pair, timeframe=self.timeframe)
        last_candle = dataframe.iloc[-1].squeeze()

        if not self.use_custom_stoploss:
            return None

        # check volume?!
        if last_candle['volume'] <= 1.0:
            return None

        # strong sell signal, in profit
        if (current_profit > 0.0) and (last_candle["fisher_wr"] >= self.cexit_fwr_overbought.value):
            return "fwr_overbought"

        # Above 0.5%, sell if Fisher/Williams in sell range
        if current_profit > 0.005:
            if last_candle["fisher_wr"] >= self.cexit_fwr_take_profit.value:
                return "take_profit"

        """
        # check profit against ROI target. This sort of emulates the freqtrade roi approach, but is much simpler
        if self.cexit_use_profit_threshold.value:
            if (current_profit >= self.cexit_profit_threshold.value):
                return 'cexit_profit_threshold'

        # check loss against threshold. This sort of emulates the freqtrade stoploss approach, but is much simpler
        if self.cexit_use_loss_threshold.value:
            if (current_profit <= self.cexit_loss_threshold.value):
                return 'cexit_loss_threshold'

        """

        # Sell any positions if open for >= 1 day with any level of profit
        if ((current_time - trade.open_date_utc).days >= 1) & (current_profit > 0):
            return "unclog_1"

        # Sell any positions at a loss if they are held for more than 7 days.
        if (current_time - trade.open_date_utc).days >= 7:
            return "unclog_7"

        # big drop predicted. Should also trigger an exit signal, but this might be quicker (and will likely be 'market' sell)
        if (current_profit > 0) and (last_candle["predicted_gain"] <= last_candle["target_loss"]):
            return "predict_drop"

        # large drop preduicted, just bail no matter profit
        if self.cexit_enable_large_drop.value:
            if last_candle["predicted_gain"] < self.cexit_large_drop.value:
                return "large_drop"

        # if in profit and exit signal is set, sell (even if exit signals are disabled)
        if (current_profit > 0) and (last_candle["exit_long"] > 0):
            return "exit_signal"

        return None