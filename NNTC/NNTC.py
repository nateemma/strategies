#pragma pylint: disable=W0105, C0103, C0301
# pylint: disable=reportMissingImports

from datetime import datetime
from enum import Enum
from functools import reduce

import numpy as np
import pandas as pd
from pandas import DataFrame

# import freqtrade.vendor.qtpylib.indicators as qtpylib
from freqtrade.exchange import timeframe_to_minutes
from freqtrade.persistence import Trade
from freqtrade.strategy import (IStrategy, IntParameter, DecimalParameter, CategoricalParameter)

pd.options.mode.chained_assignment = None  # default='warn'

# Strategy specific imports, files must reside in same folder as strategy
import sys
from pathlib import Path

group_dir = str(Path(__file__).parent)
strat_dir = str(Path(__file__).parent.parent)
sys.path.append(strat_dir)
sys.path.append(group_dir)

# print(f"sys.path: {sys.path}")

import logging

log = logging.getLogger(__name__)
# log.setLevel(logging.DEBUG)
# warnings.simplefilter(action='ignore', category=pd.errors.PerformanceWarning)
# attempts to disable annoying keras load/save logging:
log.addFilter(logging.Filter(name='loading'))
log.addFilter(logging.Filter(name='saving'))
logging.getLogger('tensorflow').setLevel(logging.WARNING)
logging.getLogger('keras').setLevel(logging.WARNING)
logging.getLogger('pickle').setLevel(logging.CRITICAL)
logging.basicConfig(level=logging.WARNING)
logging.disable(logging.WARNING)

# 
import  utils.custom_indicators as cta

from sklearn.metrics import classification_report
import sklearn.decomposition as skd

from sklearn.metrics import make_scorer
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score

import random

import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_DETERMINISTIC_OPS'] = '1'

import tensorflow as tf

seed = 42
os.environ['PYTHONHASHSEED'] = str(seed)
random.seed(seed)
tf.random.set_seed(seed)
np.random.seed(seed)

tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.WARN)

tf_logger = logging.getLogger('tensorflow')
tf_logger.setLevel(logging.WARN)

from utils.DataframeUtils import DataframeUtils, ScalerType 
from utils.DataframePopulator import DataframePopulator, DatasetType
import utils.TrainingSignals as TrainingSignals

import NNTClassifier

from utils.Environment import Environment
import utils.profiler as profiler
import pickle

"""
####################################################################################
NNTC - Neural Net Trinary Classifier
    Combines Dimensionality Reduction using Principal Component Analysis (PCA) and various
    Neural Networks set up as trinary classifiers.
      
    This works by creating a PCA model of the available technical indicators. This produces a 
    mapping of the indicators and how they affect the outcome (buy/sell/hold). We choose only the
    mappings that have a significant effect and ignore the others. This significantly reduces the size
    of the problem.
    We then train a classifier model to predict buy or sell signals based on the known outcome in the
    informative data, and use it to predict buy/sell signals based on the real-time dataframe.
    Several different Neural Network types are available, and they can either all be tested, or a pre-configured
    classifier can be used.
    
    Notes: 
    - Neural Nets need lots of data to train, and there are typically not enough buy/sell events
    in the 'normal' buffer (975 samples) to do that training sufficiently well. So, we only train in backtest mode,
    then save the resulting model. Other modes (hyperopt, plot etc.) will just load the saved model and use it.
    This means that you should run backtest with a (very) long time period (I suggest a full year).
    
    - To help avoid over-fitting, I train a single classifier across all pairs (obe for buy, another for sell). This
    should provide a more general model
    
    - models are saved in the models/ directory, relative to the current path. You will likely need to copy the models
    to the location whre you run your strategies
    
    - This is intended as a base class. Actual strategis will inherit from this class and then modify the
    buy and sell criteria
    
      
    Note that this is very slow to start up. This is mostly because we have to build the data on a rolling
    basis to avoid lookahead bias.
      
    In addition to the normal freqtrade packages, these strategies also require the installation of:
        random
        prettytable
        finta
        sklearn

####################################################################################
"""


class NNTC(IStrategy):
    plot_config = {
        'main_plot': {
            'close': {'color': 'cornflowerblue'},
        },
        'subplots': {
            "Diff": {
                '%train_buy': {'color': 'mediumaquamarine'},
                'predict_buy': {'color': 'cornflowerblue'},
                '%train_sell': {'color': 'lightsalmon'},
                'predict_sell': {'color': 'brown'},
            },
        }
    }

    # Do *not* hyperopt for the roi and stoploss spaces (unless you turn off custom stoploss)

    # ROI table:
    minimal_roi = {
        "0": 0.006
    }

    # Stoploss:
    stoploss = -0.1

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
    process_only_new_candles = False

    # ------------------------------
    # Strategy-specific global vars

    inf_mins = timeframe_to_minutes(inf_timeframe)
    data_mins = timeframe_to_minutes(timeframe)
    inf_ratio = int(inf_mins / data_mins)

    # These parameters control much of the behaviour because they control the generation of the training data
    # Unfortunately, these cannot be hyperopt params because they are used in populate_indicators, which is only run
    # once during hyperopt
    lookahead_hours = 1.0
    n_profit_stddevs = 0.0
    n_loss_stddevs = 0.0
    min_f1_score = 0.3

    compressor = None
    compress_data = True

    trinary_classifier = None

    curr_lookahead = int(12 * lookahead_hours)

    curr_pair = ""
    custom_trade_info = {}

    # the following affect training of the model. Bigger numbers give better results, but take longer and use more memory
    seq_len = 8  # 'depth' of training sequence
    num_epochs = 512  # number of iterations for training
    batch_size = 1024  # batch size for training

    COMPRESSED_SIZE = 64

    refit_model = False  # only set to True when training. If False, then existing model is used, if present
    use_full_dataset = True  # use the entire dataset for training (in backtest)
    model_per_pair = False  # single model for all pairs
    combine_models = False  # combine training across all pairs
    ignore_exit_signals = False  # set to True if you don't want to process sell/exit signals (let custom sell do it)

    scaler_type = ScalerType.Robust  # scaler type used for normalisation

    dataframeUtils = None
    dataframePopulator = None

    dwt_window = startup_candle_count

    num_pairs = 0
    # pair_model_info = {}  # holds model-related info for each pair
    # classifier_stats = {}  # holds statistics for each type of classifier (useful to rank classifiers

    # debug flags
    first_time = True  # mostly for debug
    first_run = True  # used to identify first time through buy/sell populate funcs

    dbg_scan_classifiers = False  # if True, scan all viable classifiers and choose the best. Very slow!
    dbg_test_classifier = True  # test clasifiers after fitting
    dbg_verbose = True  # controls debug output
    dbg_curr_df: DataFrame = None  # for debugging of current dataframe
    dbg_trace_memory = False  # if true, trace memory usage
    dbg_trace_pair = ""  # pair used for synching memory snapshots

    # variables to track state
    class State(Enum):
        INIT = 1
        POPULATE = 2
        STOPLOSS = 3
        RUNNING = 4

    classifier_type = NNTClassifier.ClassifierType.LSTM  # default, override in subclass

    dataset_type = DatasetType.DEFAULT
    signal_type = TrainingSignals.SignalType.Profit  # should override this
    training_signals = None

    ###################################

    # Strategy Specific Variable Storage

    ## Hyperopt Variables

   # trailing stoploss
    tstop_start = DecimalParameter(0.0, 0.06, default=0.015, decimals=3, space='sell', load=True, optimize=True)
    tstop_ratio = DecimalParameter(0.7, 0.99, default=0.9, decimals=3, space='sell', load=True, optimize=True)

    # profit threshold exit
    profit_threshold = DecimalParameter(0.005, 0.065, default=0.025, decimals=3, space='sell', load=True, optimize=True)
    use_profit_threshold = CategoricalParameter([True, False], default=True, space='sell', load=True, optimize=False)

    # loss threshold exit
    loss_threshold = DecimalParameter(-0.065, -0.005, default=-0.046, decimals=3, space='sell', load=True, optimize=True)
    use_loss_threshold = CategoricalParameter([True, False], default=True, space='sell', load=True, optimize=False)

    # use exit signal? 
    enable_exit_signal = CategoricalParameter([True, False], default=True, space='sell', load=True, optimize=False)

    ################################

    # subclasses should override the following 2 functions - this is here as an example
    # NOTE: can also just set signal_type to something else valid, and that will also work
    #       see self.training_signals.SignalType for a list of algorithms

    # Note: try to combine current/historical data (from populate_indicators) with future data
    #       If you only use future data, the ML training is just guessing
    #       Also, try to identify buy/sell ranges, rather than transitions - it gives the algorithms more chances
    #       to find a correlation. The framework will select the first one anyway.
    #       In other words, avoid using qtpylib.crossed_above() and qtpylib.crossed_below()
    #       Proably OK not to check volume, because we are just looking for patterns

    def get_train_buy_signals(self, future_df: DataFrame):

        signals = None

        if self.training_signals.check_indicators(future_df):
            signals = self.training_signals.get_entry_training_signals(future_df)
        else:
            print("    ERROR: Missing indicators in dataframe")

        if signals is None:
            signals = pd.Series(np.zeros(np.shape(future_df)[0], dtype=float))

        return signals

    def get_train_sell_signals(self, future_df: DataFrame):

        signals = None

        if self.training_signals.check_indicators(future_df):
            signals = self.training_signals.get_exit_training_signals(future_df)

        if signals is None:
            signals = pd.Series(np.zeros(np.shape(future_df)[0], dtype=float))

        return signals

    # override the following to add strategy-specific criteria to the (main) buy/sell conditions

    def get_strategy_entry_guard_conditions(self, dataframe: DataFrame):
        return self.training_signals.get_entry_guard_conditions(dataframe)

    def get_strategy_exit_guard_conditions(self, dataframe: DataFrame):
        return self.training_signals.get_exit_guard_conditions(dataframe)

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

    def print_strategy_info(self):

        print("")
        print("Strategy Parameters/Flags")
        print("")
        print(f"    Dataset Type:           {self.dataset_type} ({self.dataset_type.value})")
        print(f"    Signal Type:            {self.signal_type} ({self.training_signals.get_signal_name()})")
        print(f"    Classifier Type:        {self.classifier_type}")
        print(f"    Lookahead:              {self.lookahead_hours} hours ({self.curr_lookahead} candles)")
        print(f"    n_profit_stddevs:       {self.n_profit_stddevs}")
        print(f"    n_loss_stddevs:         {self.n_loss_stddevs}")
        print(f"    compress_data:          {self.compress_data}")
        print(f"    refit_model:            {self.refit_model}")
        print(f"    model_per_pair:         {self.model_per_pair}")
        print(f"    combine_models:         {self.combine_models}")
        print(f"    ignore_exit_signals:    {self.ignore_exit_signals}")
        print("")

    """
    Indicator Definitions
    """

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:

        # Base pair inf timeframe indicators
        curr_pair = metadata['pair']
        self.curr_pair = curr_pair

        self.set_state(curr_pair, self.State.POPULATE)
        self.curr_lookahead = int(12 * self.lookahead_hours)
        self.dbg_curr_df = dataframe

        if self.training_signals is None:
            #TODO: put params in training_signal
            self.training_signals = TrainingSignals.create_training_signals(self.signal_type, self.curr_lookahead)
            self.curr_lookahead = self.training_signals.get_lookahead()
            self.n_loss_stddevs = self.training_signals.get_n_loss_stddevs()
            self.n_profit_stddevs = self.training_signals.get_n_profit_stddevs()

        # create and initialise instances of objects shared across pairs
        if self.dataframeUtils is None:
            self.dataframeUtils = DataframeUtils()

        if self.dataframePopulator is None:

            if self.dbg_trace_memory and (self.dbg_trace_pair == self.curr_pair):
                self.dbg_trace_pair = curr_pair  # only act when we see this pair (too much otherwise)
                profiler.start(10)
                profiler.snapshot()

            self.dataframePopulator = DataframePopulator()

            self.dataframePopulator.runmode = self.dp.runmode.value
            self.dataframePopulator.win_size = min(14, self.curr_lookahead)
            self.dataframePopulator.startup_win = self.startup_candle_count
            self.dataframePopulator.n_loss_stddevs = self.n_loss_stddevs
            self.dataframePopulator.n_profit_stddevs = self.n_profit_stddevs

        # first time through? Print some debug info
        if self.first_time:
            self.first_time = False
            print("")
            print("----------------------")
            print(self.__class__.__name__)
            print("----------------------")
            print("")
            # print("***************************************")
            # print("** Warning: startup can be very slow **")
            # print("***************************************")

            Environment().print_environment()

            self.print_strategy_info()

        print("")
        print(curr_pair)

        # make sure we only retrain in backtest modes
        if self.dp.runmode.value not in ('backtest'):
            self.refit_model = False

        # (re-)set the scaler
        self.dataframeUtils.set_scaler_type(self.scaler_type)

        # populate the normal dataframe
        if self.dbg_verbose:
            print("    adding indicators...")
        dataframe = self.dataframePopulator.add_indicators(dataframe, dataset_type=self.dataset_type)

        # if number of features less than compressed size, just disable compression
        if dataframe.shape[-1] <= self.COMPRESSED_SIZE:
            self.compress_data = False
            print(f"    Disabled compression ({dataframe.shape[-1]} <= {self.COMPRESSED_SIZE})")

        # get the buy/sell training signals
        buys, sells = self.create_training_data(dataframe)

        # train the models on the populated data and signals
        if self.dbg_verbose:
            print("    training models...")
        self.train_models(curr_pair, dataframe, buys, sells)

        # add predictions
        if self.dbg_verbose:
            print("    running predictions...")

        # get predictions (Note: do not modify dataframe between calls)
        pred_buys, pred_sells = self.predict_buysell(dataframe, curr_pair)
        dataframe['predict_buy'] = pred_buys
        dataframe['predict_sell'] = pred_sells

        # Custom Stoploss
        if self.use_custom_stoploss:
            if self.dbg_verbose:
                print("    updating stoploss data...")
            self.add_stoploss_indicators(dataframe, curr_pair)

        if self.dbg_trace_memory and (self.dbg_trace_pair == self.curr_pair):
            profiler.snapshot()

        return dataframe

    ################################
    # run data augmentation techniques
    def augment_training_signals(self, buys, sells):

        # Trick 1: artificially extend positive signals one entry earlier

        bidx = np.where(buys > 0)[0]  # index of buy entrie
        # set the entry before each buy signal, unless it's the first item
        if len(bidx) > 0:  # there are some buys
            start = 1 if (bidx[0] == 0) else 0
            if len(bidx) >= start:
                buys[bidx[start:] - 1] = 1.0

        sidx = np.where(sells > 0)[0]  # index of sell entries
        # set the entry before each sell signal, unless it's the first item
        if len(sidx) > 0:  # there are some sells
            start = 1 if (sidx[0] == 0) else 0
            if len(sidx) >= start:
                sells[sidx[start:] - 1] = 1.0

        # Trick 2: sells override buys
        buys[np.where(sells > 0)[0]] = 0.0

        # Trick 3: if a buy is followed by a sell, override the buy
        buys[np.where(sells[:-1] == 1)[0] + 1] = 0.0

        return buys, sells

    # creates the buy/sell labels absed on looking ahead into the supplied dataframe
    def create_training_data(self, dataframe: DataFrame):

        # future_df = self.add_future_data(dataframe.copy())
        future_df = dataframe.copy()
        future_df = self.dataframePopulator.add_hidden_indicators(future_df)
        future_df = self.dataframePopulator.add_future_data(future_df, self.curr_lookahead)

        future_df['train_buy'] = 0.0
        future_df['train_sell'] = 0.0

        # use sequence trends as criteria
        future_df['train_buy'] = self.get_train_buy_signals(future_df)
        future_df['train_sell'] = self.get_train_sell_signals(future_df)
        # future_df['train_buy'] = np.where(future_df['train_sell']>0.0, 0.0, temp_buys) # sell takes precedence

        buys = future_df['train_buy'].copy()
        if buys.sum() < 3:
            print("OOPS! <3 ({:.0f}) buy signals generated. Check training criteria".format(buys.sum()))

        sells = future_df['train_sell'].copy()
        # sells = sells * 2.0 # sells are represented as 2.0 in this type of strategy
        if sells.sum() < 3:
            print("OOPS! <3 ({:.0f}) sell signals generated. Check training criteria".format(sells.sum()))

        # run data augmentation techniques
        buys, sells = self.augment_training_signals(buys, sells)

        # copy back to dataframe (because they likely changed)
        future_df['train_buy'] = np.where(buys > 0, 1.0, 0.0)
        future_df['train_sell'] = np.where(sells > 0, 1.0, 0.0)

        self.save_debug_data(future_df)

        return buys, sells

    def save_debug_data(self, future_df: DataFrame):

        # Debug support: add commonly used indicators so that they can be viewed
        # the list below is available for any subclass.

        # Subclasses themselves can add more by overriding the func save_debug_indicators()

        dbg_list = [
            'full_dwt', 'train_buy', 'train_sell',
            'future_gain', 'future_min', 'future_max',
            'future_profit_min', 'future_profit_max',
            'future_loss_min', 'future_loss_max', 'loss_threshold',
        ]

        if len(dbg_list) > 0:
            for indicator in dbg_list:
                self.add_debug_indicator(future_df, indicator)

        # save the indicators for this training signal type, or can override in subclass
        self.save_debug_indicators(future_df)

        return

    # save debug indicators identified by the training signal. Can also be overidden in the subclass
    def save_debug_indicators(self, future_df: DataFrame):
        dbg_list = self.training_signals.get_debug_indicators()

        if len(dbg_list) > 0:
            # print(f"    Adding debug indicators: {dbg_list}")
            for indicator in dbg_list:
                self.add_debug_indicator(future_df, indicator)

        return

    # adds an indicator to the main frame for debug (e.g. plotting). Column will be prefixed with '%', which will
    # cause it to be removed before normalisation and fitting of models
    def add_debug_indicator(self, future_df: DataFrame, indicator):
        dbg_indicator = '%' + indicator
        if not (dbg_indicator in self.dbg_curr_df):
            self.dbg_curr_df[dbg_indicator] = future_df[indicator]

    ###################

    # add indicators used by stoploss/custom sell logic
    def add_stoploss_indicators(self, dataframe, pair) -> DataFrame:
        if not pair in self.custom_trade_info:
            self.custom_trade_info[pair] = {}
            if not 'had_trend' in self.custom_trade_info[pair]:
                self.custom_trade_info[pair]['had_trend'] = False

        # Indicators used for ROI and Custom Stoploss
        dataframe = self.dataframePopulator.add_stoploss_indicators(dataframe)
        return dataframe

    # compress the supplied dataframe
    def compress_dataframe(self, dataframe: DataFrame) -> DataFrame:
        if not self.compressor:
            self.compressor = self.get_compressor(dataframe)
        # self.compressor = self.get_compressor(dataframe)
        return pd.DataFrame(self.compressor.transform(dataframe))

    # train the classification model

    def train_models(self, curr_pair, dataframe: DataFrame, buys, sells):

        # check input - need at least 2 samples or classifiers will not train
        if buys.sum() < 2:
            print("*** ERR: insufficient buys in expected results. Check training data")
            # print(buys)
            return

        # if sells.sum() < 2:
        #     print("*** ERR: insufficient sells in expected results. Check training data")
        #     return

        rand_st = 27  # use fixed number for reproducibility

        frame_size = dataframe.shape[0]

        remove_outliers = False
        if remove_outliers:
            # norm dataframe before splitting, otherwise variances are skewed
            full_df_norm = self.dataframeUtils.norm_dataframe(dataframe)
            full_df_norm, buys, sells = self.dataframeUtils.remove_outliers(full_df_norm, buys, sells)
        else:
            # full_df_norm = self.dataframeUtils.norm_dataframe(dataframe).clip(lower=-3.0, upper=3.0)  # supress outliers
            full_df_norm = self.dataframeUtils.norm_dataframe(dataframe)

        # compress data
        if self.compress_data:
            old_size = full_df_norm.shape[1]
            full_df_norm = self.compress_dataframe(full_df_norm)
            print("    Compressed data {} -> {} (features)".format(old_size, full_df_norm.shape[1]))

        # constrain size to what will be available in run modes
        if self.use_full_dataset:
            data_size = int(0.9 * frame_size)
        else:
            data_size = int(min(975, frame_size))

        # create classifiers, if necessary
        num_features = full_df_norm.shape[1]
        if (self.trinary_classifier is None) or self.model_per_pair:
            self.trinary_classifier, name = NNTClassifier.create_classifier(self.classifier_type,
                                                                            self.curr_pair,
                                                                            num_features,
                                                                            self.seq_len)

            # set additional model parameters
            # category, model_name = self.get_model_identifiers(self.curr_pair, name)
            # self.trinary_classifier.set_model_name(category, model_name)
            self.trinary_classifier.set_model_path(self.get_model_path(self.curr_pair, name))
            self.trinary_classifier.set_combine_models(self.combine_models)

        # combine holds/buys/sells into a single array
        blabels = buys.to_numpy()
        slabels = sells.to_numpy()
        blabels[np.where(slabels > 0)] = 0.0  # sells override buys

        # holds = np.ones(frame_size, dtype=float)  # init holds to 1s
        # holds[np.where(blabels > 0)] = 0.0  # if buy or sell is set, clear holds entry
        # holds[np.where(slabels > 0)] = 0.0

        holds = np.zeros(frame_size, dtype=float)  # init holds to 0s
        holds[np.where((blabels == 0) & (slabels == 0))] = 1.0

        num_samples = frame_size
        num_holds = holds.sum()
        num_buys = buys.sum()
        num_sells = sells.sum()
        hpct = 100.0 * num_holds / num_samples
        bpct = 100.0 * num_buys / num_samples
        spct = 100.0 * num_sells / num_samples
        print(f'    holds:{num_holds:.0f} ({hpct:.2f}%) ' + \
              f'buys:{num_buys:.0f} ({bpct:.2f}%) sells:{num_sells:.0f} ({spct:.2f}%)')

        # quick check
        if int(num_holds + num_buys + num_sells) != num_samples:
            print("    ** ERR: labels are inconsistent **")

        # If <1% buy/signals, issue warning. Neural net will likely not converge
        if bpct < 1.0:
            print(f'    ** WARNING: low number of buy signals ({bpct:.2f}%)')

        if spct < 1.0:
            print(f'    ** WARNING: low number of sell signals ({spct:.2f}%)')

        labels = np.array([holds, blabels, slabels]).T

        # convert to tensors
        full_tensor = self.dataframeUtils.df_to_tensor(full_df_norm, self.seq_len)
        # lbl_tensor = self.dataframeUtils.df_to_tensor(labels, self.seq_len)

        # if output is not in tensor format, don't convert
        lbl_tensor = labels

        # get training & test dataset

        pad = self.curr_lookahead  # have to allow for future results to be in range
        train_ratio = 0.8
        test_ratio = 1.0 - train_ratio
        train_size = int(train_ratio * (data_size - pad)) - 1
        test_size = int(test_ratio * (data_size - pad)) - 1

        # train_start = frame_size - train_size
        # test_start = frame_size - data_size
        train_start = 0
        test_start = train_size

        tsr_train = full_tensor[train_start:train_start + train_size]
        tsr_test = full_tensor[test_start:test_start + test_size]
        tsr_lbl_train = lbl_tensor[train_start:train_start + train_size]
        tsr_lbl_test = lbl_tensor[test_start:test_start + test_size]

        # num_buys = int(tsr_lbl_train[:, 0, 1].sum())
        # num_sells = int(tsr_lbl_train[:, 0, 2].sum())
        num_buys = int(tsr_lbl_train[:, 1].sum())
        num_sells = int(tsr_lbl_train[:, 2].sum())
        buy_pct = 100.0 * (num_buys / train_size)

        if self.dbg_verbose:
            # print("     tensor:", full_tensor.shape, ' -> train:', tsr_train.shape, " + test:", tsr_test.shape)
            # print("     labels:", lbl_tensor.shape, ' -> train:', tsr_lbl_train.shape, " + test:", tsr_lbl_test.shape)
            print("    training samples: ", train_size,
                  " #buys:", num_buys, " ({:.2f}".format(buy_pct), "%)",
                  ' #sells:', num_sells, " ({:.2f}".format(100.0 * (num_sells / train_size)), "%)", )

        # Create classifier for the model

        clf, clf_name = self.get_trinary_classifier(tsr_train, tsr_lbl_train, tsr_test, tsr_lbl_test)

        # save the models
        self.trinary_classifier = clf

        # if scan specified, test against the test dataframe
        if self.dbg_test_classifier:
            if not (clf is None):
                preds = self.get_classifier_predictions(clf, tsr_test)
                # results = np.argmax(tsr_lbl_test[:, 0], axis=1)
                results = np.argmax(tsr_lbl_test, axis=1)
                print(f"    Testing Classifier: {clf_name}, signals:{self.training_signals.get_signal_name()}, ",
                      f"pair: {curr_pair}")
                print(classification_report(results, preds, zero_division=0))
                print("")

                
                if self.dp.runmode.value in ('backtest'):
                    print("Testing pickle save/load...")

                    save_file = "clf.pkl"
                    pstr = ""

                    try:
                        print("    saving...")
                        if os.path.isfile(save_file):
                            os.remove(save_file)

                        with open(save_file, 'wb') as f:
                            pickle.dump(clf, f)

                        # print(f"pickle string: {pstr}")

                    except pickle.PicklingError as e:
                        print(f'An error occurred while pickling: {e}')


                    try:
                        print("    reloading...")
                        with open(save_file, 'rb') as f:
                            clf_loaded = pickle.load(f)
                    except pickle.UnpicklingError as e:
                        print(f'An error occurred while unpickling: {e}')
                
                

                
        return

    # get a classifier for the supplied normalised dataframe and known results
    def get_trinary_classifier(self, tensor, results, test_tensor, test_labels):

        clf = self.trinary_classifier
        name = str(self.classifier_type).split(".")[-1]

        # labels = self.get_trinary_labels(results)
        labels = results

        # if results.sum() <= 2:
        #     print("***")
        #     print("*** ERR: insufficient positive results in buy data")
        #     print("***")
        #     return clf, name

        if self.dp.runmode.value in ('backtest'):
            # If already done, just  re-fit
            if self.trinary_classifier:
                clf = self.fit_classifier(self.trinary_classifier, name, "", tensor, labels, test_tensor,
                                          test_labels)
            else:
                num_features = np.shape(tensor)[2]
                clf, name = NNTClassifier.create_classifier(self.classifier_type, self.curr_pair, num_features,
                                                            self.seq_len)

                # set the model name
                # category, model_name = self.get_model_identifiers(self.curr_pair, name)
                # clf.set_model_name(category, model_name)
                self.trinary_classifier.set_model_path(self.get_model_path(self.curr_pair, name))
                clf.set_combine_models(self.combine_models)

                # fit the classifier
                clf = self.fit_classifier(clf, name, "", tensor, labels, test_tensor, test_labels)

        return clf, name

    #######################################

    def get_compressor(self, df_norm: DataFrame):
        #  use fixed size PCA (Tensorflow models need fixed inputs)
        ncols = min(self.COMPRESSED_SIZE, df_norm.shape[-1])
        compressor = skd.PCA(n_components=ncols, whiten=True, svd_solver='full').fit(df_norm)

        num_features = np.shape(df_norm)[-1]
        if num_features > 2.0 * ncols:
            print(f"    ** WARNING: probably too much feature compression ({num_features} -> {ncols})")

        ratio_sum = compressor.explained_variance_ratio_.sum()
        if ratio_sum <= 0.9:
            print(f"    WARNING: reconstruction accuracy low: {ratio_sum} - Should be >0.9")

        return compressor

    #######################################

    def fit_classifier(self, classifier, name, tag, tensor, labels, test_tensor, test_labels):

        if classifier is None:
            print("    ERR: classifier is None")
            return None

        force_train = False if (not self.dp.runmode.value in ('backtest')) else self.refit_model
        classifier.train(tensor, test_tensor, labels, test_labels, force_train=force_train)

        return classifier

    def get_classifier_predictions(self, classifier, data):

        if self.dataframeUtils.is_dataframe(data):
            # convert dataframe to tensor
            df_tensor = self.dataframeUtils.df_to_tensor(data, self.seq_len)
        else:
            df_tensor = data

        if classifier == None:
            print("    no classifier for predictions")
            predictions = np.zeros(np.shape(df_tensor)[0], dtype=float)
            return predictions

        # run the prediction
        predictions = classifier.predict(df_tensor)
        return predictions

    #################################

    # list of potential classifier types - set to the list that you want to compare
    classifier_list = [
        NNTClassifier.ClassifierType.MLP,
        NNTClassifier.ClassifierType.Ensemble,
        NNTClassifier.ClassifierType.LSTM,
        NNTClassifier.ClassifierType.Multihead,
        NNTClassifier.ClassifierType.Transformer
    ]

    # return IDs that control model naming. Should be OK for all subclasses
    def get_model_identifiers(self, pair, clf_name, tag=""):
        # category = self.__class__.__name__

        # basic model name is built from base class (NNTC), dataset type, training signals and classifier

        if self.dataset_type == DatasetType.DEFAULT:
            ds = ""
        else:
            ds = str(self.dataset_type.value) + "_"
        model_name = "NNTC_" + \
                     ds + \
                     self.training_signals.get_signal_name() + "_" + \
                     clf_name

        category = "NNTC_" + \
                     self.training_signals.get_signal_name() + "_" + \
                     clf_name

        if self.model_per_pair:
            model_name = model_name + "_" + pair.split("/")[0]

        if len(tag) > 0:
            model_name = model_name + "_" + tag

        return category, model_name   
    
    # return IDs that control model naming. Should be OK for all subclasses
    def get_model_path(self, pair, clf_name, tag=""):

        # basic model name is built from base class (NNTC), dataset type, training signals and classifier

        if self.dataset_type == DatasetType.DEFAULT:
            ds = ""
        else:
            ds = str(self.dataset_type.value) + "_"

        model_name = "NNTC_" + \
                     ds + \
                     self.training_signals.get_signal_name() + "_" + \
                     clf_name

        root_dir = group_dir + "/models/" + model_name

        # category = "NNTC_" + \
        #              self.training_signals.get_signal_name() + "_" + \
        #              clf_name

        # add modifiers
        if self.model_per_pair:
            model_name = model_name + "_" + pair.split("/")[0]

        if len(tag) > 0:
            model_name = model_name + "_" + tag

        model_path = root_dir + "/" + model_name + ".keras"

        return model_path

    #######################################

    # tries different types of classifiers and returns the best one
    # tag parameter identifies where to save performance stats (default is not to save)
    def find_best_classifier(self, tensor, results, tag=""):

        if self.dbg_verbose:
            print("      Evaluating classifiers...")

        # Define dictionary with CLF and performance metrics
        scoring = {'accuracy': make_scorer(accuracy_score),
                   'precision': make_scorer(precision_score),
                   'recall': make_scorer(recall_score),
                   'f1_score': make_scorer(f1_score)}

        folds = 5
        clf_dict = {}
        models_scores_table = pd.DataFrame(index=['Accuracy', 'Precision', 'Recall', 'F1'])

        best_score = -0.1
        best_classifier = ""

        # labels = self.get_trinary_labels(results)
        labels = results

        # split into train & test sets
        # Note: we are taking the training data from the end (most recent data), not the beginning
        ratio = 0.8
        train_len = int(ratio * np.shape(labels)[0])
        test_len = np.shape(labels)[0] - train_len
        tsr_train = tensor[test_len + 1:, :, :]
        tsr_test = tensor[0:test_len:, :, :]
        res_train = labels[test_len + 1:, :]
        res_test = labels[0:test_len:, :]

        # print("tsr_train:", tsr_train.shape, " tsr_test:", tsr_test.shape,
        #       "res_train:", res_train.shape, "res_test:", res_test.shape)

        # check there are enough training samples
        # TODO: if low train/test samples, use k-fold sampling nstead
        if res_train.sum() < 2:
            print("    Insufficient +ve (train) results to fit: ", res_train.sum())
            return None, ""

        if res_test.sum() < 2:
            print("    Insufficient +ve (test) results: ", res_test.sum())
            return None, ""

        # scan through the list of classifiers in self.classifier_list
        num_features = np.shape(tsr_train)[2]
        for clf_id in self.classifier_list:
            clf, name = NNTClassifier.create_classifier(self.classifier_type, self.curr_pair, num_features,
                                                        self.seq_len, tag=tag)

            # set the model name
            # category, model_name = self.get_model_identifiers(self.curr_pair, name)
            # clf.set_model_name(category, model_name)
            self.trinary_classifier.set_model_path(self.get_model_path(self.curr_pair, name))
            clf.set_combine_models(self.combine_models)

            if clf is not None:

                # fit to the training data
                clf_dict[clf_id] = clf
                clf = self.fit_classifier(clf, clf_id, tag, tsr_train, res_train, tsr_test, res_test)

                # assess using the test data. Do *not* use the training data for testing
                pred_test = self.get_classifier_predictions(clf, tsr_test)

                # score = f1_score(results, prediction, average=None)[1]
                f1_scorer = make_scorer(f1_score, average='micro')
                score = f1_scorer(res_test[:, 0], pred_test)

                if self.dbg_verbose:
                    print("      {0:<20}: {1:.3f}".format(clf_id, score))

                if score > best_score:
                    best_score = score
                    best_classifier = clf_id

        if best_score <= 0.0:
            print("   No classifier found")
            return None, ""

        clf = clf_dict[best_classifier]

        # print("")
        if best_score < self.min_f1_score:
            print("!!!")
            print("!!! WARNING: F1 score below threshold ({:.3f})".format(best_score))
            print("!!!")
            return None, ""

        print("       ", tag, " model selected: ", best_classifier, " Score:{:.3f}".format(best_score))
        # print("")

        return clf, best_classifier

    # make predictions for supplied dataframe (returns column)
    def predict(self, dataframe: DataFrame, pair, clf):

        # predict = 0
        predict = None

        if clf is not None:
            # print("    predicting... - dataframe:", dataframe.shape)
            df_norm = self.dataframeUtils.norm_dataframe(dataframe)
            if self.compress_data:
                df_norm = self.compress_dataframe(df_norm)

            df_tensor = self.dataframeUtils.df_to_tensor(df_norm, self.seq_len)
            predict = self.get_classifier_predictions(clf, df_tensor)

        else:
            print("Null Classifier for pair: ", pair)

        # print (predict)
        return predict

    def predict_buysell(self, df: DataFrame, pair):
        clf = self.trinary_classifier

        if clf is None:
            print("    No Classifier for pair ", pair, " -Skipping predictions")
            predict = df['close'].copy()  # just to get the size
            predict = 0.0
            return predict, predict

        print("    predicting buys/sells...")
        preds = self.predict(df, pair, clf)

        # convert probability to buy & sell events
        # Note that I added in a 'loose'  MFI check, just to help filter out bad predictions
        buys = np.where(((preds > 0.5) & (preds < 1.4) & (df['mfi'] < 50)), 1.0, 0.0)
        sells = np.where(((preds > 1.5) & (df['mfi'] > 50)), 1.0, 0.0)

        # buys = np.where(((preds > 0.5) & (preds < 1.4)), 1.0, 0.0)
        # sells = np.where(((preds > 1.5)), 1.0, 0.0)

        return buys, sells

    ###################################
    # Debug stuff

    curr_state = {}

    def set_state(self, pair, state: State):
        # if self.dbg_verbose:
        #     if pair in self.curr_state:
        #         print("  ", pair, ": ", self.curr_state[pair], " -> ", state)
        #     else:
        #         print("  ", pair, ": ", " -> ", state)

        self.curr_state[pair] = state

    def get_state(self, pair) -> State:
        return self.curr_state[pair]

    def show_debug_info(self, pair):
        print("")

    def show_all_debug_info(self):
        print("")

    ###################################

    """
    Buy Signal
    """

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        conditions = []
        dataframe.loc[:, 'enter_tag'] = ''
        curr_pair = metadata['pair']

        self.set_state(curr_pair, self.State.RUNNING)

        if not self.dp.runmode.value in ('hyperopt'):
            if NNTC.first_run:
                NNTC.first_run = False  # note use of clas variable, not instance variable
                # self.show_debug_info(curr_pair)
                self.show_all_debug_info()

        # add some fairly loose guards, to help prevent 'bad' predictions

        # some trading volume
        conditions.append(dataframe['volume'] > 0)

        # MFI
        conditions.append(dataframe['mfi'] < 50.0)

        # Fisher/Williams in buy region
        conditions.append(dataframe['fisher_wr'] <= -0.5)

        # Classifier triggers
        # predict_cond = (
        #     (qtpylib.crossed_above(dataframe['predict_buy'], 0.0))
        # )
        predict_cond = (
            dataframe['predict_buy'] > 0.5
        )

        # print(f"Num buys: {dataframe['predict_buy'].sum()}")
        conditions.append(predict_cond)

        # add strategy-specific conditions (from subclass)
        strat_cond = self.get_strategy_entry_guard_conditions(dataframe)
        if strat_cond is not None:
            conditions.append(strat_cond)

        # set entry tags
        dataframe.loc[predict_cond, 'enter_tag'] += 'nntc_entry '

        if conditions:
            dataframe.loc[reduce(lambda x, y: x & y, conditions), 'enter_long'] = 1
            # print(f"Num buys: {dataframe['buy'].sum()}")
        else:
            dataframe['enter_long'] = 0

        if self.dbg_trace_memory and (self.dbg_trace_pair == self.curr_pair):
            profiler.snapshot()
            profiler.display_stats()
            profiler.compare()
            profiler.print_trace()

        return dataframe

    ###################################

    """
    Sell Signal
    """

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        conditions = []
        dataframe.loc[:, 'exit_tag'] = ''
        curr_pair = metadata['pair']

        self.set_state(curr_pair, self.State.RUNNING)

        if not self.dp.runmode.value in ('hyperopt'):
            if NNTC.first_run:
                NNTC.first_run = False  # note use of class variable, not instance variable
                # self.show_debug_info(curr_pair)
                self.show_all_debug_info()

        if self.ignore_exit_signals or (not self.enable_exit_signal.value):
            dataframe['exit_long'] = 0
            return dataframe

        # some volume
        conditions.append(dataframe['volume'] > 0)

        # MFI
        conditions.append(dataframe['mfi'] > 50.0)


        # Fisher/Williams in sell region
        conditions.append(dataframe['fisher_wr'] >= 0.5)

        # model triggers
        # predict_cond = (
        #     qtpylib.crossed_above(dataframe['predict_sell'], 0.5)
        # )
        predict_cond = (
                dataframe['predict_sell'] > 0.5
        )


        conditions.append(predict_cond)

        # add strategy-specific conditions (from subclass)
        strat_cond = self.get_strategy_exit_guard_conditions(dataframe)
        if strat_cond is not None:
            conditions.append(strat_cond)

        dataframe.loc[predict_cond, 'exit_tag'] += 'nntc_exit '

        if conditions:
            dataframe.loc[reduce(lambda x, y: x & y, conditions), 'exit_long'] = 1
        else:
            dataframe['exit_long'] = 0

        return dataframe

    ###################################

    """
    Custom Stoploss
    """

    # simplified version of custom trailing stoploss
    def custom_stoploss(self, pair: str, trade: Trade, current_time: datetime, current_rate: float,
                        current_profit: float, **kwargs) -> float:

        if current_profit > self.tstop_start.value:
            return current_profit * self.tstop_ratio.value

        # return min(-0.001, max(stoploss_from_open(0.05, current_profit), -0.99))
        return self.stoploss


    ###################################

    """
    Custom Exit
    (Note that this runs even if use_custom_stoploss is False)
    """

    # simplified version of custom exit

    def custom_exit(self, pair: str, trade: Trade, current_time: 'datetime', current_rate: float,
                    current_profit: float, **kwargs):

        dataframe, _ = self.dp.get_analyzed_dataframe(pair=pair, timeframe=self.timeframe)
        last_candle = dataframe.iloc[-1].squeeze()

        # trade_dur = int((current_time.timestamp() - trade.open_date_utc.timestamp()) // 60)
        # max_profit = max(0, trade.calc_profit_ratio(trade.max_rate))

        # Mod: just take the profit:
        # Above 1%, sell if Fisher/Williams in sell range
        if current_profit > 0.01:
            if last_candle['fisher_wr'] > 0.8:
                return 'take_profit'

        # Mod: strong sell signal, in profit
        if (current_profit > 0) and (last_candle['fisher_wr'] > 0.93):
            return 'fwr_high'

        # Mod: Sell any positions at a loss if they are held for more than 'N' days.
        # if (current_profit < 0.0) and (current_time - trade.open_date_utc).days >= 7:
        if (current_time - trade.open_date_utc).days >= 7:
            return 'unclog'

        # check profit against threshold. This sort of emulates the freqtrade roi approach, but is much simpler
        if self.use_profit_threshold.value:
            if (current_profit >= self.profit_threshold.value):
                return 'profit_threshold'

        # check loss against threshold. This sort of emulates the freqtrade stoploss approach, but is much simpler
        if self.use_loss_threshold.value:
            if (current_profit <= self.loss_threshold.value):
                return 'loss_threshold'

        # if in profit and exit signal is set, sell (whether or not ignore exit is active)
        if (current_profit > 0) and (last_candle['exit_long'] > 0):
            return 'exit_signal'
        
        return None

#######################
