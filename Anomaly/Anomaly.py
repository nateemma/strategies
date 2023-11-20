import operator

import numpy as np
from enum import Enum

import pywt
import talib.abstract as ta
from scipy.ndimage import gaussian_filter1d
from sklearn.manifold import LocallyLinearEmbedding

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
from pathlib import Path

sys.path.append(str(Path(__file__).parent))
sys.path.append(str(Path(__file__).parent.parent))

import logging
import warnings

log = logging.getLogger(__name__)
# log.setLevel(logging.DEBUG)
warnings.simplefilter(action='ignore', category=pd.errors.PerformanceWarning)


import  utils.custom_indicators as cta
from finta import TA as fta

from sklearn.model_selection import RandomizedSearchCV, train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.preprocessing import StandardScaler, RobustScaler
import sklearn.decomposition as skd
from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler

from sklearn.metrics import make_scorer
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.model_selection import cross_validate

import random

from prettytable import PrettyTable

import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
os.environ['TF_DETERMINISTIC_OPS'] = '1'

# import tensorflow as tf

seed = 42
os.environ['PYTHONHASHSEED'] = str(seed)
random.seed(seed)
# tf.random.set_seed(seed)
np.random.seed(seed)

# tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.WARN)

#import keras
from keras import layers
from tqdm import tqdm

from utils.CompressionAutoEncoder import CompressionAutoEncoder

from AnomalyDetector_AEnc import AnomalyDetector_AEnc
from AnomalyDetector_LOF import AnomalyDetector_LOF
from AnomalyDetector_KMeans import AnomalyDetector_KMeans
from AnomalyDetector_IFOR import AnomalyDetector_IFOR
from AnomalyDetector_EE import AnomalyDetector_EE
from AnomalyDetector_SVM import AnomalyDetector_SVM
from AnomalyDetector_LSTM import AnomalyDetector_LSTM
from AnomalyDetector_PCA import AnomalyDetector_PCA
from AnomalyDetector_GMix import AnomalyDetector_GMix
from AnomalyDetector_DBSCAN import AnomalyDetector_DBSCAN
from AnomalyDetector_Ensemble import AnomalyDetector_Ensemble

from utils.DataframeUtils import DataframeUtils, ScalerType
from utils.DataframePopulator import DataframePopulator, DatasetType
import utils.TrainingSignals as TrainingSignals
from utils.Environment import Environment

"""
####################################################################################
Anomaly - Anomaly Detection Classifier
    
    This strategy uses anomaly detection as a means to identify buys/sells
    The theory is that there are many more non-buy or non-sell signals that buy/sell signals
    So, we train an autoencoder on the 'normal' data, and any anomalies detected are then buy or sell signals.
    
    Note that the anomaly detection classifiers will look for a pre-trained model and load that if present. This
    means you can pre-train models in backtesting and use then in live runs
      
    Note that this is very slow to start up, especially if a pre-trained model is not present. 
    This is mostly because we have to build the data on a rolling basis to avoid lookahead bias.
      
    In addition to the normal freqtrade packages, these strategies also require the installation of:
        random
        prettytable
        finta
        sklearn

####################################################################################
"""


class Anomaly(IStrategy):

    plot_config = {
        'main_plot': {
            'close': {'color': 'cornflowerblue'},
            # '%recon': {'color': 'lightsalmon'},
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
        "0": 0.04
    }

    # Stoploss:
    stoploss = -0.06

    # Trailing stop:
    trailing_stop = False
    trailing_stop_positive = 0.00
    trailing_stop_positive_offset = 0.00
    trailing_only_offset_is_reached = False

    timeframe = '5m'

    inf_timeframe = '5m'

    use_custom_stoploss = True
    use_simpler_custom_stoploss = False

    # Recommended
    use_entry_signal = True
    entry_profit_only = False
    ignore_roi_if_entry_signal = True

    # Required
    startup_candle_count: int = 128  # must be power of 2
    process_only_new_candles = True

    # Strategy-specific global vars

    dwt_window = startup_candle_count

    inf_mins = timeframe_to_minutes(inf_timeframe)
    data_mins = timeframe_to_minutes(timeframe)
    inf_ratio = int(inf_mins / data_mins)

    # These parameters control much of the behaviour because they control the generation of the training data
    # Unfortunately, these cannot be hyperopt params because they are used in populate_indicators, which is only run
    # once during hyperopt
    lookahead_hours = 1.0
    n_profit_stddevs = 0.0
    n_loss_stddevs = 0.0
    min_f1_score = 0.48 # this will be low because results are anomalies :-)

    curr_lookahead = int(12 * lookahead_hours)

    curr_pair = ""
    custom_trade_info = {}

    compressor = None
    compress_data = True
    scaler_type = ScalerType.Robust # scaler type used for normalisation

    dataframeUtils = None
    dataframePopulator = None

    num_pairs = 0
    buy_classifier = None
    sell_classifier = None
    buy_classifier_list = {}
    sell_classifier_list = {}

    # debug flags
    first_time = True  # mostly for debug
    first_run = True  # used to identify first time through buy/sell populate funcs

    dbg_scan_classifiers = False  # if True, scan all viable classifiers and choose the best. Very slow!
    dbg_test_classifier = True  # test clasifiers after fitting
    dbg_verbose = True  # controls debug output
    dbg_curr_df: DataFrame = None  # for debugging of current dataframe

    # variables to track state
    class State(Enum):
        INIT = 1
        POPULATE = 2
        STOPLOSS = 3
        RUNNING = 4

    # enum of various classifiers available
    class ClassifierType(Enum):
        CompressionAutoEncoder = -1
        LSTMAutoEncoder = 1
        MLPAutoEncoder = 2
        LocalOutlierFactor = 3
        KMeans = 4
        IsolationForest = 5
        EllipticEnvelope = 6
        OneClassSVM = 7
        PCA = 8
        GaussianMixture = 9
        DBSCAN = 10 # currently not working
        Ensemble = 11

    # classifier_type = ClassifierType.IsolationForest # controls which classifier is used
    classifier_type = ClassifierType.Ensemble # controls which classifier is used
    # classifier_type = ClassifierType.DBSCAN # controls which classifier is used

    dataset_type = DatasetType.DEFAULT
    signal_type = TrainingSignals.SignalType.Profit  # should override this
    training_signals = None

    ###################################

    # Strategy Specific Variable Storage

    ## Hyperopt Variables

    '''
    # trailing stoploss
    tstop_start = DecimalParameter(0.0, 0.06, default=0.015, decimals=3, space='sell', load=True, optimize=True)
    tstop_ratio = DecimalParameter(0.7, 0.99, default=0.9, decimals=3, space='sell', load=True, optimize=True)

    # profit threshold exit
    profit_threshold = DecimalParameter(0.005, 0.065, default=0.025, decimals=3, space='sell', load=True, optimize=True)
    use_profit_threshold = CategoricalParameter([True, False], default=True, space='sell', load=True, optimize=True)

    # loss threshold exit
    loss_threshold = DecimalParameter(-0.065, -0.005, default=-0.046, decimals=3, space='sell', load=True, optimize=True)
    use_loss_threshold = CategoricalParameter([True, False], default=True, space='sell', load=True, optimize=True)

    # use exit signal? 
    enable_exit_signal = CategoricalParameter([True, False], default=False, space='sell', load=True, optimize=True)

    '''


    # enable entry/exit guards (safer vs profit)
    enable_entry_guards = CategoricalParameter([True, False], default=False, space='buy', load=True, optimize=True)
    entry_guard_fwr = DecimalParameter(-1.0, 0.0, default=-0.0, decimals=1, space='buy', load=True, optimize=True)

    enable_exit_guards = CategoricalParameter([True, False], default=False, space='sell', load=True, optimize=True)
    exit_guard_fwr = DecimalParameter(0.0, 1.0, default=0.0, decimals=1, space='sell', load=True, optimize=True)

    # use exit signal? 
    enable_exit_signal = CategoricalParameter([True, False], default=True, space='sell', load=True, optimize=True)

    # Custom Stoploss
    cstop_enable = CategoricalParameter([True, False], default=False, space='sell', load=True, optimize=True)
    cstop_start = DecimalParameter(0.0, 0.060, default=0.019, decimals=3, space='sell', load=True, optimize=True)
    cstop_ratio = DecimalParameter(0.7, 0.999, default=0.8, decimals=3, space='sell', load=True, optimize=True)

    # Custom Exit
    # profit threshold exit
    cexit_profit_threshold = DecimalParameter(0.005, 0.065, default=0.033, decimals=3, space='sell', load=True, optimize=True)
    cexit_use_profit_threshold = CategoricalParameter([True, False], default=True, space='sell', load=True, optimize=True)

    # loss threshold exit
    cexit_loss_threshold = DecimalParameter(-0.065, -0.005, default=-0.046, decimals=3, space='sell', load=True, optimize=True)
    cexit_use_loss_threshold = CategoricalParameter([True, False], default=False, space='sell', load=True, optimize=True)

    cexit_fwr_overbought = DecimalParameter(0.90, 1.00, default=0.98, decimals=2, space='sell', load=True, optimize=True)
    cexit_fwr_take_profit = DecimalParameter(0.90, 1.00, default=0.90, decimals=2, space='sell', load=True, optimize=True)
 
    ################################

    # subclasses should override the following 4 functions - this is here as an example

    # Note: try to combine current/historical data (from populate_indicators) with future data
    #       If you only use future data, the ML training is just guessing
    #       Also, try to identify buy/sell ranges, rather than transitions - it gives the algorithms more chances
    #       to find a correlation. The framework will select the first one anyway.
    #       In other words, avoid using qtpylib.crossed_above() and qtpylib.crossed_below()
    #       OK not to check volume, because we are just looking for patterns


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
        print("")
        
    ################################

    # run data augmentation techniques
    def augment_training_signals(self, buys, sells):

        # Trick 1: artificially extend positive signals one entry earlier

        bidx = np.where(buys > 0)[0]  # index of buy entries
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
    
    ################################
    """
    Indicator Definitions
    """

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:

        # Base pair inf timeframe indicators
        curr_pair = metadata['pair']
        self.curr_pair = curr_pair

        self.set_state(curr_pair, self.State.POPULATE)
        self.dbg_curr_df = dataframe

        if self.training_signals is None:
            self.training_signals = TrainingSignals.create_training_signals(self.signal_type, self.curr_lookahead)
            self.curr_lookahead = self.training_signals.get_lookahead()
            self.n_loss_stddevs = self.training_signals.get_n_loss_stddevs()
            self.n_profit_stddevs = self.training_signals.get_n_profit_stddevs()
            self.lookahead_hours = float(self.curr_lookahead / 12.0)


        if self.dataframeUtils is None:
            self.dataframeUtils = DataframeUtils()

        if self.dataframePopulator is None:
            self.dataframePopulator = DataframePopulator()

            self.dataframePopulator.runmode = self.dp.runmode.value
            self.dataframePopulator.win_size = min(14, self.curr_lookahead)
            self.dataframePopulator.startup_win = self.startup_candle_count
            self.dataframePopulator.n_loss_stddevs = self.n_loss_stddevs
            self.dataframePopulator.n_profit_stddevs = self.n_profit_stddevs

        # populate the normal dataframe
        dataframe = self.dataframePopulator.add_indicators(dataframe)
        # dataframe = self.add_indicators(dataframe)

        if Anomaly.first_time:
            Anomaly.first_time = False
            print("------------------")
            print(self.__class__.__name__)
            print("------------------")
            print("***************************************")
            print("** Warning: startup can be very slow **")
            print("***************************************")

            Environment().print_environment()

            self.print_strategy_info()

        print("")
        print(curr_pair)

        # (re-)set the scaler
        self.dataframeUtils.set_scaler_type(self.scaler_type)

        # create labels used for training
        buys, sells = self.create_training_data(dataframe)

        # # drop last group (because there cannot be a prediction)
        # df = dataframe.iloc[:-self.curr_lookahead]
        # buys = buys.iloc[:-self.curr_lookahead]
        # sells = sells.iloc[:-self.curr_lookahead]

        # train the models on the informative data
        if self.dbg_verbose:
            print("    training models...")
        df = self.train_models(curr_pair, dataframe, buys, sells)

        # add predictions

        if self.dbg_verbose:
            print("    running predictions...")

        # get predictions (Note: do not modify dataframe between calls)
        pred_buys = self.predict_buy(dataframe, curr_pair)
        pred_sells = self.predict_sell(dataframe, curr_pair)
        dataframe['predict_buy'] = pred_buys
        dataframe['predict_sell'] = pred_sells

        if self.dp.runmode.value in ('plot'):
            dataframe['%recon'] = df['%recon']

        # Custom Stoploss
        if self.dbg_verbose:
            print("    updating stoploss data...")
        self.add_stoploss_indicators(dataframe, curr_pair)

        return dataframe

    ###################################
    
    # fast curve smoothing utility
    def smooth(self, y, window):
        box = np.ones(window)/window
        y_smooth = np.convolve(np.nan_to_num(y), box, mode='same')
        y_smooth = np.round(y_smooth, decimals=3) #Hack: constrain to 3 decimal places (should be elsewhere, but convenient here)
        return np.nan_to_num(y_smooth)
    

    # add indicators used by stoploss/custom sell logic
    def add_stoploss_indicators(self, dataframe, pair) -> DataFrame:

        '''
        if not pair in self.custom_trade_info:
            self.custom_trade_info[pair] = {}
            if not 'had_trend' in self.custom_trade_info[pair]:
                self.custom_trade_info[pair]['had_trend'] = False

        dataframe = self.dataframePopulator.add_stoploss_indicators(dataframe)

        '''
        
        # add backward looking gain. (default is not waht we need)
        dataframe['bgain'] = 100.0 * (dataframe['close'] - dataframe['close'].shift(self.curr_lookahead)) / \
                            dataframe['close'].shift(self.curr_lookahead)

        dataframe['bgain'] = self.smooth(dataframe['bgain'], self.curr_lookahead) # takes care of edge effects

        dataframe['bprofit'] = dataframe['bgain'].clip(lower=0.0)
        dataframe['bloss'] = dataframe['bgain'].clip(upper=0.0)


        # add thresholds (used by buy/sell and custom exit)
        win_size = 32
        dataframe['target_profit'] = dataframe['bprofit'].rolling(window=win_size).mean() + \
            2.0 * dataframe['bprofit'].rolling(window=win_size).std()
        dataframe['target_loss'] = -abs(dataframe['bloss'].rolling(window=win_size).mean()) - \
            2.0 * abs(dataframe['bloss'].rolling(window=win_size).std())

        dataframe['target_profit'] = np.nan_to_num(dataframe['target_profit'])
        dataframe['target_loss'] = np.nan_to_num(dataframe['target_loss'])

        return dataframe



    ################################

    # creates the buy/sell labels absed on looking ahead into the supplied dataframe
    def create_training_data(self, dataframe: DataFrame):

        # future_df = self.add_future_data(dataframe.copy())
        future_df = self.dataframePopulator.add_hidden_indicators(dataframe.copy())
        future_df = self.dataframePopulator.add_future_data(future_df, self.curr_lookahead)

        future_df['train_buy'] = 0.0
        future_df['train_sell'] = 0.0

        # use sequence trends as criteria
        buys = self.get_train_buy_signals(future_df)
        sells = self.get_train_sell_signals(future_df)
        # run data augmentation techniques
        buys, sells = self.augment_training_signals(buys, sells)

        # use sequence trends as criteria
        future_df['train_buy'] = buys
        future_df['train_sell'] = sells

        # buys = future_df['train_buy'].copy()
        if buys.sum() < 3:
            print("OOPS! <3 ({:.0f}) buy signals generated. Check training criteria".format(buys.sum()))

        # sells = future_df['train_sell'].copy()
        if buys.sum() < 3:
            print("OOPS! <3 ({:.0f}) sell signals generated. Check training criteria".format(sells.sum()))

        self.save_debug_data(future_df)
        self.save_debug_indicators(future_df)

        return buys, sells

    def save_debug_data(self, future_df: DataFrame):

        # Debug support: add commonly used indicators so that they can be viewed
        # the list below is available for any subclass. Subclasses themselves can add more by overriding
        # the func save_debug_indicators()

        dbg_list = [
            'full_dwt', 'train_buy', 'train_sell',
            'future_gain', 'future_min', 'future_max',
            'future_profit_min', 'future_profit_max', 'future_profit_threshold',
            'future_loss_min', 'future_loss_max', 'future_loss_threshold',
        ]

        if len(dbg_list) > 0:
            for indicator in dbg_list:
                self.add_debug_indicator(future_df, indicator)

        return

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


    ############################

    def get_classifier(self, nfeatures, tag):
        clf = None
        # clf_type = 4

        if self.classifier_type == self.ClassifierType.CompressionAutoEncoder:
            # This is a special case used to train the compression autoencoder. It is not actually a classifier
            # NOTE: have to be careful about dimensions (must match training)
            # ONLY use this option to train the compression autoencoder
            if self.compress_data:
                print("ERROR: self.compress_data should be False")
                return None
            clf = CompressionAutoEncoder(nfeatures, tag=tag)

        elif self.classifier_type == self.ClassifierType.MLPAutoEncoder:
            clf = AnomalyDetector_AEnc(nfeatures, tag=tag)

        elif self.classifier_type == self.ClassifierType.LocalOutlierFactor:
            clf = AnomalyDetector_LOF(self.curr_pair, tag=tag)

        elif self.classifier_type == self.ClassifierType.KMeans:
            clf = AnomalyDetector_KMeans(self.curr_pair, tag=tag)

        elif self.classifier_type == self.ClassifierType.IsolationForest:
            clf = AnomalyDetector_IFOR(self.curr_pair, tag=tag)

        elif self.classifier_type == self.ClassifierType.EllipticEnvelope:
            clf = AnomalyDetector_EE(self.curr_pair, tag=tag)

        elif self.classifier_type == self.ClassifierType.OneClassSVM:
            clf = AnomalyDetector_SVM(self.curr_pair, tag=tag)

        elif self.classifier_type == self.ClassifierType.PCA:
            clf = AnomalyDetector_PCA(self.curr_pair, tag=tag)

        elif self.classifier_type == self.ClassifierType.LSTMAutoEncoder:
            clf = AnomalyDetector_LSTM(nfeatures, tag=tag)

        elif self.classifier_type == self.ClassifierType.GaussianMixture:
            clf = AnomalyDetector_GMix(self.curr_pair, tag=tag)

        elif self.classifier_type == self.ClassifierType.DBSCAN:
            clf = AnomalyDetector_DBSCAN(self.curr_pair, tag=tag)

        elif self.classifier_type == self.ClassifierType.Ensemble:
            clf = AnomalyDetector_Ensemble(self.curr_pair, tag=tag)

        else:
            print("    ERR: unknown classifier type ({})".format(self.classifier_type))

        return clf

    ############################

    # train the PCA reduction and classification models

    def train_models(self, curr_pair, dataframe: DataFrame, buys, sells) -> DataFrame:

        # check input - need at least 2 samples or classifiers will not train
        if buys.sum() < 2:
            print("*** ERR: insufficient buys in expected results. Check training data")
            # print(buys)
            return

        if sells.sum() < 2:
            print("*** ERR: insufficient sells in expected results. Check training data")
            return

        rand_st = 27  # use fixed number for reproducibility

        full_df_norm = self.dataframeUtils.norm_dataframe(dataframe)


        if self.compress_data:
            old_size = full_df_norm.shape[1]
            full_df_norm = self.compress_dataframe(full_df_norm)
            print("    Compressed data {} -> {} (features)".format(old_size, full_df_norm.shape[1]))
        else:
            if self.dbg_verbose:
                print("    Not compressing data")

        # create classifiers, if necessary

        if self.curr_pair not in self.buy_classifier_list:
            self.buy_classifier = self.get_classifier(full_df_norm.shape[1], "Buy")
            self.buy_classifier_list[self.curr_pair] = self.buy_classifier
        else:
            self.buy_classifier = self.buy_classifier_list[self.curr_pair]

        if self.enable_exit_signal.value:
            if self.curr_pair not in self.sell_classifier_list:
                self.sell_classifier = self.get_classifier(full_df_norm.shape[1], "Sell")
                self.sell_classifier_list[self.curr_pair] = self.sell_classifier
            else:
                self.sell_classifier = self.sell_classifier_list[self.curr_pair]

        # constrain sample size to what will be available in run modes
        data_size = int(min(975, full_df_norm.shape[0]))

        train_ratio = 0.8
        train_size = int(train_ratio * data_size)
        test_size = data_size - train_size

        # df_train, df_test, train_buys, test_buys, train_sells, test_sells, = train_test_split(full_df_norm,
        #                                                                                       buys,
        #                                                                                       sells,
        #                                                                                       train_size=train_size,
        #                                                                                       random_state=rand_st,
        #                                                                                       shuffle=False)
        # use the back portion of data for training, front for testing
        # df_test, df_train = self.dataframeUtils.split_dataframe(full_df_norm, (1.0 - train_ratio))
        # test_buys, train_buys = self.dataframeUtils.split_array(buys, (1.0 - train_ratio))
        # test_sells, train_sells = self.dataframeUtils.split_array(sells, (1.0 - train_ratio))

        df_train, df_test = self.dataframeUtils.split_dataframe(full_df_norm, train_ratio)
        train_buys, test_buys = self.dataframeUtils.split_array(buys, train_ratio)
        train_sells, test_sells = self.dataframeUtils.split_array(sells, train_ratio)

        if self.dbg_verbose:
            print("     dataframe:", full_df_norm.shape, ' -> train:', df_train.shape, " + test:", df_test.shape)
            print("     buys:", buys.shape, ' -> train:', train_buys.shape, " + test:", test_buys.shape)
            print("     sells:", sells.shape, ' -> train:', train_sells.shape, " + test:", test_sells.shape)

        print("    #training samples:", len(df_train), " #buys:", int(train_buys.sum()), ' #sells:',
              int(train_sells.sum()))

        train_buy_labels = self.dataframeUtils.get_binary_labels(train_buys)
        train_sell_labels = self.dataframeUtils.get_binary_labels(train_sells)
        test_buy_labels = self.dataframeUtils.get_binary_labels(test_buys)
        test_sell_labels = self.dataframeUtils.get_binary_labels(test_sells)

        # force train/fit the classifiers in backtest mode only
        force_train = True if (self.dp.runmode.value in ('backtest')) else False

        # Buy Classifier
        self.buy_classifier.train(df_train, df_test, train_buys, test_buys, force_train=force_train)

        # Sell Classifier
        if self.enable_exit_signal.value:
            self.sell_classifier.train(df_train, df_test, train_sells, test_sells, force_train=force_train)


        # if scan specified, test against the test dataframe
        if self.dbg_test_classifier:

            if not (self.buy_classifier is None):
                pred_buys = self.buy_classifier.predict(df_test)
                print("")
                print("Testing - Buy Signals (", type(self.buy_classifier).__name__, ")")
                print(classification_report(test_buy_labels, pred_buys, zero_division=0))
                print("")

            if not (self.sell_classifier is None):
                pred_sells = self.sell_classifier.predict(df_test)
                print("")
                print("Testing - Sell Signals (", type(self.sell_classifier).__name__, ")")
                print(classification_report(test_sell_labels, pred_sells, zero_division=0))
                print("")

        # if running 'plot', reconstruct the original dataframe for display
        if self.dp.runmode.value in ('plot'):
            if self.compress_data:
                df_norm = self.dataframeUtils.norm_dataframe(dataframe)  # this also resets the scaler
                df_compressed = self.compress_dataframe(df_norm)
                df_recon_compressed = self.buy_classifier.reconstruct(df_compressed)
                df_recon_norm = self.compressor.inverse_transform(df_recon_compressed)
                df_recon_norm = pd.DataFrame(df_recon_norm, columns=df_norm.columns)
                df_recon = self.dataframeUtils.denorm_dataframe(df_recon_norm)
                dataframe['%recon'] = df_recon['close']
            else:
                # debug: get reconstructed dataframe and save 'close' as a comparison
                tmp = self.dataframeUtils.norm_dataframe(dataframe)  # this just resets the scaler
                df_recon_norm = self.buy_classifier.reconstruct(tmp)
                df_recon = self.dataframeUtils.denorm_dataframe(df_recon_norm)
                dataframe['%recon'] = df_recon['close']
        return dataframe

    # compress the supplied dataframe
    def compress_dataframe(self, df_norm: DataFrame) -> DataFrame:
        if not self.compressor:
            self.compressor = self.get_compressor(df_norm)
        return pd.DataFrame(self.compressor.transform(df_norm))

    # get the compressor model for the supplied dataframe (dataframe must be normalised)
    # use .transform() to compress the dataframe
    def get_compressor(self, df_norm: DataFrame):

        ncols = df_norm.shape[1]  # allow all components to get the full variance matrix
        whiten = True

        compressor_type = 0

        # there are various types of PCA, plus alternatives like ICA and Feature Extraction
        if compressor_type == 0:
            # just use fixed size PCA (easier for classifiers to deal with)
            ncols = 64
            compressor = skd.PCA(n_components=ncols, whiten=True, svd_solver='full').fit(df_norm)

        elif compressor_type == 1:
            # accurate, but slow
            print("    Using KernelPCA...")
            compressor = skd.KernelPCA(n_components=ncols, remove_zero_eig=True).fit(df_norm)
            eigenvalues = compressor.eigenvalues_
            # print(var_ratios)
            # Note: eigenvalues are not bounded, so just have to go by min value
            ncols = 0
            val_threshold = 0.5
            while ((eigenvalues[ncols] > val_threshold) & (ncols < len(eigenvalues))):
                ncols = ncols + 1

            # if necessary, re-calculate compressor with reduced column set
            if (ncols != df_norm.shape[1]):
                # compressor = skd.PCA(n_components=ncols, svd_solver="randomized", whiten=True).fit(df_norm)
                compressor = skd.KernelPCA(n_components=ncols, remove_zero_eig=True).fit(df_norm)

        elif compressor_type == 2:
            # fast, but not as accurate
            print("    Using Locally Linear Embedding...")
            # compressor = LocallyLinearEmbedding(n_neighbors=32, n_components=16, eigen_solver='dense',
            #                              method="modified").fit(df_norm)
            compressor = LocallyLinearEmbedding().fit(df_norm)

        elif compressor_type == 3:
            # a bit slow, still debugging...
            print("    Using Autoencoder...")
            compressor = CompressionAutoEncoder(df_norm.shape[1], tag="Buy")

        else:
            print("*** ERR - unknown compressor type ({}) ***".format(compressor_type))
            compressor = None

        return compressor

    # make predictions for supplied dataframe (returns column)
    def predict(self, dataframe: DataFrame, pair, clf):

        # predict = 0
        predict = None

        if clf:
            # print("    predicting... - dataframe:", dataframe.shape)
            df_norm = self.dataframeUtils.norm_dataframe(dataframe)
            if self.compress_data:
                df_norm = self.compress_dataframe(df_norm)
            predict = clf.predict(df_norm)

        else:
            print("Null CLF for pair: ", pair)

        # print (predict)
        return predict

    def predict_buy(self, df: DataFrame, pair):
        clf = self.buy_classifier

        if clf is None:
            print("    No Buy Classifier for pair ", pair, " -Skipping predictions")
            predict = df['close'].copy()  # just to get the size
            predict = 0.0
            return predict

        print("    predicting buys...")
        predict = self.predict(df, pair, clf)

        # anomaly detection tends to flag both buys and sells, so filter based on MFI
        predict = np.where((predict > 0) & (df['mfi'] < 50), 1.0, 0.0)

        return predict

    def predict_sell(self, df: DataFrame, pair):
        clf = self.sell_classifier
        if clf is None:
            print("    No Sell Classifier for pair ", pair, " -Skipping predictions")
            predict = df['close']  # just to get the size
            predict = 0.0
            return predict

        print("    predicting sells...")
        predict = self.predict(df, pair, clf)

        # anomaly detection tends to flag both buys and sells, so filter based on MFI
        predict = np.where((predict > 0) & (df['mfi'] > 50), 1.0, 0.0)

        return predict

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
            if Anomaly.first_run:
                Anomaly.first_run = False  # note use of clas variable, not instance variable
                # self.show_debug_info(curr_pair)
                self.show_all_debug_info()

       # add some fairly loose guards, to help prevent 'bad' predictions

        if self.enable_entry_guards.value:
            # some trading volume
            conditions.append(dataframe['volume'] > 0)

            # # MFI
            # conditions.append(dataframe['mfi'] < 50.0)

            # Fisher/Williams in buy region
            conditions.append(dataframe['fisher_wr'] <= -0.5)

        # add strategy-specific conditions (from subclass)
        strat_cond = self.get_strategy_entry_guard_conditions(dataframe)
        if strat_cond is not None:
            conditions.append(strat_cond)

        # # sell signal is not active
        # conditions.append(dataframe['predict_sell'] < 0.1)

        # PCA/Classifier triggers
        anomaly_cond = (
            # (qtpylib.crossed_above(dataframe['predict_buy'], 0.5))
            dataframe['predict_buy'] > 0.5
 
        )
        conditions.append(anomaly_cond)

        # set entry tags
        dataframe.loc[anomaly_cond, 'enter_tag'] += 'anom_entry '

        if conditions:
            dataframe.loc[reduce(lambda x, y: x & y, conditions), 'enter_long'] = 1
        else:
            dataframe['enter_long'] = 0

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

        if not self.enable_exit_signal.value:
            dataframe['exit_long'] = 0
            return dataframe
        
        if not self.dp.runmode.value in ('hyperopt'):
            if Anomaly.first_run:
                Anomaly.first_run = False  # note use of clas variable, not instance variable
                # self.show_debug_info(curr_pair)
                self.show_all_debug_info()
        # if we are to ignore exit signals, just set exit column to 0s and return
        if not self.enable_exit_signal.value:
            dataframe['exit_long'] = 0
            return dataframe
        
        if self.enable_exit_guards.value:
            conditions.append(dataframe['volume'] > 0)

            # # MFI
            # conditions.append(dataframe['mfi'] > 50.0)

            # Fisher/Williams in sell region
            conditions.append(dataframe['fisher_wr'] >= 0.5)


        # add strategy-specific conditions (from subclass)
        strat_cond = self.get_strategy_exit_guard_conditions(dataframe)
        if strat_cond is not None:
            conditions.append(strat_cond)

        # PCA triggers
        anomaly_cond = (
            # qtpylib.crossed_above(dataframe['predict_sell'], 0.5)
            dataframe['predict_sell'] > 0.5
        )

        conditions.append(anomaly_cond)

        dataframe.loc[anomaly_cond, 'exit_tag'] += 'anom_exit '

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

        # if enable, use custom trailing ratio, else use default system
        if self.cstop_enable.value:
            # if current profit is above start value, then set stoploss at fraction of current profit
            if current_profit > self.cstop_start.value:
                return current_profit * self.cstop_ratio.value

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
        

        if not self.use_custom_stoploss:
            return None

        # strong sell signal, in profit
        if (current_profit > 0) and (last_candle['fisher_wr'] >= self.cexit_fwr_overbought.value):
            return 'fwr_overbought'

        # Above 1%, sell if Fisher/Williams in sell range
        if current_profit > 0.01:
            if last_candle['fisher_wr'] >= self.cexit_fwr_take_profit.value:
                return 'take_profit'
 

        # check profit against ROI target. This sort of emulates the freqtrade roi approach, but is much simpler
        if self.cexit_use_profit_threshold.value:
            if (current_profit >= self.cexit_profit_threshold.value):
                return 'cexit_profit_threshold'

        # check loss against threshold. This sort of emulates the freqtrade stoploss approach, but is much simpler
        if self.cexit_use_loss_threshold.value:
            if (current_profit <= self.cexit_loss_threshold.value):
                return 'cexit_loss_threshold'
              
        # Sell any positions if open for >= 1 day with any level of profit
        if ((current_time - trade.open_date_utc).days >= 1) & (current_profit > 0):
            return 'unclog_1'
        
        # Sell any positions at a loss if they are held for more than 7 days.
        if (current_time - trade.open_date_utc).days >= 7:
            return 'unclog_7'
        
        ''' not a gain-based strat
        # big drop predicted. Should also trigger an exit signal, but this might be quicker (and will likely be 'market' sell)
        if (current_profit > 0) and (last_candle['predicted_gain'] <= last_candle['target_loss']):
            return 'predict_drop'
        
        '''

        # if in profit and exit signal is set, sell (even if exit signals are disabled)
        if (current_profit > 0) and (last_candle['exit_long'] > 0):
            return 'exit_signal'

        return None

    #######################