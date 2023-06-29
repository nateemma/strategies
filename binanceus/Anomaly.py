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

import logging
import warnings

log = logging.getLogger(__name__)
# log.setLevel(logging.DEBUG)
warnings.simplefilter(action='ignore', category=pd.errors.PerformanceWarning)

import custom_indicators as cta
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

from CompressionAutoEncoder import CompressionAutoEncoder

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

from DataframeUtils import DataframeUtils, ScalerType
from DataframePopulator import DataframePopulator

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

    ignore_exit_signals = False # set to True if you don't want to process sell/exit signals (let custom sell do it)

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

    ###################################

    # Strategy Specific Variable Storage

    ## Hyperopt Variables

    if use_simpler_custom_stoploss:
        sell_params = {
            "pHSL": -0.068,
            "pPF_1": 0.008,
            "pPF_2": 0.098,
            "pSL_1": 0.02,
            "pSL_2": 0.065,
        }

        # hard stoploss profit
        pHSL = DecimalParameter(-0.200, -0.010, default=-0.08, decimals=3, space='sell', load=True)

        # profit threshold 1, trigger point, SL_1 is used
        pPF_1 = DecimalParameter(0.008, 0.020, default=0.016, decimals=3, space='sell', load=True)
        pSL_1 = DecimalParameter(0.008, 0.020, default=0.011, decimals=3, space='sell', load=True)

        # profit threshold 2, SL_2 is used
        pPF_2 = DecimalParameter(0.040, 0.100, default=0.080, decimals=3, space='sell', load=True)
        pSL_2 = DecimalParameter(0.020, 0.070, default=0.040, decimals=3, space='sell', load=True)

    else:

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
        cstop_bail_time_trend = CategoricalParameter([True, False], default=True, space='sell', load=True,
                                                     optimize=True)
        cstop_max_stoploss = DecimalParameter(-0.30, -0.01, default=-0.10, space='sell', load=True, optimize=True)

    ################################

    # subclasses should override the following 4 functions - this is here as an example

    # Note: try to combine current/historical data (from populate_indicators) with future data
    #       If you only use future data, the ML training is just guessing
    #       Also, try to identify buy/sell ranges, rather than transitions - it gives the algorithms more chances
    #       to find a correlation. The framework will select the first one anyway.
    #       In other words, avoid using qtpylib.crossed_above() and qtpylib.crossed_below()
    #       OK not to check volume, because we are just looking for patterns


    def get_train_buy_signals(self, future_df: DataFrame):
        series = np.where(
            (
                    (future_df['mfi'] <= 30) & # loose guard
                    (future_df['dwt_gain'] <= future_df['future_loss_threshold']) &  # loss exceeds threshold

                    (future_df['future_profit_max'] >= future_df['future_profit_threshold']) & # future profit exceeds threshold
                    (future_df['future_max'] > future_df['dwt_recent_max']) # future window max exceeds prior window max
            ), 1.0, 0.0)

        return series

    def get_train_sell_signals(self, future_df: DataFrame):
        series = np.where(
            (
                    (future_df['mfi'] >= 70) & # loose guard
                    (future_df['dwt_gain'] >= future_df['future_profit_threshold']) & # profit exceeds threshold

                    (future_df['future_loss_min'] <= future_df['future_loss_threshold']) & # future loss exceeds threshold
                    (future_df['future_min'] < future_df['dwt_recent_min']) # future window max exceeds prior window max
            ), 1.0, 0.0)

        return series


    # provide additional buy/sell signals
    # These usually repeat the (backward-looking) guards from the training signal criteria. This helps filter out
    # bad predictions (Machine Learning is not perfect)


    def get_strategy_entry_guard_conditions(self, dataframe: DataFrame):

        # buys = None
        buys = np.where(
            (
                    (dataframe['mfi'] <= 40) #&
                    # (dataframe['dwt_loss'] <= dataframe['loss_threshold'])   #  loss exceeds threshold
            ), 1.0, 0.0)

        return buys

    def get_strategy_exit_guard_conditions(self, dataframe: DataFrame):

        # sells = None

        sells = np.where(
            (
                    (dataframe['mfi'] >= 60) &
                    (dataframe['dwt_profit'] >= dataframe['profit_threshold'])  # profit exceeds threshold
            ), 1.0, 0.0)

        return sells

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

        self.set_state(curr_pair, self.State.POPULATE)
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

        # populate the normal dataframe
        dataframe = self.dataframePopulator.add_indicators(dataframe)
        # dataframe = self.add_indicators(dataframe)

        if Anomaly.first_time:
            Anomaly.first_time = False
            print("")
            print("***************************************")
            print("** Warning: startup can be very slow **")
            print("***************************************")

            print("    Lookahead: ", self.curr_lookahead, " candles (", self.lookahead_hours, " hours)")

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

    # add indicators used by stoploss/custom sell logic
    def add_stoploss_indicators(self, dataframe, pair) -> DataFrame:
        if not pair in self.custom_trade_info:
            self.custom_trade_info[pair] = {}
            if not 'had_trend' in self.custom_trade_info[pair]:
                self.custom_trade_info[pair]['had_trend'] = False

        dataframe = self.dataframePopulator.add_stoploss_indicators(dataframe)

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
        future_df['train_buy'] = self.get_train_buy_signals(future_df)
        future_df['train_sell'] = self.get_train_sell_signals(future_df)

        buys = future_df['train_buy'].copy()
        if buys.sum() < 3:
            print("OOPS! <3 ({:.0f}) buy signals generated. Check training criteria".format(buys.sum()))

        sells = future_df['train_sell'].copy()
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

    # empty func. Meant to be overridden by subclass
    def save_debug_indicators(self, future_df: DataFrame):
        pass
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

        if not self.ignore_exit_signals:
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
        df_test, df_train = self.dataframeUtils.split_dataframe(full_df_norm, (1.0 - train_ratio))
        test_buys, train_buys = self.dataframeUtils.split_array(buys, (1.0 - train_ratio))
        test_sells, train_sells = self.dataframeUtils.split_array(sells, (1.0 - train_ratio))

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
        if not self.ignore_exit_signals:
            self.sell_classifier.train(df_train, df_test, train_sells, test_sells, force_train=force_train)


        # if scan specified, test against the test dataframe
        if self.dbg_test_classifier:

            if not (self.buy_classifier is None):
                pred_buys = self.buy_classifier.predict(df_test)
                print("")
                print("Testing - Buy Signals (", type(self.buy_classifier).__name__, ")")
                print(classification_report(test_buy_labels, pred_buys))
                print("")

            if not (self.sell_classifier is None):
                pred_sells = self.sell_classifier.predict(df_test)
                print("")
                print("Testing - Sell Signals (", type(self.sell_classifier).__name__, ")")
                print(classification_report(test_sell_labels, pred_sells))
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

        # # ATR in buy range
        # conditions.append(dataframe['atr_signal'] > 0.0)

        # some trading volume
        conditions.append(dataframe['volume'] > 0)

        # # Fisher RSI + Williams combo
        # conditions.append(dataframe['fisher_wr'] < -0.7)

        # MFI
        # conditions.append(dataframe['mfi'] < 30.0)

        # # below TEMA
        # conditions.append(dataframe['close'] < dataframe['tema'])

        # add strategy-specific conditions (from subclass)
        strat_cond = self.get_strategy_entry_guard_conditions(dataframe)
        if strat_cond is not None:
            conditions.append(strat_cond)

        # # sell signal is not active
        # conditions.append(dataframe['predict_sell'] < 0.1)

        # PCA/Classifier triggers
        anomaly_cond = (
            (qtpylib.crossed_above(dataframe['predict_buy'], 0.5))
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

        if not self.dp.runmode.value in ('hyperopt'):
            if Anomaly.first_run:
                Anomaly.first_run = False  # note use of clas variable, not instance variable
                # self.show_debug_info(curr_pair)
                self.show_all_debug_info()

        if self.ignore_exit_signals:
            dataframe['exit_long'] = 0
            return dataframe

        # conditions.append(dataframe['volume'] > 0)

        # # ATR in sell range
        # conditions.append(dataframe['atr_signal'] <= 0.0)

        # # above Bollinger mid-point
        # conditions.append(dataframe['close'] > dataframe['bb_middleband'])

        # # Fisher RSI + Williams combo
        # conditions.append(dataframe['fisher_wr'] > 0.5)

        # MFI
        # conditions.append(dataframe['mfi'] > 70.0)

        # add strategy-specific conditions (from subclass)
        strat_cond = self.get_strategy_exit_guard_conditions(dataframe)
        if strat_cond is not None:
            conditions.append(strat_cond)

        # PCA triggers
        anomaly_cond = (
            qtpylib.crossed_above(dataframe['predict_sell'], 0.5)
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

    def custom_stoploss(self, pair: str, trade: 'Trade', current_time: datetime, current_rate: float,
                        current_profit: float, **kwargs) -> float:

        if self.use_simpler_custom_stoploss:
            return self.simpler_custom_stoploss(pair, trade, current_time, current_rate, current_profit)
        else:
            return self.complex_custom_stoploss(pair, trade, current_time, current_rate, current_profit)

    def complex_custom_stoploss(self, pair: str, trade: 'Trade', current_time: datetime, current_rate: float,
                                current_profit: float) -> float:

        # self.set_state(pair, self.State.STOPLOSS)

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

    ## Custom Trailing stoploss ( credit to Perkmeister for this custom stoploss to help the strategy ride a green candle )
    def simpler_custom_stoploss(self, pair: str, trade: 'Trade', current_time: datetime,
                                current_rate: float, current_profit: float) -> float:

        # hard stoploss profit
        HSL = self.pHSL.value
        PF_1 = self.pPF_1.value
        SL_1 = self.pSL_1.value
        PF_2 = self.pPF_2.value
        SL_2 = self.pSL_2.value

        # For profits between PF_1 and PF_2 the stoploss (sl_profit) used is linearly interpolated
        # between the values of SL_1 and SL_2. For all profits above PL_2 the sl_profit value
        # rises linearly with current profit, for profits below PF_1 the hard stoploss profit is used.

        if (current_profit > PF_2):
            sl_profit = SL_2 + (current_profit - PF_2)
        elif (current_profit > PF_1):
            sl_profit = SL_1 + ((current_profit - PF_1) * (SL_2 - SL_1) / (PF_2 - PF_1))
        else:
            sl_profit = HSL

        # Only for hyperopt invalid return
        if (sl_profit >= current_profit):
            return -0.99

        return min(-0.01, max(stoploss_from_open(sl_profit, current_profit), -0.99))

        # if current_profit < 0.02:
        #     return -1  # return a value bigger than the initial stoploss to keep using the initial stoploss

        # # After reaching the desired offset, allow the stoploss to trail by half the profit
        # desired_stoploss = current_profit / 4
        #
        # # Use a minimum of 0.5% and a maximum of 10%
        # return max(min(desired_stoploss, 0.10), 0.005)

    ###################################

    """
    Custom Sell
    """

    def custom_exit(self, pair: str, trade: 'Trade', current_time: 'datetime', current_rate: float,
                    current_profit: float, **kwargs):

        # Mod: just take the profit:

        dataframe, _ = self.dp.get_analyzed_dataframe(pair=pair, timeframe=self.timeframe)
        last_candle = dataframe.iloc[-1].squeeze()

        # Above 3%, sell if MFA > 90
        if current_profit > 0.03:
            if last_candle['mfi'] > 90:
                return 'mfi_90'

        # Mod: strong sell signal, in profit
        if (current_profit > 0) and (last_candle['fisher_wr'] > 0.98):
                return 'fwr_98'

        # Mod: Sell any positions at a loss if they are held for more than two days.
        if current_profit < 0.0 and (current_time - trade.open_date_utc).days >= 2:
            return 'unclog'

        if self.use_simpler_custom_stoploss:
            return self.simpler_custom_exit(pair, trade, current_time, current_rate, current_profit)
        else:
            return self.complex_custom_exit(pair, trade, current_time, current_rate, current_profit)

    def complex_custom_exit(self, pair: str, trade: 'Trade', current_time: 'datetime', current_rate: float,
                            current_profit: float):

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

        # Mod: strong sell signal, in profit
        if (current_profit > 0) and (last_candle['fisher_wr'] > 0.98):
                return 'fwr_98'

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

    def simpler_custom_exit(self, pair: str, trade: 'Trade', current_time: 'datetime', current_rate: float,
                            current_profit: float):
        dataframe, _ = self.dp.get_analyzed_dataframe(pair=pair, timeframe=self.timeframe)
        last_candle = dataframe.iloc[-1].squeeze()

        # Above 5% profit, sell
        if current_profit > 0.05:
            return 'profit_5'

        # Above 2%, sell if MFA > 90
        if current_profit > 0.02:
            if last_candle['mfi'] > 90:
                return 'mfi_90'

        # Mod: strong sell signal, in profit
        if (current_profit > 0) and (last_candle['fisher_wr'] > 0.98):
                return 'fwr_98'

        # Sell any positions at a loss if they are held for more than one day.
        if current_profit < 0.0 and (current_time - trade.open_date_utc).days >= 2:
            return 'unclog'

        return None


#######################
