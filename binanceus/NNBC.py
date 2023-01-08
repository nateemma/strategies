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

import tensorflow as tf

seed = 42
os.environ['PYTHONHASHSEED'] = str(seed)
random.seed(seed)
tf.random.set_seed(seed)
np.random.seed(seed)

tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.WARN)

import keras
from keras import layers
from tqdm import tqdm
import Attention
import RBM


from DataframeUtils import DataframeUtils
from DataframePopulator import DataframePopulator

from NNBClassifier_MLP import NNBClassifier_MLP
from NNBClassifier_MLP2 import NNBClassifier_MLP2
from NNBClassifier_LSTM import NNBClassifier_LSTM
from NNBClassifier_LSTM2 import NNBClassifier_LSTM2
from NNBClassifier_Attention import NNBClassifier_Attention
from NNBClassifier_Multihead import NNBClassifier_Multihead
from NNBClassifier_Transformer import NNBClassifier_Transformer
from NNBClassifier_RBM import NNBClassifier_RBM

"""
####################################################################################
NNBC - Neural Net Binary Classifier
    Combines Dimensionality Reduction using Principal Component Analysis (PCA) and various
    Neural Networks set up as binary classifiers.
      
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


class NNBC(IStrategy):

    plot_config = {
        'main_plot': {
            'close': {'color': 'cornflowerblue'},
        },
        'subplots': {
            "Diff": {
                '%train_buy': {'color': 'mediumaquamarine'},
                'predict_buy': {'color': 'cornflowerblue'},
                '%train_sell': {'color': 'salmon'},
                'predict_sell': {'color': 'orange'},
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
    process_only_new_candles = True

    #------------------------------
    # Strategy-specific global vars

    inf_mins = timeframe_to_minutes(inf_timeframe)
    data_mins = timeframe_to_minutes(timeframe)
    inf_ratio = int(inf_mins / data_mins)

    # These parameters control much of the behaviour because they control the generation of the training data
    # Unfortunately, these cannot be hyperopt params because they are used in populate_indicators, which is only run
    # once during hyperopt
    lookahead_hours = 0.5
    n_profit_stddevs = 1.0
    n_loss_stddevs = 1.0
    min_f1_score = 0.5

    compressor = None
    compress_data = True
    classifier_name = 'Transformer'  # select based on testing
    # classifier_name = 'Multihead'  # select based on testing
    # classifier_name = 'LSTM'  # for debug
    buy_classifier = None
    sell_classifier = None

    curr_lookahead = int(12 * lookahead_hours)

    curr_pair = ""
    custom_trade_info = {}

    # the following affect training of the model. Bigger numbers give better results, but take longer and use more memory
    seq_len = 8  # 'depth' of training sequence
    num_epochs = 512  # number of iterations for training
    batch_size = 1024  # batch size for training

    refit_model = False  # only set to True when training. If False, then existing model is used, if present
    use_full_dataset = True  # use the entire dataset for training (in backtest)

    dataframeUtils = None
    dataframePopulator = None

    buy_tag = 'Buy'
    sell_tag = 'Sell'

    dwt_window = startup_candle_count

    num_pairs = 0
    # pair_model_info = {}  # holds model-related info for each pair
    # classifier_stats = {}  # holds statistics for each type of classifier (useful to rank classifiers

    # debug flags
    first_time = True  # mostly for debug
    first_run = True  # used to identify first time through buy/sell populate funcs

    dbg_scan_classifiers = True  # if True, scan all viable classifiers and choose the best. Very slow!
    dbg_test_classifier = True  # test clasifiers after fitting
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

    #  hyperparams
    # buy_gain = IntParameter(1, 50, default=4, space='buy', load=True, optimize=True)
    #
    # sell_gain = IntParameter(-1, -15, default=-4, space='sell', load=True, optimize=True)

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

    # subclasses should oiverride the following 2 functions - this is here as an example

    # Note: try to combine current/historical data (from populate_indicators) with future data
    #       If you only use future data, the ML training is just guessing
    #       Also, try to identify buy/sell ranges, rather than transitions - it gives the algorithms more chances
    #       to find a correlation. The framework will select the first one anyway.
    #       In other words, avoid using qtpylib.crossed_above() and qtpylib.crossed_below()
    #       Proably OK not to check volume, because we are just looking for patterns

    def get_train_buy_signals(self, future_df: DataFrame):

        print("!!! WARNING: using base class (buy) training implementation !!!")

        series = np.where(
            (
                # (future_df['mfi'] <= 30) &  # loose oversold threshold
                (future_df['future_gain'] >= future_df['profit_threshold'])  # future gain above threshold
            ), 1.0, 0.0)

        return series

    def get_train_sell_signals(self, future_df: DataFrame):

        print("!!! WARNING: using base class (sell) training implementation !!!")

        series = np.where(
            (
                # (future_df['mfi'] >= 70) &  # loose overbought threshold
                (future_df['future_gain'] <= future_df['loss_threshold'])  # future loss above threshold
            ), 1.0, 0.0)

        return series

    # override the following to add strategy-specific criteria to the (main) buy/sell conditions

    def get_strategy_buy_conditions(self, dataframe: DataFrame):
        return None

    def get_strategy_sell_conditions(self, dataframe: DataFrame):
        return None

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

        if self.first_time:
            self.first_time = False
            print("")
            print("***************************************")
            print("** Warning: startup can be very slow **")
            print("***************************************")

            print("    Lookahead: ", self.curr_lookahead, " candles (", self.lookahead_hours, " hours)")


        print("")
        print(curr_pair)

        # make sure we only retrain in backtest modes
        if self.dp.runmode.value not in ('backtest'):
            self.refit_model = False

        # populate the normal dataframe
        dataframe = self.dataframePopulator.add_indicators(dataframe)

        buys, sells = self.create_training_data(dataframe)

        # # drop last group (because there cannot be a prediction)
        # df = dataframe.iloc[:-self.curr_lookahead]
        # buys = buys.iloc[:-self.curr_lookahead]
        # sells = sells.iloc[:-self.curr_lookahead]

        # Principal Component Analysis of inf data

        # train the models on the informative data
        if self.dbg_verbose:
            print("    training models...")
        self.train_models(curr_pair, dataframe, buys, sells)
        # add predictions

        if self.dbg_verbose:
            print("    running predictions...")

        # get predictions (Note: do not modify dataframe between calls)
        pred_buys = self.predict_buy(dataframe, curr_pair)
        pred_sells = self.predict_sell(dataframe, curr_pair)
        dataframe['predict_buy'] = pred_buys
        dataframe['predict_sell'] = pred_sells

        # Custom Stoploss
        if self.dbg_verbose:
            print("    updating stoploss data...")
        self.add_stoploss_indicators(dataframe, curr_pair)

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
        if sells.sum() < 3:
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
            'future_profit_min', 'future_profit_max', 'profit_threshold',
            'future_loss_min', 'future_loss_max', 'loss_threshold',
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
        return pd.DataFrame(self.compressor.transform(dataframe))


    # train the PCA reduction and classification models

    def train_models(self, curr_pair, dataframe: DataFrame, buys, sells):

        if self.dp.runmode.value not in ('backtest'):
            return

        # check input - need at least 2 samples or classifiers will not train
        if buys.sum() < 2:
            print("*** ERR: insufficient buys in expected results. Check training data")
            # print(buys)
            return

        # if sells.sum() < 2:
        #     print("*** ERR: insufficient sells in expected results. Check training data")
        #     return

        rand_st = 27  # use fixed number for reproducibility

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
            data_size = int(0.9 * full_df_norm.shape[0])
        else:
            data_size = int(min(975, full_df_norm.shape[0]))


        # create classifiers, if necessary
        num_features = full_df_norm.shape[1]
        if self.buy_classifier is None:
            self.buy_classifier, _ = self.classifier_factory(self.classifier_name, num_features, tag=self.buy_tag)
        if self.sell_classifier is None:
            self.sell_classifier, _ = self.classifier_factory(self.classifier_name, num_features, tag=self.sell_tag)


        # get training dataset
        # Note: this returns tensors, not dataframes

        v_tensor, v_buys, v_sells = self.dataframeUtils.build_standard_dataset(data_size, full_df_norm, buys, sells,
                                                                          self.curr_lookahead, self.seq_len)

        tsr_train, tsr_test, train_buys, test_buys, \
        train_sells, test_sells, = self.dataframeUtils.split_tensor(v_tensor,
                                                               v_buys,
                                                               v_sells,
                                                               0.8,
                                                               self.curr_lookahead)

        num_buys = int(train_buys[:, 0].sum())
        num_sells = int(train_sells[:, 0].sum())

        if self.dbg_verbose:
            # print("     df_norm:", df_norm.shape, ' v_tensor:', v_tensor.shape)
            print("     tensor:", v_tensor.shape, ' -> train:', tsr_train.shape, " + test:", tsr_test.shape)
            print("     buys:", v_buys.shape, ' -> train:', train_buys.shape, " + test:", test_buys.shape)
            print("     sells:", v_sells.shape, ' -> train:', train_sells.shape, " + test:", test_sells.shape)

        print("    #training samples:", len(tsr_train), " #buys:", num_buys, ' #sells:', num_sells)

        # TODO: if low number of buys/sells, try k-fold sampling

        buy_labels = self.dataframeUtils.get_binary_labels(buys)
        sell_labels = self.dataframeUtils.get_binary_labels(sells)
        # train_buy_labels = self.dataframeUtils.get_binary_labels(train_buys)
        # train_sell_labels = self.dataframeUtils.get_binary_labels(train_sells)
        # test_buy_labels = self.dataframeUtils.get_binary_labels(test_buys)
        # test_sell_labels = self.dataframeUtils.get_binary_labels(test_sells)

        train_buy_labels = train_buys
        train_sell_labels = train_sells
        test_buy_labels = test_buys
        test_sell_labels = test_sells

        # Create buy/sell classifiers for the model

        # check that we have enough positives to train
        buy_ratio = 100.0 * (num_buys / len(train_buys))
        # if (buy_ratio < 0.5):
        #     print("*** ERR: insufficient number of positive buy labels ({:.2f}%)".format(buy_ratio))
        #     return

        buy_clf, buy_clf_name = self.get_buy_classifier(tsr_train, train_buy_labels, tsr_test, test_buy_labels)

        sell_ratio = 100.0 * (num_sells / len(train_sells))
        # if (sell_ratio < 0.5):
        #     print("*** ERR: insufficient number of positive sell labels ({:.2f}%)".format(sell_ratio))
        #     return

        sell_clf, sell_clf_name = self.get_sell_classifier(tsr_train, train_sell_labels, tsr_test, test_sell_labels)

        # save the models
        self.buy_classifier = buy_clf
        self.sell_classifier = sell_clf

        # if scan specified, test against the test dataframe
        if self.dbg_test_classifier:

            if not (buy_clf is None):
                pred_buys = self.get_classifier_predictions(buy_clf, tsr_test)
                print("")
                print("Testing Buy Classifier (", buy_clf_name, ")")
                print(classification_report(test_buy_labels[:, 0], pred_buys))
                print("")

            if not (sell_clf is None):
                pred_sells = self.get_classifier_predictions(sell_clf, tsr_test)
                print("")
                print("Testing Sell Classifier (", sell_clf_name, ")")
                print(classification_report(test_sell_labels[:, 0], pred_sells))
                print("")

        return

    # get a classifier for the supplied normalised dataframe and known results
    def get_buy_classifier(self, tensor, results, test_tensor, test_labels):

        clf = None
        name = self.classifier_name

        # labels = self.get_binary_labels(results)
        labels = results

        # if results.sum() <= 2:
        #     print("***")
        #     print("*** ERR: insufficient positive results in buy data")
        #     print("***")
        #     return clf, name

        if self.dp.runmode.value in ('backtest'):
            # If already done, just  re-fit
            if self.buy_classifier:
                clf = self.fit_classifier(self.buy_classifier, name, self.buy_tag, tensor, labels, test_tensor, test_labels)
            else:
                num_features = np.shape(tensor)[2]
                clf, name = self.classifier_factory(name, num_features, tag=self.buy_tag)
                clf = self.fit_classifier(clf, name, self.buy_tag, tensor, labels, test_tensor, test_labels)

        return clf, name

    # get a classifier for the supplied normalised dataframe and known results
    def get_sell_classifier(self, tensor, results, test_tensor, test_labels):

        clf = None
        name = self.classifier_name
        # labels = self.get_binary_labels(results)
        labels = results

        # if results.sum() <= 2:
        #     print("***")
        #     print("*** ERR: insufficient positive results in sell data")
        #     print("***")
        #     return clf, name

        if self.dp.runmode.value in ('backtest'):
            # If already done, just  re-fit
            if self.sell_classifier:
                clf = self.fit_classifier(self.sell_classifier, name, self.sell_tag, tensor, labels, test_tensor, test_labels)
            else:
                num_features = np.shape(tensor)[2]
                clf, name = self.classifier_factory(name, num_features, tag=self.buy_tag)
                clf = self.fit_classifier(clf, name, self.sell_tag, tensor, labels, test_tensor, test_labels)

        return clf, name


    #######################################

    def get_compressor(self, df_norm: DataFrame):
        # just use fixed size PCA (easier for classifiers to deal with)
        ncols = 64
        compressor = skd.PCA(n_components=ncols, whiten=True, svd_solver='full').fit(df_norm)
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

        if not isinstance(data, (np.ndarray, np.array)):
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
        # 'MLP', 'LSTM', 'Attention', 'Multihead'
        'MLP', 'MLP2', 'LSTM', 'Multihead', 'Transformer'
    ]

    # factory to create classifier based on name
    def classifier_factory(self, name, nfeatures, tag=""):
        clf = None

        if name == 'MLP':
            clf = NNBClassifier_MLP(self.curr_pair, self.seq_len, nfeatures, tag=tag)
        elif name == 'MLP2':
            clf = NNBClassifier_MLP2(self.curr_pair, self.seq_len, nfeatures, tag=tag)
        elif name == 'LSTM':
            clf = NNBClassifier_LSTM(self.curr_pair, self.seq_len, nfeatures, tag=tag)
        elif name == 'LSTM2':
            clf = NNBClassifier_LSTM2(self.curr_pair, self.seq_len, nfeatures, tag=tag)
        elif name == 'Attention':
            clf = NNBClassifier_Attention(self.curr_pair, self.seq_len, nfeatures, tag=tag)
        elif name == 'Multihead':
            clf = NNBClassifier_Multihead(self.curr_pair, self.seq_len, nfeatures, tag=tag)
        elif name == 'Transformer':
            clf = NNBClassifier_Transformer(self.curr_pair, self.seq_len, nfeatures, tag=tag)
        elif name == 'RBM':
            clf = NNBClassifier_RBM(self.curr_pair, self.seq_len, nfeatures, tag=tag)

        else:
            print("Unknown classifier: ", name)
            clf = None
        return clf, name


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

        # labels = self.get_binary_labels(results)
        labels = results

        # # TODO: replace train_test_spli?!
        #
        # # split into test/train for evaluation, then re-fit once selected
        # # tsr_train, df_test, res_train, res_test = train_test_split(df, results, train_size=0.5)
        # tsr_train, tsr_test, res_train, res_test = train_test_split(tensor, labels, train_size=0.8,
        #                                                             random_state=27, shuffle=False)

        ratio = 0.8
        train_len = int(ratio * np.shape(labels)[0])
        test_len = np.shape(labels)[0] - train_len
        tsr_train = tensor[test_len+1:, :, :]
        tsr_test = tensor[0:test_len:, :, :]
        res_train = labels[test_len+1:, :]
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

        num_features = np.shape(tsr_train)[2]
        for clf_name in self.classifier_list:
            clf, _ = self.classifier_factory(clf_name, num_features, tag=tag)

            if clf is not None:

                # fit to the training data
                clf_dict[clf_name] = clf
                clf = self.fit_classifier(clf, clf_name, tag, tsr_train, res_train, tsr_test, res_test)

                # assess using the test data. Do *not* use the training data for testing
                pred_test = self.get_classifier_predictions(clf, tsr_test)

                # score = f1_score(results, prediction, average=None)[1]
                score = f1_score(res_test[:, 0], pred_test, average='macro')

                if self.dbg_verbose:
                    print("      {0:<20}: {1:.3f}".format(clf_name, score))

                if score > best_score:
                    best_score = score
                    best_classifier = clf_name

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

        # if self.dbg_test_classifier:
        #     # DEBUG: check accuracy
        #     signals = df['train_buy_signal']
        #     labels = self.get_binary_labels(signals)
        #
        #     if  self.dbg_verbose:
        #         print("")
        #         print("Predict - Buy Signals (", type(clf).__name__, ")")
        #         print(classification_report(labels, predict))
        #         print("")
        #
        #     score = f1_score(labels, predict, average='macro')
        #     if score <= 0.5:
        #         print("")
        #         print("!!! WARNING: (buy) F1 score below 51% ({:.3f})".format(score))
        #         print("    Classifier:", type(clf).__name__)
        #         print("")

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

        # if self.dbg_test_classifier:
        #     # DEBUG: check accuracy
        #     signals = df['train_sell_signal']
        #     labels = self.get_binary_labels(signals)
        #
        #     if self.dbg_verbose:
        #         print("")
        #         print("Predict - Sell Signals (", type(clf).__name__, ")")
        #         print(classification_report(labels, predict))
        #         print("")
        #
        #     score = f1_score(labels, predict, average='macro')
        #     if score <= 0.5:
        #         print("")
        #         print("!!! WARNING: (buy) F1 score below 51% ({:.3f})".format(score))
        #         print("    Classifier:", type(clf).__name__)
        #         print("")

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
            if NNBC.first_run:
                NNBC.first_run = False  # note use of clas variable, not instance variable
                # self.show_debug_info(curr_pair)
                self.show_all_debug_info()


        # add some fairly loose guards, to help prevent 'bad' predictions

        # # some trading volume
        # conditions.append(dataframe['volume'] > 0)

        # MFI
        conditions.append(dataframe['mfi'] < 50.0)

        # # above TEMA
        # conditions.append(dataframe['dwt'] < dataframe['tema'])


        # Classifier triggers
        predict_cond = (
            (qtpylib.crossed_above(dataframe['predict_buy'], 0.5))
        )
        conditions.append(predict_cond)

        # add strategy-specific conditions (from subclass)
        strat_cond = self.get_strategy_buy_conditions(dataframe)
        if strat_cond is not None:
            conditions.append(strat_cond)

        # set entry tags
        dataframe.loc[predict_cond, 'enter_tag'] += 'nnbc_entry '

        if conditions:
            dataframe.loc[reduce(lambda x, y: x & y, conditions), 'buy'] = 1
        else:
            dataframe['entry'] = 0

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
            if NNBC.first_run:
                NNBC.first_run = False  # note use of clas variable, not instance variable
                # self.show_debug_info(curr_pair)
                self.show_all_debug_info()

        # # some volume
        # conditions.append(dataframe['volume'] > 0)

        # MFI
        conditions.append(dataframe['mfi'] > 50.0)

        # # below TEMA
        # conditions.append(dataframe['dwt'] > dataframe['tema'])


        # PCA triggers
        predict_cond = (
            qtpylib.crossed_above(dataframe['predict_sell'], 0.5)
        )

        conditions.append(predict_cond)

        # add strategy-specific conditions (from subclass)
        strat_cond = self.get_strategy_sell_conditions(dataframe)
        if strat_cond is not None:
            conditions.append(strat_cond)

        dataframe.loc[predict_cond, 'exit_tag'] += 'nnbc_exit '

        if conditions:
            dataframe.loc[reduce(lambda x, y: x & y, conditions), 'sell'] = 1
        else:
            dataframe['exit'] = 0

        return dataframe

    ###################################

    """
    Custom Stoploss
    """

    def custom_stoploss(self, pair: str, trade: 'Trade', current_time: datetime, current_rate: float,
                        current_profit: float, **kwargs) -> float:

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
