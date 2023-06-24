import operator

import numpy as np
from enum import Enum

import pywt
import talib.abstract as ta
from scipy.ndimage import gaussian_filter1d
from xgboost import XGBClassifier

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
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler
import sklearn.decomposition as skd
from sklearn.svm import SVC, SVR
from sklearn.utils.fixes import loguniform
from sklearn import preprocessing
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier, VotingClassifier, \
    StackingClassifier
from sklearn.naive_bayes import GaussianNB, MultinomialNB
from sklearn.neural_network import MLPClassifier, BernoulliRBM
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis, LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.manifold import LocallyLinearEmbedding

from sklearn.metrics import make_scorer
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.model_selection import cross_validate

import random

from prettytable import PrettyTable

from CompressionAutoEncoder import CompressionAutoEncoder
from RBMEncoder import RBMEncoder


from DataframeUtils import DataframeUtils, ScalerType
from DataframePopulator import DataframePopulator

"""
####################################################################################
PCA - uses Principal Component Analysis to try and reduce the total set of indicators
      to more manageable dimensions, and predict the next gain step.
      
      This works by creating a PCA model of the available technical indicators. This produces a 
      mapping of the indicators and how they affect the outcome (buy/sell/hold). We choose only the
      mappings that have a significant effect and ignore the others. This significantly reduces the size
      of the problem.
      We then train a classifier model to predict buy or sell signals based on the known outcome in the
      informative data, and use it to predict buy/sell signals based on the real-time dataframe.
      
      Note that this is very slow to start up. This is mostly because we have to build the data on a rolling
      basis to avoid lookahead bias.
      
      In addition to the normal freqtrade packages, these strategies also require the installation of:
        random
        prettytable
        finta

####################################################################################
"""


# enum of various classifiers available
class ClassifierType(Enum):
    LogisticRegression = 0
    DecisionTree = 1
    RandomForest = 2
    GaussianNB = 3
    MLP = 4
    IsolationForest = 5
    EllipticEnvelope = 6
    OneClassSVM = 7
    PCA = 8
    GaussianMixture = 9
    KNeighbors = 10
    StochasticGradientDescent = 11
    GradientBoosting = 12
    AdaBoost = 13
    QuadraticDiscriminantAnalysis = 14
    LinearSVC = 15
    GaussianSVC = 16
    PolySVC = 17
    SigmoidSVC = 18
    Voting = 19
    LinearDiscriminantAnalysis = 20
    XGBoost = 21
    Stacking = 22
    

class PCA(IStrategy):

    # default plot config
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
    process_only_new_candles = False

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
    min_f1_score = 0.70

    curr_lookahead = int(12 * lookahead_hours)

    curr_pair = ""
    custom_trade_info = {}

    num_pairs = 0
    pair_model_info = {}  # holds model-related info for each pair
    classifier_stats = {}  # holds statistics for each type of classifier (useful to rank classifiers

    ignore_exit_signals = False # set to True if you don't want to process sell/exit signals (let custom sell do it)

    # debug flags
    first_time = True  # mostly for debug
    first_run = True  # used to identify first time through buy/sell populate funcs

    scaler_type = ScalerType.Robust # scaler type used for normalisation

    dataframeUtils = None
    dataframePopulator = None

    dbg_scan_classifiers = False  # if True, scan all viable classifiers and choose the best. Very slow!
    dbg_test_classifier = True  # test clasifiers after fitting
    dbg_analyse_pca = False  # analyze PCA weights
    dbg_verbose = False  # controls debug output
    dbg_curr_df: DataFrame = None  # for debugging of current dataframe

    # variables to track state
    class State(Enum):
        INIT = 1
        POPULATE = 2
        STOPLOSS = 3
        RUNNING = 4


    # default classifier
    # default_classifier = ClassifierType.LinearDiscriminantAnalysis  # select based on testing
    default_classifier = ClassifierType.Stacking  # select based on testing

    ###################################

    # Strategy Specific Variable Storage

    ## Hyperopt Variables

    # PCA hyperparams
    # buy_pca_gain = IntParameter(1, 50, default=4, space='buy', load=True, optimize=True)
    #
    # sell_pca_gain = IntParameter(-1, -15, default=-4, space='sell', load=True, optimize=True)

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
                    (future_df['mfi'] >= 80) &  # classic oversold threshold
                    (future_df['dwt'] <= future_df['future_min'])  # at min of future window
            ), 1.0, 0.0)

        return series

    def get_train_sell_signals(self, future_df: DataFrame):

        print("!!! WARNING: using base class (sell) training implementation !!!")

        series = np.where(
            (
                    (future_df['mfi'] <= 20) &  # classic overbought threshold
                    (future_df['dwt'] >= future_df['future_max'])  # at max of future window
            ), 1.0, 0.0)

        return series


    # override the following to add strategy-specific criteria to the (main) buy/sell conditions

    def get_strategy_entry_guard_conditions(self, dataframe: DataFrame):
        return None

    def get_strategy_exit_guard_conditions(self, dataframe: DataFrame):
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
            print(self.__class__.__name__)
            print("")
            print("***************************************")
            print("** Warning: startup can be very slow **")
            print("***************************************")

            print("    Lookahead: ", self.curr_lookahead, " candles (", self.lookahead_hours, " hours)")

        print("")
        print(curr_pair)

        # if first time through for this pair, add entry to pair_model_info
        if not (curr_pair in self.pair_model_info):
            self.pair_model_info[curr_pair] = {
                'interval': 0,
                'pca_size': 0,
                'pca': None,
                'clf_buy_name': "",
                'clf_buy': None,
                'clf_sell_name': "",
                'clf_sell': None
            }
        else:
            # decrement interval. When this reaches 0 it will trigger re-fitting of the data
            self.pair_model_info[curr_pair]['interval'] = self.pair_model_info[curr_pair]['interval'] - 1

        # (re-)set the scaler
        self.dataframeUtils.set_scaler_type(self.scaler_type)

        # populate the normal dataframe
        # dataframe = self.add_indicators(dataframe)
        dataframe = self.dataframePopulator.add_indicators(dataframe)

        buys, sells = self.create_training_data(dataframe)

        # # drop last group (because there cannot be a prediction)
        # df = dataframe.iloc[:-self.curr_lookahead]
        # buys = buys.iloc[:-self.curr_lookahead]
        # sells = sells.iloc[:-self.curr_lookahead]

        # Principal Component Analysis of inf data

        # train the models on the informative data
        if self.dbg_verbose:
            print("    training models..")
        self.train_models(curr_pair, dataframe, buys, sells)
        # add predictions

        if self.dbg_verbose:
            print("    running predictions..")

        # get predictions (Note: do not modify dataframe between calls)
        pred_buys = self.predict_buy(dataframe, curr_pair)
        pred_sells = self.predict_sell(dataframe, curr_pair)
        dataframe['predict_buy'] = pred_buys
        dataframe['predict_sell'] = pred_sells

        # Custom Stoploss
        if self.dbg_verbose:
            print("    updating stoploss data..")
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
        if indicator in future_df:
            if not (dbg_indicator in self.dbg_curr_df):
                self.dbg_curr_df[dbg_indicator] = future_df[indicator]

    ###################

    # train the PCA reduction and classification models

    def train_models(self, curr_pair, dataframe: DataFrame, buys, sells):

        # only run if interval reaches 0 (no point retraining every camdle)
        count = self.pair_model_info[curr_pair]['interval']
        if (count > 0):
            self.pair_model_info[curr_pair]['interval'] = count - 1
            return
        else:
            # reset interval to a random number between 1 and the amount of lookahead
            # self.pair_model_info[curr_pair]['interval'] = random.randint(1, self.curr_lookahead)
            self.pair_model_info[curr_pair]['interval'] = random.randint(2, max(32, self.curr_lookahead))

        # Reset models for this pair. Makes it safe to just return on error
        self.pair_model_info[curr_pair]['pca_size'] = 0
        self.pair_model_info[curr_pair]['pca'] = None
        self.pair_model_info[curr_pair]['clf_buy_name'] = ""
        self.pair_model_info[curr_pair]['clf_buy'] = None
        self.pair_model_info[curr_pair]['clf_sell_name'] = ""
        self.pair_model_info[curr_pair]['clf_sell'] = None

        # check input - need at least 2 samples or classifiers will not train
        if buys.sum() < 2:
            print("*** ERR: insufficient buys in expected results. Check training data")
            # print(buys)
            return

        if sells.sum() < 2:
            print("*** ERR: insufficient sells in expected results. Check training data")
            return

        rand_st = 27  # use fixed number for reproducibility

        remove_outliers = False
        if remove_outliers:
            # norm dataframe before splitting, otherwise variances are skewed
            full_df_norm = self.dataframeUtils.norm_dataframe(dataframe)
            full_df_norm, buys, sells = self.dataframeUtils.remove_outliers(full_df_norm, buys, sells)
        else:
            full_df_norm = self.dataframeUtils.norm_dataframe(dataframe).clip(lower=-3.0, upper=3.0)  # supress outliers

        # constrain size to what will be available in run modes
        data_size = int(min(975, full_df_norm.shape[0]))

        # get 'viable' data set (includes all buys/sells)
        v_df_norm, v_buys, v_sells = self.dataframeUtils.build_viable_dataset(data_size, full_df_norm, buys, sells)

        train_size = int(0.8 * data_size)
        test_size = data_size - train_size

        df_train, df_test, train_buys, test_buys, train_sells, test_sells, = train_test_split(v_df_norm,
                                                                                              v_buys,
                                                                                              v_sells,
                                                                                              train_size=train_size,
                                                                                              random_state=rand_st,
                                                                                              shuffle=True)
        if self.dbg_verbose:
            print("     dataframe:", v_df_norm.shape, ' -> train:', df_train.shape, " + test:", df_test.shape)
            print("     buys:", buys.shape, ' -> train:', train_buys.shape, " + test:", test_buys.shape)
            print("     sells:", sells.shape, ' -> train:', train_sells.shape, " + test:", test_sells.shape)

        print("    #training samples:", len(df_train), " #buys:", int(train_buys.sum()), ' #sells:',
              int(train_sells.sum()))

        # TODO: if low number of buys/sells, try k-fold sampling

        buy_labels = self.dataframeUtils.get_binary_labels(buys)
        sell_labels = self.dataframeUtils.get_binary_labels(sells)
        train_buy_labels = self.dataframeUtils.get_binary_labels(train_buys)
        train_sell_labels = self.dataframeUtils.get_binary_labels(train_sells)
        test_buy_labels = self.dataframeUtils.get_binary_labels(test_buys)
        test_sell_labels = self.dataframeUtils.get_binary_labels(test_sells)

        # create the PCA analysis model

        pca = self.get_pca(df_train)

        df_train_pca = DataFrame(pca.transform(df_train))

        # DEBUG:
        # print("")
        print("   ", curr_pair, " - input: ", df_train.shape, " -> pca: ", df_train_pca.shape)

        if df_train_pca.shape[1] <= 1:
            print("***")
            print("** ERR: PCA reduced to 1. Must be training data still in dataframe!")
            print("df_train columns: ", df_train.columns.values)
            print("df_train_pca columns: ", df_train_pca.columns.values)
            print("***")
            return

        # Create buy/sell classifiers for the model

        # check that we have enough positives to train
        buy_ratio = 100.0 * (train_buys.sum() / len(train_buys))
        if (buy_ratio < 0.5):
            print("*** ERR: insufficient number of positive buy labels ({:.2f}%)".format(buy_ratio))
            return

        buy_clf, buy_clf_name = self.get_buy_classifier(df_train_pca, train_buy_labels)

        sell_ratio = 100.0 * (train_sells.sum() / len(train_sells))
        if (sell_ratio < 0.5):
            print("*** ERR: insufficient number of positive sell labels ({:.2f}%)".format(sell_ratio))
            return

        if not self.ignore_exit_signals:
            sell_clf, sell_clf_name = self.get_sell_classifier(df_train_pca, train_sell_labels)

            if self.dbg_verbose:
                print(f'    Classifiers - sell: {buy_clf_name} sell: {sell_clf_name}')
        else:
            sell_clf = None
            sell_clf_name = "None"

        # save the models
        self.pair_model_info[curr_pair]['pca'] = pca
        self.pair_model_info[curr_pair]['pca_size'] = df_train_pca.shape[1]
        self.pair_model_info[curr_pair]['clf_buy_name'] = buy_clf_name
        self.pair_model_info[curr_pair]['clf_buy'] = buy_clf
        self.pair_model_info[curr_pair]['clf_sell_name'] = sell_clf_name
        self.pair_model_info[curr_pair]['clf_sell'] = sell_clf

        # if scan specified, test against the test dataframe
        if self.dbg_test_classifier and self.dbg_verbose:

            df_test_pca = DataFrame(pca.transform(df_test))
            if not (buy_clf is None):
                pred_buys = buy_clf.predict(df_test_pca)
                print("")
                print("Predict - Buy Signals (", type(buy_clf).__name__, ")")
                print(classification_report(test_buy_labels, pred_buys))
                print("")

            if not (sell_clf is None):
                pred_sells = sell_clf.predict(df_test_pca)
                print("")
                print("Predict - Sell Signals (", type(sell_clf).__name__, ")")
                print(classification_report(test_sell_labels, pred_sells))
                print("")

    autoencoder = None

    # get the PCA model for the supplied dataframe (dataframe must be normalised)
    def get_pca(self, df_norm: DataFrame):

        ncols = df_norm.shape[1]  # allow all components to get the full variance matrix
        whiten = True

        # change this variable to select the type of PCA used.
        # Regular old PCA (type 0) seems to perform best, but I leave the others here for reference
        pca_type = 0

        # there are various types of PCA, plus alternatives like ICA and Feature Extraction
        if pca_type == 0:
            pca = skd.PCA(n_components=ncols, whiten=whiten, svd_solver='full').fit(df_norm)
            var_ratios = pca.explained_variance_ratio_

            # if self.dbg_verbose:
            #     print ("PCA variance_ratio: ", pca.explained_variance_ratio_)

            # scan variance and only take if column contributes >x%
            ncols = 0
            var_sum = 0.0
            variance_threshold = 0.999
            # variance_threshold = 0.99
            while ((var_sum < variance_threshold) & (ncols < len(var_ratios))):
                var_sum = var_sum + var_ratios[ncols]
                ncols = ncols + 1

            # if necessary, re-calculate pca with reduced column set
            if (ncols != df_norm.shape[1]):
                # pca = skd.PCA(n_components=ncols, svd_solver="randomized", whiten=True).fit(df_norm)
                pca = skd.PCA(n_components=ncols, whiten=whiten, svd_solver='full').fit(df_norm)

            self.check_pca(pca, df_norm)

            if self.dbg_analyse_pca and self.dbg_verbose:
                self.analyse_pca(pca, df_norm)

        elif pca_type == 1:
            # accurate, but slow
            print("    Using KernelPCA..")
            pca = skd.KernelPCA(n_components=ncols, remove_zero_eig=True).fit(df_norm)
            eigenvalues = pca.eigenvalues_
            # print(var_ratios)
            # Note: eigenvalues are not bounded, so just have to go by min value
            ncols = 0
            val_threshold = 0.5
            while ((eigenvalues[ncols] > val_threshold) & (ncols < len(eigenvalues))):
                ncols = ncols + 1

            # if necessary, re-calculate pca with reduced column set
            if (ncols != df_norm.shape[1]):
                # pca = skd.PCA(n_components=ncols, svd_solver="randomized", whiten=True).fit(df_norm)
                pca = skd.KernelPCA(n_components=ncols, remove_zero_eig=True).fit(df_norm)

        elif pca_type == 2:
            print("    Using FactorAnalysis..")
            # Note: this is SUPER slow, do not recommend using it :-(
            # it is useful to run this once, and check to see which indicators are useful or not
            pca = skd.FactorAnalysis(n_components=ncols).fit(df_norm)
            var_ratios = pca.noise_variance_
            # print(var_ratios)

            # if self.dbg_verbose:
            #     print ("PCA variance_ratio: ", pca.explained_variance_ratio_)

            # scan variance and only take if column contributes >x%
            ncols = 0
            variance_threshold = 0.09
            col_names = df_norm.columns
            for i in range(len(var_ratios)):
                if var_ratios[i] > variance_threshold:
                    c = '#' if (var_ratios[i] > 0.5) else '+'
                    print("    {} ({:.3f}) {:<24}".format(c, var_ratios[i], col_names[i]))
                    ncols = ncols + 1
                else:
                    c = '!' if (var_ratios[i] < 0.05) else '-'
                    print("                                {} ({:.3f}) {:<20}".format(c, var_ratios[i], col_names[i]))

            # if necessary, re-calculate pca with reduced column set
            if (ncols != df_norm.shape[1]):
                # pca = skd.PCA(n_components=ncols, svd_solver="randomized", whiten=True).fit(df_norm)
                pca = skd.FactorAnalysis(n_components=ncols).fit(df_norm)

        elif pca_type == 3:
            # fast, but not as accurate
            print("    Using Locally Linear Embedding..")
            pca = LocallyLinearEmbedding(n_components=4, eigen_solver='dense', method="modified").fit(df_norm)

        elif pca_type == 4:
            # a bit slow, still debugging..
            print("    Using Autoencoder..")
            # pca = LocallyLinearEmbedding(n_components=4, eigen_solver='dense', method="modified").fit(df_norm)
            if self.autoencoder is None:
                # self.autoencoder = AutoEncoder(df_norm.shape[1])
                self.autoencoder = CompressionAutoEncoder(df_norm.shape[1], tag="Buy")
            pca = self.autoencoder

        elif pca_type == 5:
            # Restricted Boltzmann Machine
            print("    Using Restricted Boltzmann Machine..")
            pca = RBMEncoder()


        else:
            print("*** ERR - unknown PCA type ***")
            pca = None

        return pca

    # does a quick for suspicious values. Separate func because we always want to call this
    def check_pca(self, pca, df):

        ratios = pca.explained_variance_ratio_
        loadings = pd.DataFrame(pca.components_.T, index=df.columns.values)

        # check variance ratios
        var_big = np.where(ratios >= 0.5)[0]
        if len(var_big) > 0:
            # print("    !!! high variance in columns: ", var_big)
            print("    !!! high variance in columns: ", df.columns.values[var_big])
            # print("    !!! variances: ", ratios)

        var_0 = np.where(ratios == 0)[0]
        if len(var_0) > 0:
            print("    !!! zero variance in columns: ", var_0)

        # check PCA rows
        inf_rows = loadings[(np.isinf(loadings)).any(axis=1)].index.values.tolist()

        if len(inf_rows) > 0:
            print("    !!! inf values in rows: ", inf_rows)

        na_rows = loadings[loadings.isna().any(axis=1)].index.values.tolist()
        if len(na_rows) > 0:
            print("    !!! na values in rows: ", na_rows)

        zero_rows = loadings[(loadings == 0).any(axis=1)].index.values.tolist()
        if len(zero_rows) > 0:
            print("    !!! No contribution from indicator(s) - remove ?! : ", zero_rows)

        return

    def analyse_pca(self, pca, df):
        print("")
        print("Variance Ratios:")
        ratios = pca.explained_variance_ratio_
        print(ratios)
        print("")

        # print matrix of weightings for selected components
        loadings = pd.DataFrame(pca.components_.T, index=df.columns.values)

        l2 = loadings.abs()
        l3 = loadings.mul(ratios)
        ranks = loadings.rank()

        loadings['Score'] = l2.sum(axis=1)
        loadings['Score0'] = loadings[loadings.columns.values[0]].abs()
        loadings['Rank'] = loadings['Score'].rank(ascending=False)
        loadings['Rank0'] = loadings['Score0'].rank(ascending=False)
        print("Loadings, by PC0:")
        print(loadings.sort_values('Rank0').head(n=30))
        print("")
        # print("Loadings, by All Columns:")
        # print(loadings.sort_values('Rank').head(n=30))
        # print("")

        # weighted by variance ratios
        l3a = l3.abs()
        l3['Score'] = l3a.sum(axis=1)
        l3['Rank'] = loadings['Score'].rank(ascending=False)
        print("Loadings, Weighted by Variance Ratio")
        print(l3.sort_values('Rank').head(n=20))

        # # rankings per column
        ranks['Score'] = ranks.sum(axis=1)
        ranks['Rank'] = ranks['Score'].rank(ascending=True)
        print("Rankings per column")
        print(ranks.sort_values('Rank', ascending=True).head(n=30))

        # print(loadings.head())
        # print(l3.head())

    # get a classifier for the supplied dataframe (normalised) and known results
    def get_buy_classifier(self, df_norm: DataFrame, results):

        clf = None
        name = ""
        labels = self.dataframeUtils.get_binary_labels(results)

        if results.sum() <= 2:
            print("***")
            print("*** ERR: insufficient positive results in buy data")
            print("***")
            return clf, name

        # If already done, just get previous result and re-fit
        if self.pair_model_info[self.curr_pair]['clf_buy']:
            clf = self.pair_model_info[self.curr_pair]['clf_buy']
            clf = clf.fit(df_norm, labels)
            name = self.pair_model_info[self.curr_pair]['clf_buy_name']
        else:
            if self.dbg_scan_classifiers:
                if self.dbg_verbose:
                    print("    Finding best buy classifier:")
                clf, name = self.find_best_classifier(df_norm, labels, tag="buy")
            else:
                clf, name = self.classifier_factory(self.default_classifier, df_norm, labels)
                clf = clf.fit(df_norm, labels)

        return clf, name

    # get a classifier for the supplied dataframe (normalised) and known results
    def get_sell_classifier(self, df_norm: DataFrame, results):

        clf = None
        name = ""
        labels = self.dataframeUtils.get_binary_labels(results)

        if results.sum() <= 2:
            print("***")
            print("*** ERR: insufficient positive results in sell data")
            print("***")
            return clf, name

        # If already done, just get previous result and re-fit
        if self.pair_model_info[self.curr_pair]['clf_sell']:
            clf = self.pair_model_info[self.curr_pair]['clf_sell']
            clf = clf.fit(df_norm, labels)
            name = self.pair_model_info[self.curr_pair]['clf_sell_name']
        else:
            if self.dbg_scan_classifiers:
                if self.dbg_verbose:
                    print("    Finding best sell classifier:")
                clf, name = self.find_best_classifier(df_norm, labels, tag="sell")
            else:
                clf, name = self.classifier_factory(self.default_classifier, df_norm, labels)
                clf = clf.fit(df_norm, labels)

        return clf, name


    # list of potential classifier types - set to the list that you want to compare
    classifier_list = [
        ClassifierType.LogisticRegression, ClassifierType.GaussianNB,
        ClassifierType.StochasticGradientDescent, ClassifierType.GradientBoosting,
        ClassifierType.AdaBoost, ClassifierType.LinearSVC, ClassifierType.SigmoidSVC,
        ClassifierType.LinearDiscriminantAnalysis, ClassifierType.XGBoost, ClassifierType.Stacking
    ]

    # factory to create classifier based on name
    def classifier_factory(self, name, data, labels):
        clf = None

        if name == ClassifierType.LogisticRegression:
            clf = LogisticRegression(max_iter=10000)
        elif name == ClassifierType.DecisionTree:
            clf = DecisionTreeClassifier()
        elif name == ClassifierType.RandomForest:
            clf = RandomForestClassifier()
        elif name == ClassifierType.GaussianNB:
            clf = GaussianNB()
        elif name == ClassifierType.MLP:
            clf = MLPClassifier(hidden_layer_sizes=(64, 32, 1),
                                max_iter=50,
                                activation='relu',
                                learning_rate='adaptive',
                                alpha=1e-5,
                                solver='adam',
                                verbose=0)

        elif name == ClassifierType.KNeighbors:
            clf = KNeighborsClassifier(n_neighbors=3)
        elif name == ClassifierType.StochasticGradientDescent:
            clf = SGDClassifier()
        elif name == ClassifierType.GradientBoosting:
            clf = GradientBoostingClassifier()
        elif name == ClassifierType.AdaBoost:
            clf = AdaBoostClassifier()
        elif name == ClassifierType.LinearSVC:
            clf = LinearSVC(dual=False)
        elif name == ClassifierType.GaussianSVC:
            clf = SVC(kernel='rbf')
        elif name == ClassifierType.PolySVC:
            clf = SVC(kernel='poly')
        elif name == ClassifierType.SigmoidSVC:
            clf = SVC(kernel='sigmoid')
        elif name == ClassifierType.Voting:
            # choose 4 decent classifiers
            c1, _ = self.classifier_factory(ClassifierType.LinearDiscriminantAnalysis, data, labels)
            c2, _ = self.classifier_factory(ClassifierType.SigmoidSVC, data, labels)
            c3, _ = self.classifier_factory(ClassifierType.XGBoost, data, labels)
            c4, _ = self.classifier_factory(ClassifierType.AdaBoost, data, labels)
            clf = VotingClassifier(estimators=[('c1', c1), ('c2', c2), ('c3', c3), ('c4', c4)], voting='hard')
        elif name == ClassifierType.LinearDiscriminantAnalysis:
            clf = LinearDiscriminantAnalysis()
        elif name == ClassifierType.QuadraticDiscriminantAnalysis:
            clf = QuadraticDiscriminantAnalysis()
        elif name == ClassifierType.XGBoost:
            clf = XGBClassifier()
        elif name == ClassifierType.Stacking:
            # Stacked 'ensemble' of classifiers
            c1, _ = self.classifier_factory(ClassifierType.LinearDiscriminantAnalysis, data, labels)
            c2, _ = self.classifier_factory(ClassifierType.LinearSVC, data, labels)
            c3, _ = self.classifier_factory(ClassifierType.StochasticGradientDescent, data, labels)
            c4, _ = self.classifier_factory(ClassifierType.LogisticRegression, data, labels)
            c5, _ = self.classifier_factory(ClassifierType.AdaBoost, data, labels)
            estimators = [('c1', c1), ('c2', c2), ('c3', c3), ('c4', c4), ('c5', c5)]
            clf = StackingClassifier(estimators=estimators,
                                     final_estimator=LogisticRegression())
        else:
            print("Unknown classifier: ", name)
            clf = None
        return clf, name

    # tries different types of classifiers and returns the best one
    # tag parameter identifies where to save performance stats (default is not to save)
    def find_best_classifier(self, df, results, tag=""):

        if self.dbg_verbose:
            print("      Evaluating classifiers..")

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

        labels = self.dataframeUtils.get_binary_labels(results)

        # split into test/train for evaluation, then re-fit once selected
        # df_train, df_test, res_train, res_test = train_test_split(df, results, train_size=0.5)
        df_train, df_test, res_train, res_test = train_test_split(df, labels, train_size=0.8,
                                                                  random_state=27, shuffle=True)
        # print("df_train:",  df_train.shape, " df_test:", df_test.shape,
        #       "res_train:", res_train.shape, "res_test:", res_test.shape)

        # check there are enough training samples
        # TODO: if low train/test samples, use k-fold sampling nstead
        if res_train.sum() < 2:
            print("    Insufficient +ve (train) results to fit: ", res_train.sum())
            return None, ""

        if res_test.sum() < 2:
            print("    Insufficient +ve (test) results: ", res_test.sum())
            return None, ""

        for cname in self.classifier_list:
            clf, _ = self.classifier_factory(cname, df_train, res_train)

            if clf is not None:

                # fit to the training data
                clf_dict[cname] = clf
                clf = clf.fit(df_train, res_train)

                # assess using the test data. Do *not* use the training data for testing
                pred_test = clf.predict(df_test)
                # score = f1_score(results, prediction, average=None)[1]
                score = f1_score(res_test, pred_test, average='macro')

                if self.dbg_verbose:
                    print("      {0:<20}: {1:.3f}".format(cname, score))

                if score > best_score:
                    best_score = score
                    best_classifier = cname

                # update classifier stats
                if tag:
                    if not (tag in self.classifier_stats):
                        self.classifier_stats[tag] = {}

                    if not (cname in self.classifier_stats[tag]):
                        self.classifier_stats[tag][cname] = {'count': 0, 'score': 0.0, 'selected': 0}

                    curr_count = self.classifier_stats[tag][cname]['count']
                    curr_score = self.classifier_stats[tag][cname]['score']
                    self.classifier_stats[tag][cname]['count'] = curr_count + 1
                    self.classifier_stats[tag][cname]['score'] = (curr_score * curr_count + score) / (curr_count + 1)

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

        # update stats for selected classifier
        if tag:
            if best_classifier in self.classifier_stats[tag]:
                self.classifier_stats[tag][best_classifier]['selected'] = self.classifier_stats[tag][best_classifier] \
                                                                              ['selected'] + 1

        print("       ", tag, " model selected: ", best_classifier, " Score:{:.3f}".format(best_score))
        # print("")

        return clf, best_classifier

    # make predictions for supplied dataframe (returns column)
    def predict(self, dataframe: DataFrame, pair, clf):

        # predict = 0
        predict = None

        pca = self.pair_model_info[pair]['pca']

        if clf:
            # print("    predicting.. - dataframe:", dataframe.shape)
            df_norm = self.dataframeUtils.norm_dataframe(dataframe)
            df_norm_pca = pca.transform(df_norm)
            predict = clf.predict(df_norm_pca)

        else:
            print("Null CLF for pair: ", pair)

        # print (predict)
        return predict

    def predict_buy(self, df: DataFrame, pair):
        clf = self.pair_model_info[pair]['clf_buy']

        if clf is None:
            print("    No Buy Classifier for pair ", pair, " -Skipping predictions")
            self.pair_model_info[pair]['interval'] = min(self.pair_model_info[pair]['interval'], 4)
            predict = df['close'].copy()  # just to get the size
            predict = 0.0
            return predict

        print("    predicting buys..")
        predict = self.predict(df, pair, clf)

        # if self.dbg_test_classifier:
        #     # DEBUG: check accuracy
        #     signals = df['train_buy_signal']
        #     labels = self.dataframeUtils.get_binary_labels(signals)
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
        clf = self.pair_model_info[pair]['clf_sell']
        if clf is None:
            print("    No Sell Classifier for pair ", pair, " -Skipping predictions")
            self.pair_model_info[pair]['interval'] = min(self.pair_model_info[pair]['interval'], 4)
            predict = df['close']  # just to get the size
            predict = 0.0
            return predict

        print("    predicting sells..")
        predict = self.predict(df, pair, clf)

        # if self.dbg_test_classifier:
        #     # DEBUG: check accuracy
        #     signals = df['train_sell_signal']
        #     labels = self.dataframeUtils.get_binary_labels(signals)
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
        # print("")
        # print("pair_model_info:")
        print("  ", pair, ": ", self.pair_model_info[pair])
        # print("")

    def show_all_debug_info(self):
        print("")
        if (len(self.pair_model_info) > 0):
            # print("Model Info:")
            # print("----------")
            table = PrettyTable(["Pair", "PCA Size", "Buy Classifier", "Sell Classifier"])
            table.title = "Model Information"
            table.align = "l"
            table.align["PCA Size"] = "c"
            table.reversesort = False
            table.sortby = 'Pair'

            for pair in self.pair_model_info:
                table.add_row([pair,
                               self.pair_model_info[pair]['pca_size'],
                               self.pair_model_info[pair]['clf_buy_name'],
                               self.pair_model_info[pair]['clf_sell_name']
                               ])

            print(table)

        if len(self.classifier_stats) > 0:
            # print("Classifier Statistics:")
            # print("---------------------")
            print("")
            if 'buy' in self.classifier_stats:
                print("")
                table = PrettyTable(["Classifier", "Mean Score", "Selected"])
                table.title = "Buy Classifiers"
                table.align["Classifier"] = "l"
                table.align["Mean Score"] = "c"
                table.float_format = '.4'
                for cls in self.classifier_stats['buy']:
                    table.add_row([cls,
                                   self.classifier_stats['buy'][cls]['score'],
                                   self.classifier_stats['buy'][cls]['selected']])
                table.reversesort = True
                # table.sortby = 'Mean Score'
                print(table.get_string(sort_key=operator.itemgetter(2, 1), sortby="Selected"))
                print("")

            if 'sell' in self.classifier_stats:
                print("")
                table = PrettyTable(["Classifier", "Mean Score", "Selected"])
                table.title = "Sell Classifiers"
                table.align["Classifier"] = "l"
                table.align["Mean Score"] = "c"
                table.float_format = '.4'
                for cls in self.classifier_stats['sell']:
                    table.add_row([cls,
                                   self.classifier_stats['sell'][cls]['score'],
                                   self.classifier_stats['sell'][cls]['selected']])
                table.reversesort = True
                # table.sortby = 'Mean Score'
                print(table.get_string(sort_key=operator.itemgetter(2, 1), sortby="Selected"))
                print("")

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
            if PCA.first_run and self.dbg_scan_classifiers:
                PCA.first_run = False  # note use of clas variable, not instance variable
                # self.show_debug_info(curr_pair)
                self.show_all_debug_info()

        # add some fairly loose guards, to help prevent 'bad' predictions

        # # ATR in buy range
        # conditions.append(dataframe['atr_signal'] > 0.0)

        # some trading volume
        conditions.append(dataframe['volume'] > 0)

        # MFI
        conditions.append(dataframe['mfi'] < 30.0)

        # below TEMA
        conditions.append(dataframe['close'] < dataframe['tema'])

        # PCA/Classifier triggers
        pca_cond = (
            (qtpylib.crossed_above(dataframe['predict_buy'], 0.5))
        )
        conditions.append(pca_cond)

        # add strategy-specific conditions (from subclass)
        strat_cond = self.get_strategy_entry_guard_conditions(dataframe)
        if strat_cond is not None:
            conditions.append(strat_cond)

        # set entry tags
        dataframe.loc[pca_cond, 'enter_tag'] += 'pca_entry '

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
            if PCA.first_run and self.dbg_scan_classifiers:
                PCA.first_run = False  # note use of clas variable, not instance variable
                # self.show_debug_info(curr_pair)
                self.show_all_debug_info()

        # if we are to ignore exit signals, just set exit column to 0s and return
        if self.ignore_exit_signals:
            dataframe['exit_long'] = 0
            return dataframe

        conditions.append(dataframe['volume'] > 0)

        # MFI
        conditions.append(dataframe['mfi'] > 70.0)

        # above TEMA
        conditions.append(dataframe['close'] > dataframe['tema'])

        # PCA triggers
        pca_cond = (
            qtpylib.crossed_above(dataframe['predict_sell'], 0.5)
        )

        conditions.append(pca_cond)

        # add strategy-specific conditions (from subclass)
        strat_cond = self.get_strategy_exit_guard_conditions(dataframe)
        if strat_cond is not None:
            conditions.append(strat_cond)

        dataframe.loc[pca_cond, 'exit_tag'] += 'pca_exit '

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


#######################
