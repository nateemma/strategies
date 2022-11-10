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
from datetime import datetime, timedelta
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
from sklearn.svm import SVC
from sklearn.utils.fixes import loguniform
from sklearn import preprocessing
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier, VotingClassifier
from sklearn.naive_bayes import GaussianNB, MultinomialNB
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis, LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import make_scorer
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.model_selection import cross_validate

import random

from prettytable import PrettyTable

"""
####################################################################################
PCA - uses Principal Component Analysis to try and reduce the total set of indicators
      to more manageable dimensions, and predict the next gain step.
      
      This works by creating a PCA model of the available technical indicators. This produces a 
      mapping of the indicators and how they affect the outcome (buy/sell/hold). We choose only the
      mappings that have a signficant effect and ignore the others. This significantly reduces the size
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

class PCA2(IStrategy):
    # Do *not* hyperopt for the roi and stoploss spaces (unless you turn off custom stoploss)

    # ROI table:
    minimal_roi = {
        "0": 0.05
    }

    # Stoploss:
    stoploss = -0.10

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
    min_f1_score = 0.70

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

    dwt_window = startup_candle_count

    num_pairs = 0
    pair_model_info = {}  # holds model-related info for each pair
    classifier_stats = {} # holds statistics for each type of classifier (useful to rank classifiers

    # debug flags
    first_time = True  # mostly for debug
    first_run = True  # used to identify first time through buy/sell populate funcs

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

    ###################################

    # Strategy Specific Variable Storage

    ## Hyperopt Variables

    sell_params = {
        "pHSL": -0.186,
        "pPF_1": 0.011,
        "pPF_2": 0.071,
        "pSL_1": 0.02,
        "pSL_2": 0.063
    }

    # hard stoploss profit
    pHSL = DecimalParameter(-0.200, -0.040, default=-0.08, decimals=3, space='sell', load=True)
    # profit threshold 1, trigger point, SL_1 is used
    pPF_1 = DecimalParameter(0.008, 0.020, default=0.016, decimals=3, space='sell', load=True)
    pSL_1 = DecimalParameter(0.008, 0.020, default=0.011, decimals=3, space='sell', load=True)

    # profit threshold 2, SL_2 is used
    pPF_2 = DecimalParameter(0.040, 0.100, default=0.080, decimals=3, space='sell', load=True)
    pSL_2 = DecimalParameter(0.020, 0.070, default=0.040, decimals=3, space='sell', load=True)

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
                    (future_df['rsi'] >= 80) & # classic oversold threshold
                    (future_df['future_gain'] > future_df['profit_threshold'])
            ), 1.0, 0.0)

        return series

    def get_train_sell_signals(self, future_df: DataFrame):

        print("!!! WARNING: using base class (sell) training implementation !!!")

        series = np.where(
            (
                    (future_df['rsi'] <= 20) &  # classic overbought threshold
                    (future_df['future_gain'] < future_df['loss_threshold'])
            ), 1.0, 0.0)

        return series

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


        self.set_state(curr_pair, self.State.POPULATE)
        self.curr_lookahead = int(12 * self.lookahead_hours)
        self.dbg_curr_df = dataframe

        # reset profit/loss thresholds
        self.profit_threshold = self.default_profit_threshold
        self.loss_threshold = self.default_loss_threshold

        if PCA2.first_time:
            PCA2.first_time = False
            print("")
            print("***************************************")
            print("** Warning: startup can be very slow **")
            print("***************************************")

            print("    Lookahead: ", self.curr_lookahead, " candles (", self.lookahead_hours, " hours)")
            print("    Thresholds - Profit:{:.2f}% Loss:{:.2f}%".format(self.profit_threshold,
                                                                        self.loss_threshold))

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

        # populate the normal dataframe
        dataframe = self.add_indicators(dataframe)

        buys, sells = self.create_training_data(dataframe)

        # drop last group (because there cannot be a prediction)
        df = dataframe.iloc[:-self.curr_lookahead]
        buys = buys.iloc[:-self.curr_lookahead]
        sells = sells.iloc[:-self.curr_lookahead]

        # Principal Component Analysis of inf data

        # train the models on the informative data
        if self.dbg_verbose:
            print("    training models...")
        self.train_models(curr_pair, df, buys, sells)
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

    ###################################

    # populate dataframe with desired technical indicators
    # NOTE: OK to throw (almost) anything in here, just add it to the parameter list
    # The whole idea is to create a dimension-reduced mapping anyway
    # Warning: do not use indicators that might produce 'inf' results, it messes up the scaling
    def add_indicators(self, dataframe: DataFrame) -> DataFrame:

        win_size = max(self.curr_lookahead, 14)

        # these averages are used internally, do not remove!
        dataframe['sma'] = ta.SMA(dataframe, timeperiod=win_size)
        dataframe['ema'] = ta.EMA(dataframe, timeperiod=win_size)
        dataframe['tema'] = ta.TEMA(dataframe, timeperiod=win_size)
        dataframe['tema_stddev'] = dataframe['tema'].rolling(win_size).std()

        # RSI
        period = 14
        smoothD = 3
        SmoothK = 3
        dataframe['rsi'] = ta.RSI(dataframe, timeperiod=win_size)
        stochrsi = (dataframe['rsi'] - dataframe['rsi'].rolling(period).min()) / (
                dataframe['rsi'].rolling(period).max() - dataframe['rsi'].rolling(period).min())
        dataframe['srsi_k'] = stochrsi.rolling(SmoothK).mean() * 100
        dataframe['srsi_d'] = dataframe['srsi_k'].rolling(smoothD).mean()

        # Bollinger Bands (must include these)
        bollinger = qtpylib.bollinger_bands(dataframe['close'], window=20, stds=2)
        dataframe['bb_lowerband'] = bollinger['lower']
        dataframe['bb_middleband'] = bollinger['mid']
        dataframe['bb_upperband'] = bollinger['upper']
        dataframe['bb_width'] = ((dataframe['bb_upperband'] - dataframe['bb_lowerband']) / dataframe['bb_middleband'])
        dataframe["bb_gain"] = ((dataframe["bb_upperband"] - dataframe["close"]) / dataframe["close"])
        dataframe["bb_loss"] = ((dataframe["bb_lowerband"] - dataframe["close"]) / dataframe["close"])

        # Donchian Channels
        dataframe['dc_upper'] = ta.MAX(dataframe['high'], timeperiod=win_size)
        dataframe['dc_lower'] = ta.MIN(dataframe['low'], timeperiod=win_size)
        dataframe['dc_mid'] = ta.TEMA(((dataframe['dc_upper'] + dataframe['dc_lower']) / 2), timeperiod=win_size)

        dataframe["dcbb_dist_upper"] = (dataframe["dc_upper"] - dataframe['bb_upperband'])
        dataframe["dcbb_dist_lower"] = (dataframe["dc_lower"] - dataframe['bb_lowerband'])

        # Fibonacci Levels (of Donchian Channel)
        dataframe['dc_dist'] = (dataframe['dc_upper'] - dataframe['dc_lower'])
        dataframe['dc_hf'] = dataframe['dc_upper'] - dataframe['dc_dist'] * 0.236  # Highest Fib
        dataframe['dc_chf'] = dataframe['dc_upper'] - dataframe['dc_dist'] * 0.382  # Centre High Fib
        dataframe['dc_clf'] = dataframe['dc_upper'] - dataframe['dc_dist'] * 0.618  # Centre Low Fib
        dataframe['dc_lf'] = dataframe['dc_upper'] - dataframe['dc_dist'] * 0.764  # Low Fib

         # Keltner Channels
        keltner = qtpylib.keltner_channel(dataframe)
        dataframe["kc_upper"] = keltner["upper"]
        dataframe["kc_lower"] = keltner["lower"]
        dataframe["kc_mid"] = keltner["mid"]

        # Williams %R
        dataframe['wr'] = 0.02 * (williams_r(dataframe, period=14) + 50.0)

        # Fisher RSI
        rsi = 0.1 * (dataframe['rsi'] - 50)
        dataframe['fisher_rsi'] = (np.exp(2 * rsi) - 1) / (np.exp(2 * rsi) + 1)

        # Combined Fisher RSI and Williams %R
        dataframe['fisher_wr'] = (dataframe['wr'] + dataframe['fisher_rsi']) / 2.0

        # RSI
        # dataframe['rsi'] = ta.RSI(dataframe, timeperiod=win_size)
        dataframe['rsi_14'] = ta.RSI(dataframe, timeperiod=14)

        # RMI: https://www.tradingview.com/script/kwIt9OgQ-Relative-Momentum-Index/
        dataframe['rmi'] = cta.RMI(dataframe, length=24, mom=5)

        # MA Streak: https://www.tradingview.com/script/Yq1z7cIv-MA-Streak-Can-Show-When-a-Run-Is-Getting-Long-in-the-Tooth/
        dataframe['mastreak'] = cta.mastreak(dataframe, period=4)

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

        # Indicators used only for ROI and Custom Stoploss
        ssldown, sslup = cta.SSLChannels_ATR(dataframe, length=21)
        dataframe['sroc'] = cta.SROC(dataframe, roclen=21, emalen=13, smooth=21)
        dataframe['ssl_dir'] = 0
        dataframe['ssl_dir'] = np.where(sslup > ssldown, 1.0, -1.0)

        # EMAs
        dataframe['ema_12'] = ta.EMA(dataframe, timeperiod=12)
        dataframe['ema_20'] = ta.EMA(dataframe, timeperiod=20)
        dataframe['ema_25'] = ta.EMA(dataframe, timeperiod=25)
        dataframe['ema_35'] = ta.EMA(dataframe, timeperiod=35)
        dataframe['ema_50'] = ta.EMA(dataframe, timeperiod=50)
        dataframe['ema_100'] = ta.EMA(dataframe, timeperiod=100)
        dataframe['ema_200'] = ta.EMA(dataframe, timeperiod=200)

        # SMA
        dataframe['sma_200'] = ta.SMA(dataframe, timeperiod=200)
        dataframe['sma_200_dec_20'] = np.where(dataframe['sma_200'] < dataframe['sma_200'].shift(20), 1.0, -1.0)
        dataframe['sma_200_dec_24'] = np.where(dataframe['sma_200'] < dataframe['sma_200'].shift(24), 1.0, -1.0)

        # # CMF
        # dataframe['cmf'] = chaikin_money_flow(dataframe, 20)

        # # CTI
        # dataframe['cti'] = pta.cti(dataframe["close"], length=20)

        # CRSI (3, 2, 100)
        crsi_closechange = dataframe['close'] / dataframe['close'].shift(1)
        crsi_updown = np.where(crsi_closechange.gt(1), 1.0, np.where(crsi_closechange.lt(1), -1.0, -1.0))
        dataframe['crsi'] = (ta.RSI(dataframe['close'], timeperiod=3) + ta.RSI(crsi_updown, timeperiod=2) + ta.ROC(
            dataframe['close'],
            100)) / 3

        # Williams %R
        dataframe['r_14'] = williams_r(dataframe, period=14)
        dataframe['r_480'] = williams_r(dataframe, period=480)

        # ROC
        dataframe['roc_9'] = ta.ROC(dataframe, timeperiod=9)

        # T3 Average
        dataframe['t3_avg'] = t3_average(dataframe)

        # S/R
        res_series = dataframe['high'].rolling(window=5, center=True).apply(lambda row: is_resistance(row),
                                                                            raw=True).shift(2)
        sup_series = dataframe['low'].rolling(window=5, center=True).apply(lambda row: is_support(row),
                                                                           raw=True).shift(2)
        dataframe['res_level'] = Series(
            np.where(res_series,
                     np.where(dataframe['close'] > dataframe['open'], dataframe['close'], dataframe['open']),
                     float('NaN'))).ffill()
        dataframe['res_hlevel'] = Series(np.where(res_series, dataframe['high'], float('NaN'))).ffill()
        dataframe['sup_level'] = Series(
            np.where(sup_series,
                     np.where(dataframe['close'] < dataframe['open'], dataframe['close'], dataframe['open']),
                     float('NaN'))).ffill()

        # Pump protections
        dataframe['hl_pct_change_48'] = range_percent_change(dataframe, 'HL', 48)
        dataframe['hl_pct_change_36'] = range_percent_change(dataframe, 'HL', 36)
        dataframe['hl_pct_change_24'] = range_percent_change(dataframe, 'HL', 24)
        dataframe['hl_pct_change_12'] = range_percent_change(dataframe, 'HL', 12)
        dataframe['hl_pct_change_6'] = range_percent_change(dataframe, 'HL', 6)

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

        # SAR Parabol
        dataframe['sar'] = ta.SAR(dataframe)

        dataframe['mom'] = ta.MOM(dataframe, timeperiod=14)

        # priming indicators
        dataframe['color'] = np.where((dataframe['close'] > dataframe['open']), 1.0, -1.0)
        dataframe['rsi_7'] = ta.RSI(dataframe, timeperiod=7)
        dataframe['roc_6'] = ta.ROC(dataframe, timeperiod=6)
        dataframe['primed'] = np.where(dataframe['color'].rolling(3).sum() == 3.0, 1.0, -1.0)
        dataframe['in_the_mood'] = np.where(dataframe['rsi_7'] > dataframe['rsi_7'].rolling(12).mean(), 1.0, -1.0)
        dataframe['moist'] = np.where(qtpylib.crossed_above(dataframe['macd'], dataframe['macdsignal']), 1.0, -1.0)
        dataframe['throbbing'] = np.where(dataframe['roc_6'] > dataframe['roc_6'].rolling(12).mean(), 1.0, -1.0)

        ## sqzmi to detect quiet periods
        dataframe['sqzmi'] = np.where(fta.SQZMI(dataframe), 1.0, -1.0)
        dataframe['sqz_on'] = np.where(
            (
                (dataframe["kc_upper"] > dataframe['bb_upperband']) &
                (dataframe["kc_lower"] < dataframe['bb_lowerband'])
            ), 1.0, -1.0
        )
        dataframe['sqz_off'] = np.where(
            (
                (dataframe["kc_upper"] < dataframe['bb_upperband']) &
                (dataframe["kc_lower"] > dataframe['bb_lowerband'])
            ), 1.0, -1.0
        )
        dataframe['sqz_none'] = np.where(
            (
                (dataframe['sqz_on'] < 0) &
                (dataframe["sqz_off"] < 0)
            ), 1.0, -1.0
        )

        # MFI
        dataframe['mfi'] = ta.MFI(dataframe)
        dataframe['mfi_norm'] = self.norm_column(dataframe['mfi'])
        dataframe['mfi_buy'] = np.where((dataframe['mfi_norm'] > 0.5), 1.0, 0.0)
        dataframe['mfi_sell'] = np.where((dataframe['mfi_norm'] <= -0.5), 1.0, 0.0)
        dataframe['mfi_signal'] = dataframe['mfi_buy'] - dataframe['mfi_sell']

        # Volume Flow Indicator (MFI) for volume based on the direction of price movement
        dataframe['vfi'] = fta.VFI(dataframe, period=14)
        dataframe['vfi_norm'] = self.norm_column(dataframe['vfi'])
        dataframe['vfi_buy'] = np.where((dataframe['vfi_norm'] > 0.5), 1.0, 0.0)
        dataframe['vfi_sell'] = np.where((dataframe['vfi_norm'] <= -0.5), 1.0, 0.0)
        dataframe['vfi_signal'] = dataframe['vfi_buy'] - dataframe['vfi_sell']

        # ATR
        dataframe['atr'] = ta.ATR(dataframe, timeperiod=win_size)
        dataframe['atr_norm'] = self.norm_column(dataframe['atr'])
        dataframe['atr_buy'] = np.where((dataframe['atr_norm'] > 0.5), 1.0, 0.0)
        dataframe['atr_sell'] = np.where((dataframe['atr_norm'] <= -0.5), 1.0, 0.0)
        dataframe['atr_signal'] = dataframe['atr_buy'] - dataframe['atr_sell']

        # Hilbert Transform Indicator - SineWave
        hilbert = ta.HT_SINE(dataframe)
        dataframe['htsine'] = hilbert['sine']
        dataframe['htleadsine'] = hilbert['leadsine']

        # Oscillators

        # EWO
        dataframe['ewo'] = ewo(dataframe, 50, 200)

        # Ultimate Oscillator
        dataframe['uo'] = ta.ULTOSC(dataframe)

        # Aroon, Aroon Oscillator
        aroon = ta.AROON(dataframe)
        dataframe['aroonup'] = aroon['aroonup']
        dataframe['aroondown'] = aroon['aroondown']
        dataframe['aroonosc'] = ta.AROONOSC(dataframe)

        # Awesome Oscillator
        dataframe['ao'] = qtpylib.awesome_oscillator(dataframe)

        # Commodity Channel Index: values [Oversold:-100, Overbought:100]
        dataframe['cci'] = ta.CCI(dataframe)

        # DWT model
        # get rolling DWT. Probably OK to just apply to the whole dataframe, but be careful anyway
        dataframe['dwt'] = dataframe['close'].rolling(window=self.dwt_window).apply(self.roll_get_dwt)
        dataframe['smooth'] = dataframe['close'].rolling(window=self.dwt_window).apply(self.roll_smooth)
        dataframe['dwt_smooth'] = dataframe['dwt'].rolling(window=self.dwt_window).apply(self.roll_smooth)

        # smoothed version - useful for trends
        # dataframe['dwt_smooth'] = gaussian_filter1d(dataframe['dwt'], 8)

        dataframe['dwt_deriv'] = np.gradient(dataframe['dwt_smooth'])
        dataframe['dwt_top'] = np.where(qtpylib.crossed_below(dataframe['dwt_deriv'], 0.0), 1, 0)
        dataframe['dwt_bottom'] = np.where(qtpylib.crossed_above(dataframe['dwt_deriv'], 0.0), 1, 0)

        dataframe['dwt_diff'] = 100.0 * (dataframe['dwt'] - dataframe['close']) / dataframe['close']
        dataframe['dwt_smooth_diff'] = 100.0 * (dataframe['dwt'] - dataframe['dwt_smooth']) / dataframe['dwt_smooth']

        # up/down direction
        dataframe['dwt_dir'] = 0.0
        # dataframe['dwt_dir'] = np.where(dataframe['dwt'].diff() >= 0, 1.0, -1.0)
        dataframe['dwt_dir'] = np.where(dataframe['dwt_smooth'].diff() >= 0, 1.0, -1.0)

        dataframe['dwt_trend'] = np.where(dataframe['dwt_dir'].rolling(5).sum() > 0.0, 1.0, -1.0)

        dataframe['dwt_gain'] = 100.0 * (dataframe['dwt'] - dataframe['dwt'].shift()) / dataframe['dwt'].shift()

        dataframe['dwt_profit'] = dataframe['dwt_gain'].clip(lower=0.0)
        dataframe['dwt_loss'] = dataframe['dwt_gain'].clip(upper=0.0)

        # get rolling mean & stddev so that we have a localised estimate of (recent) activity
        dataframe['dwt_mean'] = dataframe['dwt'].rolling(win_size).mean()
        dataframe['dwt_std'] = dataframe['dwt'].rolling(win_size).std()
        dataframe['dwt_profit_mean'] = dataframe['dwt_profit'].rolling(win_size).mean()
        dataframe['dwt_profit_std'] = dataframe['dwt_profit'].rolling(win_size).std()
        dataframe['dwt_loss_mean'] = dataframe['dwt_loss'].rolling(win_size).mean()
        dataframe['dwt_loss_std'] = dataframe['dwt_loss'].rolling(win_size).std()

        # Sequences of consecutive up/downs
        dataframe['dwt_nseq'] = dataframe['dwt_dir'].rolling(window=win_size, min_periods=1).sum()

        dataframe['dwt_nseq_up'] = dataframe['dwt_nseq'].clip(lower=0.0)
        dataframe['dwt_nseq_up_mean'] = dataframe['dwt_nseq_up'].rolling(window=win_size).mean()
        dataframe['dwt_nseq_up_std'] = dataframe['dwt_nseq_up'].rolling(window=win_size).std()
        dataframe['dwt_nseq_up_thresh'] = dataframe['dwt_nseq_up_mean'] + \
                                          self.n_profit_stddevs * dataframe['dwt_nseq_up_std']
        dataframe['dwt_nseq_sell'] = np.where(dataframe['dwt_nseq_up'] > dataframe['dwt_nseq_up_thresh'], 1.0, 0.0)

        dataframe['dwt_nseq_dn'] = dataframe['dwt_nseq'].clip(upper=0.0)
        dataframe['dwt_nseq_dn_mean'] = dataframe['dwt_nseq_dn'].rolling(window=win_size).mean()
        dataframe['dwt_nseq_dn_std'] = dataframe['dwt_nseq_dn'].rolling(window=win_size).std()
        dataframe['dwt_nseq_dn_thresh'] = dataframe['dwt_nseq_dn_mean'] - self.n_loss_stddevs * dataframe[
            'dwt_nseq_dn_std']
        dataframe['dwt_nseq_buy'] = np.where(dataframe['dwt_nseq_dn'] < dataframe['dwt_nseq_dn_thresh'], 1.0, 0.0)

        # Recent min/max
        dataframe['dwt_recent_min'] = dataframe['dwt_smooth'].rolling(window=win_size).min()
        dataframe['dwt_recent_max'] = dataframe['dwt_smooth'].rolling(window=win_size).max()
        dataframe['dwt_maxmin'] = 100.0 * (dataframe['dwt_recent_max'] - dataframe['dwt_recent_min']) / \
                                  dataframe['dwt_recent_max']

        # longer term high/low
        dataframe['dwt_low'] = dataframe['dwt_smooth'].rolling(window=self.startup_candle_count).min()
        dataframe['dwt_high'] = dataframe['dwt_smooth'].rolling(window=self.startup_candle_count).max()

        # # these are (primarily) clues for the ML algorithm:
        dataframe['dwt_at_min'] = np.where(dataframe['dwt_smooth'] <= dataframe['dwt_recent_min'], 1.0, 0.0)
        dataframe['dwt_at_max'] = np.where(dataframe['dwt_smooth'] >= dataframe['dwt_recent_max'], 1.0, 0.0)
        dataframe['dwt_at_low'] = np.where(dataframe['dwt_smooth'] <= dataframe['dwt_low'], 1.0, 0.0)
        dataframe['dwt_at_high'] = np.where(dataframe['dwt_smooth'] >= dataframe['dwt_high'], 1.0, 0.0)

        # TODO: remove/fix any columns that contain 'inf'
        self.check_inf(dataframe)

        # TODO: fix NaNs
        dataframe.fillna(0.0, inplace=True)

        return dataframe

    # calculate future gains. Used for setting targets
    def add_future_data(self, dataframe: DataFrame) -> DataFrame:

        # yes, we lookahead in the data!
        lookahead = self.curr_lookahead

        win_size = max(lookahead, 14)

        # make a copy of the dataframe so that we do not put any forward looking data into the main dataframe
        future_df = dataframe.copy()

        # we can either use the actual closing price, or the DWT model (smoother)

        use_dwt = True
        if use_dwt:
            price_col = 'full_dwt'

            # get the 'full' DWT transform. This models the entire dataframe, so cannot be used in the 'main' dataframe
            future_df['full_dwt'] = self.get_dwt(dataframe['close'])

        else:
            price_col = 'close'
            future_df['full_dwt'] = 0.0

        # calculate future gains
        future_df['future_close'] = future_df[price_col].shift(-lookahead)

        future_df['future_gain'] = 100.0 * (future_df['future_close'] - future_df[price_col]) / future_df[price_col]
        future_df['future_gain'].clip(lower=-5.0, upper=5.0, inplace=True)

        future_df['future_profit'] = future_df['future_gain'].clip(lower=0.0)
        future_df['future_loss'] = future_df['future_gain'].clip(upper=0.0)

        # get rolling mean & stddev so that we have a localised estimate of (recent) future activity
        # Note: window in past because we already looked forward
        future_df['profit_mean'] = future_df['future_profit'].rolling(win_size).mean()
        future_df['profit_std'] = future_df['future_profit'].rolling(win_size).std()
        future_df['profit_max'] = future_df['future_profit'].rolling(win_size).max()
        future_df['profit_min'] = future_df['future_profit'].rolling(win_size).min()
        future_df['loss_mean'] = future_df['future_loss'].rolling(win_size).mean()
        future_df['loss_std'] = future_df['future_loss'].rolling(win_size).std()
        future_df['loss_max'] = future_df['future_loss'].rolling(win_size).max()
        future_df['loss_min'] = future_df['future_loss'].rolling(win_size).min()

        # future_df['profit_threshold'] = future_df['profit_mean'] + self.n_profit_stddevs * abs(future_df['profit_std'])
        # future_df['loss_threshold'] = future_df['loss_mean'] - self.n_loss_stddevs * abs(future_df['loss_std'])

        future_df['profit_threshold'] = future_df['dwt_profit_mean'] + self.n_profit_stddevs * abs(
            future_df['dwt_profit_std'])
        future_df['loss_threshold'] = future_df['dwt_loss_mean'] - self.n_loss_stddevs * abs(future_df['dwt_loss_std'])

        future_df['profit_diff'] = (future_df['future_profit'] - future_df['profit_threshold']) * 10.0
        future_df['loss_diff'] = (future_df['future_loss'] - future_df['loss_threshold']) * 10.0

        # future_df['buy_signal'] = np.where(future_df['profit_diff'] > 0.0, 1.0, 0.0)
        # future_df['sell_signal'] = np.where(future_df['loss_diff'] < 0.0, -1.0, 0.0)

        # these explicitly uses dwt
        future_df['future_dwt'] = future_df['full_dwt'].shift(-lookahead)
        # future_df['curr_trend'] = np.where(future_df['full_dwt'].shift(-1) > future_df['full_dwt'], 1.0, -1.0)
        # future_df['future_trend'] = np.where(future_df['future_dwt'].shift(-1) > future_df['future_dwt'], 1.0, -1.0)

        future_df['trend'] = np.where(future_df[price_col] >= future_df[price_col].shift(), 1.0, -1.0)
        future_df['ftrend'] = np.where(future_df['future_close'] >= future_df['future_close'].shift(), 1.0, -1.0)

        future_df['curr_trend'] = np.where(future_df['trend'].rolling(3).sum() > 0.0, 1.0, -1.0)
        future_df['future_trend'] = np.where(future_df['ftrend'].rolling(3).sum() > 0.0, 1.0, -1.0)

        future_df['dwt_dir'] = 0.0
        future_df['dwt_dir'] = np.where(dataframe['dwt'].diff() >= 0, 1, -1)
        future_df['dwt_dir_up'] = np.where(dataframe['dwt'].diff() >= 0, 1, 0)
        future_df['dwt_dir_dn'] = np.where(dataframe['dwt'].diff() < 0, 1, 0)

        # build forward-looking sum of up/down trends
        future_win = pd.api.indexers.FixedForwardWindowIndexer(window_size=int(win_size))  # don't use a big window

        future_df['future_nseq'] = future_df['curr_trend'].rolling(window=future_win, min_periods=1).sum()

        # future_df['future_nseq_up'] = 0.0
        # future_df['future_nseq_up'] = np.where(
        #     future_df["dwt_dir_up"].eq(1),
        #     future_df.groupby(future_df["dwt_dir_up"].ne(future_df["dwt_dir_up"].shift(1)).cumsum()).cumcount() + 1,
        #     0.0
        # )
        future_df['future_nseq_up'] = future_df['future_nseq'].clip(lower=0.0)

        future_df['future_nseq_up_mean'] = future_df['future_nseq_up'].rolling(window=future_win).mean()
        future_df['future_nseq_up_std'] = future_df['future_nseq_up'].rolling(window=future_win).std()
        future_df['future_nseq_up_thresh'] = future_df['future_nseq_up_mean'] + self.n_profit_stddevs * future_df[
            'future_nseq_up_std']

        # future_df['future_nseq_dn'] = -np.where(
        #     future_df["dwt_dir_dn"].eq(1),
        #     future_df.groupby(future_df["dwt_dir_dn"].ne(future_df["dwt_dir_dn"].shift(1)).cumsum()).cumcount() + 1,
        #     0.0
        # )
        # print(future_df['future_nseq_dn'])
        future_df['future_nseq_dn'] = future_df['future_nseq'].clip(upper=0.0)

        future_df['future_nseq_dn_mean'] = future_df['future_nseq_dn'].rolling(future_win).mean()
        future_df['future_nseq_dn_std'] = future_df['future_nseq_dn'].rolling(future_win).std()
        future_df['future_nseq_dn_thresh'] = future_df['future_nseq_dn_mean'] \
                                             - self.n_loss_stddevs * future_df['future_nseq_dn_std']

        # Recent min/max
        # future_df['future_min'] = future_df[price_col].rolling(window=future_win).min()
        # future_df['future_max'] = future_df[price_col].rolling(window=future_win).max()
        future_df['future_min'] = future_df['dwt_smooth'].rolling(window=future_win).min()
        future_df['future_max'] = future_df['dwt_smooth'].rolling(window=future_win).max()

        # get average gain & stddev
        profit_mean = future_df['future_profit'].mean()
        profit_std = future_df['future_profit'].std()
        loss_mean = future_df['future_loss'].mean()
        loss_std = future_df['future_loss'].std()

        # update thresholds
        if self.profit_threshold != profit_mean:
            newval = profit_mean + self.n_profit_stddevs * profit_std
            print("    Profit threshold {:.4f} -> {:.4f}".format(self.profit_threshold, newval))
            self.profit_threshold = newval

        if self.loss_threshold != loss_mean:
            newval = loss_mean - self.n_loss_stddevs * abs(loss_std)
            print("    Loss threshold {:.4f} -> {:.4f}".format(self.loss_threshold, newval))
            self.loss_threshold = newval

        return future_df

    ################################

    # creates the buy/sell labels absed on looking ahead into the supplied dataframe
    def create_training_data(self, dataframe: DataFrame):

        future_df = self.add_future_data(dataframe.copy())

        future_df['train_buy'] = 0.0
        future_df['train_sell'] = 0.0

        # use seqquence trends as criteria
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
            'profit_min', 'profit_max', 'profit_threshold',
            'loss_min', 'loss_max', 'loss_threshold',
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

    # returns (rolling) smoothed version of input column
    def roll_smooth(self, col) -> np.float:
        # must return scalar, so just calculate prediction and take last value

        # smooth = gaussian_filter1d(col, 4)
        smooth = gaussian_filter1d(col, 2)

        length = len(smooth)
        if length > 0:
            return smooth[length - 1]
        else:
            print("model:", smooth)
            return 0.0

    def get_dwt(self, col):

        a = np.array(col)

        # de-trend the data
        w_mean = a.mean()
        w_std = a.std()
        a_notrend = (a - w_mean) / w_std
        # a_notrend = a_notrend.clip(min=-3.0, max=3.0)

        # get DWT model of data
        restored_sig = self.dwtModel(a_notrend)

        # re-trend
        model = (restored_sig * w_std) + w_mean

        return model

    def roll_get_dwt(self, col) -> np.float:
        # must return scalar, so just calculate prediction and take last value

        model = self.get_dwt(col)

        length = len(model)
        if length > 0:
            return model[length - 1]
        else:
            # cannot calculate DWT (e.g. at startup), just return original value
            return col[len(col)-1]

    def dwtModel(self, data):

        # the choice of wavelet makes a big difference
        # for an overview, check out: https://www.kaggle.com/theoviel/denoising-with-direct-wavelet-transform
        # wavelet = 'db1'
        wavelet = 'db8'
        # wavelet = 'bior1.1'
        # wavelet = 'haar'  # deals well with harsh transitions
        level = 1
        wmode = "smooth"
        tmode = "hard"
        length = len(data)

        # Apply DWT transform
        coeff = pywt.wavedec(data, wavelet, mode=wmode)

        # remove higher harmonics
        std = np.std(coeff[level])
        sigma = (1 / 0.6745) * self.madev(coeff[-level])
        # sigma = self.madev(coeff[-level])
        uthresh = sigma * np.sqrt(2 * np.log(length))

        coeff[1:] = (pywt.threshold(i, value=uthresh, mode=tmode) for i in coeff[1:])

        # inverse DWT transform
        model = pywt.waverec(coeff, wavelet, mode=wmode)

        # there is a known bug in waverec where odd numbered lengths result in an extra item at the end
        diff = len(model) - len(data)
        return model[0:len(model) - diff]
        # return model[diff:]

    def madev(self, d, axis=None):
        """ Mean absolute deviation of a signal """
        return np.mean(np.absolute(d - np.mean(d, axis)), axis)

    # add indicators used by stoploss/custom sell logic
    def add_stoploss_indicators(self, dataframe, pair) -> DataFrame:
        if not pair in self.custom_trade_info:
            self.custom_trade_info[pair] = {}
            if not 'had_trend' in self.custom_trade_info[pair]:
                self.custom_trade_info[pair]['had_trend'] = False

        return dataframe

    def norm_column(self, col):
        return self.zscore_column(col)

    # normalises a column. Data is in units of 1 stddev, i.e. a value of 1.0 represents 1 stdev above mean
    def zscore_column(self, col):
        return (col - col.mean()) / col.std()

    # applies MinMax scaling to a column. Returns all data in range [0,1]
    def minmax_column(self, col):
        result = col
        # print(col)

        if (col.dtype == 'str') or (col.dtype == 'object'):
            result = 0.0
        else:
            result = col
            cmax = max(col)
            cmin = min(col)
            denom = float(cmax - cmin)
            if denom == 0.0:
                result = 0.0
            else:
                result = (col - col.min()) / denom

        return result

    def check_inf(self, dataframe):
        col_name = dataframe.columns.to_series()[np.isinf(dataframe).any()]
        if len(col_name) > 0:
            print("***")
            print("*** Infinity in cols: ", col_name)
            print("***")

    def remove_debug_columns(self, dataframe: DataFrame) -> DataFrame:
        drop_list = dataframe.filter(regex='^%').columns
        if len(drop_list) > 0:
            for col in drop_list:
                dataframe = dataframe.drop(col, axis=1)
            dataframe.reindex()
        return dataframe

    # Normalise a dataframe
    def norm_dataframe(self, dataframe: DataFrame) -> DataFrame:
        self.check_inf(dataframe)

        temp = dataframe.copy()
        if 'date' in temp.columns:
            temp['date'] = pd.to_datetime(temp['date']).astype('int64')

        temp = self.remove_debug_columns(temp)

        temp.set_index('date')
        temp.reindex()

        return self.zscore_dataframe(temp).fillna(0.0)

    # Normalise a dataframe using Z-Score normalisation (mean=0, stddev=1)
    def zscore_dataframe(self, dataframe: DataFrame) -> DataFrame:
        self.check_inf(dataframe)
        return ((dataframe - dataframe.mean()) / dataframe.std())

    # Scale a dataframe using sklearn builtin scaler
    def scale_dataframe(self, dataframe: DataFrame) -> DataFrame:
        self.check_inf(dataframe)

        temp = dataframe.copy()
        if 'date' in temp.columns:
            temp['date'] = pd.to_datetime(temp['date']).astype('int64')

        temp = self.remove_debug_columns(temp)

        temp.reindex()

        scaler = RobustScaler()
        scaler = scaler.fit(temp)
        temp = scaler.transform(temp)
        return temp

    # remove outliers from normalised dataframe
    def remove_outliers(self, df_norm: DataFrame, buys, sells):

        # for col in df_norm.columns.values:
        #     if col != 'date':
        #         df_norm = df_norm[(df_norm[col] <= 3.0)]
        # return df_norm
        df = df_norm.copy()
        df['%temp_buy'] = buys.copy()
        df['%temp_sell'] = sells.copy()
        #
        df2 = df[((df >= -3.0) & (df <= 3.0)).all(axis=1)]
        # df_out = df[~((df >= -3.0) & (df <= 3.0)).all(axis=1)] # for debug
        ndrop = df_norm.shape[0] - df2.shape[0]
        if ndrop > 0:
            b = df2['%temp_buy'].copy()
            s = df2['%temp_sell'].copy()
            df2.drop('%temp_buy', axis=1, inplace=True)
            df2.drop('%temp_sell', axis=1, inplace=True)
            df2.reindex()
            # if self.dbg_verbose:
            print("    Removed ", ndrop, " outliers")
            # print(" df2:", df2)
            # print(" df_out:", df_out)
            # print ("df_norm:", df_norm.shape, "df2:", df2.shape, "df_out:", df_out.shape)
        else:
            # no outliers, just return originals
            df2 = df_norm
            b = buys
            s = sells
        return df2, b, s


    # build a 'viable' dataframe sample set. Needed because the positive labels are sparse
    def build_viable_dataset(self, size: int, df_norm: DataFrame, buys, sells):

        # copy and combine the data into one dataframe
        df = df_norm.copy()
        df['%temp_buy'] = buys.copy()
        df['%temp_sell'] = sells.copy()

        # df_buy = df[( (df['%temp_buy'] > 0) ).all(axis=1)]
        # df_sell = df[((df['%temp_sell'] > 0)).all(axis=1)]
        # df_nosig = df[((df['%temp_buy'] == 0) & (df['%temp_sell'] == 0)).all(axis=1)]

        df_buy = df.loc[df['%temp_buy'] == 1]
        df_sell = df.loc[df['%temp_sell'] == 1]
        df_nosig = df.loc[(df['%temp_buy'] == 0) & (df['%temp_sell'] == 0)]

        # make sure there aren't too many buys & sells
        # We are aiming for a roughly even split between buys, sells, and 'no signal' (no buy or sell)
        max_signals = int(2*size/3)
        if ((df_buy.shape[0] + df_sell.shape[0]) > max_signals):
            # both exceed max?
            sig_size = int(size/3)
            if (df_buy.shape[0] > sig_size) & (df_sell.shape[0] > sig_size):
                # resize both buy & sell to 1/3 of requested size
                df_buy, _ = train_test_split(df_buy, train_size=sig_size, shuffle=True)
                df_sell, _ = train_test_split(df_sell, train_size=sig_size, shuffle=True)
            else:
                # only one them is too big, so figure out which
                if (df_buy.shape[0] > df_sell.shape[0]):
                    df_buy, _ = train_test_split(df_buy, train_size=max_signals-df_sell.shape[0], shuffle=True)
                else:
                    df_sell, _ = train_test_split(df_sell, train_size=max_signals-df_buy.shape[0], shuffle=True)

        # extract enough rows to fill the requested size
        fill_size = size - min(df_buy.shape[0], int(size/3)) - min(df_sell.shape[0], int(size/3))
        df_nosig, _ = train_test_split(df_nosig, train_size=fill_size, shuffle=True)
        # print("viable df - buys:{} sells:{} fill:{}".format(df_buy.shape[0], df_sell.shape[0], df_nosig.shape[0]))

        # concatenate the dataframes
        frames = [df_buy, df_sell, df_nosig]
        df2 = pd.concat(frames)

        # shuffle rows
        df2 = df2.sample(frac=1)

        # separate out the data, buys & sells
        b = df2['%temp_buy'].copy()
        s = df2['%temp_sell'].copy()
        df2.drop('%temp_buy', axis=1, inplace=True)
        df2.drop('%temp_sell', axis=1, inplace=True)
        df2.reindex()

        return df2, b, s

    # map column into [0,1]
    def get_binary_labels(self, col):
        binary_encoder = LabelEncoder().fit([min(col), max(col)])
        result = binary_encoder.transform(col)
        # print ("label input:  ", col)
        # print ("label output: ", result)
        return result

    # train the PCA reduction and classification models

    def train_models(self, curr_pair, dataframe: DataFrame, buys, sells):

        # only run if interval reaches 0 (no point retraining every camdle)
        count = self.pair_model_info[curr_pair]['interval']
        if (count > 0):
            self.pair_model_info[curr_pair]['interval'] = count - 1
            return
        else:
            # reset interval to a random number between 1 and the amount of lookahead
            self.pair_model_info[curr_pair]['interval'] = random.randint(1, self.curr_lookahead)

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
            full_df_norm = self.norm_dataframe(dataframe)
            full_df_norm, buys, sells = self.remove_outliers(full_df_norm, buys, sells)
        else:
            full_df_norm = self.norm_dataframe(dataframe).clip(lower=-3.0, upper=3.0)  # supress outliers

        # constrain size to what will be available in run modes
        data_size = int(min(975, full_df_norm.shape[0]))

        # get 'viable' data set (includes all buys/sells)
        v_df_norm, v_buys, v_sells = self.build_viable_dataset(data_size, full_df_norm, buys, sells)

        train_size = int(0.6 * data_size)
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

        print("    #training samples:", len(df_train), " #buys:", int(train_buys.sum()), ' #sells:', int(train_sells.sum()))

        #TODO: if low number of buys/sells, try k-fold sampling

        buy_labels = self.get_binary_labels(buys)
        sell_labels = self.get_binary_labels(sells)
        train_buy_labels = self.get_binary_labels(train_buys)
        train_sell_labels = self.get_binary_labels(train_sells)
        test_buy_labels = self.get_binary_labels(test_buys)
        test_sell_labels = self.get_binary_labels(test_sells)

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

        sell_clf, sell_clf_name = self.get_sell_classifier(df_train_pca, train_sell_labels)

        # save the models

        self.pair_model_info[curr_pair]['pca'] = pca
        self.pair_model_info[curr_pair]['pca_size'] = df_train_pca.shape[1]
        self.pair_model_info[curr_pair]['clf_buy_name'] = buy_clf_name
        self.pair_model_info[curr_pair]['clf_buy'] = buy_clf
        self.pair_model_info[curr_pair]['clf_sell_name'] = sell_clf_name
        self.pair_model_info[curr_pair]['clf_sell'] = sell_clf

        # if scan specified, test against the test dataframe
        if self.dbg_scan_classifiers and self.dbg_verbose:

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

    # get the PCA model for the supplied dataframe (dataframe must be normalised)
    def get_pca(self, df_norm: DataFrame):

        ncols = df_norm.shape[1]  # allow all components to get the full variance matrix
        whiten = True
        # pca = skd.PCA(n_components=ncols, svd_solver="randomized", whiten=whiten).fit(df_norm)
        pca = skd.PCA(n_components=ncols, whiten=whiten, svd_solver='full').fit(df_norm)

        # if self.dbg_verbose:
        #     print ("PCA variance_ratio: ", pca.explained_variance_ratio_)

        # scan variance and only take if column contributes >x%
        ncols = 0
        var_sum = 0.0
        variance_threshold = 0.999
        # variance_threshold = 0.99
        while ((var_sum < variance_threshold) & (ncols < len(pca.explained_variance_ratio_))):
            var_sum = var_sum + pca.explained_variance_ratio_[ncols]
            ncols = ncols + 1

        # if necessary, re-calculate pca with reduced column set
        if (ncols != df_norm.shape[1]):
            # pca = skd.PCA(n_components=ncols, svd_solver="randomized", whiten=True).fit(df_norm)
            pca = skd.PCA(n_components=ncols, whiten=whiten, svd_solver='full').fit(df_norm)

        self.check_pca(pca, df_norm)

        if self.dbg_analyse_pca and self.dbg_verbose:
            self.analyse_pca(pca, df_norm)

        return pca

    # does a quick for suspicious values. Separate func because we always want to call this
    def check_pca(self, pca, df):

        ratios = pca.explained_variance_ratio_
        loadings = pd.DataFrame(pca.components_.T, index=df.columns.values)

        # check variance ratios
        var_big = np.where(ratios >= 0.5)[0]
        if len(var_big) > 0:
            print("    !!! high variance in columns: ", var_big)
            print("    !!! variances: ", ratios)

        var_0  = np.where(ratios == 0)[0]
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
            print("    !!! zero values in rows (remove indicator?!) : ", zero_rows)

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
        print (l3.sort_values('Rank').head(n=20))

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
        labels = self.get_binary_labels(results)

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
        labels = self.get_binary_labels(results)

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

    # default classifier
    default_classifier = 'LDA'  # select based on testing

    # list of potential classifier types - set to the list that you want to compare
    classifier_list = [
        'LogisticRegression', 'GaussianNB', 'SGD',
        'GradientBoosting', 'AdaBoost', 'linearSVC', 'sigmoidSVC',
        'LDA'
    ]

    # factory to create classifier based on name
    def classifier_factory(self, name, data, labels):
        clf = None

        if name == 'LogisticRegression':
            clf = LogisticRegression(max_iter=10000)
        elif name == 'DecisionTree':
            clf = DecisionTreeClassifier()
        elif name == 'RandomForest':
            clf = RandomForestClassifier()
        elif name == 'GaussianNB':
            clf = GaussianNB()
        elif name == 'MLP':
            param_grid = {
                'hidden_layer_sizes': [(30, 2), (30, 80, 2), (30, 60, 30, 2)],
                'max_iter': [50, 100, 150],
                'activation': ['tanh', 'relu'],
                'solver': ['sgd', 'adam', 'lbfgs'],
                'alpha': [0.0001, 0.05],
                'learning_rate': ['constant', 'adaptive'],
            }
            clf = MLPClassifier(hidden_layer_sizes=(16, 4, 2),
                                max_iter=50,
                                activation='relu',
                                learning_rate='adaptive',
                                alpha=1e-5,
                                solver='lbfgs',
                                verbose=0)


        elif name == 'KNeighbors':
            clf = KNeighborsClassifier(n_neighbors=3)
        elif name == 'SGD':
            clf = SGDClassifier()
        elif name == 'GradientBoosting':
            clf = GradientBoostingClassifier()
        elif name == 'AdaBoost':
            clf = AdaBoostClassifier()
        elif name == 'QDA':
            clf = QuadraticDiscriminantAnalysis()
        elif name == 'linearSVC':
            clf = LinearSVC(dual=False)
        elif name == 'gaussianSVC':
            clf = SVC(kernel='rbf')
        elif name == 'polySVC':
            clf = SVC(kernel='poly')
        elif name == 'sigmoidSVC':
            clf = SVC(kernel='sigmoid')
        elif name == 'Voting':
            # choose 4 decent classifiers
            c1, _ = self.classifier_factory('AdaBoost', data, labels)
            c2, _ = self.classifier_factory('GaussianNB', data, labels)
            c3, _ = self.classifier_factory('KNeighbors', data, labels)
            c4, _ = self.classifier_factory('DecisionTree', data, labels)
            clf = VotingClassifier(estimators=[('c1', c1), ('c2', c2), ('c3', c3), ('c4', c4)], voting='hard')
        elif name == 'LDA':
            clf = LinearDiscriminantAnalysis()
        elif name == 'QDA':
            clf = QuadraticDiscriminantAnalysis()


        else:
            print("Unknown classifier: ", name)
            clf = None
        return clf, name

    # tries different types of classifiers and returns the best one
    # tag parameter identifies where to save performance stats (default is not to save)
    def find_best_classifier(self, df, results, tag=""):

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

        labels = self.get_binary_labels(results)

        # split into test/train for evaluation, then re-fit once selected
        # df_train, df_test, res_train, res_test = train_test_split(df, results, train_size=0.5)
        df_train, df_test, res_train, res_test = train_test_split(df, labels, train_size=0.8,
                                                                  random_state=27, shuffle=True)
        # print("df_train:",  df_train.shape, " df_test:", df_test.shape,
        #       "res_train:", res_train.shape, "res_test:", res_test.shape)

        # check there are enough training samples
        #TODO: if low train/test samples, use k-fold sampling nstead
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
                        self.classifier_stats[tag][cname] = { 'count': 0, 'score': 0.0, 'selected': 0}

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
            # print("predict - dataframe:", dataframe.shape)
            df_norm = self.norm_dataframe(dataframe)
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
        clf = self.pair_model_info[pair]['clf_sell']
        if clf is None:
            print("    No Sell Classifier for pair ", pair, " -Skipping predictions")
            self.pair_model_info[pair]['interval'] = min(self.pair_model_info[pair]['interval'], 4)
            predict = df['close']  # just to get the size
            predict = 0.0
            return predict

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

    def populate_buy_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        conditions = []
        dataframe.loc[:, 'enter_tag'] = ''
        curr_pair = metadata['pair']

        self.set_state(curr_pair, self.State.RUNNING)

        if not self.dp.runmode.value in ('hyperopt'):
            if PCA2.first_run:
                PCA2.first_run = False # note use of clas variable, not instance variable
                # self.show_debug_info(curr_pair)
                self.show_all_debug_info()

        conditions.append(dataframe['volume'] > 0)

        # add some fairly loose guards, to help prevent 'bad' predictions

        # # ATR in buy range
        # conditions.append(dataframe['atr_signal'] > 0.0)

        # some trading volume
        conditions.append(dataframe['volume'] > 0)

        # Fisher RSI + Williams combo
        conditions.append(dataframe['fisher_wr'] < -0.7)

        # below Bollinger mid-point
        conditions.append(dataframe['close'] < dataframe['bb_middleband'])

        # PCA/Classifier triggers
        pca_cond = (
            (qtpylib.crossed_above(dataframe['predict_buy'], 0.5))
        )
        conditions.append(pca_cond)

        # set entry tags
        dataframe.loc[pca_cond, 'enter_tag'] += 'pca_entry '

        if conditions:
            dataframe.loc[reduce(lambda x, y: x & y, conditions), 'buy'] = 1
        else:
            dataframe['buy'] = 0

        return dataframe

    ###################################

    """
    Sell Signal
    """

    def populate_sell_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        conditions = []
        dataframe.loc[:, 'exit_tag'] = ''
        curr_pair = metadata['pair']

        self.set_state(curr_pair, self.State.RUNNING)

        if not self.dp.runmode.value in ('hyperopt'):
            if PCA2.first_run:
                PCA2.first_run = False # note use of clas variable, not instance variable
                # self.show_debug_info(curr_pair)
                self.show_all_debug_info()

        conditions.append(dataframe['volume'] > 0)

        # # ATR in sell range
        # conditions.append(dataframe['atr_signal'] <= 0.0)

        # above Bollinger mid-point
        conditions.append(dataframe['close'] > dataframe['bb_middleband'])

        # Fisher RSI + Williams combo
        conditions.append(dataframe['fisher_wr'] > 0.5)

        # PCA triggers
        pca_cond = (
            qtpylib.crossed_above(dataframe['predict_sell'], 0.5)
        )

        conditions.append(pca_cond)

        dataframe.loc[pca_cond, 'exit_tag'] += 'pca_exit '

        if conditions:
            dataframe.loc[reduce(lambda x, y: x & y, conditions), 'sell'] = 1
        else:
            dataframe['sell'] = 0

        return dataframe

    ###################################

    ## Custom Trailing stoploss ( credit to Perkmeister for this custom stoploss to help the strategy ride a green candle )
    def custom_stoploss(self, pair: str, trade: 'Trade', current_time: datetime,
                        current_rate: float, current_profit: float, **kwargs) -> float:

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

