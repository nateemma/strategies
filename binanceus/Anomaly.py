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

from CompressionAutoEncoder import CompressionAutoEncoder

from AnomalyDetector_AEnc import AnomalyDetector_AEnc
from AnomalyDetector_LOF import AnomalyDetector_LOF
from AnomalyDetector_KMeans import AnomalyDetector_KMeans
from AnomalyDetector_IFOR import AnomalyDetector_IFOR
from AnomalyDetector_EE import AnomalyDetector_EE
from AnomalyDetector_SVM import AnomalyDetector_SVM
from AnomalyDetector_PCA import AnomalyDetector_PCA

from AnomalyDetector_LSTM import AnomalyDetector_LSTM

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
            '%recon': {'color': 'lightsalmon'},
        },
        'subplots': {
            "Diff": {
                '%train_buy': {'color': 'green'},
                'predict_buy': {'color': 'blue'},
                '%train_sell': {'color': 'red'},
                'predict_sell': {'color': 'orange'},
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
    n_profit_stddevs = 1.0
    n_loss_stddevs = 1.0
    min_f1_score = 0.49

    curr_lookahead = int(12 * lookahead_hours)

    curr_pair = ""
    custom_trade_info = {}

    compressor = None
    compress_data = False

    # profit/loss thresholds used for assessing buy/sell signals. Keep these realistic!
    # Note: if self.dynamic_gain_thresholds is True, these will be adjusted for each pair, based on historical mean
    default_profit_threshold = 0.3
    default_loss_threshold = -0.3
    profit_threshold = default_profit_threshold
    loss_threshold = default_loss_threshold
    dynamic_gain_thresholds = True  # dynamically adjust gain thresholds based on actual mean (beware, training data could be bad)

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
        cstop_bail_time_trend = CategoricalParameter([True, False], default=True, space='sell', load=True,
                                                     optimize=True)
        cstop_max_stoploss = DecimalParameter(-0.30, -0.01, default=-0.10, space='sell', load=True, optimize=True)

    ################################

    # subclasses should override the following 2 functions - this is here as an example

    # Note: try to combine current/historical data (from populate_indicators) with future data
    #       If you only use future data, the ML training is just guessing
    #       Also, try to identify buy/sell ranges, rather than transitions - it gives the algorithms more chances
    #       to find a correlation. The framework will select the first one anyway.
    #       In other words, avoid using qtpylib.crossed_above() and qtpylib.crossed_below()
    #       Proably OK not to check volume, because we are just looking for patterns

    def get_train_buy_signals(self, future_df: DataFrame):

        # print("!!! WARNING: using base class (buy) training implementation !!!")

        series = np.where(
            (
                    # (future_df['mfi'] <= 20) &  # loose oversold threshold
                    # (future_df['close'] < future_df['tema']) &  # below average
                    # (future_df['close'] < future_df['close'].shift(self.curr_lookahead)) &
                    (future_df['future_gain'] >= self.profit_threshold) &  # future gain above threshold
                    (future_df['dwt_bottom'] > 0)  # bottom of trend
            ), 1.0, 0.0)

        return series

    def get_train_sell_signals(self, future_df: DataFrame):

        # print("!!! WARNING: using base class (sell) training implementation !!!")

        series = np.where(
            (
                    # (future_df['mfi'] >= 80) &  # loose overbought threshold
                    # (future_df['close'] > future_df['tema']) &  # above average
                    # (future_df['close'] > future_df['close'].shift(self.curr_lookahead)) &
                    (future_df['future_gain'] <= self.loss_threshold) &  # future loss above threshold
                    (future_df['dwt_top'] > 0)  # top of trend
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

        print("")
        print(curr_pair)

        # populate the normal dataframe
        dataframe = self.add_indicators(dataframe)

        if Anomaly.first_time:
            Anomaly.first_time = False
            print("")
            print("***************************************")
            print("** Warning: startup can be very slow **")
            print("***************************************")

            print("    Lookahead: ", self.curr_lookahead, " candles (", self.lookahead_hours, " hours)")

        # create labels used for training
        buys, sells = self.create_training_data(dataframe)

        # drop last group (because there cannot be a prediction)
        df = dataframe.iloc[:-self.curr_lookahead]
        buys = buys.iloc[:-self.curr_lookahead]
        sells = sells.iloc[:-self.curr_lookahead]

        # train the models on the informative data
        if self.dbg_verbose:
            print("    training models...")
        df = self.train_models(curr_pair, df, buys, sells)

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

        # if self.dbg_verbose:
        #     print("    saving models...")
        # if self.buy_classifier:
        #     self.buy_classifier.save()
        # if self.sell_classifier:
        #     self.sell_classifier.save()

        return dataframe

    ###################################

    # add indicators used by stoploss/custom sell logic
    def add_stoploss_indicators(self, dataframe, pair) -> DataFrame:
        if not pair in self.custom_trade_info:
            self.custom_trade_info[pair] = {}
            if not 'had_trend' in self.custom_trade_info[pair]:
                self.custom_trade_info[pair]['had_trend'] = False

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
        # dataframe['dc_hf'] = dataframe['dc_upper'] - dataframe['dc_dist'] * 0.236  # Highest Fib
        # dataframe['dc_chf'] = dataframe['dc_upper'] - dataframe['dc_dist'] * 0.382  # Centre High Fib
        # dataframe['dc_clf'] = dataframe['dc_upper'] - dataframe['dc_dist'] * 0.618  # Centre Low Fib
        # dataframe['dc_lf'] = dataframe['dc_upper'] - dataframe['dc_dist'] * 0.764  # Low Fib

        #  # Keltner Channels
        # keltner = qtpylib.keltner_channel(dataframe)
        # dataframe["kc_upper"] = keltner["upper"]
        # dataframe["kc_lower"] = keltner["lower"]
        # dataframe["kc_mid"] = keltner["mid"]

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

        # # EMAs
        # dataframe['ema_12'] = ta.EMA(dataframe, timeperiod=12)
        # dataframe['ema_20'] = ta.EMA(dataframe, timeperiod=20)
        # dataframe['ema_25'] = ta.EMA(dataframe, timeperiod=25)
        # dataframe['ema_35'] = ta.EMA(dataframe, timeperiod=35)
        # dataframe['ema_50'] = ta.EMA(dataframe, timeperiod=50)
        # dataframe['ema_100'] = ta.EMA(dataframe, timeperiod=100)
        # dataframe['ema_200'] = ta.EMA(dataframe, timeperiod=200)

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

        # # T3 Average
        # dataframe['t3_avg'] = t3_average(dataframe)

        # # S/R
        # res_series = dataframe['high'].rolling(window=5, center=True).apply(lambda row: is_resistance(row),
        #                                                                     raw=True).shift(2)
        # sup_series = dataframe['low'].rolling(window=5, center=True).apply(lambda row: is_support(row),
        #                                                                    raw=True).shift(2)
        # dataframe['res_level'] = Series(
        #     np.where(res_series,
        #              np.where(dataframe['close'] > dataframe['open'], dataframe['close'], dataframe['open']),
        #              float('NaN'))).ffill()
        # dataframe['res_hlevel'] = Series(np.where(res_series, dataframe['high'], float('NaN'))).ffill()
        # dataframe['sup_level'] = Series(
        #     np.where(sup_series,
        #              np.where(dataframe['close'] < dataframe['open'], dataframe['close'], dataframe['open']),
        #              float('NaN'))).ffill()

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

        # # SAR Parabol
        # dataframe['sar'] = ta.SAR(dataframe)

        dataframe['mom'] = ta.MOM(dataframe, timeperiod=14)

        # priming indicators
        dataframe['color'] = np.where((dataframe['close'] > dataframe['open']), 1.0, -1.0)
        dataframe['rsi_7'] = ta.RSI(dataframe, timeperiod=7)
        dataframe['roc_6'] = ta.ROC(dataframe, timeperiod=6)
        dataframe['primed'] = np.where(dataframe['color'].rolling(3).sum() == 3.0, 1.0, -1.0)
        dataframe['in_the_mood'] = np.where(dataframe['rsi_7'] > dataframe['rsi_7'].rolling(12).mean(), 1.0, -1.0)
        dataframe['moist'] = np.where(qtpylib.crossed_above(dataframe['macd'], dataframe['macdsignal']), 1.0, -1.0)
        dataframe['throbbing'] = np.where(dataframe['roc_6'] > dataframe['roc_6'].rolling(12).mean(), 1.0, -1.0)

        # MFI
        dataframe['mfi'] = ta.MFI(dataframe)

        # Volume Flow Indicator (MFI) for volume based on the direction of price movement
        dataframe['vfi'] = fta.VFI(dataframe, period=14)

        # ATR
        dataframe['atr'] = ta.ATR(dataframe, timeperiod=win_size)

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
        # if in backtest or hyperopt, then we have to do rolling calculations
        if self.dp.runmode.value in ('hyperopt', 'backtest'):
            dataframe['dwt'] = dataframe['close'].rolling(window=self.dwt_window).apply(self.roll_get_dwt)
            dataframe['smooth'] = dataframe['close'].rolling(window=self.dwt_window).apply(self.roll_smooth)
            dataframe['dwt_smooth'] = dataframe['dwt'].rolling(window=self.dwt_window).apply(self.roll_smooth)
        else:
            dataframe['dwt'] = self.get_dwt(dataframe['close'])
            dataframe['smooth'] = gaussian_filter1d(dataframe['close'], 2)
            dataframe['dwt_smooth'] = gaussian_filter1d(dataframe['dwt'], 2)

        # smoothed version - useful for trends
        # dataframe['dwt_smooth'] = gaussian_filter1d(dataframe['dwt'], 8)

        # dataframe['dwt_deriv'] = np.gradient(dataframe['dwt_smooth'])
        dataframe['dwt_deriv'] = np.gradient(dataframe['dwt'])
        dataframe['dwt_top'] = np.where(qtpylib.crossed_below(dataframe['dwt_deriv'], 0.0), 1, 0)
        dataframe['dwt_bottom'] = np.where(qtpylib.crossed_above(dataframe['dwt_deriv'], 0.0), 1, 0)

        dataframe['dwt_diff'] = 100.0 * (dataframe['dwt'] - dataframe['close']) / dataframe['close']
        dataframe['dwt_smooth_diff'] = 100.0 * (dataframe['dwt'] - dataframe['dwt_smooth']) / dataframe['dwt_smooth']

        # up/down direction
        dataframe['dwt_dir'] = 0.0
        # dataframe['dwt_dir'] = np.where(dataframe['dwt'].diff() >= 0, 1.0, -1.0)
        dataframe['dwt_dir'] = np.where(dataframe['dwt_smooth'].diff() > 0, 1.0, -1.0)

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
        # dataframe['dwt_nseq'] = dataframe['dwt_dir'].rolling(window=win_size, min_periods=1).sum()
        dataframe['dwt_nseq'] = dataframe['dwt_trend'].rolling(window=win_size, min_periods=1).sum()

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
            if self.dbg_verbose:
                print("    Profit threshold {:.4f} -> {:.4f}".format(self.profit_threshold, newval))
            self.profit_threshold = newval

        if self.loss_threshold != loss_mean:
            newval = loss_mean - self.n_loss_stddevs * abs(loss_std)
            if self.dbg_verbose:
                print("    Loss threshold {:.4f} -> {:.4f}".format(self.loss_threshold, newval))
            self.loss_threshold = newval

        return future_df

    ################################

    # creates the buy/sell labels absed on looking ahead into the supplied dataframe
    def create_training_data(self, dataframe: DataFrame):

        future_df = self.add_future_data(dataframe.copy())

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
            return col[len(col) - 1]

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

    ############################

    def get_classifier(self, nfeatures, tag):
        clf = None
        clf_type = 4

        if clf_type == 0:
            # NOTE: have to be careful about dimensions (must match training)
            # ONLY use this option to train the compression autoencoder
            if self.compress_data:
                print("WARNING: self.compress_data should be False")
            clf = CompressionAutoEncoder(nfeatures, tag=tag)
        elif clf_type == 1:
            clf = AnomalyDetector_AEnc("BuyAnomalyDetector", nfeatures, tag=tag)
        elif clf_type == 2:
            clf = AnomalyDetector_LOF()
        elif clf_type == 3:
            clf = AnomalyDetector_KMeans()
        elif clf_type == 4:
            clf = AnomalyDetector_IFOR(self.curr_pair, tag=tag)
        elif clf_type == 5:
            clf = AnomalyDetector_EE()
        elif clf_type == 6:
            clf = AnomalyDetector_SVM()
        elif clf_type == 7:
            clf = AnomalyDetector_PCA(nfeatures)
        elif clf_type == 8:
            clf = AnomalyDetector_LSTM(nfeatures, self.curr_pair)

        else:
            print("    ERR: unknown classifier type ({})".format(clf_type))

        return clf

    ############################

    # get a scaler for scaling/normalising the data (in a func because I change it routinely)
    def get_scaler(self):
        # uncomment the one yu want
        # return StandardScaler()
        # return RobustScaler()
        return MinMaxScaler()

    def norm_column(self, col):
        return self.zscore_column(col)

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

    scaler = None

    # Normalise a dataframe
    def norm_dataframe(self, dataframe: DataFrame) -> DataFrame:
        self.check_inf(dataframe)

        temp = dataframe.copy()
        if 'date' in temp.columns:
            temp['date'] = pd.to_datetime(temp['date']).astype('int64')

        temp = self.remove_debug_columns(temp)

        temp.set_index('date')
        temp.reindex()

        cols = temp.columns
        self.scaler = self.get_scaler()

        temp = pd.DataFrame(self.scaler.fit_transform(temp), columns=cols)

        return temp

    # De-Normalise a dataframe - note this relies on the scaler still being valid
    def denorm_dataframe(self, dataframe: DataFrame) -> DataFrame:

        temp = dataframe.copy()

        cols = temp.columns

        df = pd.DataFrame(self.scaler.inverse_transform(dataframe), columns=cols)

        return df

    # slit a dataframe into two, based on the supplied ratio
    def split_dataframe(self, dataframe: DataFrame, ratio: float) -> (DataFrame, DataFrame):
        split_row = int(ratio * dataframe.shape[0])
        df1 = dataframe.iloc[0:split_row].copy()
        df2 = dataframe.iloc[split_row + 1:].copy()
        return df1, df2

    # slit an array into two, based on the supplied ratio
    def split_array(self, array, ratio: float) -> (DataFrame, DataFrame):
        split_row = int(ratio * np.shape(array)[0])
        a1 = array[0:split_row].copy()
        a2 = array[split_row + 1:].copy()
        return a1, a2

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
        # if self.dbg_verbose:
        #     print("     df_norm:{} size:{} buys:{} sells:{}".format(df_norm.shape, size, buys.shape[0], sells.shape[0]))

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
        max_signals = int(2 * size / 3)
        buy_train_size = df_buy.shape[0]
        sell_train_size = df_sell.shape[0]

        if max_signals > df_nosig.shape[0]:
            max_signals = int((size - df_nosig.shape[0])) - 1

        if ((df_buy.shape[0] + df_sell.shape[0]) > max_signals):
            # both exceed max?
            sig_size = int(max_signals / 2)
            # if self.dbg_verbose:
            #     print("     sig_size:{} max_signals:{} buys:{} sells:{}".format(sig_size, max_signals, df_buy.shape[0],
            #                                                                     df_sell.shape[0]))

            if (df_buy.shape[0] > sig_size) & (df_sell.shape[0] > sig_size):
                # resize both buy & sell to 1/3 of requested size
                buy_train_size = sig_size
                sell_train_size = sig_size
            else:
                # only one them is too big, so figure out which
                if (df_buy.shape[0] > df_sell.shape[0]):
                    buy_train_size = max_signals - df_sell.shape[0]
                else:
                    sell_train_size = max_signals - df_buy.shape[0]

            # if self.dbg_verbose:
            #     print("     buy_train_size:{} sell_train_size:{}".format(buy_train_size, sell_train_size))

        if buy_train_size < df_buy.shape[0]:
            df_buy, _ = train_test_split(df_buy, train_size=buy_train_size, shuffle=False)
        if sell_train_size < df_sell.shape[0]:
            df_sell, _ = train_test_split(df_sell, train_size=sell_train_size, shuffle=False)

        # extract enough rows to fill the requested size
        fill_size = size - buy_train_size - sell_train_size - 1
        # if self.dbg_verbose:
        #     print("     df_nosig:{} fill_size:{}".format(df_nosig.shape, fill_size))

        if fill_size < df_nosig.shape[0]:
            df_nosig, _ = train_test_split(df_nosig, train_size=fill_size, shuffle=False)

        # print("viable df - buys:{} sells:{} fill:{}".format(df_buy.shape[0], df_sell.shape[0], df_nosig.shape[0]))

        # concatenate the dataframes
        frames = [df_buy, df_sell, df_nosig]
        df2 = pd.concat(frames)

        # # shuffle rows
        # df2 = df2.sample(frac=1)

        # separate out the data, buys & sells
        b = df2['%temp_buy'].copy()
        s = df2['%temp_sell'].copy()
        df2.drop('%temp_buy', axis=1, inplace=True)
        df2.drop('%temp_sell', axis=1, inplace=True)
        df2.reindex()

        if self.dbg_verbose:
            print("     df2:", df2.shape, " b:", b.shape, " s:", s.shape)

        return df2, b, s

    # map column into [0,1]
    def get_binary_labels(self, col):
        binary_encoder = LabelEncoder().fit([min(col), max(col)])
        result = binary_encoder.transform(col)
        # print ("label input:  ", col)
        # print ("label output: ", result)
        return result

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

        full_df_norm = self.norm_dataframe(dataframe)

        if self.compress_data:
            old_size = full_df_norm.shape[1]
            full_df_norm = self.compress_dataframe(full_df_norm)
            print("    Compressed data {} -> {} (features)".format(old_size, full_df_norm.shape[1]))
        else:
            if self.dbg_verbose:
                print("    Not compressing data")

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
        df_test, df_train = self.split_dataframe(full_df_norm, (1.0 - train_ratio))
        test_buys, train_buys = self.split_array(buys, (1.0 - train_ratio))
        test_sells, train_sells = self.split_array(sells, (1.0 - train_ratio))

        if self.dbg_verbose:
            print("     dataframe:", full_df_norm.shape, ' -> train:', df_train.shape, " + test:", df_test.shape)
            print("     buys:", buys.shape, ' -> train:', train_buys.shape, " + test:", test_buys.shape)
            print("     sells:", sells.shape, ' -> train:', train_sells.shape, " + test:", test_sells.shape)

        print("    #training samples:", len(df_train), " #buys:", int(train_buys.sum()), ' #sells:',
              int(train_sells.sum()))

        train_buy_labels = self.get_binary_labels(train_buys)
        train_sell_labels = self.get_binary_labels(train_sells)
        test_buy_labels = self.get_binary_labels(test_buys)
        test_sell_labels = self.get_binary_labels(test_sells)

        # Buy Classifier

        # # create classifiers, if necessary

        if self.curr_pair not in self.buy_classifier_list:
            self.buy_classifier = self.get_classifier(full_df_norm.shape[1], "Buy")
            self.buy_classifier_list[self.curr_pair] = self.buy_classifier
        else:
            self.buy_classifier = self.buy_classifier_list[self.curr_pair]

        # if self.dp.runmode.value not in ('plot'):
        # train/fit the classifiers (note, this is cumulative)
        force_train = True if (self.dp.runmode.value in ('backtest')) else False

        self.buy_classifier.train(df_train, df_test, train_buys, test_buys, force_train=force_train)

        # Sell Classifier

        if self.curr_pair not in self.sell_classifier_list:
            self.sell_classifier = self.get_classifier(full_df_norm.shape[1], "Sell")
            self.sell_classifier_list[self.curr_pair] = self.sell_classifier
        else:
            self.sell_classifier = self.sell_classifier_list[self.curr_pair]

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
                df_norm = self.norm_dataframe(dataframe)  # this also resets the scaler
                df_compressed = self.compress_dataframe(df_norm)
                df_recon_compressed = self.buy_classifier.reconstruct(df_compressed)
                df_recon_norm = self.compressor.inverse_transform(df_recon_compressed)
                df_recon_norm = pd.DataFrame(df_recon_norm, columns=df_norm.columns)
                df_recon = self.denorm_dataframe(df_recon_norm)
                dataframe['%recon'] = df_recon['close']
            else:
                # debug: get reconstructed dataframe and save 'close' as a comparison
                tmp = self.norm_dataframe(dataframe)  # this just resets the scaler
                df_recon_norm = self.buy_classifier.reconstruct(tmp)
                df_recon = self.denorm_dataframe(df_recon_norm)
                dataframe['%recon'] = df_recon['close']
        return dataframe

    # remove any rows thatr are buys or sells from training data (needed by some classifiers)
    def clean_training_data(self, df_train, train_buys, train_sells) -> DataFrame:

        # print("    removing anomalies from training data")
        df = df_train.copy()
        df['%buys'] = train_buys
        df['%sells'] = train_sells
        df = df[((df['%buys'] < 1.0) & (df['%sells'] < 1.0))]
        df = df.drop(['%buys', '%sells'], axis=1)

        return df

    # compress the suplied dataframe
    def compress_dataframe(self, dataframe: DataFrame) -> DataFrame:
        if not self.compressor:
            self.compressor = self.get_compressor(dataframe)
        return pd.DataFrame(self.compressor.transform(dataframe))

    # get the compressor model for the supplied dataframe (dataframe must be normalised)
    # use .transform() to compress the dataframe
    def get_compressor(self, df_norm: DataFrame):

        ncols = df_norm.shape[1]  # allow all components to get the full variance matrix
        whiten = True

        compressor_type = 0

        # there are various types of PCA, plus alternatives like ICA and Feature Extraction
        if compressor_type == 0:
            '''
            compressor = skd.PCA(n_components=ncols, whiten=whiten, svd_solver='full').fit(df_norm)
            var_ratios = compressor.explained_variance_ratio_

            # if self.dbg_verbose:
            #     print ("PCA variance_ratio: ", compressor.explained_variance_ratio_)

            # scan variance and only take if column contributes >x%
            ncols = 0
            var_sum = 0.0
            variance_threshold = 0.98  # bias towards compression rather than accuracy
            # variance_threshold = 0.99
            while ((var_sum < variance_threshold) & (ncols < len(var_ratios))):
                var_sum = var_sum + var_ratios[ncols]
                ncols = ncols + 1

            # if necessary, re-calculate compressor with reduced column set
            if (ncols != df_norm.shape[1]):
                # compressor = skd.PCA(n_components=ncols, svd_solver="randomized", whiten=True).fit(df_norm)
                compressor = skd.PCA(n_components=ncols, whiten=whiten, svd_solver='full').fit(df_norm)
            '''
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
            print("*** ERR - unknown PCA type ***")
            compressor = None

        return compressor

    # make predictions for supplied dataframe (returns column)
    def predict(self, dataframe: DataFrame, pair, clf):

        # predict = 0
        predict = None

        if clf:
            # print("    predicting... - dataframe:", dataframe.shape)
            df_norm = self.norm_dataframe(dataframe)
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

    def populate_buy_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
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
        conditions.append(dataframe['mfi'] < 20.0)

        # below TEMA
        conditions.append(dataframe['close'] < dataframe['tema'])

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

    def populate_sell_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        conditions = []
        dataframe.loc[:, 'exit_tag'] = ''
        curr_pair = metadata['pair']

        self.set_state(curr_pair, self.State.RUNNING)

        if not self.dp.runmode.value in ('hyperopt'):
            if Anomaly.first_run:
                Anomaly.first_run = False  # note use of clas variable, not instance variable
                # self.show_debug_info(curr_pair)
                self.show_all_debug_info()

        conditions.append(dataframe['volume'] > 0)

        # # ATR in sell range
        # conditions.append(dataframe['atr_signal'] <= 0.0)

        # above Bollinger mid-point
        conditions.append(dataframe['close'] > dataframe['bb_middleband'])

        # # Fisher RSI + Williams combo
        # conditions.append(dataframe['fisher_wr'] > 0.5)

        # MFI
        conditions.append(dataframe['mfi'] > 80.0)

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

        # # hard stoploss profit
        # HSL = self.pHSL.value
        # PF_1 = self.pPF_1.value
        # SL_1 = self.pSL_1.value
        # PF_2 = self.pPF_2.value
        # SL_2 = self.pSL_2.value
        #
        # # For profits between PF_1 and PF_2 the stoploss (sl_profit) used is linearly interpolated
        # # between the values of SL_1 and SL_2. For all profits above PL_2 the sl_profit value
        # # rises linearly with current profit, for profits below PF_1 the hard stoploss profit is used.
        #
        # if (current_profit > PF_2):
        #     sl_profit = SL_2 + (current_profit - PF_2)
        # elif (current_profit > PF_1):
        #     sl_profit = SL_1 + ((current_profit - PF_1) * (SL_2 - SL_1) / (PF_2 - PF_1))
        # else:
        #     sl_profit = HSL
        #
        # # Only for hyperopt invalid return
        # if (sl_profit >= current_profit):
        #     return -0.99
        #
        # return min(-0.01, max(stoploss_from_open(sl_profit, current_profit), -0.99))

        if current_profit < 0.02:
            return -1  # return a value bigger than the initial stoploss to keep using the initial stoploss

        # After reaching the desired offset, allow the stoploss to trail by half the profit
        desired_stoploss = current_profit / 4

        # Use a minimum of 1% and a maximum of 10%
        return max(min(desired_stoploss, 0.10), 0.01)

    ###################################

    """
    Custom Sell
    """

    def custom_sell(self, pair: str, trade: 'Trade', current_time: 'datetime', current_rate: float,
                    current_profit: float, **kwargs):
        if self.use_simpler_custom_stoploss:
            return self.simpler_custom_sell(pair, trade, current_time, current_rate, current_profit)
        else:
            return self.complex_custom_sell(pair, trade, current_time, current_rate, current_profit)

    def complex_custom_sell(self, pair: str, trade: 'Trade', current_time: 'datetime', current_rate: float,
                            current_profit: float):

        dataframe, _ = self.dp.get_analyzed_dataframe(pair=pair, timeframe=self.timeframe)
        last_candle = dataframe.iloc[-1].squeeze()

        trade_dur = int((current_time.timestamp() - trade.open_date_utc.timestamp()) // 60)
        max_profit = max(0, trade.calc_profit_ratio(trade.max_rate))
        pullback_value = max(0, (max_profit - self.csell_pullback_amount.value))
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

    def simpler_custom_sell(self, pair: str, trade: 'Trade', current_time: 'datetime', current_rate: float,
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

        # Sell any positions at a loss if they are held for more than one day.
        if current_profit < 0.0 and (current_time - trade.open_date_utc).days >= 2:
            return 'unclog'

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
