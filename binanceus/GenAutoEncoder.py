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
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler
from sklearn.preprocessing import LabelEncoder, StandardScaler

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

from AutoEncoder import AutoEncoder
from LSTMAutoEncoder import LSTMAutoEncoder
from LSTM2AutoEncoder import LSTM2AutoEncoder

"""
####################################################################################
GenAutoEncoder - Generates an auto encoder for dimensional reduction of stock data.
                This strategy does not actually do any trading, it is intended to generate a compression
                model for stock data. It works using the auto-encoder technique, which basically 'encodes'
                data down to lower dimensions using an RNN, then decodes it using the inverse transform and compares
                the results to the original. The network is thenm trained until it starts matching the input data.
                You can then save just the encoder portion and use that independently as a way of reducing the
                dimensions of the input data.
                See: https://ekamperi.github.io/machine%20learning/2021/01/21/encoder-decoder-model.html
                
                Note that the list of indicators need to match, otherwise you will get a mismatch in model vs data
                dimensions.
      
      In addition to the normal freqtrade packages, these strategies also require the installation of:
        random
        prettytable
        finta

####################################################################################
"""


class GenAutoEncoder(IStrategy):
    # the following are included just to keep freqtrade happy. There is no need to hyperopt

    # ROI table:
    minimal_roi = {
        "0": 0.05
    }

    # Stoploss:
    stoploss = -0.05

    # Trailing stop:
    trailing_stop = False
    trailing_stop_positive = None
    trailing_stop_positive_offset = 0.0
    trailing_only_offset_is_reached = False

    timeframe = '5m'

    inf_timeframe = '5m'

    use_custom_stoploss = False

    # Recommended
    use_entry_signal = True
    entry_profit_only = False
    ignore_roi_if_entry_signal = True

    # Required
    startup_candle_count: int = 128  # must be power of 2
    process_only_new_candles = True

    # compatibility with PCA strats
    curr_lookahead = 12
    dwt_window = startup_candle_count
    n_profit_stddevs = 2.0
    n_loss_stddevs = 2.0
    min_f1_score = 0.6
    default_profit_threshold = 0.3
    default_loss_threshold = -0.3
    profit_threshold = default_profit_threshold
    loss_threshold = default_loss_threshold
    dynamic_gain_thresholds = True  # dynamically adjust gain thresholds based on actual mean (beware, training data could be bad)

    # strategy-specific vars

    num_pairs = 0
    model_info = {}  # holds model-related info

    # This is the dimension of the latent space (encoding space)
    latent_dim = 32
    num_features = 128  # will change based on populate_indicators

    # the following affect training of the model. Bigger numbers give better model, but take longer and use more memory
    seq_len = 8  # 'depth' of training sequence
    num_epochs = 64
    # num_epochs = 512  # number of iterations for training
    batch_size = 1024  # batch size for training

    # debug flags
    first_time = True  # mostly for debug
    first_run = True  # used to identify first time through buy/sell populate funcs

    dbg_scan_models = True  # if True, scan all viable models and choose the best. Very slow!
    dbg_test_model = True  # test clasifiers after fitting
    dbg_verbose = True  # controls debug output
    dbg_curr_df: DataFrame = None  # for debugging of current dataframe

    ###################################

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

        print("")
        print(curr_pair)

        # populate the normal dataframe
        dataframe = self.add_indicators(dataframe)

        self.dbg_curr_df = dataframe
        self.num_features = dataframe.shape[1]

        if GenAutoEncoder.first_time:
            GenAutoEncoder.first_time = False
            print("Creating auto encoder model...")
            self.create_models()

        # update the models with the current dataframe
        self.train_models(dataframe)

        return dataframe

    ###################################

    # populate dataframe with desired technical indicators
    # NOTE: OK to throw (almost) anything in here, just add it to the parameter list
    # The whole idea is to create a dimension-reduced mapping anyway
    # Warning: do not use indicators that might produce 'inf' results, it messes up the scaling
    def add_indicators(self, dataframe: DataFrame) -> DataFrame:

        # reset profit/loss thresholds
        self.profit_threshold = self.default_profit_threshold
        self.loss_threshold = self.default_loss_threshold

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
        # dataframe['dc_model'] = dataframe['dc_upper'] - dataframe['dc_dist'] * 0.618  # Centre Low Fib
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

        dataframe['dwt_deriv'] = np.gradient(dataframe['dwt_smooth'])
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

        # TODO: fix NaNs
        dataframe.fillna(0.0, inplace=True)

        return dataframe

    ############################

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

    # get a scaler for scaling/normalising the data (in a func because I change it routinely)
    def get_scaler(self):
        # uncomment the one yu want
        # return StandardScaler()
        # return RobustScaler()
        return MinMaxScaler()

    def remove_debug_columns(self, dataframe: DataFrame) -> DataFrame:
        drop_list = dataframe.filter(regex='^%').columns
        if len(drop_list) > 0:
            for col in drop_list:
                dataframe = dataframe.drop(col, axis=1)
            dataframe.reindex()
        return dataframe

    # Normalise a dataframe
    def norm_dataframe(self, dataframe: DataFrame) -> DataFrame:

        temp = dataframe.copy()
        if 'date' in temp.columns:
            temp['date'] = pd.to_datetime(temp['date']).astype('int64')

        temp = self.remove_debug_columns(temp)
        cols = temp.columns
        drop_list = ['buy_tag', 'enter_tag', 'sell_tag', 'exit_tag', 'buy', 'sell']
        for col in drop_list:
            if col in cols:
                temp = temp.drop(col, axis=1)

        temp.set_index('date')
        temp.reindex()

        cols = temp.columns
        scaler = self.get_scaler()

        temp = scaler.fit_transform(temp)

        return pd.DataFrame(temp, columns=cols)

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

    # map column into [0,1]
    def get_binary_labels(self, col):
        binary_encoder = LabelEncoder().fit([min(col), max(col)])
        result = binary_encoder.transform(col)
        # print ("label input:  ", col)
        # print ("label output: ", result)
        return result

    ############################

    # create the model list
    def create_models(self):
        if len(self.model_list) > 0:
            for model_name in self.model_list:
                self.model_info[model_name] = self.model_factory(model_name)

    # updates the training of each model with the supplied dataframe
    def train_models(self, dataframe: DataFrame):

        rand_st = 27  # use fixed number for reproducibility

        # normalise the dataframe
        full_df_norm = self.norm_dataframe(dataframe)

        data_size = full_df_norm.shape[0]

        # get training and test datasets
        train_size = int(0.8 * data_size)
        test_size = data_size - train_size

        df_train, df_test = train_test_split(full_df_norm,
                                             train_size=train_size,
                                             random_state=rand_st,
                                             shuffle=True)
        # if self.dbg_verbose:
        #     print("    dataframe:", full_df_norm.shape, ' -> train:', df_train.shape, " + test:", df_test.shape)

        # loop through the available models and train
        if len(self.model_info) > 0:
            # print("model_info: ", self.model_info)
            for name in self.model_info:
                if self.model_info[name] is None:
                    print("    ERR: no model for ", name)
                else:
                    self.fit_model(self.model_info[name], df_train, df_test)
        else:
            print("ERR: No models defined")

        return

    # evaluate the available models (after training)
    def evaluate_models(self, dataframe: DataFrame):
        print("")
        if len(self.model_info) > 0:

            # normalise the dataframe
            full_df_norm = self.norm_dataframe(dataframe)

            # get training and test datasets
            train_size = int(0.5 * full_df_norm.shape[0])
            df_train, df_test = train_test_split(full_df_norm,
                                                 train_size=train_size,
                                                 shuffle=True)
            for name in self.model_info:
                model = self.model_info[name]
                self.evaluate_model(name, model, df_test)
                model.save()
        else:
            print("ERR: No models defined")

        print("")
        return

    ###################################
    # versions using AutoEncoder class
    def fit_model(self, model: keras.Model, df_train: DataFrame, df_test: DataFrame):
        if model != None:
            model.train(df_train, df_test)
        else:
            print("    ERR: model is  None")
        return

    def evaluate_model(self, name, model, df_test: DataFrame):
        model.evaluate(df_test)
        return

    ###################################
    # default model
    default_model = 'Dense1'  # select based on testing

    # list of potential model types - set to the list that you want to compare
    model_list = [
        'Dense1', 'LSTM1', 'LSTM2'
        # 'LSTM2'

    ]

    # factory to create model based on name
    def model_factory(self, name) -> keras.Model:
        model = None

        if name == 'Dense1':
            model = AutoEncoder(self.num_features)
        elif name == 'LSTM1':
            model = LSTMAutoEncoder(self.num_features)
        elif name == 'LSTM2':
            model = LSTM2AutoEncoder(self.num_features)


        else:
            print("Unknown model: ", name)
            model = None

        # if model != None:
        #     model = self.get_model_weights(model, name)

        return model


    def build_Conv1(self, nfeatures):

        # This is the dimension of the latent space (encoding space)
        latent_dim = 32

        encoder = keras.Sequential()
        encoder.add(layers.Dense(128, activation='relu', input_shape=(1, nfeatures)))
        encoder.add(layers.Conv1D(64, 2, padding='same', activation='relu'))
        encoder.add(layers.MaxPooling1D(2, padding='same'))
        encoder.add(layers.Conv1D(32, 3, padding='same', activation='relu'))
        encoder.add(layers.MaxPooling1D(2, padding='same'))
        encoder.add(layers.Dense(64, activation='relu'))
        encoder.add(layers.Dropout(rate=0.1))
        encoder.add(layers.Dense(self.latent_dim, activation='relu', name='encoder_output'))

        decoder = keras.Sequential()
        decoder.add(layers.Dense(64, activation='relu', input_shape=(1, self.latent_dim)))
        decoder.add(layers.Conv1D(32, 3, padding='same', activation='relu'))
        # decoder.add(layers.UpSampling1D(2))
        decoder.add(layers.Conv1D(64, 2, padding='same', activation='relu'))
        # decoder.add(layers.UpSampling1D(2))
        decoder.add(layers.Conv1D(64, 3, padding='same', activation='sigmoid'))
        decoder.add(layers.Dense(nfeatures, activation=None))

        autoencoder = keras.Model(inputs=encoder.input, outputs=decoder(encoder.output))
        autoencoder.compile(metrics=['accuracy', 'mse'], loss='mse', optimizer='adam')

        autoencoder.summary()

        return autoencoder

    ###################################

    """
    Buy Signal
    """

    def populate_buy_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:

        # clear buy/entry signals
        dataframe.loc[:, 'enter_tag'] = ''
        dataframe['buy'] = 0

        # if first call, summarise models
        if not self.dp.runmode.value in ('hyperopt'):
            if GenAutoEncoder.first_run:
                GenAutoEncoder.first_run = False  # note use of clas variable, not instance variable
                self.evaluate_models(dataframe)

        return dataframe

    ###################################

    """
    Sell Signal
    """

    def populate_sell_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:

        # do nothing except clear sell signals
        dataframe.loc[:, 'exit_tag'] = ''
        dataframe['sell'] = 0

        return dataframe

    ###################################


# Elliot Wave Oscillator
def ewo(dataframe, sma1_length=5, sma2_length=35):
    sma1 = ta.EMA(dataframe, timeperiod=sma1_length)
    sma2 = ta.EMA(dataframe, timeperiod=sma2_length)
    smadif = (sma1 - sma2) / dataframe['close'] * 100
    return smadif


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
