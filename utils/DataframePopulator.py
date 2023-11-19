#
# This is a set of functions used to populate the indicators in a dataframe
# They are in a seperate file because I use the same set of indicators across several types of strategies, so it's
# just convenient to do it this way
#
import math

import numpy as np
import pandas as pd

import pywt
import talib.abstract as ta
from scipy.ndimage import gaussian_filter1d

from pandas import DataFrame, Series

import sys
from pathlib import Path
from functools import reduce
from enum import Enum

sys.path.append(str(Path(__file__).parent))

import logging
import warnings

log = logging.getLogger(__name__)
# log.setLevel(logging.DEBUG)
warnings.simplefilter(action='ignore', category=pd.errors.PerformanceWarning)

import freqtrade.vendor.qtpylib.indicators as qtpylib

import custom_indicators as cta
from finta import TA as fta

import legendary_ta as lta

from DataframeUtils import DataframeUtils
from scipy.stats import linregress


#################

# the type of dataset used for input
class DatasetType(Enum):
    DEFAULT = 0
    MINIMAL = 1
    SMALL = 2
    MEDIUM = 3
    LARGE = 4
    CUSTOM1 = 5
    CUSTOM2 = 6

class DataframePopulator():
    # global vars that control data generation. Ok to set these from a strategy

    startup_win = 128  # should be a power of 2
    win_size = 14
    runmode = ""  # set this to self.dp.runmode.value

    n_profit_stddevs = 0.0
    n_loss_stddevs = 0.0

    dataframeUtils = None

    def __init__(self):
        super().__init__()
        self.dataframeUtils = DataframeUtils()

    #################

    # populate dataframe with desired technical indicators
    # NOTE: OK to throw (almost) anything in here, just add it to the parameter list
    # The whole idea is to create a dimension-reduced mapping anyway
    # Warning: do not use indicators that might produce 'inf' results, it messes up the scaling


    # use the minimal set of indicators needed to satisfy the trading signals logic and base class checks
    def add_minimal_indicators(self, dataframe: DataFrame) -> DataFrame:

        dataframe['mid'] = (dataframe['open'] + dataframe['close']) / 2.0
        dataframe['gain'] = 100.0 * (dataframe['close'] - dataframe['open']) / dataframe['open']
        dataframe['profit'] = dataframe['gain'].clip(lower=0.0)
        dataframe['loss'] = dataframe['gain'].clip(upper=0.0)


        # RSI
        dataframe['rsi'] = ta.RSI(dataframe, timeperiod=self.win_size)

        # Williams %R
        dataframe['wr'] = 0.02 * (self.williams_r(dataframe, period=14) + 50.0)

        # Fisher RSI
        rsi = 0.1 * (dataframe['rsi'] - 50)
        dataframe['fisher_rsi'] = (np.exp(2 * rsi) - 1) / (np.exp(2 * rsi) + 1)

        # Combined Fisher RSI and Williams %R
        dataframe['fisher_wr'] = (dataframe['wr'] + dataframe['fisher_rsi']) / 2.0


        return dataframe

 
    # ------------------------------
    # 'small' set of indicators - basically, the best-known ones
    def add_small_indicators(self, dataframe: DataFrame) -> DataFrame:

        dataframe = self.add_minimal_indicators(dataframe)


        # MFI - Chaikin Money Flow Indicator
        dataframe['mfi'] = ta.MFI(dataframe)

        # Recent min/max
        dataframe['recent_min'] = dataframe['close'].rolling(window=self.win_size).min()
        dataframe['recent_max'] = dataframe['close'].rolling(window=self.win_size).max()


        # Bollinger Bands (must include these)
        bollinger = qtpylib.bollinger_bands(dataframe['close'], window=20, stds=2)
        dataframe['bb_lowerband'] = bollinger['lower']
        dataframe['bb_middleband'] = bollinger['mid']
        dataframe['bb_upperband'] = bollinger['upper']
        dataframe['bb_width'] = ((dataframe['bb_upperband'] - dataframe['bb_lowerband']) / dataframe['bb_middleband'])
        dataframe["bb_gain"] = ((dataframe["bb_upperband"] - dataframe["close"]) / dataframe["close"])
        dataframe["bb_loss"] = ((dataframe["bb_lowerband"] - dataframe["close"]) / dataframe["close"])


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


        # DWT model
        # if in backtest or hyperopt, then we have to do rolling calculations
        if self.runmode in ('hyperopt', 'backtest', 'plot'):
            # dataframe['dwt'] = dataframe['close'].rolling(window=self.startup_win).apply(self.roll_get_dwt)
            dataframe['dwt'] = dataframe['mid'].rolling(window=self.startup_win).apply(self.roll_get_dwt)
        else:
            # dataframe['dwt'] = self.get_dwt(dataframe['close'])
            dataframe['dwt'] = self.get_dwt(dataframe['mid'])

        dataframe['dwt_gain'] = 100.0 * (dataframe['dwt'] - dataframe['dwt'].shift()) / dataframe['dwt'].shift()
        dataframe['dwt_profit'] = dataframe['dwt_gain'].clip(lower=0.0)
        dataframe['dwt_loss'] = dataframe['dwt_gain'].clip(upper=0.0)

        dataframe['dwt_profit_mean'] = dataframe['dwt_profit'].rolling(self.win_size).mean()
        dataframe['dwt_profit_std'] = dataframe['dwt_profit'].rolling(self.win_size).std()
        dataframe['dwt_loss_mean'] = dataframe['dwt_loss'].rolling(self.win_size).mean()
        dataframe['dwt_loss_std'] = dataframe['dwt_loss'].rolling(self.win_size).std()

        # (Local) Profit & Loss thresholds are used extensively, do not remove!
        dataframe['profit_threshold'] = dataframe['dwt_profit_mean'] + self.n_profit_stddevs * abs(
            dataframe['dwt_profit_std'])

        dataframe['loss_threshold'] = dataframe['dwt_loss_mean'] - self.n_loss_stddevs * abs(dataframe['dwt_loss_std'])

        # Sequences of consecutive up/downs
        dataframe['dwt_dir'] = 0.0
        dataframe['dwt_dir'] = np.where(dataframe['dwt'].diff() > 0, 1.0, -1.0)

        dataframe['dwt_dir_up'] = dataframe['dwt_dir'].clip(lower=0.0)
        dataframe['dwt_nseq_up'] = dataframe['dwt_dir_up'] * (dataframe['dwt_dir_up'].groupby(
            (dataframe['dwt_dir_up'] != dataframe['dwt_dir_up'].shift()).cumsum()).cumcount() + 1)
        dataframe['dwt_nseq_up'] = dataframe['dwt_nseq_up'].clip(lower=0.0, upper=20.0)  # removes startup artifacts

        dataframe['dwt_dir_dn'] = abs(dataframe['dwt_dir'].clip(upper=0.0))
        dataframe['dwt_nseq_dn'] = dataframe['dwt_dir_dn'] * (dataframe['dwt_dir_dn'].groupby(
            (dataframe['dwt_dir_dn'] != dataframe['dwt_dir_dn'].shift()).cumsum()).cumcount() + 1)
        dataframe['dwt_nseq_dn'] = dataframe['dwt_nseq_dn'].clip(lower=0.0, upper=20.0)

        # rolling linear slope of the DWT (i.e. average trend) of near-past
        dataframe['dwt_slope'] = dataframe['dwt'].rolling(window=6).apply(self.roll_get_slope)

        # moving averages
        dataframe['sma'] = ta.SMA(dataframe, timeperiod=self.win_size)
        dataframe['ema'] = ta.EMA(dataframe, timeperiod=self.win_size)
        dataframe['tema'] = ta.TEMA(dataframe, timeperiod=self.win_size)
        # dataframe['tema_stddev'] = dataframe['tema'].rolling(self.win_size).std()

        # Donchian Channels
        dataframe['dc_upper'] = ta.MAX(dataframe['high'], timeperiod=self.win_size)
        dataframe['dc_lower'] = ta.MIN(dataframe['low'], timeperiod=self.win_size)
        dataframe['dc_mid'] = ta.TEMA(((dataframe['dc_upper'] + dataframe['dc_lower']) / 2), timeperiod=self.win_size)

        dataframe["dcbb_dist_upper"] = (dataframe["dc_upper"] - dataframe['bb_upperband'])
        dataframe["dcbb_dist_lower"] = (dataframe["dc_lower"] - dataframe['bb_lowerband'])

        # Fibonacci Levels (of Donchian Channel)
        dataframe['dc_dist'] = (dataframe['dc_upper'] - dataframe['dc_lower'])
        # dataframe['dc_hf'] = dataframe['dc_upper'] - dataframe['dc_dist'] * 0.236  # Highest Fib
        # dataframe['dc_chf'] = dataframe['dc_upper'] - dataframe['dc_dist'] * 0.382  # Centre High Fib
        # dataframe['dc_clf'] = dataframe['dc_upper'] - dataframe['dc_dist'] * 0.618  # Centre Low Fib
        # dataframe['dc_lf'] = dataframe['dc_upper'] - dataframe['dc_dist'] * 0.764  # Low Fib

        # Keltner Channels (these can sometimes produce inf results)
        keltner = qtpylib.keltner_channel(dataframe)
        dataframe["kc_upper"] = keltner["upper"]
        dataframe["kc_lower"] = keltner["lower"]
        dataframe["kc_mid"] = keltner["mid"]

        return dataframe
    
   # ------------------------------

    def add_default_indicators(self, dataframe: DataFrame) -> DataFrame:

        dataframe = self.add_small_indicators(dataframe)

        # Recent min/max
        dataframe['recent_min'] = dataframe['close'].rolling(window=self.win_size).min()
        dataframe['recent_max'] = dataframe['close'].rolling(window=self.win_size).max()

        # Bollinger Bands (must include these)
        bollinger = qtpylib.bollinger_bands(dataframe['close'], window=20, stds=2)
        dataframe['bb_lowerband'] = bollinger['lower']
        dataframe['bb_middleband'] = bollinger['mid']
        dataframe['bb_upperband'] = bollinger['upper']
        dataframe['bb_width'] = ((dataframe['bb_upperband'] - dataframe['bb_lowerband']) / dataframe['bb_middleband'])
        dataframe["bb_gain"] = ((dataframe["bb_upperband"] - dataframe["close"]) / dataframe["close"])
        dataframe["bb_loss"] = ((dataframe["bb_lowerband"] - dataframe["close"]) / dataframe["close"])

        # moving averages
        dataframe['sma'] = ta.SMA(dataframe, timeperiod=self.win_size)
        dataframe['ema'] = ta.EMA(dataframe, timeperiod=self.win_size)
        dataframe['tema'] = ta.TEMA(dataframe, timeperiod=self.win_size)
        # dataframe['tema_stddev'] = dataframe['tema'].rolling(self.win_size).std()

        # Stochastic
        period = 14
        smoothD = 3
        SmoothK = 3
        stochrsi = (dataframe['rsi'] - dataframe['rsi'].rolling(period).min()) / (
                dataframe['rsi'].rolling(period).max() - dataframe['rsi'].rolling(period).min())
        dataframe['srsi_k'] = stochrsi.rolling(SmoothK).mean() * 100
        dataframe['srsi_d'] = dataframe['srsi_k'].rolling(smoothD).mean()

        # Donchian Channels
        dataframe['dc_upper'] = ta.MAX(dataframe['high'], timeperiod=self.win_size)
        dataframe['dc_lower'] = ta.MIN(dataframe['low'], timeperiod=self.win_size)
        dataframe['dc_mid'] = ta.TEMA(((dataframe['dc_upper'] + dataframe['dc_lower']) / 2), timeperiod=self.win_size)

        dataframe["dcbb_dist_upper"] = (dataframe["dc_upper"] - dataframe['bb_upperband'])
        dataframe["dcbb_dist_lower"] = (dataframe["dc_lower"] - dataframe['bb_lowerband'])

        # Fibonacci Levels (of Donchian Channel)
        dataframe['dc_dist'] = (dataframe['dc_upper'] - dataframe['dc_lower'])
        # dataframe['dc_hf'] = dataframe['dc_upper'] - dataframe['dc_dist'] * 0.236  # Highest Fib
        # dataframe['dc_chf'] = dataframe['dc_upper'] - dataframe['dc_dist'] * 0.382  # Centre High Fib
        # dataframe['dc_clf'] = dataframe['dc_upper'] - dataframe['dc_dist'] * 0.618  # Centre Low Fib
        # dataframe['dc_lf'] = dataframe['dc_upper'] - dataframe['dc_dist'] * 0.764  # Low Fib

        # Keltner Channels (these can sometimes produce inf results)
        keltner = qtpylib.keltner_channel(dataframe)
        dataframe["kc_upper"] = keltner["upper"]
        dataframe["kc_lower"] = keltner["lower"]
        dataframe["kc_mid"] = keltner["mid"]

        # RSI
        # dataframe['rsi'] = ta.RSI(dataframe, timeperiod=self.win_size)
        dataframe['rsi_14'] = ta.RSI(dataframe, timeperiod=14)

        # SMA
        dataframe['sma_200'] = ta.SMA(dataframe, timeperiod=200)

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

        # Volume Flow Indicator (MFI) for volume based on the direction of price movement
        dataframe['vfi'] = fta.VFI(dataframe, period=14)

        # ATR
        dataframe['atr'] = ta.ATR(dataframe, timeperiod=self.win_size)

        # Hilbert Transform Indicator - SineWave
        hilbert = ta.HT_SINE(dataframe)
        dataframe['htsine'] = hilbert['sine']
        dataframe['htleadsine'] = hilbert['leadsine']

        # Oscillators

        # EWO
        dataframe['ewo'] = self.ewo(dataframe, 50, 200)

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

        # Legenadry TA indicators
        dataframe = lta.fisher_cg(dataframe)
        dataframe = lta.exhaustion_bars(dataframe)
        dataframe = lta.smi_momentum(dataframe)
        dataframe = lta.pinbar(dataframe, dataframe["smi"])
        dataframe = lta.breakouts(dataframe)

        return dataframe

    # ------------------------------

    def add_medium_indicators(self, dataframe: DataFrame) -> DataFrame:

        dataframe = self.add_small_indicators(dataframe)


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

        # Volume Flow Indicator (MFI) for volume based on the direction of price movement
        dataframe['vfi'] = fta.VFI(dataframe, period=14)

        # ATR
        dataframe['atr'] = ta.ATR(dataframe, timeperiod=self.win_size)

        # Oscillators

        # EWO
        dataframe['ewo'] = self.ewo(dataframe, 50, 200)

        # Ultimate Oscillator
        dataframe['uo'] = ta.ULTOSC(dataframe)

        # Aroon, Aroon Oscillator
        aroon = ta.AROON(dataframe)
        dataframe['aroonup'] = aroon['aroonup']
        dataframe['aroondown'] = aroon['aroondown']
        dataframe['aroonosc'] = ta.AROONOSC(dataframe)

        # Awesome Oscillator
        dataframe['ao'] = qtpylib.awesome_oscillator(dataframe)

        return dataframe

    # ------------------------------

    def add_large_indicators(self, dataframe: DataFrame) -> DataFrame:

        dataframe = self.add_medium_indicators(dataframe)

        # Stochastic
        period = 14
        smoothD = 3
        SmoothK = 3
        stochrsi = (dataframe['rsi'] - dataframe['rsi'].rolling(period).min()) / (
                dataframe['rsi'].rolling(period).max() - dataframe['rsi'].rolling(period).min())
        dataframe['srsi_k'] = stochrsi.rolling(SmoothK).mean() * 100
        dataframe['srsi_d'] = dataframe['srsi_k'].rolling(smoothD).mean()

        # RSI
        dataframe['rsi_14'] = ta.RSI(dataframe, timeperiod=14)

        # SMA
        dataframe['sma_200'] = ta.SMA(dataframe, timeperiod=200)

        # Hilbert Transform Indicator - SineWave
        hilbert = ta.HT_SINE(dataframe)
        dataframe['htsine'] = hilbert['sine']
        dataframe['htleadsine'] = hilbert['leadsine']

        # Commodity Channel Index: values [Oversold:-100, Overbought:100]
        dataframe['cci'] = ta.CCI(dataframe)

        # Legendary TA indicators
        dataframe = lta.fisher_cg(dataframe)
        dataframe = lta.exhaustion_bars(dataframe)
        dataframe = lta.smi_momentum(dataframe)
        dataframe = lta.pinbar(dataframe, dataframe["smi"])
        dataframe = lta.breakouts(dataframe)


        return dataframe

    # ------------------------------

    # for experimentation
    def add_custom1_indicators(self, dataframe: DataFrame) -> DataFrame:

        dataframe = self.add_minimal_indicators(dataframe)

        # Bollinger Bands 
        bollinger = qtpylib.bollinger_bands(dataframe['close'], window=20, stds=2)
        dataframe['bb_lowerband'] = bollinger['lower']
        dataframe['bb_middleband'] = bollinger['mid']
        dataframe['bb_upperband'] = bollinger['upper']
        dataframe['bb_width'] = ((dataframe['bb_upperband'] - dataframe['bb_lowerband']) / dataframe['bb_middleband'])
        dataframe["bb_gain"] = ((dataframe["bb_upperband"] - dataframe["close"]) / dataframe["close"])
        dataframe["bb_loss"] = ((dataframe["bb_lowerband"] - dataframe["close"]) / dataframe["close"])


        # MACD
        macd = ta.MACD(dataframe)
        dataframe['macd'] = macd['macd']
        dataframe['macdsignal'] = macd['macdsignal']
        dataframe['macdhist'] = macd['macdhist']

        '''
        # Keltner Channels (Warning: these can sometimes produce inf results)
        keltner = qtpylib.keltner_channel(dataframe)
        dataframe["kc_upper"] = keltner["upper"]
        dataframe["kc_lower"] = keltner["lower"]
        dataframe["kc_mid"] = keltner["mid"]


        # Donchian Channels
        dataframe['dc_upper'] = ta.MAX(dataframe['high'], timeperiod=self.win_size)
        dataframe['dc_lower'] = ta.MIN(dataframe['low'], timeperiod=self.win_size)
        dataframe['dc_mid'] = ta.TEMA(((dataframe['dc_upper'] + dataframe['dc_lower']) / 2), timeperiod=self.win_size)
        
        '''
        
        return dataframe

    # ------------------------------

    def add_custom2_indicators(self, dataframe: DataFrame) -> DataFrame:

        dataframe = self.add_minimal_indicators(dataframe)
        return dataframe

    #------------------------------

    def add_indicators(self, dataframe: DataFrame, dataset_type=DatasetType.DEFAULT) -> DataFrame:

        # print(f"dataset_type: {dataset_type}")

        if dataset_type == DatasetType.DEFAULT:
            dataframe = self.add_default_indicators(dataframe)
        elif dataset_type == DatasetType.MINIMAL:
            dataframe = self.add_minimal_indicators(dataframe)
        elif dataset_type == DatasetType.SMALL:
            dataframe = self.add_small_indicators(dataframe)
        elif dataset_type == DatasetType.MEDIUM:
            dataframe = self.add_medium_indicators(dataframe)
        elif dataset_type == DatasetType.LARGE:
            dataframe = self.add_large_indicators(dataframe)
        elif dataset_type == DatasetType.CUSTOM1:
            dataframe = self.add_custom1_indicators(dataframe)
        elif dataset_type == DatasetType.CUSTOM2:
            dataframe = self.add_custom2_indicators(dataframe)
        else:
            print(f"    ERROR: Unknown dataset type: {dataset_type}")

        # TODO: remove/fix any columns that contain 'inf'
        self.dataframeUtils.check_inf(dataframe)

        # fix NaNs
        dataframe.fillna(0.0, inplace=True)

        return dataframe

    ################################

    # 'hidden' indicators. These are ostensibly backward looking, but may inadvertently use means, smoothing etc.
    def add_hidden_indicators(self, dataframe: DataFrame) -> DataFrame:

        if 'mid' not in dataframe:
            print("")
            print("*** ERR: missing columns (mid)")
            print(f"    cols: {dataframe.columns.values}")
            print("")

        dataframe['fwd_dwt'] = self.get_dwt(dataframe['mid'])

        dataframe['dwt_deriv'] = np.gradient(dataframe['dwt'])
        # dataframe['dwt_deriv'] = np.gradient(dataframe['dwt'])
        dataframe['dwt_top'] = np.where(qtpylib.crossed_below(dataframe['dwt_deriv'], 0.0), 1, 0)
        dataframe['dwt_bottom'] = np.where(qtpylib.crossed_above(dataframe['dwt_deriv'], 0.0), 1, 0)

        # dataframe['dwt_diff'] = 100.0 * (dataframe['dwt'] - dataframe['mid']) / dataframe['mid']
        dataframe['dwt_diff'] = 100.0 * (dataframe['fwd_dwt'] - dataframe['mid']) / dataframe['mid']
        # dataframe['dwt_diff'] = 100.0 * (dataframe['dwt'] - dataframe['dwt']) / dataframe['dwt']

        dataframe['dwt_trend'] = np.where(dataframe['dwt_dir'].rolling(5).sum() > 3.0, 1.0, -1.0)

        # get rolling mean & stddev so that we have a localised estimate of (recent) activity
        dataframe['dwt_mean'] = dataframe['dwt'].rolling(self.win_size).mean()
        dataframe['dwt_std'] = dataframe['dwt'].rolling(self.win_size).std()

        # Recent min/max
        dataframe['dwt_recent_min'] = dataframe['dwt'].rolling(window=self.win_size).min()
        dataframe['dwt_recent_max'] = dataframe['dwt'].rolling(window=self.win_size).max()
        dataframe['dwt_maxmin'] = 100.0 * (dataframe['dwt_recent_max'] - dataframe['dwt_recent_min']) / \
                                  dataframe['dwt_recent_max']
        dataframe['dwt_delta_min'] = (100.0 * (dataframe['dwt_recent_min'] - dataframe['close']) / \
                                      dataframe['close']).clip(lower=-5.0)
        dataframe['dwt_delta_max'] = (100.0 * (dataframe['dwt_recent_max'] - dataframe['close']) / \
                                      dataframe['close']).clip(upper=5.0)
        # longer term high/low
        dataframe['dwt_low'] = dataframe['dwt'].rolling(window=self.startup_win).min()
        dataframe['dwt_high'] = dataframe['dwt'].rolling(window=self.startup_win).max()

        # # these are (primarily) clues for the ML algorithm:
        # dataframe['dwt_at_min'] = np.where(dataframe['dwt'] <= dataframe['dwt_recent_min'], 1.0, 0.0)
        # dataframe['dwt_at_max'] = np.where(dataframe['dwt'] >= dataframe['dwt_recent_max'], 1.0, 0.0)
        dataframe['dwt_at_low'] = np.where(dataframe['dwt'] <= dataframe['dwt_low'], 1.0, 0.0)
        dataframe['dwt_at_high'] = np.where(dataframe['dwt'] >= dataframe['dwt_high'], 1.0, 0.0)

        # dataframe['loss_threshold'] = dataframe['dwt_loss_mean'] - self.n_loss_stddevs * abs(dataframe['dwt_loss_std'])

        return dataframe

    # calculate future gains. Used for setting targets. Yes, we lookahead in the data!
    def add_future_data(self, dataframe: DataFrame, lookahead: int) -> DataFrame:

        lookahead_win = max(lookahead, 14)

        # make a copy of the dataframe so that we do not put any forward looking data into the main dataframe
        # Also, use a different name to avoid cut & paste errors
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
        future_df['future_close'] = future_df[price_col].shift(-lookahead_win)

        future_df['future_gain'] = 100.0 * (future_df['future_close'] - future_df[price_col]) / future_df[price_col]
        future_df['future_gain'].clip(lower=-5.0, upper=5.0, inplace=True)

        future_df['future_profit'] = future_df['future_gain'].clip(lower=0.0)
        future_df['future_loss'] = future_df['future_gain'].clip(upper=0.0)

        # get rolling mean & stddev so that we have a localised estimate of (recent) future activity
        # Note: window in past because we already looked forward (with 'future_close')
        future_df['future_gain_mean'] = future_df['future_gain'].rolling(lookahead_win).mean()
        future_df['future_gain_std'] = future_df['future_gain'].rolling(lookahead_win).std()
        future_df['future_gain_sum'] = future_df['future_gain'].rolling(lookahead_win).sum()

        future_df['future_profit_mean'] = future_df['future_profit'].rolling(lookahead_win).mean()
        future_df['future_profit_std'] = future_df['future_profit'].rolling(lookahead_win).std()
        future_df['future_loss_mean'] = future_df['future_loss'].rolling(lookahead_win).mean()
        future_df['future_loss_std'] = future_df['future_loss'].rolling(lookahead_win).std()

        future_df['future_profit_max'] = future_df['future_profit'].rolling(lookahead_win).max()
        future_df['future_profit_min'] = future_df['future_profit'].rolling(lookahead_win).min()
        future_df['future_loss_max'] = future_df['future_loss'].rolling(lookahead_win).max()
        future_df['future_loss_min'] = future_df['future_loss'].rolling(lookahead_win).min()

        # future_df['profit_threshold'] = future_df['profit_mean'] + self.n_profit_stddevs * abs(future_df['profit_std'])
        # future_df['loss_threshold'] = future_df['loss_mean'] - self.n_loss_stddevs * abs(future_df['loss_std'])

        future_df['future_profit_threshold'] = future_df['dwt_profit_mean'] + self.n_profit_stddevs * abs(
            future_df['dwt_profit_std'])
        future_df['future_loss_threshold'] = future_df['dwt_loss_mean'] - self.n_loss_stddevs * abs(
            future_df['dwt_loss_std'])

        future_df['future_profit_diff'] = (future_df['future_profit'] - future_df['future_profit_threshold']) * 10.0
        future_df['future_loss_diff'] = (future_df['future_loss'] - future_df['future_loss_threshold']) * 10.0

        # future_df['buy_signal'] = np.where(future_df['profit_diff'] > 0.0, 1.0, 0.0)
        # future_df['sell_signal'] = np.where(future_df['loss_diff'] < 0.0, -1.0, 0.0)

        # these explicitly uses dwt
        future_df['future_dwt'] = future_df['full_dwt'].shift(-lookahead_win)
        # future_df['curr_trend'] = np.where(future_df['full_dwt'].shift(-1) > future_df['full_dwt'], 1.0, -1.0)
        # future_df['future_trend'] = np.where(future_df['future_dwt'].shift(-1) > future_df['future_dwt'], 1.0, -1.0)

        future_df['trend'] = np.where(future_df[price_col] >= future_df[price_col].shift(), 1.0, -1.0)
        future_df['ftrend'] = np.where(future_df['future_close'] >= future_df['future_close'].shift(), 1.0, -1.0)

        future_df['curr_trend'] = np.where(future_df['trend'].rolling(3).sum() > 0.0, 1.0, -1.0)
        future_df['future_trend'] = np.where(future_df['ftrend'].rolling(3).sum() > 0.0, 1.0, -1.0)

        # Sequences of consecutive up/downs (using full_dwt)
        future_df['full_dwt_dir'] = 0.0
        future_df['full_dwt_dir'] = np.where(future_df['full_dwt'].diff() > 0, 1.0, -1.0)

        future_df['full_dwt_dir_up'] = future_df['full_dwt_dir'].clip(lower=0.0)
        future_df['full_dwt_nseq_up'] = future_df['full_dwt_dir_up'] * (future_df['full_dwt_dir_up'].groupby(
            (future_df['full_dwt_dir_up'] != future_df['full_dwt_dir_up'].shift()).cumsum()).cumcount() + 1)
        future_df['full_dwt_nseq_up'] = future_df['full_dwt_nseq_up'].clip(lower=0.0,
                                                                           upper=20.0)  # removes startup artifacts

        future_df['full_dwt_dir_dn'] = abs(future_df['full_dwt_dir'].clip(upper=0.0))
        future_df['full_dwt_nseq_dn'] = future_df['full_dwt_dir_dn'] * (future_df['full_dwt_dir_dn'].groupby(
            (future_df['full_dwt_dir_dn'] != future_df['full_dwt_dir_dn'].shift()).cumsum()).cumcount() + 1)
        future_df['full_dwt_nseq_dn'] = future_df['full_dwt_nseq_dn'].clip(lower=0.0, upper=20.0)

        # build forward-looking sum of up/down trends
        future_win = pd.api.indexers.FixedForwardWindowIndexer(window_size=int(self.win_size))  # don't use a big window

        future_df['future_nseq_up'] = future_df['full_dwt_nseq_up'].shift(-self.win_size)

        future_df['future_nseq_up_mean'] = future_df['future_nseq_up'].rolling(window=future_win).mean()
        future_df['future_nseq_up_std'] = future_df['future_nseq_up'].rolling(window=future_win).std()
        future_df['future_nseq_up_thresh'] = future_df['future_nseq_up_mean'] + self.n_profit_stddevs * future_df[
            'future_nseq_up_std']

        future_df['future_nseq_dn'] = future_df['full_dwt_nseq_dn'].shift(-self.win_size)

        future_df['future_nseq_dn_mean'] = future_df['future_nseq_dn'].rolling(future_win).mean()
        future_df['future_nseq_dn_std'] = future_df['future_nseq_dn'].rolling(future_win).std()
        future_df['future_nseq_dn_thresh'] = future_df['future_nseq_dn_mean'] \
                                             - self.n_loss_stddevs * future_df['future_nseq_dn_std']

        # Recent min/max
        # future_df['future_min'] = future_df[price_col].rolling(window=future_win).min()
        # future_df['future_max'] = future_df[price_col].rolling(window=future_win).max()
        future_df['future_min'] = future_df['dwt'].rolling(window=future_win).min()
        future_df['future_max'] = future_df['dwt'].rolling(window=future_win).max()

        future_df['future_maxmin'] = 100.0 * (future_df['future_max'] - future_df['future_min']) / \
                                     future_df['future_max']
        future_df['future_delta_min'] = 100.0 * (future_df['future_min'] - future_df['close']) / \
                                        future_df['close']
        future_df['future_delta_max'] = 100.0 * (future_df['future_max'] - future_df['close']) / \
                                        future_df['close']

        future_df['future_maxmin'] = future_df['future_maxmin'].clip(lower=0.0, upper=10.0)


        # rolling linear slope of the DWT (i.e. average trend) of near-past (shifted forward)
        future_df['future_slope'] = future_df['future_dwt'].rolling(window=6).apply(self.roll_get_slope)

        # get average gain & stddev
        profit_mean = future_df['future_profit'].mean()
        profit_std = future_df['future_profit'].std()
        loss_mean = future_df['future_loss'].mean()
        loss_std = future_df['future_loss'].std()

        return future_df

    ###################################

    # add indicators used by stoploss/custom sell logic
    def add_stoploss_indicators(self, dataframe) -> DataFrame:

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

    ##################

    # returns (rolling) smoothed version of input column
    def roll_smooth(self, col) -> float:
        # must return scalar, so just calculate prediction and take last value

        smooth = gaussian_filter1d(col, 4)
        # smooth = gaussian_filter1d(col, 2)

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

    def roll_get_dwt(self, col) -> float:
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
        # sigma = madev(coeff[-level])
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

    def roll_get_slope(self, col) -> float:
        # must return scalar, so just calculate prediction and take last value

        slope = np.polyfit(col.index, col, 1)[0]

        if np.isnan(slope) or np.isinf(slope):
            slope = 10.0

        if (slope < 0) and math.isinf(slope):
            slope = -10.0

        return slope

    #######################

    # Utility functions

    # Elliot Wave Oscillator
    def ewo(self, dataframe, sma1_length=5, sma2_length=35):
        sma1 = ta.EMA(dataframe, timeperiod=sma1_length)
        sma2 = ta.EMA(dataframe, timeperiod=sma2_length)
        smadif = (sma1 - sma2) / dataframe['close'] * 100
        return smadif

    # Chaikin Money Flow
    def chaikin_money_flow(self, dataframe, n=20, fillna=False) -> Series:
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
        mfv *= dataframe['l1' \
                         'volume']
        cmf = (mfv.rolling(n, min_periods=0).sum()
               / dataframe['volume'].rolling(n, min_periods=0).sum())
        if fillna:
            cmf = cmf.replace([np.inf, -np.inf], np.nan).fillna(0)
        return Series(cmf, name='cmf')

    # Williams %R
    def williams_r(self, dataframe: DataFrame, period: int = 14) -> Series:
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
    def vwma(self, dataframe: DataFrame, length: int = 10):
        """Indicator: Volume Weighted Moving Average (VWMA)"""
        # Calculate Result
        pv = dataframe['close'] * dataframe['volume']
        vwma = Series(ta.SMA(pv, timeperiod=length) / ta.SMA(dataframe['volume'], timeperiod=length))
        vwma = vwma.fillna(0, inplace=True)
        return vwma

    # Exponential moving average of a volume weighted simple moving average
    def ema_vwma_osc(self, dataframe, len_slow_ma):
        slow_ema = Series(ta.EMA(self.vwma(dataframe, len_slow_ma), len_slow_ma))
        return ((slow_ema - slow_ema.shift(1)) / slow_ema.shift(1)) * 100

    def t3_average(self, dataframe, length=5):
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

    def pivot_points(self, dataframe: DataFrame, mode='fibonacci') -> Series:
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
                                    np.where((dataframe['close'] > dataframe['open']), demark_pivot_gt,
                                             demark_pivot_eq))
            dm_pivot = demark_pivot / 4
            dm_res = demark_pivot / 2 - dataframe['low']
            dm_sup = demark_pivot / 2 - dataframe['high']
            return dm_pivot, dm_res, dm_sup

    def heikin_ashi(self, dataframe, smooth_inputs=False, smooth_outputs=False, length=10):
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
    def is_support(self, row_data) -> bool:
        conditions = []
        for row in range(len(row_data) - 1):
            if row < len(row_data) // 2:
                conditions.append(row_data[row] > row_data[row + 1])
            else:
                conditions.append(row_data[row] < row_data[row + 1])
        result = reduce(lambda x, y: x & y, conditions)
        return result

    # Range midpoint acts as Resistance
    def is_resistance(self, row_data) -> bool:
        conditions = []
        for row in range(len(row_data) - 1):
            if row < len(row_data) // 2:
                conditions.append(row_data[row] < row_data[row + 1])
            else:
                conditions.append(row_data[row] > row_data[row + 1])
        result = reduce(lambda x, y: x & y, conditions)
        return result

    def range_percent_change(self, dataframe: DataFrame, method, length: int) -> float:
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
