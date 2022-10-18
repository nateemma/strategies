import numpy as np
from enum import Enum

import talib.abstract as ta
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
from sklearn.preprocessing import StandardScaler
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
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA, QuadraticDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import make_scorer
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.model_selection import cross_validate

import random

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
      
      Note that this is very sow to start up. This is mostly because we have to build the data on a rolling
      basis to avoid lookahead bias.

####################################################################################
"""


class PCA(IStrategy):
    # Do *not* hyperopt for the roi and stoploss spaces

    # ROI table:
    minimal_roi = {
        "0": 0.1
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
    use_sell_signal = True
    sell_profit_only = False
    ignore_roi_if_buy_signal = True

    # Required
    startup_candle_count: int = 128
    process_only_new_candles = True

    # Strategy-specific global vars

    inf_mins = timeframe_to_minutes(inf_timeframe)
    data_mins = timeframe_to_minutes(timeframe)
    inf_ratio = int(inf_mins / data_mins)

    # These parameters control much of the behaviour because they control the generation of the training data
    # Unfortunately, these cannot be hyperopt params because they are used in populate_indicators, which is only run
    # once during hyperopt
    lookahead_hours = 0.5
    n_profit_stddevs = 2.0
    n_loss_stddevs = 2.0
    param_list = []  # list of parameters to use (technical indicators)

    inf_lookahead = int((12 / inf_ratio) * lookahead_hours)
    curr_lookahead = inf_lookahead

    curr_pair = ""
    custom_trade_info = {}

    # profit/loss thresholds used for assessing buy/sell signals. Keep these realistic!
    # Note: if self.dynamic_gain_thresholds is True, these will be adjusted for each pair, based on historical mean
    default_profit_threshold = 0.008
    default_loss_threshold = -1.2 * default_profit_threshold # bias towards buys, let custom sell/stoploss take care of sells
    profit_threshold = default_profit_threshold
    loss_threshold = default_loss_threshold
    dynamic_gain_thresholds = True # dynamically adjust gain thresholds based on actual mean (beware, training data could be bad)

    num_pairs = 0
    pair_model_info = {} # holds model-related info for each pair

    # debug flags
    first_time = True # mostly for debug
    dbg_scan_classifiers = True  # if True, scan all viable classifiers and choose the best. Very slow!
    dbg_test_classifier = True # test clasifiers after fitting
    dbg_analyse_pca = False # analyze PCA weights
    dbg_verbose = False

    # variables to track state
    class State(Enum):
        INIT = 1
        POPULATE = 2
        STOPLOSS = 3
        RUNNING = 4


    ###################################

    # Strategy Specific Variable Storage

    ## Hyperopt Variables

    # PCA hyperparams
    # buy_pca_gain = IntParameter(1, 50, default=4, space='buy', load=True, optimize=True)
    #
    # sell_pca_gain = IntParameter(-1, -15, default=-4, space='sell', load=True, optimize=True)

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
    cstop_bail_time_trend = CategoricalParameter([True, False], default=True, space='sell', load=True, optimize=True)
    cstop_max_stoploss = DecimalParameter(-0.30, -0.01, default=-0.10, space='sell', load=True, optimize=True)

    ###################################

    """
    inf Pair Definitions
    """

    def inf_pairs(self):
        # all pairs in the whitelist are also in the informative list
        pairs = self.dp.current_whitelist()
        inf_pairs = [(pair, self.inf_timeframe) for pair in pairs]
        return inf_pairs

    ###################################

    """
    Indicator Definitions
    """

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:


        # Base pair inf timeframe indicators
        curr_pair = metadata['pair']
        self.curr_pair = curr_pair

        if self.first_time:
            self.first_time = False
            print("")
            print("***************************************")
            print("** Warning: startup can be very slow **")
            print("***************************************")

            print("    Lookahead: ", self.inf_lookahead, " candles (", self.lookahead_hours, " hours)")
            print("    Thresholds - Profit:{:.2f}% Loss:{:.2f}%".format(self.profit_threshold * 100.0,
                                                                        self.loss_threshold * 100.0))

        print("")
        print(curr_pair)

        self.set_state(curr_pair, self.State.POPULATE)



        # reset profit/loss thresholds
        self.profit_threshold = self.default_profit_threshold
        self.loss_threshold = self.default_loss_threshold

        # if first time through for this pair, add entry to pair_model_info
        if not (curr_pair in self.pair_model_info):
            self.pair_model_info[curr_pair] = {
                'interval': 0,
                'pca': None,
                'clf_buy': None,
                'clf_sell': None
            }
        else:
            # decrement interval. When this reaches 0 it will trigger re-fitting of the data
            self.pair_model_info[curr_pair]['interval'] = self.pair_model_info[curr_pair]['interval'] - 1

        inf = self.dp.get_pair_dataframe(pair=curr_pair, timeframe=self.inf_timeframe)

        # Calculate all technical indicators.
        # # This is done on the inf dataframe because that is what we will use to create the model

        inf = self.add_indicators(inf).fillna(0.0)

        # set global vars for use by rolling functions
        self.curr_lookahead = self.inf_lookahead

        if self.dbg_verbose:
            print("    generating training data...")

        inf = self.create_training_data(inf)

        inf = inf.iloc[:-self.inf_lookahead]  # drop last group (because there cannot be a prediction)

        # Principal Component Analysis of inf data

        # train the models on the informative data
        if self.dbg_verbose:
            print("    training models...")
        self.train_models(inf, curr_pair)

        # populate the normal dataframe
        dataframe = self.add_indicators(dataframe)

        # add predictions
        self.curr_lookahead = self.inf_lookahead * self.inf_ratio

        # this is only needed for debug:
        dataframe = self.create_training_data(dataframe)

        if self.dbg_verbose:
            print("    running predictions...")

        dataframe['predict_buy'] = 0
        dataframe['predict_sell'] = 0
        if self.dp.runmode.value in ('live', 'dry_run'):
            dataframe['predict_buy'] = self.predict_buy(dataframe, curr_pair)
            dataframe['predict_sell'] = self.predict_sell(dataframe, curr_pair)
        else:
            # I think we can just apply - check against rolling solution
            dataframe['predict_buy'] = self.predict_buy(dataframe, curr_pair)
            dataframe['predict_sell'] = self.predict_sell(dataframe, curr_pair)

        # Custom Stoploss
        if self.dbg_verbose:
            print("    updating stoploss data...")
        self.add_stoploss_indicators(dataframe, curr_pair)

        return dataframe

    ###################################

    # populate dataframe with desired technical indicators
    # NOTE: OK to thriow anything in here, just add it to the parameter list
    # The whole idea is to create a dimension-reduced mapping anyway
    def add_indicators(self, dataframe: DataFrame) -> DataFrame:

        self.param_list = []

        win_size = max(self.curr_lookahead, 14)

        # these averages are used internally, do not remove!
        dataframe['sma'] = ta.SMA(dataframe, timeperiod=win_size)
        dataframe['ema'] = ta.EMA(dataframe, timeperiod=win_size)
        dataframe['tema'] = ta.TEMA(dataframe, timeperiod=win_size)
        self.param_list += ['sma', 'ema', 'tema']

        # RSI
        period = 14
        smoothD = 3
        SmoothK = 3
        dataframe['rsi'] = ta.RSI(dataframe, timeperiod=win_size)
        stochrsi = (dataframe['rsi'] - dataframe['rsi'].rolling(period).min()) / (
                    dataframe['rsi'].rolling(period).max() - dataframe['rsi'].rolling(period).min())
        dataframe['srsi_k'] = stochrsi.rolling(SmoothK).mean() * 100
        dataframe['srsi_d'] = dataframe['srsi_k'].rolling(smoothD).mean()
        self.param_list += ['rsi', 'srsi_k', 'srsi_d']

        # Bollinger Bands (must include these)
        bollinger = qtpylib.bollinger_bands(dataframe['close'], window=20, stds=2)
        dataframe['bb_lowerband'] = bollinger['lower']
        dataframe['bb_middleband'] = bollinger['mid']
        dataframe['bb_upperband'] = bollinger['upper']
        dataframe['bb_width'] = ((dataframe['bb_upperband'] - dataframe['bb_lowerband']) / dataframe['bb_middleband'])
        dataframe["bb_gain"] = ((dataframe["bb_upperband"] - dataframe["close"]) / dataframe["close"])
        self.param_list += ['bb_lowerband', 'bb_middleband', 'bb_upperband', 'bb_width', 'bb_gain']

        # Donchian Channels
        dataframe['dc_upper'] = ta.MAX(dataframe['high'], timeperiod=win_size)
        dataframe['dc_lower'] = ta.MIN(dataframe['low'], timeperiod=win_size)
        dataframe['dc_mid'] = ta.TEMA(((dataframe['dc_upper'] + dataframe['dc_lower']) / 2), timeperiod=win_size)
        self.param_list += ["dc_upper", "dc_lower", "dc_mid"]

        dataframe["dcbb_dist_upper"] = (dataframe["dc_upper"] - dataframe['bb_upperband'])
        dataframe["dcbb_dist_lower"] = (dataframe["dc_lower"] - dataframe['bb_lowerband'])
        self.param_list += ["dcbb_dist_upper", "dcbb_dist_lower"]

        # Fibonacci Levels (of Donchian Channel)
        dataframe['dc_dist'] = (dataframe['dc_upper'] - dataframe['dc_lower'])
        dataframe['dc_hf'] = dataframe['dc_upper'] - dataframe['dc_dist'] * 0.236 # Highest Fib
        dataframe['dc_chf'] = dataframe['dc_upper'] - dataframe['dc_dist'] * 0.382 # Centre High Fib
        dataframe['dc_clf'] = dataframe['dc_upper'] - dataframe['dc_dist'] * 0.618 # Centre Low Fib
        dataframe['dc_lf'] = dataframe['dc_upper'] - dataframe['dc_dist'] * 0.764 # Low Fib
        self.param_list += ["dc_dist", "dc_hf", 'dc_chf', 'dc_clf', 'dc_lf']

        # Keltner Channels
        keltner = qtpylib.keltner_channel(dataframe)
        dataframe["kc_upper"] = keltner["upper"]
        dataframe["kc_lower"] = keltner["lower"]
        dataframe["kc_mid"] = keltner["mid"]
        dataframe["kc_percent"] = (
            (dataframe["close"] - dataframe["kc_lower"]) /
            (dataframe["kc_upper"] - dataframe["kc_lower"])
        )
        dataframe["kc_width"] = (
            (dataframe["kc_upper"] - dataframe["kc_lower"]) / dataframe["kc_mid"]
        )
        dataframe['kc_dist'] = (dataframe['kc_upper'] - dataframe['close'])
        self.param_list += ["kc_upper", "kc_lower", 'kc_mid', 'kc_percent', 'kc_width', 'kc_dist']


        # Williams %R
        dataframe['wr'] = 0.02 * (williams_r(dataframe, period=14) + 50.0)
        self.param_list += ['wr']

        # Fisher RSI
        rsi = 0.1 * (dataframe['rsi'] - 50)
        dataframe['fisher_rsi'] = (np.exp(2 * rsi) - 1) / (np.exp(2 * rsi) + 1)
        self.param_list += ['fisher_rsi']

        # Combined Fisher RSI and Williams %R
        dataframe['fisher_wr'] = (dataframe['wr'] + dataframe['fisher_rsi']) / 2.0
        self.param_list += ['fisher_wr']

        # RSI
        dataframe['rsi'] = ta.RSI(dataframe, timeperiod=win_size)
        dataframe['rsi_14'] = ta.RSI(dataframe, timeperiod=14)
        self.param_list += ['rsi', 'rsi_14']


        # RMI: https://www.tradingview.com/script/kwIt9OgQ-Relative-Momentum-Index/
        dataframe['rmi'] = cta.RMI(dataframe, length=24, mom=5)
        self.param_list += ['rmi']

        # MA Streak: https://www.tradingview.com/script/Yq1z7cIv-MA-Streak-Can-Show-When-a-Run-Is-Getting-Long-in-the-Tooth/
        dataframe['mastreak'] = cta.mastreak(dataframe, period=4)
        self.param_list += ['mastreak']

        # Trends
        dataframe['candle_up'] = np.where(dataframe['close'] >= dataframe['close'].shift(), 1.0, 0.0)
        dataframe['candle_up_trend'] = np.where(dataframe['candle_up'].rolling(5).sum() >= 3.0, 1.0, 0.0)
        dataframe['candle_up_seq'] = dataframe['candle_up'].rolling(5).sum()

        dataframe['candle_dn'] = np.where(dataframe['close'] < dataframe['close'].shift(), 1.0, 0.0)
        dataframe['candle_dn_trend'] = np.where(dataframe['candle_up'].rolling(5).sum() >= 3.0, 1.0, 0.0)
        dataframe['candle_dn_seq'] = dataframe['candle_up'].rolling(5).sum()

        self.param_list += ['candle_up', 'candle_up_trend', 'candle_up_seq', 'candle_dn', 'candle_dn_trend',
                            'candle_dn_seq']

        dataframe['rmi_up'] = np.where(dataframe['rmi'] >= dataframe['rmi'].shift(), 1.0, 0.0)
        dataframe['rmi_up_trend'] = np.where(dataframe['rmi_up'].rolling(5).sum() >= 3, 1.0, 0.0)

        dataframe['rmi_dn'] = np.where(dataframe['rmi'] <= dataframe['rmi'].shift(), 1.0, 0.0)
        dataframe['rmi_dn_count'] = dataframe['rmi_dn'].rolling(8).sum()
        self.param_list += ['candle_up_trend', 'rmi_up_trend', 'rmi_dn_count']

        # Indicators used only for ROI and Custom Stoploss
        ssldown, sslup = cta.SSLChannels_ATR(dataframe, length=21)
        dataframe['sroc'] = cta.SROC(dataframe, roclen=21, emalen=13, smooth=21)
        dataframe['ssl_dir'] = 0
        dataframe['ssl_dir'] = np.where(sslup > ssldown, 1.0, 0.0)
        self.param_list += ['sroc', 'ssl_dir']

        # EMAs
        dataframe['ema_12'] = ta.EMA(dataframe, timeperiod=12)
        dataframe['ema_20'] = ta.EMA(dataframe, timeperiod=20)
        dataframe['ema_25'] = ta.EMA(dataframe, timeperiod=25)
        dataframe['ema_35'] = ta.EMA(dataframe, timeperiod=35)
        dataframe['ema_50'] = ta.EMA(dataframe, timeperiod=50)
        dataframe['ema_100'] = ta.EMA(dataframe, timeperiod=100)
        dataframe['ema_200'] = ta.EMA(dataframe, timeperiod=200)
        self.param_list += ['ema_12', 'ema_20', 'ema_25', 'ema_35', 'ema_50', 'ema_100', 'ema_200']

        # SMA
        dataframe['sma_200'] = ta.SMA(dataframe, timeperiod=200)
        dataframe['sma_200_dec_20'] = dataframe['sma_200'] < dataframe['sma_200'].shift(20)
        dataframe['sma_200_dec_24'] = dataframe['sma_200'] < dataframe['sma_200'].shift(24)
        self.param_list += ['sma_200', 'sma_200_dec_20', 'sma_200_dec_24']

        # CMF
        dataframe['cmf'] = chaikin_money_flow(dataframe, 20)
        self.param_list += ['cmf']

        # CTI
        dataframe['cti'] = pta.cti(dataframe["close"], length=20)
        self.param_list += ['cti']

        # CRSI (3, 2, 100)
        crsi_closechange = dataframe['close'] / dataframe['close'].shift(1)
        crsi_updown = np.where(crsi_closechange.gt(1), 1.0, np.where(crsi_closechange.lt(1), -1.0, 0.0))
        dataframe['crsi'] = (ta.RSI(dataframe['close'], timeperiod=3) + ta.RSI(crsi_updown, timeperiod=2) + ta.ROC(
            dataframe['close'],
            100)) / 3
        self.param_list += ['crsi']

        # Williams %R
        dataframe['r_14'] = williams_r(dataframe, period=14)
        dataframe['r_480'] = williams_r(dataframe, period=480)
        self.param_list += ['r_14', 'r_480']
        # ROC
        dataframe['roc_9'] = ta.ROC(dataframe, timeperiod=9)
        self.param_list += ['roc_9']

        # T3 Average
        dataframe['t3_avg'] = t3_average(dataframe)
        self.param_list += ['t3_avg']

        # S/R
        res_series = dataframe['high'].rolling(window=5, center=True).apply(lambda row: is_resistance(row),
                                                                            raw=True).shift(2)
        sup_series = dataframe['low'].rolling(window=5, center=True).apply(lambda row: is_support(row), raw=True).shift(
            2)
        dataframe['res_level'] = Series(
            np.where(res_series,
                     np.where(dataframe['close'] > dataframe['open'], dataframe['close'], dataframe['open']),
                     float('NaN'))).ffill()
        dataframe['res_hlevel'] = Series(np.where(res_series, dataframe['high'], float('NaN'))).ffill()
        dataframe['sup_level'] = Series(
            np.where(sup_series,
                     np.where(dataframe['close'] < dataframe['open'], dataframe['close'], dataframe['open']),
                     float('NaN'))).ffill()
        # self.param_list += ['res_level', 'res_hlevel', 'sup_level']

        # Pump protections
        dataframe['hl_pct_change_48'] = range_percent_change(dataframe, 'HL', 48)
        dataframe['hl_pct_change_36'] = range_percent_change(dataframe, 'HL', 36)
        dataframe['hl_pct_change_24'] = range_percent_change(dataframe, 'HL', 24)
        dataframe['hl_pct_change_12'] = range_percent_change(dataframe, 'HL', 12)
        dataframe['hl_pct_change_6'] = range_percent_change(dataframe, 'HL', 6)
        self.param_list += ['hl_pct_change_6', 'hl_pct_change_12', 'hl_pct_change_24', 'hl_pct_change_36',
                            'hl_pct_change_48']

        # ADX
        dataframe['adx'] = ta.ADX(dataframe)
        self.param_list += ["adx"]

        # Plus Directional Indicator / Movement
        dataframe['dm_plus'] = ta.PLUS_DM(dataframe)
        dataframe['di_plus'] = ta.PLUS_DI(dataframe)
        self.param_list += ['dm_plus', 'di_plus']

        # Minus Directional Indicator / Movement
        dataframe['dm_minus'] = ta.MINUS_DM(dataframe)
        dataframe['di_minus'] = ta.MINUS_DI(dataframe)
        dataframe['dm_delta'] = dataframe['dm_plus'] - dataframe['dm_minus']
        dataframe['di_delta'] = dataframe['di_plus'] - dataframe['di_minus']
        self.param_list += ['dm_minus', 'di_minus', 'dm_delta', 'di_delta']

        # MACD
        macd = ta.MACD(dataframe)
        dataframe['macd'] = macd['macd']
        dataframe['macdsignal'] = macd['macdsignal']
        dataframe['macdhist'] = macd['macdhist']
        self.param_list += ['macd', 'macdsignal', 'macdhist']

        # Stoch fast
        stoch_fast = ta.STOCHF(dataframe)
        dataframe['fastd'] = stoch_fast['fastd']
        dataframe['fastk'] = stoch_fast['fastk']
        self.param_list += ['fastd', 'fastk']

        # SAR Parabol
        dataframe['sar'] = ta.SAR(dataframe)
        self.param_list += ['sar']

        dataframe['mom'] = ta.MOM(dataframe, timeperiod=14)
        self.param_list += ['mom']

        # priming indicators
        dataframe['color'] = np.where((dataframe['close'] > dataframe['open']), 1.0, 0.0)
        dataframe['rsi_7'] = ta.RSI(dataframe, timeperiod=7)
        dataframe['roc_6'] = ta.ROC(dataframe, timeperiod=6)
        dataframe['primed'] = np.where(dataframe['color'].rolling(3).sum() == 3.0, 1.0, 0.0)
        dataframe['in_the_mood'] = dataframe['rsi_7'] > dataframe['rsi_7'].rolling(12).mean()
        dataframe['moist'] = qtpylib.crossed_above(dataframe['macd'], dataframe['macdsignal'])
        dataframe['throbbing'] = dataframe['roc_6'] > dataframe['roc_6'].rolling(12).mean()
        self.param_list += ['color', 'primed', 'in_the_mood', 'moist', 'throbbing']

        ## sqzmi to detect quiet periods
        dataframe['sqzmi'] = fta.SQZMI(dataframe)
        self.param_list += ['sqzmi']

        # MFI
        dataframe['mfi'] = ta.MFI(dataframe)
        dataframe['mfi_norm'] = self.norm_column(dataframe['mfi'])
        dataframe['mfi_buy'] = np.where((dataframe['mfi_norm'] > 0.5), 1.0, 0.0)
        dataframe['mfi_sell'] = np.where((dataframe['mfi_norm'] < -0.5), 1.0, 0.0)
        dataframe['mfi_signal'] = dataframe['mfi_buy'] - dataframe['mfi_sell']
        self.param_list += ['mfi', 'mfi_signal']

        # Volume Flow Indicator (MFI) for volume based on the direction of price movement
        dataframe['vfi'] = fta.VFI(dataframe, period=14)
        dataframe['vfi_norm'] = self.norm_column(dataframe['vfi'])
        dataframe['vfi_buy'] = np.where((dataframe['vfi_norm'] > 0.5), 1.0, 0.0)
        dataframe['vfi_sell'] = np.where((dataframe['vfi_norm'] < -0.5), 1.0, 0.0)
        dataframe['vfi_signal'] = dataframe['vfi_buy'] - dataframe['vfi_sell']
        self.param_list += ['vfi', 'vfi_signal']

        #ATR
        dataframe['atr'] = ta.ATR(dataframe, timeperiod=win_size)
        dataframe['atr_norm'] = self.norm_column(dataframe['atr'])
        dataframe['atr_buy'] = np.where((dataframe['atr_norm'] > 0.5), 1.0, 0.0)
        dataframe['atr_sell'] = np.where((dataframe['atr_norm'] < -0.5), 1.0, 0.0)
        dataframe['atr_signal'] = dataframe['atr_buy'] - dataframe['atr_sell']
        self.param_list += ['atr', 'atr_signal']

        # Hilbert Transform Indicator - SineWave
        hilbert = ta.HT_SINE(dataframe)
        dataframe['htsine'] = hilbert['sine']
        dataframe['htleadsine'] = hilbert['leadsine']
        self.param_list += ['htsine', 'htleadsine']

        # Oscillators

        # EWO
        dataframe['ewo'] = ewo(dataframe, 50, 200)
        self.param_list += ['ewo']

        # Ultimate Oscillator
        dataframe['uo'] = ta.ULTOSC(dataframe)
        self.param_list += ['uo']

        # Aroon, Aroon Oscillator
        aroon = ta.AROON(dataframe)
        dataframe['aroonup'] = aroon['aroonup']
        dataframe['aroondown'] = aroon['aroondown']
        dataframe['aroonosc'] = ta.AROONOSC(dataframe)
        self.param_list += ['aroonup', 'aroondown', 'aroonosc']

        # Awesome Oscillator
        dataframe['ao'] = qtpylib.awesome_oscillator(dataframe)
        self.param_list += ['ao']

        # Commodity Channel Index: values [Oversold:-100, Overbought:100]
        dataframe['cci'] = ta.CCI(dataframe)
        self.param_list += ['cci']

        return dataframe

    def create_training_data(self, dataframe: DataFrame) -> DataFrame:

        dataframe['train_buy_signal'] = 0.0
        dataframe['train_sell_signal'] = 0.0

        dataframe = self.add_future_data(dataframe)

        # set targets based on data just generated
        self.set_gain_targets(dataframe)

        # set the buy and sell signals used for training
        dataframe['train_buy_signal'] = np.where(
            (
                (  # price went up above threshold
                    (dataframe['future_profit'] > self.profit_threshold) &
                    (dataframe['future_trend'] > 0)
                )
            ), 1.0, 0.0)

        use_tema = True
        if use_tema:
            # I have no idea why, but sometimes sell signals work better using TEMA
            dataframe['train_sell_signal'] = np.where(
                (
                    (  # price went went down below threshold
                        (dataframe['future_tema_loss'] < self.loss_threshold) &
                        (dataframe['future_trend'] <= 0)
                    )
                ), 1.0, 0.0)
        else:
            # just use the 'close' data
            dataframe['train_sell_signal'] = np.where(
                (
                    (  # price went down below threshold
                            (dataframe['future_loss'] < self.loss_threshold) &
                            (dataframe['future_trend'] <= 0)
                    )
                ), 1.0, 0.0)

        return dataframe


    # calculate future gains. Used for setting targets
    def add_future_data(self, dataframe: DataFrame) -> DataFrame:

        # yes, we lookahead in the data!
        lookahead = self.curr_lookahead

        dataframe['future_close'] = 0.0
        dataframe['future_trend'] = 0.0

        dataframe['future_profit'] = 0.0
        dataframe['future_loss'] = 0.0

        dataframe['future_tema'] = 0.0
        dataframe['future_tema_profit'] = 0.0
        dataframe['future_tema_loss'] = 0.0

        # to try: run rolling window on the reversed Series (and reverse the answer)
        # dataframe.rolling_sum(ts[::-1], window=lookahead, min_periods=0)[::-1]

        # calculate future gains
        dataframe['future_close'] = dataframe['close'].shift(-lookahead)
        dataframe['future_tema'] = dataframe['tema'].shift(-lookahead)

        dataframe['future_gain'] = (dataframe['future_close'] - dataframe['close']) / dataframe['close']
        dataframe['future_gain'].clip(lower=-1.0, upper=1.0, inplace=True)

        dataframe['future_tema_gain'] = (dataframe['future_tema'] - dataframe['tema']) / dataframe['tema']
        dataframe['future_tema_gain'].clip(lower=-1.0, upper=1.0, inplace=True)

        # dataframe['future_trend'] = dataframe['candle_up_trend'].shift(-lookahead)
        dataframe['future_trend'] = dataframe['throbbing'].shift(-lookahead)

        dataframe['future_profit'] = dataframe['future_gain'].apply(lambda x: x if x > 0.0 else 0.0)
        dataframe['future_loss'] = dataframe['future_gain'].apply(lambda x: x if x < 0.0 else 0.0)

        dataframe['future_tema_profit'] = dataframe['future_tema_gain'].apply(lambda x: x if x > 0.0 else 0.0)
        dataframe['future_tema_loss'] = dataframe['future_tema_gain'].apply(lambda x: x if x < 0.0 else 0.0)

        dataframe['future_gain'].fillna(0.0, inplace=True)
        dataframe['future_profit'].fillna(0.0, inplace=True)
        dataframe['future_loss'].fillna(0.0, inplace=True)

        dataframe['future_tema_gain'].fillna(0.0, inplace=True)
        dataframe['future_tema_profit'].fillna(0.0, inplace=True)
        dataframe['future_tema_loss'].fillna(0.0, inplace=True)

        dataframe['future_close'].fillna(0.0, inplace=True)
        dataframe['future_trend'].fillna(0.0, inplace=True)

        return dataframe

    # calculate profit/loss targets for training
    def set_gain_targets(self, dataframe: DataFrame) -> DataFrame:

        # Debug: show mean & stddev, and potential target
        pmean = dataframe['future_profit'].mean()
        pstd = dataframe['future_profit'].std()
        ptgt = pmean + self.n_profit_stddevs * abs(pstd)
        lmean = dataframe['future_loss'].mean()
        lstd = dataframe['future_loss'].std()
        ltgt = lmean - self.n_loss_stddevs * abs(lstd)

        # print("    Profit - target:{:.3f}".format(ptgt))
        # print("    Loss   - target:{:.3f}".format(ltgt))

        if self.dynamic_gain_thresholds:
            ptgt = min(ptgt, 0.05)
            if (ptgt > 0.0) & (ptgt != self.profit_threshold):
                print("    Profit - target:{:.3f} -> {:.3f}".format(self.profit_threshold, ptgt))
                self.profit_threshold = ptgt
            ltgt = max(ltgt, -0.05)
            if (ltgt < 0.0) & (ltgt != self.loss_threshold):
                print("    Loss   - target:{:.3f} -> {:.3f}".format(self.loss_threshold, ltgt))
                self.loss_threshold = ltgt

        return dataframe


    # add indicators used by stoploss/custom sell logic
    def add_stoploss_indicators(self, dataframe, pair) -> DataFrame:
        if not pair in self.custom_trade_info:
            self.custom_trade_info[pair] = {}
            if not 'had_trend' in self.custom_trade_info[pair]:
                self.custom_trade_info[pair]['had_trend'] = False

        return dataframe

    # normalises a column. Data is in units of 1 stddev, i.e. a value of 1.0 represents 1 stdev above mean
    def norm_column(self, col:Series) -> Series:
        return (col - col.mean()) / col.std()

    # applies MinMax scaling to a column. Returns all data in range [0,1]
    def scale_column(self, col:Series) -> Series:
        return (col - col.min()) / (col.max() - col.min())

    # Normalise a dataframe
    def norm_dataframe(self, dataframe: DataFrame) -> DataFrame:
        return ((dataframe - dataframe.mean()) / dataframe.std()).fillna(0.0)

    # Scale a dataframe
    def scale_dataframe(self, dataframe: DataFrame) -> DataFrame:
        scaler = StandardScaler()
        # scaler.fit(dataframe)
        df_scaled = DataFrame(scaler.fit_transform(dataframe))
        return df_scaled.fillna(0.0)

    binary_encoder: LabelEncoder = None
    def get_binary_labels(self, col):
        if not self.binary_encoder:
            self.binary_encoder = LabelEncoder().fit([0.0, 1.0])

        return  self.binary_encoder.transform(col)

    # train the reduction and classification models

    def train_models(self, dataframe: DataFrame, curr_pair):

        # only run if interval reaches 0
        count = self.pair_model_info[curr_pair]['interval']
        if (count > 0):
            self.pair_model_info[curr_pair]['interval'] = count - 1
            return
        else:
            # reset interval to a random number between 1 and the amount of lookahead
            self.pair_model_info[curr_pair]['interval'] = random.randint(1, self.inf_lookahead)

        # if in non-run mode, reduce size of dataframe to match run-time buffer size (975)
        if self.dp.runmode.value in ('live', 'dry_run'):
            df_train = dataframe
            df_test = dataframe
        else:
            train_size = int(min(975, 0.66*dataframe.shape[0]))
            random_st = 27  # use fixed number for reproducibility
            df_train, df_test = train_test_split(dataframe, train_size=train_size, random_state=random_st)
            if  self.dbg_verbose:
                print("     dataframe:", dataframe.shape, ' -> train:', df_train.shape, " + test:", df_test.shape)

        # extract buy/sell signals
        buys = dataframe['train_buy_signal'].fillna(0.0)
        sells = dataframe['train_sell_signal'].fillna(0.0)
        train_buys = df_train['train_buy_signal'].fillna(0.0)
        train_sells = df_train['train_sell_signal'].fillna(0.0)
        test_buys = df_test['train_buy_signal'].fillna(0.0)
        test_sells = df_test['train_sell_signal'].fillna(0.0)

        buy_labels = self.get_binary_labels(buys)
        sell_labels = self.get_binary_labels(sells)
        train_buy_labels = self.get_binary_labels(train_buys)
        train_sell_labels = self.get_binary_labels(train_sells)
        test_buy_labels = self.get_binary_labels(test_buys)
        test_sell_labels = self.get_binary_labels(test_sells)

        # extract (only) indicator columns
        df = df_train[self.param_list].fillna(0.0)

        # normalise the data (required by the PCA/Classifier libraries)
        df_norm = self.norm_dataframe(df)

        # create the PCA analysis model

        pca = self.get_pca(df_norm)

        df_norm_pca = DataFrame(pca.transform(df_norm))

        # DEBUG:
        # print("")
        print("   ", curr_pair, " - input: ", df_norm.shape, " -> pca: ", df_norm_pca.shape)

        # Create buy/sell classifiers for the model

        buy_clf = self.get_buy_classifier(df_norm_pca, train_buy_labels)
        sell_clf = self.get_sell_classifier(df_norm_pca, train_sell_labels)

        # save the models

        self.pair_model_info[curr_pair]['pca'] = pca
        self.pair_model_info[curr_pair]['clf_buy'] = buy_clf
        self.pair_model_info[curr_pair]['clf_sell'] = sell_clf

        # if scan specified, test against the test dataframe
        if self.dbg_scan_classifiers and self.dbg_verbose:
            df = df_test[self.param_list].fillna(0.0)
            df_norm = self.norm_dataframe(df)
            df_norm_pca = DataFrame(pca.transform(df_norm))
            pred_buys = buy_clf.predict(df_norm_pca)
            print("")
            print("Predict - Buy Signals (", type(buy_clf).__name__, ")")
            print(classification_report(test_buy_labels, pred_buys))
            print("")

            pred_sells = sell_clf.predict(df_norm_pca)
            print("")
            print("Predict - Buy Signals (", type(sell_clf).__name__, ")")
            print(classification_report(test_sell_labels, pred_sells))
            print("")

    # get the PCA model for the supplied dataframe (dataframe must be normalised)
    def get_pca(self, df_norm: DataFrame):

        ncols = df_norm.shape[1] # allow all components to get the full variance matrix
        whiten = False
        # pca = skd.PCA(n_components=ncols, svd_solver="randomized", whiten=whiten).fit(df_norm)
        pca = skd.PCA(n_components=ncols, whiten=whiten, svd_solver='full').fit(df_norm)

        # scan variance and only take if column contributes >x%
        ncols = 1
        variance_threshold = 0.005
        while ((pca.explained_variance_ratio_[ncols] >= variance_threshold) & (ncols < len(pca.explained_variance_ratio_))):
            ncols = ncols + 1

        # if necessary, re-calculate pca with reduced column set
        if (ncols != df_norm.shape[1]):
            # pca = skd.PCA(n_components=ncols, svd_solver="randomized", whiten=True).fit(df_norm)
            pca = skd.PCA(n_components=ncols, whiten=whiten, svd_solver='full').fit(df_norm)

        if self.dbg_analyse_pca and self.dbg_verbose:
            self.analyse_pca(pca, df_norm)

        return pca

    def analyse_pca(self, pca, df):
        print("")
        print("Variance Ratios:")
        ratios = pca.explained_variance_ratio_
        print(ratios)
        print("")

        # print matrix of weightings for selected components
        loadings = pd.DataFrame(pca.components_.T, index=df.columns.values)
        l2 = loadings.abs()
        # l3 = loadings.mul(ratios)
        ranks = loadings.rank()

        loadings['Score'] = l2.sum(axis=1)
        loadings['Score0'] = loadings[loadings.columns.values[0]].abs()
        loadings['Rank'] = loadings['Score'].rank(ascending=False)
        loadings['Rank0'] = loadings['Score0'].rank(ascending=False)
        print("Loadings, by PC0:")
        print(loadings.sort_values('Rank0').head(n=30))
        print("")
        print("Loadings, by All Columns:")
        print(loadings.sort_values('Rank').head(n=30))
        print("")

        # # weighted by variance ratios
        # l3a = l3.abs()
        # l3['Score'] = l3a.sum(axis=1)
        # l3['Rank'] = loadings['Score'].rank(ascending=False)
        # print("Loadings, Weighted by Variance Ratio")
        # print (l3.sort_values('Rank').head(n=20))

        # # rankings per column
        ranks['Score'] = ranks.sum(axis=1)
        ranks['Rank'] = ranks['Score'].rank(ascending=True)
        print("Rankings per column")
        print(ranks.sort_values('Rank', ascending=True).head(n=30))

        # print(loadings.head())
        # print(l3.head())

    # get a classifier for the supplied dataframe (normalised) and known results
    def get_buy_classifier(self, df_norm: DataFrame, results):

        labels = self.get_binary_labels(results)

        # If already done, just get previous result and re-fit
        if self.pair_model_info[self.curr_pair]['clf_buy']:
            clf = self.pair_model_info[self.curr_pair]['clf_buy']
            clf = clf.fit(df_norm, labels)
        else:
            if self.dbg_scan_classifiers:
                print("    Finding best buy classifier:")
                clf = self.find_best_classifier(df_norm, labels)
            else:
                clf = self.classifier_factory(self.default_classifier)
                clf = clf.fit(df_norm, labels)

        return clf

    # get a classifier for the supplied dataframe (normalised) and known results
    def get_sell_classifier(self, df_norm: DataFrame, results):

        labels = self.get_binary_labels(results)

        # If already done, just get previous result and re-fit
        if self.pair_model_info[self.curr_pair]['clf_sell']:
            clf = self.pair_model_info[self.curr_pair]['clf_sell']
            clf = clf.fit(df_norm, labels)
        else:
            if self.dbg_scan_classifiers:
                print("    Finding best sell classifier:")
                clf = self.find_best_classifier(df_norm, labels)
            else:
                clf = self.classifier_factory(self.default_classifier)
                clf = clf.fit(df_norm, labels)

        return clf

    # default classifier
    default_classifier = 'GaussianNB' # select based on testing

    # list of potential classifier types - set to the list that you want to compare
    classifier_list = [
        # 'LogisticRegression', 'LinearSVC', 'DecisionTree', 'RandomForest', 'GaussianNB',
        # 'MLP', 'KNeighbors'
        'LogisticRegression', 'KNeighbors', 'DecisionTree', 'RandomForest', 'GaussianNB', 'SGD',
        'GradientBoosting', 'AdaBoost', 'QDA', 'linearSVC', 'gaussianSVC', 'polySVC', 'sigmoidSVC',
        'Voting'
    ]

    # factory to create classifier based on name
    def classifier_factory(self, name):
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
            clf = MLPClassifier(alpha=1, max_iter=1000)
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
            c1 = self.classifier_factory('AdaBoost')
            c2 = self.classifier_factory('GaussianNB')
            c3 = self.classifier_factory('KNeighbors')
            c4 = self.classifier_factory('DecisionTree')
            clf = VotingClassifier(estimators=[('c1', c1), ('c2', c2), ('c3', c3), ('c4', c4)], voting='hard')


        else:
            print("Unknown classifier: ", name)
            clf = None
        return clf

    # tries different types of classifiers and returns the best one
    def find_best_classifier(self, df, results):

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

        #split into test/train for evaluation, then re-fit once selected
        # df_train, df_test, res_train, res_test = train_test_split(df, results, train_size=0.5)
        df_train, df_test, res_train, res_test = train_test_split(df, labels, train_size=0.5, random_state=27)
        # print("df_train:",  df_train.shape, " df_test:", df_test.shape,
        #       "res_train:", res_train.shape, "res_test:", res_test.shape)

        for cname in self.classifier_list:
            clf = self.classifier_factory(cname)

            if clf is not None:
                # fit to the training data
                clf_dict[cname] = clf
                clf = clf.fit(df_train, res_train)

                # assess using the test data. Do *not* use the training data for testing
                pred_test = clf.predict(df_test)
                # score = f1_score(results, prediction, average=None)[1]
                score = f1_score(res_test, pred_test, average='macro')

                if self.dbg_verbose:
                    print("      {0:<20}: {1:.2f}".format(cname, score))

                if score > best_score:
                    best_score = score
                    best_classifier = cname

        clf = clf_dict[best_classifier]
        # clf = clf.fit(df, results)
        clf = clf.fit(df, labels) # re-fit to full dataframe

        # print("")
        if best_score <= 0.5:
            print("")
            print("!!! WARNING: F1 score below 51% ({:.3f})".format(best_score))
            clf = None
            print("")

        print("       Model selected: ", best_classifier, " Score:{:.3f}".format(best_score))
        # print("")

        return clf

    # make predictions for supplied dataframe (returns column)
    def predict(self, dataframe, pair, clf):

        # predict = 0
        predict = None

        pca = self.pair_model_info[pair]['pca']

        if clf:
            df = dataframe[self.param_list].fillna(0)
            # df_norm = (df - df.mean()) / df.std()
            # df_norm.fillna(0, inplace=True)
            df_norm = self.norm_dataframe(df)
            df_norm_pca = pca.transform(df_norm)
            predict = clf.predict(df_norm_pca)

        else:
            print("Null CLF for pair: ", pair)

        # print (predict)
        return predict

    def predict_buy(self, df, pair):
        clf = self.pair_model_info[pair]['clf_buy']

        if clf is None:
            print ("    No Classifier for pair ", pair, " -Skipping predictions")
            self.pair_model_info[pair]['interval'] = min(self.pair_model_info[pair]['interval'], 4)
            predict = df['close'] # just to get the size
            predict = 0.0
            return predict

        predict = self.predict(df, pair, clf)

        if self.dbg_test_classifier:
            # DEBUG: check accuracy
            signals = df['train_buy_signal']
            labels = self.get_binary_labels(signals)

            if  self.dbg_verbose:
                print("")
                print("Predict - Buy Signals (", type(clf).__name__, ")")
                print(classification_report(labels, predict))
                print("")

            score = f1_score(labels, predict, average='macro')
            if score <= 0.5:
                print("")
                print("!!! WARNING: (buy) F1 score below 51% ({:.3f})".format(score))
                print("    Classifier:", type(clf).__name__)
                print("")

        return predict

    def predict_sell(self, df, pair):
        clf = self.pair_model_info[pair]['clf_sell']
        if clf is None:
            print ("    No Classifier for pair ", pair, " -Skipping predictions")
            self.pair_model_info[pair]['interval'] = min(self.pair_model_info[pair]['interval'], 4)
            predict = df['close'] # just to get the size
            predict = 0.0
            return predict

        predict = self.predict(df, pair, clf)

        if self.dbg_test_classifier:
            # DEBUG: check accuracy
            signals = df['train_sell_signal']
            labels = self.get_binary_labels(signals)

            if self.dbg_verbose:
                print("")
                print("Predict - Sell Signals (", type(clf).__name__, ")")
                print(classification_report(labels, predict))
                print("")

            score = f1_score(labels, predict, average='macro')
            if score <= 0.5:
                print("")
                print("!!! WARNING: (buy) F1 score below 51% ({:.3f})".format(score))
                print("    Classifier:", type(clf).__name__)
                print("")

        return predict

    ###################################
    # Debug stuff

    curr_state = {}

    def set_state(self, pair, state: State):
        if pair in self.curr_state:
            print("  ", pair, ": ", self.curr_state[pair], " -> ", state)
        else:
            print("  ", pair, ": ", " -> ", state)

        self.curr_state[pair] = state

    def get_state(self, pair) -> State:
        return self.curr_state[pair]

    def show_debug_info(self, pair):

        print("")
        # print("pair_model_info:")
        print("  ", pair, ": ", self.pair_model_info[pair])
        print("")

    ###################################

    """
    Buy Signal
    """

    def populate_buy_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        conditions = []
        dataframe.loc[:, 'enter_tag'] = ''
        curr_pair = metadata['pair']

        #TODO: only run once in hyperopt mode
        if not self.dp.runmode.value in ('hyperopt'):
            if self.get_state(curr_pair) != self.State.RUNNING:
                self.set_state(curr_pair, self.State.RUNNING)
                self.show_debug_info(curr_pair)


        # conditions.append(dataframe['volume'] > 0)

        # # currently below TEMA
        # conditions.append(dataframe['close'] < dataframe['tema'])

        # # ATR in buy range
        # conditions.append(dataframe['atr_signal'] > 0.0)

        # PCA triggers
        pca_cond = (
            (qtpylib.crossed_above(dataframe['predict_buy'], 0.5))
        )
        conditions.append(pca_cond)

        # set entry tags
        dataframe.loc[pca_cond, 'enter_tag'] += 'pca_entry '

        if conditions:
            dataframe.loc[reduce(lambda x, y: x & y, conditions), 'buy'] = 1

        return dataframe

    ###################################

    """
    Sell Signal
    """

    def populate_sell_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        conditions = []
        dataframe.loc[:, 'exit_tag'] = ''
        curr_pair = metadata['pair']

        if not self.dp.runmode.value in ('hyperopt'):
            if self.get_state(curr_pair) != self.State.RUNNING:
                self.set_state(curr_pair, self.State.RUNNING)
                self.show_debug_info(curr_pair)

        # conditions.append(dataframe['volume'] > 0)

        # # only sell if currently above TEMA
        # conditions.append(dataframe['close'] > dataframe['tema'])

        # # ATR in sell range
        # conditions.append(dataframe['atr_signal'] <= 0.0)

        # PCA triggers
        pca_cond = (
            qtpylib.crossed_above(dataframe['predict_sell'], 0.5)
        )

        conditions.append(pca_cond)

        dataframe.loc[pca_cond, 'exit_tag'] += 'pca_exit '

        if conditions:
            dataframe.loc[reduce(lambda x, y: x & y, conditions), 'sell'] = 1

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

    def custom_sell(self, pair: str, trade: 'Trade', current_time: 'datetime', current_rate: float,
                    current_profit: float, **kwargs):

        dataframe, _ = self.dp.get_analyzed_dataframe(pair=pair, timeframe=self.timeframe)
        last_candle = dataframe.iloc[-1].squeeze()

        trade_dur = int((current_time.timestamp() - trade.open_date_utc.timestamp()) // 60)
        max_profit = max(0, trade.calc_profit_ratio(trade.max_rate))
        pullback_value = max(0, (max_profit - self.csell_pullback_amount.value))
        in_trend = False

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
