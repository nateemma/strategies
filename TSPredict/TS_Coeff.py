"""
####################################################################################
TS_Coeff - base class for 'simple' time series prediction
             Handles most of the logic for time series prediction. Subclasses should
             override the model-related functions

             This strategy uses only the 'base' indicators (open, close etc.) to estimate future gain.
             Note that I use gain rather than price because it is a normalised value, and works better with prediction algorithms.
             I use the actual (future) gain to train a base model, which is then further refined for each individual pair.
             The model is created if it does not exist, and is trained on all available data before being saved.
             Models are saved in user_data/strategies/TSPredict/models/<class>/<class>.sav, where <class> is the name of the current class
             (TS_Coeff if running this directly, or the name of the subclass). 
             If the model already exits, then it is just loaded and used.
             So, it makes sense to do initial training over a long period of time to create the base model. 
             If training, then no backtesting or tuning for individual pairs is performed (way faster).
             If you want to retrain (e.g. you changed indicators), then delete the model and run the strategy over a long time period

####################################################################################
"""

#pragma pylint: disable=W0105, C0103, C0114, C0115, C0116, C0301, C0303, C0325, W1203

from datetime import datetime
from functools import reduce

import cProfile
import pstats

import numpy as np
# Get rid of pandas warnings during backtesting
import pandas as pd
from pandas import DataFrame, Series

import pywt

# from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import RobustScaler


# import freqtrade.vendor.qtpylib.indicators as qtpylib

from freqtrade.strategy import (IStrategy, DecimalParameter, CategoricalParameter)
from freqtrade.persistence import Trade
import freqtrade.vendor.qtpylib.indicators as qtpylib

pd.options.mode.chained_assignment = None  # default='warn'

# Strategy specific imports, files must reside in same folder as strategy
import sys
from pathlib import Path

import os
import joblib
import copy

group_dir = str(Path(__file__).parent)
strat_dir = str(Path(__file__).parent.parent)
sys.path.append(strat_dir)
sys.path.append(group_dir)

# # this adds  ../utils
# sys.path.append("../utils")

import logging
import warnings

log = logging.getLogger(__name__)
log.setLevel(logging.DEBUG)
warnings.simplefilter(action='ignore', category=pd.errors.PerformanceWarning)

from xgboost import XGBRegressor
# from lightgbm import LGBMRegressor
from sklearn.linear_model import PassiveAggressiveRegressor
from sklearn.linear_model import SGDRegressor

from utils.DataframeUtils import DataframeUtils, ScalerType # pylint: disable=E0401
# import pywt
import talib.abstract as ta





class TS_Coeff(IStrategy):
    # Do *not* hyperopt for the roi and stoploss spaces


    plot_config = {
        'main_plot': {
            'close': {'color': 'cornflowerblue'}
        },
        'subplots': {
            "Diff": {
                'predicted_gain': {'color': 'orange'},
                'future_gain': {'color': 'lightblue'},
                'target_profit': {'color': 'lightgreen'},
                'target_loss': {'color': 'lightsalmon'}
            },
        }
    }


    # ROI table:
    minimal_roi = {
        "0": 0.06
    }

    # Stoploss:
    stoploss = -0.10

    # Trailing stop:
    trailing_stop = False
    trailing_stop_positive = None
    trailing_stop_positive_offset = 0.0
    trailing_only_offset_is_reached = False

    timeframe = '5m'
    inf_timeframe = '15m'

    use_custom_stoploss = True

    # Recommended
    use_exit_signal = True
    exit_profit_only = False
    ignore_roi_if_entry_signal = True

    # Required
    startup_candle_count: int = 128  # must be power of 2
    win_size = 14

    process_only_new_candles = True

    custom_trade_info = {} # pair-specific data
    curr_pair = ""

    ###################################

    # Strategy Specific Variable Storage

    ## Hyperopt Variables

    # model_window = startup_candle_count
    model_window = 128

    lookahead = 6

    df_coeffs: DataFrame = None
    coeff_table = None
    coeff_array = None

    training_data = None
    training_labels = None
    training_mode = False # do not set manually
    supports_incremental_training = True
    model_per_pair = False
    combine_models = True
    model_trained = False
    new_model = False

    norm_data = True
    # retrain_period = 12 # number of candles before retrining
    retrain_period = 2 # for testing only!

    dataframeUtils = None
    scaler = RobustScaler()
    model = None
    base_model = None

    curr_dataframe: DataFrame = None

    target_profit = 0.0
    target_loss = 0.0

    # hyperparams
    # NOTE: this strategy does not hyperopt well, no idea why. Note that some vars are turned off (optimize=False)

    # the defaults are set for fairly frequent trades, and get out quickly
    # if you want bigger trades, then increase entry_model_gain, decrese exit_model_gain and adjust profit_threshold and
    # loss_threshold accordingly. 
    # Note that there is also a corellation to self.lookahead, but that cannot be a hyperopt parameter (because it is 
    # used in populate_indicators). Larger lookahead implies bigger differences between the model and actual price
    # entry_model_gain = DecimalParameter(0.5, 3.0, decimals=1, default=1.0, space='buy', load=True, optimize=True)
    # exit_model_gain = DecimalParameter(-5.0, 0.0, decimals=1, default=-1.0, space='sell', load=True, optimize=True)

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
 

    ###################################

    def bot_start(self, **kwargs) -> None:

        if self.dataframeUtils is None:
            self.dataframeUtils = DataframeUtils()
            self.dataframeUtils.set_scaler_type(ScalerType.Robust)

        return

    ###################################

    """
    Informative Pair Definitions
    """

    def informative_pairs(self):
        return []

    ###################################

    """
    Indicator Definitions
    """

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:


        # Base pair dataframe timeframe indicators
        curr_pair = metadata['pair']

        self.curr_dataframe = dataframe
        self.curr_pair = curr_pair

        # print("")
        # print(curr_pair)
        # print("")

        # if self.dataframeUtils is None:
        #     self.dataframeUtils = DataframeUtils()
        #     self.dataframeUtils.set_scaler_type(ScalerType.Robust)


        # backward looking gain
        dataframe['gain'] = 100.0 * (dataframe['close'] - dataframe['close'].shift(self.lookahead)) / dataframe['close']
        dataframe['gain'].fillna(0.0, inplace=True)
        dataframe['gain'] = self.smooth(dataframe['gain'], 8)
        dataframe['gain'] = self.detrend_array(dataframe['gain'])

        # target profit/loss thresholds        
        dataframe['profit'] = dataframe['gain'].clip(lower=0.0)
        dataframe['loss'] = dataframe['gain'].clip(upper=0.0)
        win_size = 32
        dataframe['target_profit'] = dataframe['profit'].rolling(window=win_size).mean() + \
            2.0 * dataframe['profit'].rolling(window=win_size).std()
        dataframe['target_loss'] = dataframe['loss'].rolling(window=win_size).mean() - \
            2.0 * abs(dataframe['loss'].rolling(window=win_size).std())

        # RSI
        dataframe['rsi'] = ta.RSI(dataframe, timeperiod=self.win_size)

        # Williams %R
        dataframe['wr'] = 0.02 * (self.williams_r(dataframe, period=self.win_size) + 50.0)

        # Fisher RSI
        rsi = 0.1 * (dataframe['rsi'] - 50)
        dataframe['fisher_rsi'] = (np.exp(2 * rsi) - 1) / (np.exp(2 * rsi) + 1)

        # Combined Fisher RSI and Williams %R
        dataframe['fisher_wr'] = (dataframe['wr'] + dataframe['fisher_rsi']) / 2.0

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

        # Exponential Moving Average
        dataframe['tema'] = ta.TEMA(dataframe, timeperiod=self.win_size)

        # init prediction column
        dataframe['model_gain'] = 0.0

        # create and init the model, if first time (dataframe has to be populated first)
        if self.model is None:
            # print("    Loading model")
            self.load_model(np.shape(dataframe))

        # add the predictions
        # print("    Making predictions...")
        dataframe = self.add_predictions(dataframe)

        dataframe.fillna(0.0, inplace=True)

        # #DBG (cannot include this in 'real' strat because it's forward looking):
        # dataframe['dwt'] = self.get_dwt(dataframe['gain'])

        return dataframe
   
    ###################################
    
    def smooth(self, y, window):
        box = np.ones(window)/window
        y_smooth = np.convolve(y, box, mode='same')
        y_smooth = np.round(y_smooth, decimals=3) #Hack: constrain to 3 decimal places (should be elsewhere, but convenient here)
        return np.nan_to_num(y_smooth)
    
    # look ahead to get future gain. Do *not* put this into the main dataframe!
    def get_future_gain(self, dataframe):
        # future_gain = 100.0 * (dataframe['close'].shift(-self.lookahead) - dataframe['close']) / dataframe['close']
        # future_gain.fillna(0.0, inplace=True)
        # future_gain = np.array(future_gain)
        # future_gain = self.smooth(future_gain, 8)
        # # print(f'future_gain:{future_gain}')
        # return self.detrend_array(future_gain)
        df = self.convert_dataframe(dataframe)
        future_gain = df['gain'].shift(-self.lookahead).to_numpy()
        return self.smooth(future_gain, 8)
    
    ###################################

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


    ###################################


    #-------------

    def convert_dataframe(self, dataframe: DataFrame) -> DataFrame:
        df = dataframe.copy()

        # convert date column so that it can be scaled.
        if 'date' in df.columns:
            dates = pd.to_datetime(df['date'], utc=True)
            df['date'] = dates.astype('int64')

        df.fillna(0.0, inplace=True)

        df.set_index('date')
        df.reindex()

        if self.norm_data:
            # scale the dataframe
            self.scaler.fit(df)
            df = pd.DataFrame(self.scaler.transform(df), columns=df.columns)

        return df


    ###################################

    # DWT functions
 
    def madev(self, d, axis=None):
        """ Mean absolute deviation of a signal """
        return np.mean(np.absolute(d - np.mean(d, axis)), axis)

    def detrend_array(self, a):
        '''
        if self.norm_data:
            # de-trend the data
            w_mean = a.mean()
            w_std = a.std()
            a_notrend = (a - w_mean) / w_std
            return a_notrend
        else:
            return a
        '''
        return a

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
    

    def get_dwt(self, col):

        a = np.array(col)

        # # de-trend the data
        # w_mean = a.mean()
        # w_std = a.std()
        # a_notrend = (a - w_mean) / w_std

        # # get DWT model of data
        # restored_sig = self.dwtModel(a_notrend)

        # # re-trend
        # dwt_model = (restored_sig * w_std) + w_mean

        dwt_model = self.dwtModel(a)

        return dwt_model


    ###################################

 
    # function to get  coefficients (override in subclass)
    def get_coeffs(self, data: np.array) -> np.array:


        num_items = 64

        # this is a very naiive implementation that just does a polynomial fit and extrapolation
        # replace this with more viable algorithms (DWT, SWT) in subclass
        items = data[-num_items:]
        x_values = np.arange(num_items)
        y_values = items
        coefficients = np.polyfit(x_values, y_values, 8)

        # extrapolate by self.lookahead steps
        f = np.poly1d(coefficients)
        x_new = np.arange(num_items+self.lookahead)
        y_new = f(x_new)

        # print(f'coefficients:{coefficients}, y_new:{y_new}')
        features = np.array((coefficients[0], coefficients[1], y_new[-1]))

        return features

    #-------------

    # builds a numpy array of coefficients
    def add_coefficients(self, dataframe) -> DataFrame:

        # normalise the dataframe
        df_norm = self.convert_dataframe(dataframe)

        # copy the gain data into an np.array (faster)
        gain_data = np.array(dataframe['gain']).copy()

        init_done = False
        
        # roll through the close data and create DWT coefficients for each step
        nrows = np.shape(gain_data)[0]

        start = 0
        end = start + self.model_window
        dest = end

        # print(f"nrows:{nrows} start:{start} end:{end} dest:{dest} nbuffs:{nbuffs}")

        self.coeff_array = None
        num_coeffs = 0

        while end < nrows:
            dslice = gain_data[start:end]

            # print(f"start:{start} end:{end} dest:{dest} len:{len(dslice)}")

            features = self.get_coeffs(dslice)
            
            # initialise the np.array (need features first to know size)
            if not init_done:
                init_done = True
                num_coeffs = len(features)
                self.coeff_array = np.zeros((nrows, num_coeffs), dtype=float)
                # print(f"coeff_array:{np.shape(self.coeff_array)}")

            # copy the features to the appropriate row of the coefficient array (offset due to startup window)
            self.coeff_array[dest] = features

            start = start + 1
            dest = dest + 1
            end = end + 1

        # # normalise the coefficients
        # self.scaler.fit(self.coeff_array)
        # self.coeff_array = self.scaler.transform(self.coeff_array)

        return dataframe
    
    #-------------


    # builds a numpy array of coefficients
    def build_coefficient_table(self, data: np.array):
        
        # roll through the  data and create coefficients for each step
        nrows = np.shape(data)[0]

        # print(f'build_coefficient_table() data:{np.shape(data)}')

        start = 0
        if nrows > self.model_window:
            end = start + self.model_window - 1
        else:
            end = start + 32
        dest = end

        # print(f"nrows:{nrows} start:{start} end:{end} dest:{dest} nbuffs:{nbuffs}")

        self.coeff_table = None
        num_coeffs = 0
        init_done = False

        while end < nrows:
            dslice = data[start:end]

            # print(f"start:{start} end:{end} dest:{dest} len:{len(dslice)}")

            features = self.get_coeffs(dslice)
            # print(f'build_coefficient_table() features: {np.shape(features)}')
            
            # initialise the np.array (need features first to know size)
            if not init_done:
                init_done = True
                num_coeffs = len(features)
                self.coeff_table = np.zeros((nrows, num_coeffs), dtype=float)
                # print(f"coeff_table:{np.shape(self.coeff_table)}")

            # copy the features to the appropriate row of the coefficient array (offset due to startup window)
            self.coeff_table[dest] = features

            start = start + 1
            dest = dest + 1
            end = end + 1


        # print(f'build_coefficient_table() self.coeff_table: {np.shape(self.coeff_table)}')

        return

    #-------------

    # merge the supplied dataframe with the coefficient table. Number of rows must match
    def merge_coeff_table(self, dataframe: DataFrame):

        # print(f'merge_coeff_table() self.coeff_table: {np.shape(self.coeff_table)}')

        num_coeffs = np.shape(self.coeff_table)[1]

        '''
        # set up the column names
        col_names = []
        for i in range(num_coeffs):
            col = "coeff_" + str(i)
            col_names.append(col)
        
        # print(f'self.coeff_table: {np.shape(self.coeff_table)} col_names:{np.shape(col_names)}')

        # convert the np.array into a dataframe
        df_coeff = pd.DataFrame(self.coeff_table, columns=col_names)

        # merge into the main dataframe
        dataframe = self.merge_data(dataframe, df_coeff)

        return dataframe
        '''

        merged_table = np.concatenate([np.array(dataframe), self.coeff_table], axis=1)

        return merged_table
    
    #-------------


    def merge_data(self, df1: DataFrame, df2: DataFrame) -> DataFrame:

        # merge df_coeffs into the main dataframe

        l1 = df1.shape[0]
        l2 = df2.shape[0]

        if l1 != l2:
            print(f"    **** size mismatch. len(df1)={l1} len(df2)={l2}")
        dataframe = pd.concat([df1, df2], axis=1, ignore_index=False).fillna(0.0)
        # print(f"    df1={df1.shape} df2={df2.shape} df={dataframe.shape}")

        # concat sometimes adds an extra row, so trim to original size
        dataframe = dataframe.iloc[:l1]

        # print(f'cols:{dataframe.columns}')

        return dataframe


    #-------------

    ###################################

    # Model-related funcs. Override in subclass to use a different type of model


    def get_model_path(self, pair):
        category = self.__class__.__name__
        root_dir = group_dir + "/models/" + category
        model_name = category
        if self.model_per_pair and (len(pair) > 0):
            model_name = model_name + "_" + pair.split("/")[0]
        path = root_dir + "/" + model_name + ".sav"
        return path
    
    def load_model(self, df_shape):

        model_path = self.get_model_path("")

        # load from file or create new model
        if os.path.exists(model_path):
            # use joblib to reload model state
            print("    loading from: ", model_path)
            self.model = joblib.load(model_path)
            self.model_trained = True
            self.new_model = False
            self.training_mode = False
        else:
            self.create_model(df_shape)
            self.model_trained = False
            self.new_model = True
            self.training_mode = True


        # if supporting incremental training, try to set the warm_start attribute
        if self.supports_incremental_training:
            # check that attribute exists
            if hasattr(self.model, 'warm_start'):
                self.model.warm_start = True

        if self.model is None:
            print("***    ERR: model was not created properly ***")

        return

    #-------------

    def save_model(self):

        # save trained model (but only if didn't already exist)

        model_path = self.get_model_path("")

        # create directory if it doesn't already exist
        save_dir = os.path.dirname(model_path)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        # use joblib to save model state
        print("    saving to: ", model_path)
        joblib.dump(self.model, model_path)

        return

    #-------------

    def create_model(self, df_shape):

        # print("    creating new model using: XGBRegressor")
        # params = {'n_estimators': 100, 'max_depth': 4, 'learning_rate': 0.1}
        # self.model = XGBRegressor(**params)


        # self.model = PassiveAggressiveRegressor(warm_start=True)
        self.model = SGDRegressor(loss='huber')

        print(f"    creating new model using: {type(self.model)}")

        if self.model is None:
            print("***    ERR: create_model() - model was not created ***")
        return

    #-------------

    # train the model. Override if not an sklearn-compatible algorithm
    # set save_model=False if you don't want to save the model (needed for ML algorithms)
    def train_model(self, model, data: np.array, results: np.array, save_model):

        if self.model is None:
            print("***    ERR: no model ***")
            return
        
        # # train on the supplied data (XGBoost-specific)
        # if self.new_model and (not self.model_trained):
        #     model.fit(data, results)
        # else:
        #     model.fit(data, results, xgb_model=self.model)

        x = np.nan_to_num(data)
        model = model.partial_fit(x, results)

        return


    #-------------

    # single prediction (for use in rolling calculation)
    def predict(self, df) -> float:

        data = np.array(self.convert_dataframe(df))

        # y_pred = self.model.predict(data)[-1]
        y_pred = self.predict_data(self.model, data)[-1]

        return y_pred


    #-------------
    
    # initial training of the model
    def init_model(self, dataframe: DataFrame):
        # if not self.training_mode:
        #     # limit training to what we would see in a live environment, otherwise results are too good
        #     live_buffer_size = 974
        #     if dataframe.shape[0] > live_buffer_size:
        #         df = dataframe.iloc[-live_buffer_size:]
        #     else:
        #         df = dataframe
        # else:
        #     df = dataframe

        # if model is not yet trained, or this is a new model and we want to combine across pairs, then train
        if (not self.model_trained) or (self.new_model and self.combine_models):

            df = dataframe

            future_gain_data = self.get_future_gain(df)

            df_norm = self.convert_dataframe(dataframe)
            gain_data = df_norm['gain'].to_numpy()
            self.build_coefficient_table(gain_data)
            data = self.merge_coeff_table(self.convert_dataframe(df_norm))

            training_data = data[:-self.lookahead-1].copy()
            training_labels = future_gain_data[:-self.lookahead-1].copy()

            if not self.model_trained:
                print(f'    initial training ({self.curr_pair})')
            else:
                print(f'    incremental training ({self.curr_pair})')

            self.train_model(self.model, training_data, training_labels, True)

            self.model_trained = True

            if self.new_model:
                self.save_model()
        
        # print(f'    model_trained:{self.model_trained} new_model:{self.new_model}  combine_models:{self.combine_models}')

        return
    
    #-------------
    
    # generate predictions for an np array (intended to be overriden if needed)
    def predict_data(self, model, data):
        x = np.nan_to_num(data)
        preds = model.predict(x)
        preds = np.clip(preds, -3.0, 3.0)
        return preds


    #-------------
    
    # add predictions in a jumping fashion. This is a compromise - the rolling version is very slow
    # Note: you probably need to manually tune the parameters, since there is some limited lookahead here
    def add_jumping_predictions(self, dataframe: DataFrame) -> DataFrame:

        # limit training to what we would see in a live environment, otherwise results are too good
        live_buffer_size = 974
        df = dataframe

        # roll through the close data and predict for each step
        nrows = np.shape(df)[0]


        # build the coefficient table and merge into the dataframe (OK outside the main loop since it's built incrementally anyway)


        # set up training data
        future_gain_data = self.get_future_gain(df)
        df_norm = self.convert_dataframe(dataframe)
        self.build_coefficient_table(df_norm['gain'].to_numpy()) 

        self.training_data = self.merge_coeff_table(df_norm)
        self.training_labels = np.zeros(np.shape(future_gain_data), dtype=float)
        # self.training_labels[:-self.lookahead] = gain_data[self.lookahead:].copy()
        self.training_labels = future_gain_data.copy()

        # initialise the prediction array, using the close data
        pred_array = np.zeros(np.shape(future_gain_data), dtype=float)

        # win_size = 974
        win_size = 128

        # loop until we get to/past the end of the buffer
        # start = 0
        # end = start + win_size
        start = win_size
        end = start + win_size
        train_start = 0

        base_model = copy.deepcopy(self.model) # make a deep copy so that we don't override the baseline model

        while end < nrows:

            # extract the data and coefficients from the current window
            start = end - win_size

            # get the training data. Use data prior to current prediction data, limited to live buffer size (so backtest resembles live modes)


            # if start > live_buffer_size:
            #     train_start = start - live_buffer_size
            # print(f'train_start: {train_start} start:{start}')

            # print(f"start:{start} end:{end} train_start:{train_start}")
            # print(f'train_data: {train_data}')
            # print(f'train_results: {train_results}')

            # (re-)train the model on prior data and get predictions

            if (not self.training_mode) and (self.supports_incremental_training):
                train_data = self.training_data[train_start:start-1].copy()
                train_results = self.training_labels[train_start:start-1].copy()
                base_model = copy.deepcopy(self.model)
                self.train_model(base_model, train_data, train_results, False)


            # rebuild data up to end of current window
            dslice = self.training_data[start:end].copy()
            preds = self.predict_data(base_model, dslice)

            # copy the predictions for this window into the main predictions array
            pred_array[start:end] = preds.copy()

            # move the window to the next segment
            end = end + win_size - 1

        # make sure the last section gets processed (the loop above may not exactly fit the data)
        # Note that we cannot use the last section for training because we don't have forward looking data

        '''
        if (not self.training_mode) and (self.supports_incremental_training):
            dslice = self.training_data[-(win_size+self.lookahead):-self.lookahead]
            cslice = self.training_labels[-(win_size+self.lookahead):-self.lookahead]
            self.train_model(base_model, dslice, cslice, False)
        '''

        # predict for last window
        dslice = self.training_data[-win_size:]
        # preds = self.model.predict(dslice)
        preds = self.predict_data(base_model, dslice)
        pred_array[-win_size:] = preds.copy()

        dataframe['predicted_gain'] = pred_array.copy()

        # add gain to dataframe for display purposes
        dataframe['future_gain'] = future_gain_data.copy()

        # # update pair-specific model
        # self.custom_trade_info[self.curr_pair]['model'] = model

        # # restore model
        # self.model = copy.deepcopy(base_model)

        return dataframe

    #-------------
    
    # add the latest prediction, and update training periodically
    def add_latest_prediction(self, dataframe: DataFrame) -> DataFrame:

        df = dataframe

        # set up training data
        #TODO: see if we can do this incrementally instead of rebuilding every time, or just use portion of data
        future_gain_data = self.get_future_gain(df)
        df_norm = self.convert_dataframe(dataframe)
        self.build_coefficient_table(df_norm['gain'].to_numpy()) 

        data = self.merge_coeff_table(df_norm)

        plen = len(self.custom_trade_info[self.curr_pair]['predictions'])

        self.training_data = data[-plen:].copy()
        self.training_labels = future_gain_data[-plen:].copy()

        pred_array = np.zeros(plen, dtype=float)

        # print(f"[predictions]:{np.shape(self.custom_trade_info[self.curr_pair]['predictions'])}  pred_array:{np.shape(pred_array)}")

        # copy previous predictions and shift down by 1
        pred_array = self.custom_trade_info[self.curr_pair]['predictions'].copy()
        pred_array = np.roll(pred_array, -1)
        pred_array[-1] = 0.0

        # cannot use last portion because we are looking ahead
        dslice = self.training_data[:-self.lookahead]
        tslice = self.training_labels[:-self.lookahead]

        # retrain base model and get predictions
        base_model = copy.deepcopy(self.model)
        self.train_model(base_model, dslice, tslice, False)
        preds = self.predict_data(base_model, self.training_data)

        # self.model = copy.deepcopy(base_model) # restore original model

        # only replace last prediction (i.e. don't overwrite the historical predictions)
        pred_array[-1] = preds[-1]

        dataframe['predicted_gain'] = 0.0
        dataframe['predicted_gain'][-plen:] = pred_array.copy()
        self.custom_trade_info[self.curr_pair]['predictions'] = pred_array.copy()

        # add gain to dataframe for display purposes
        dataframe['future_gain'] = future_gain_data.copy()

        print(f'    {self.curr_pair}:   predict {preds[-1]:.2f}% gain')

        return dataframe

    
    #-------------
    
    # add predictions to dataframe['predicted_gain']
    def add_predictions(self, dataframe: DataFrame) -> DataFrame:

        # print(f"    {self.curr_pair} adding predictions")

        run_profiler = False

        if run_profiler:
            prof = cProfile.Profile()
            prof.enable()

        self.scaler = RobustScaler() # reset scaler each time

        self.init_model(dataframe)

        if self.curr_pair not in self.custom_trade_info:
            self.custom_trade_info[self.curr_pair] = {
                # 'model': self.model,
                'initialised': False,
                'predictions': None
            }


        if self.training_mode:
            print(f'    Training mode. Skipping backtest for {self.curr_pair}')
            dataframe['predicted_gain'] = 0.0
        else:
            if not self.custom_trade_info[self.curr_pair]['initialised']:
                print(f'    backtesting {self.curr_pair}')
                dataframe = self.add_jumping_predictions(dataframe)
                # dataframe = self.add_rolling_predictions(dataframe)
                self.custom_trade_info[self.curr_pair]['initialised'] = True

                # init target values to hyperopt values. Will be dynamic after this point
                # dataframe['target_profit'] = float(self.entry_model_gain.value)
                # # dataframe['target_loss'] = float(self.exit_model_gain.value)
                # self.target_profit = self.entry_model_gain.value
                # self.target_loss = self.exit_model_gain.value
            else:
                # print(f'    updating latest prediction for: {self.curr_pair}')
                dataframe = self.add_latest_prediction(dataframe)

            # save the predictions and targets
            self.custom_trade_info[self.curr_pair]['predictions'] = dataframe['predicted_gain'].to_numpy()
            # self.custom_trade_info[self.curr_pair]['target_profit'] = dataframe['target_profit'].to_numpy()
            # self.custom_trade_info[self.curr_pair]['target_loss'] = dataframe['target_loss'].to_numpy()


        if run_profiler:
            prof.disable()
            # print profiling output
            stats = pstats.Stats(prof).strip_dirs().sort_stats("cumtime")
            stats.print_stats(20) # top 20 rows

        return dataframe
    
    ###################################

    """
    entry Signal
    """

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        conditions = []
        dataframe.loc[:, 'enter_tag'] = ''
       
        if self.training_mode:
            dataframe['enter_long'] = 0
            return dataframe
        
        if self.enable_entry_guards.value:
            # Fisher/Williams in oversold region
            conditions.append(dataframe['fisher_wr'] < self.entry_guard_fwr.value)

            # some trading volume
            conditions.append(dataframe['volume'] > 0)


        fwr_cond = (
            (dataframe['fisher_wr'] < -0.98)
        )


        # model triggers
        model_cond = (
            (
                # model predicts a rise above the entry threshold
                qtpylib.crossed_above(dataframe['predicted_gain'], dataframe['target_profit']) #&
                # (dataframe['predicted_gain'] >= dataframe['target_profit']) &
                # (dataframe['predicted_gain'].shift() >= dataframe['target_profit'].shift()) &

                # Fisher/Williams in oversold region
                # (dataframe['fisher_wr'] < -0.5)
            )
            # |
            # (
            #     # large gain predicted (ignore fisher_wr)
            #     qtpylib.crossed_above(dataframe['predicted_gain'], 2.0 * dataframe['target_profit']) 
            # )
        )
        
        

        # conditions.append(fwr_cond)
        conditions.append(model_cond)


        # set entry tags
        dataframe.loc[fwr_cond, 'enter_tag'] += 'fwr_entry '
        dataframe.loc[model_cond, 'enter_tag'] += 'model_entry '

        if conditions:
            dataframe.loc[reduce(lambda x, y: x & y, conditions), 'enter_long'] = 1

        return dataframe


    ###################################

    """
    exit Signal
    """

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        conditions = []
        dataframe.loc[:, 'exit_tag'] = ''

        if self.training_mode or (not self.enable_exit_signal.value):
            dataframe['exit_long'] = 0
            return dataframe

        # if self.enable_entry_guards.value:

        if self.enable_exit_guards.value:
            # Fisher/Williams in overbought region
            conditions.append(dataframe['fisher_wr'] > self.exit_guard_fwr.value)

            # some trading volume
            conditions.append(dataframe['volume'] > 0)

        # model triggers
        model_cond = (
            (

                 qtpylib.crossed_below(dataframe['predicted_gain'], dataframe['target_loss'] )
           )
        )

        conditions.append(model_cond)


        # set exit tags
        dataframe.loc[model_cond, 'exit_tag'] += 'model_exit '

        if conditions:
            dataframe.loc[reduce(lambda x, y: x & y, conditions), 'exit_long'] = 1

        return dataframe



    def confirm_trade_exit(self, pair: str, trade: Trade, order_type: str, amount: float,
                           rate: float, time_in_force: str, exit_reason: str,
                           current_time: datetime, **kwargs) -> bool:
                
        if self.dp.runmode.value not in ('backtest', 'plot', 'hyperopt'):
            print(f'Trade Exit: {pair}, rate: {round(rate, 4)}')

        return True
    
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
        
        
        # big drop predicted. Should also trigger an exit signal, but this might be quicker (and will likely be 'market' sell)
        if (current_profit > 0) and (last_candle['predicted_gain'] <= last_candle['target_loss']):
            return 'predict_drop'
        

        # if in profit and exit signal is set, sell (even if exit signals are disabled)
        if (current_profit > 0) and (last_candle['exit_long'] > 0):
            return 'exit_signal'

        return None