# pragma pylint: disable=W0105, C0103, C0114, C0115, C0116, C0301, C0302, C0303, C0411, C0413,  W1203

"""
####################################################################################
TS_Wavelet - predicts each component/coefficient of a wavelet and reconstructs the signal to produce a prediction
            NOTE: this is *very* compute intensive

####################################################################################
"""


from datetime import datetime
from functools import reduce

import cProfile
import pstats
import traceback

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
from typing import Any, List, Optional

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

import logging
import warnings

log = logging.getLogger(__name__)
log.setLevel(logging.DEBUG)
warnings.simplefilter(action='ignore', category=pd.errors.PerformanceWarning)
warnings.simplefilter(action='ignore', category=FutureWarning)

from xgboost import XGBRegressor
# from lightgbm import LGBMRegressor
from sklearn.linear_model import PassiveAggressiveRegressor
from sklearn.linear_model import SGDRegressor

from utils.DataframeUtils import DataframeUtils, ScalerType # pylint: disable=E0401
# import pywt
import talib.abstract as ta


from TSPredict import TSPredict


class TS_Wavelet(TSPredict):
   

    custom_trade_info = {} # pair-specific data
    curr_pair = ""

    ###################################

    # Strategy Specific Variable Storage


    # model_window = startup_candle_count
    model_window = 128

    # lookahead = 6
    lookahead = 9

    df_coeffs: DataFrame = None
    coeff_table = None
    coeff_array = None
    coeff_start_col = 0
    gain_data = None

    single_col_prediction = False

    norm_data = False
 
   
  
    ###################################

    def add_strategy_indicators(self, dataframe):

        # add some extra indicators

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


        # moving averages
        dataframe['sma'] = ta.SMA(dataframe, timeperiod=14)
        dataframe['ema'] = ta.EMA(dataframe, timeperiod=14)
        dataframe['tema'] = ta.TEMA(dataframe, timeperiod=14)

        # Donchian Channels
        dataframe['dc_upper'] = ta.MAX(dataframe['high'], timeperiod=self.win_size)
        dataframe['dc_lower'] = ta.MIN(dataframe['low'], timeperiod=self.win_size)
        dataframe['dc_mid'] = ta.TEMA(((dataframe['dc_upper'] + dataframe['dc_lower']) / 2), timeperiod=self.win_size)

        return dataframe

    ###################################

    # DWT functions
 
    def madev(self, d, axis=None):
        """ Mean absolute deviation of a signal """
        return np.mean(np.absolute(d - np.mean(d, axis)), axis)


    ###################################

    # Coefficient functions
 
    wavelet = "db2"
    mode = "smooth"
    coeff_slices = None
    coeff_shapes = None
    coeff_format = "wavedec"

    # function to get dwt coefficients
    def get_coeffs(self, data: np.array) -> np.array:

        length = len(data)

        x = data

        # data must be of even length, so trim if necessary
        if (len(x) % 2) != 0:
            x = x[1:]

        # print(pywt.wavelist(kind='discrete'))

        # get the DWT coefficients
        self.wavelet = 'bior3.9'
        # self.wavelet = 'db2'
        # self.mode = 'smooth'
        self.mode = 'zero'
        self.coeff_format = "wavedec"
        level = 2
        coeffs = pywt.wavedec(x, self.wavelet, mode=self.mode, level=level)

        ''''''
        # remove higher harmonics
        std = np.std(coeffs[level])
        sigma = (1 / 0.6745) * self.madev(coeffs[-level])
        # sigma = madev(coeff[-level])
        uthresh = sigma * np.sqrt(2 * np.log(length))

        coeffs[1:] = (pywt.threshold(i, value=uthresh, mode='hard') for i in coeffs[1:])

        ''''''

        return self.coeff_to_array(coeffs)

    #-------------


    def coeff_to_array(self, coeffs):
        # flatten the coefficient arrays

        # faster:
        # arr, self.coeff_slices, self.coeff_shapes = pywt.ravel_coeffs(coeffs)

        # more general purpose (can use with many waveforms)
        arr, self.coeff_slices = pywt.coeffs_to_array(coeffs)

        return np.array(arr)

    #-------------

    def array_to_coeff(self, array):

        # faster:
        # coeffs = pywt.unravel_coeffs(array, self.coeff_slices, self.coeff_shapes, output_format='wavedec')

        # more general purpose (can use with many waveforms)
        coeffs = pywt.array_to_coeffs(array, self.coeff_slices, output_format=self.coeff_format)

        # print(f'    coeff_slices:{self.coeff_slices}, coeff_shapes:{self.coeff_shapes} array:{np.shape(array)}')
        return coeffs

    #-------------

    def get_value(self, coeffs):
        # series = pywt.waverec(coeffs, self.wavelet)

        series = pywt.waverec(coeffs, wavelet=self.wavelet, mode=self.mode)
        # print(f'    coeff_slices:{self.coeff_slices}, coeff_shapes:{self.coeff_shapes} series:{np.shape(series)}')

        return series

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
            dslice = data[start:end].copy()

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

        # merged_table = np.concatenate([np.array(dataframe), self.coeff_table], axis=1)
        merged_table = self.coeff_table

        # save the start column for later use
        # self.coeff_start_col = np.shape(dataframe)[1]
        self.coeff_start_col = 0

        return merged_table

    #-------------
    # Normalisation

    scaler = RobustScaler()

    def update_scaler(self, data):

        if not self.scaler:
            self.scaler = RobustScaler()

        self.scaler.fit(data.reshape(-1,1))

    def norm_array(self, a):
            return self.scaler.transform(a.reshape(-1, 1))

    def denorm_array(self, a):
            return self.scaler.inverse_transform(a.reshape(-1, 1)).squeeze()

    #-------------

    # override func to get data for this strategy
    def get_data(self, dataframe):

        df_norm = self.convert_dataframe(dataframe)
        gain_data = df_norm['gain'].to_numpy()
        self.build_coefficient_table(gain_data)
        data = self.merge_coeff_table(df_norm)
        return data
    
    #####################################

    # Model-related functions - note that this family of strategies does not save/reload models



    # regression/prediction model. For this family, it must be *fast*
    def create_model(self, df_shape):

        # # XGBoost is the best regression algorithm, but it runs too slowly for this strategy
        # params = {'n_estimators': 100, 'max_depth': 4, 'learning_rate': 0.1}
        # self.model = XGBRegressor(**params)

        self.model = PassiveAggressiveRegressor(warm_start=False)

        print(f"    creating new model using: {type(self.model)}")

        if self.model is None:
            print("***    ERR: create_model() - model was not created ***")
        return

    # -------------

    # the following are here just to maintain compatibilkity with TSPredict

    def init_model(self, dataframe: DataFrame):
        pass

    def train_model(self, model, data: np.array, results: np.array, save_model):
        pass

    def save_model(self):
        pass

    def load_model(self, df_shape):
        self.model_trained = True
        self.new_model = False
        self.training_mode = False
        return

    #####################################

    # Prediction-related functions

    #-------------

    # generate predictions for an np array 
    # data should be the entire data array, not a slice
    # since we both train and predict, supply indices to allow trining and predicting in different regions
    def predict_data(self, data, train_start, train_end, predict_start, predict_end):

        # a little different than other strats, since we train a model for each column
        x = np.nan_to_num(data)

        ncols = np.shape(data)[1]

        coeff_arr = []

        if not self.single_col_prediction:
            # all columns - better, but much slower
            training_data = data[train_start:train_end - self.lookahead].copy()
            predict_data = data[predict_start:predict_end].copy()

        # create the model
        if self.model is None:
            self.create_model(np.shape(data))

        # print(f'    train_start:{train_start}, train_end:{train_end}, predict_start:{predict_start}, predict_end:{predict_end}')

        # train/predict for each coefficient individually
        for i in range(self.coeff_start_col, ncols):

            if self.single_col_prediction:
                training_data = data[train_start:train_end, i][:-self.lookahead].reshape(-1,1) # 1 column - less accurate, but much faster
            training_labels = data[train_start:train_end, i][self.lookahead:].copy()
            training_labels = np.nan_to_num(training_labels)

            # print(f'    data:{np.shape(training_data)} labels:{np.shape(training_labels)}')


            # fit the model
            self.model.fit(training_data, training_labels)

            # get a prediction
            if self.single_col_prediction:
                predict_data = data[predict_start:predict_end, i].reshape(-1,1) # 1 column - less accurate, but much faster

            pred = self.model.predict(predict_data)[-1]
            coeff_arr.append(pred)

        # convert back to wavelet coefficients (different for each type of wavelet)
        wcoeffs = self.array_to_coeff(np.array(coeff_arr))

        # convert back to gain
        preds = self.get_value(wcoeffs)
        # print(f'    preds[-1]:{preds[-1]}')

        if self.norm_data:
            preds = self.denorm_array(preds)

        return preds


    #-------------
    
    # add predictions in a jumping fashion. This is a compromise - the rolling version is very slow
    # Note: you probably need to manually tune the parameters, since there is some limited lookahead here
    def add_jumping_predictions(self, dataframe: DataFrame) -> DataFrame:

        df = dataframe

        # roll through the close data and predict for each step
        nrows = np.shape(df)[0]


        # build the coefficient table and merge into the dataframe (OK outside the main loop since it's built incrementally anyway)


        # set up training data
        future_gain_data = self.get_future_gain(df)
        data = self.get_data(dataframe)

        # initialise the prediction array, using the close data
        pred_array = np.zeros(np.shape(future_gain_data), dtype=float)

        # NOTE: win_size must be 126 because waverec returns a fixed size signal (you guessed it, length=126)
        win_size = 126

        # loop until we get to/past the end of the buffer
        # start = 0
        # end = start + win_size
        start = win_size
        end = start + win_size
        train_end = start - 1
        train_size = 2 * win_size
        train_start = max(0, train_end-train_size)

        # base_model = copy.deepcopy(self.model) # make a deep copy so that we don't override the baseline model
        # base_model = self.model

        while end < nrows:

             # set the (unmodified) gain data for scaling
            # self.gain_data = np.array(dataframe['gain'].iloc[start:end])
            self.update_scaler(np.array(dataframe['gain'].iloc[start:end]))

            # rebuild data up to end of current window
            preds = self.predict_data(data, train_start, train_end, start, end)

            # print(f'    preds:{np.shape(preds)}')
            # copy the predictions for this window into the main predictions array
            pred_array[start:end] = preds[-win_size:].copy()

            # move the window to the next segment
            # end = end + win_size - 1
            end = end + win_size
            start = start + win_size
            train_end = start - 1
            train_start = max(0, train_end - train_size)


        # predict for last window

        # self.gain_data = np.array(dataframe['gain'].iloc[-win_size:])
        self.update_scaler(np.array(dataframe['gain'].iloc[-win_size:]))

        preds = self.predict_data(data, -(train_size + win_size + 1), -(win_size + 1), -win_size, -1)
        # preds = self.predict_data(data, 0, -(win_size + 1), -win_size, -1)
        plen = len(preds)
        pred_array[-plen:] = preds.copy()

        palen = len(pred_array)
        dataframe['predicted_gain'][-palen:] = pred_array.copy()

        return dataframe

    #-------------
    
    # add the latest prediction, and update training periodically
    def add_latest_prediction(self, dataframe: DataFrame) -> DataFrame:

        df = dataframe
        win_size = 64
        nrows = np.shape(df)[0]
        train_size = 256

        try:
            # set up training data
            #TODO: see if we can do this incrementally instead of rebuilding every time, or just use portion of data
            future_gain_data = self.get_future_gain(df)
            data = self.get_data(dataframe)

            plen = len(self.custom_trade_info[self.curr_pair]['predictions'])
            dlen = len(dataframe['gain'])
            clen = min(plen, dlen)

            training_data = data[-clen:].copy()
            training_labels = future_gain_data[-clen:].copy()

            pred_array = np.zeros(clen, dtype=float)

            # print(f"[predictions]:{np.shape(self.custom_trade_info[self.curr_pair]['predictions'])}  pred_array:{np.shape(pred_array)}")

            # copy previous predictions and shift down by 1
            pred_array[-clen:] = self.custom_trade_info[self.curr_pair]['predictions'][-clen:].copy()
            pred_array = np.roll(pred_array, -1)
            pred_array[-1] = 0.0

            # # save original model
            # base_model = copy.deepcopy(self.model)

            # get predictions
            train_end = nrows - (win_size + 1)
            # train_start = max(0, train_end - train_size)
            train_start = 0
            # self.gain_data = np.array(dataframe['gain'].iloc[-win_size:])
            self.update_scaler(np.array(dataframe['gain'].iloc[-win_size:]))

            preds = self.predict_data(data, 
                                      train_start, train_end, 
                                      -win_size, -1)

            # self.model = base_model # restore original model

            # only replace last prediction (i.e. don't overwrite the historical predictions)
            pred_array[-1] = preds[-1].copy()

            dataframe['predicted_gain'] = 0.0
            dataframe['predicted_gain'][-clen:] = pred_array[-clen:].copy()
            self.custom_trade_info[self.curr_pair]['predictions'][-clen:] = pred_array[-clen:].copy()

            '''
            pg = preds[-1]
            if pg <= dataframe['target_loss'].iloc[-1]:
                tag = "(*)"
            elif pg >= dataframe['target_profit'].iloc[-1]:
                tag = " * "
            else:
                tag = "   "
            print(f'    {tag} predict {pg:6.2f}% gain for: {self.curr_pair}')

            '''

        except Exception as e:
            print("*** Exception in add_latest_prediction()")
            print(e) # prints the error message
            print(traceback.format_exc()) # prints the full traceback

        return dataframe

    
    #-------------
    
    