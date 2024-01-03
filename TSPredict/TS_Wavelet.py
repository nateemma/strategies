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
from sklearn.svm import SVR

from utils.DataframeUtils import DataframeUtils, ScalerType # pylint: disable=E0401
# import pywt
import talib.abstract as ta
import utils.custom_indicators as cta

import utils.Wavelets as Wavelets
import utils.Forecasters as Forecasters

from TSPredict import TSPredict


class TS_Wavelet(TSPredict):
   

    custom_trade_info = {} # pair-specific data
    curr_pair = ""

    ###################################

    # Strategy Specific Variable Storage

    lookahead = 6
    # lookahead = 9

    # model_window = startup_candle_count
    model_window = 32 # longer = slower
    train_len = 32 # longer = slower
    scale_len = 16 # no. recent candles to use when scaling

    use_rolling = True # True = rolling (slow but realistic), False = Jumping (much faster, less realistic)
    single_col_prediction = False # True = use only gain. False = use all columns (better, but much slower)



    df_coeffs: DataFrame = None
    coeff_table = None
    coeff_table_offset = 0
    coeff_array = None
    coeff_start_col = 0
    coeff_num_cols = 0
    gain_data = None


    norm_data = True

    wavelet_type:Wavelets.WaveletType = Wavelets.WaveletType.MODWT
    wavelet = None

    forecaster_type:Forecasters.ForecasterType = Forecasters.ForecasterType.PA
    forecaster = None

    data = None

    min_wavelet_size = 16 # needed for consistently-sized transforms
    win_size = model_window # this can vary

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

        # # reset expanding window size
        # self.win_size = self.min_wavelet_size

        return dataframe

    ###################################

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

    
    #####################################

    # Model-related functions - note that this family of strategies does not save/reload models

    # the following are here just to maintain compatibilkity with TSPredict

    def create_model(self, df_shape):
        pass

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

    # override func to get data for this strategy
    def set_data(self, dataframe):

        df_norm = self.convert_dataframe(dataframe)
        self.gain_data = df_norm['gain'].to_numpy()

        # data for building coeff_table
        self.data = self.gain_data.copy() # copy becaise gain_data changes

        # copy of dataframe
        self.curr_dataframe = dataframe

        return


    # builds a numpy array of coefficients
    def build_coefficient_table(self, start, end):

        # print(f'start:{start} end:{end} self.win_size:{self.win_size}')


        # lazy initialisation of vars (so thatthey can be changed in subclasses)
        if self.wavelet is None:
           self.wavelet = Wavelets.make_wavelet(self.wavelet_type)

        self.wavelet.set_lookahead(self.lookahead)

        if self.forecaster is None:
            self.forecaster = Forecasters.make_forecaster(self.forecaster_type)

        # if forecaster does not require pre-training, then just set training length to 0
        if not self.forecaster.requires_pretraining():
            self.train_len = 0

        if self.wavelet is None:
            print('    **** ERR: wavelet not specified')
            return

        # print(f'    Wavelet:{self.wavelet_type.name} Forecaster:{self.forecaster_type.name}')

        # double check forecaster/multicolumn combo
        if (not self.single_col_prediction) and (not self.forecaster.supports_multiple_columns()):
            print('    **** ERR: forecaster does not support multiple columns')
            print('              Reverting to single column predictionss')
            self.single_col_prediction = True

        self.coeff_table = None
        self.coeff_table_offset = start
        num_coeffs = 0
        init_done = False

        features = None
        nrows = end - start
        row_start = max(self.min_wavelet_size, start) # don't run until we have enough data for the transform

        for row in range(row_start, end):
            # dslice = data[start:end].copy()
            win_start = max(0, row-self.min_wavelet_size)
            dslice = self.data[win_start:row]
            # print(f'data[{win_start}:{row}]: {dslice}')

            coeffs = self.wavelet.get_coeffs(dslice)
            features = self.wavelet.coeff_to_array(coeffs)
            # print(f'features: {np.shape(features)}')

            # initialise the np.array (need features first to know size)
            if not init_done:
                init_done = True
                num_coeffs = len(features)
                self.coeff_table = np.zeros((nrows, num_coeffs), dtype=float)
                # print(f"coeff_table:{np.shape(self.coeff_table)}")

            # copy the features to the appropriate row of the coefficient array 
            # print(f'row: {row} start:{start}')
            self.coeff_table[row-start] = features

        # merge data from main dataframe
        self.merge_coeff_table(start, end)

        return
    #-------------

    # merge the supplied dataframe with the coefficient table. Number of rows must match
    def merge_coeff_table(self, start, end):

        # print(f'merge_coeff_table() self.coeff_table: {np.shape(self.coeff_table)}')

        self.coeff_num_cols = np.shape(self.coeff_table)[1]

        # if using single column prediction, no need to merge in dataframe column because they won't be used
        if self.single_col_prediction or (not self.merge_indicators):
            merged_table = self.coeff_table
            self.coeff_start_col = 0
        else:
            self.coeff_start_col = np.shape(self.curr_dataframe)[1]
            df = self.curr_dataframe.iloc[start:end]
            df_norm = self.convert_dataframe(df)
            merged_table = np.concatenate([np.array(df_norm), self.coeff_table], axis=1)

        self.coeff_table = merged_table

        return

    #-------------

    # generate predictions for an np array 
    def predict_data(self, predict_start, predict_end):

        # a little different than other strats, since we train a model for each column

        # check that we have enough data to run a prediction, if not return zeros
        if predict_start < (self.train_len + self.lookahead):
            return np.zeros(predict_end-predict_start, dtype=float)

        ncols = np.shape(self.coeff_table)[1]
        coeff_arr: np.array = []

        # train on previous data (*not* current data!)
        train_end = predict_start - 1
        train_start = max(0, train_end-self.train_len)
        results_start = train_start + self.lookahead
        results_end = train_end + self.lookahead

        # coefficient table may only be partial, so adjust start/end positions
        start = predict_start - self.coeff_table_offset
        end = predict_end - self.coeff_table_offset

        # print(f'self.coeff_table_offset:{self.coeff_table_offset} start:{start} end:{end}')

        # get the data buffers from self.coeff_table
        if not self.single_col_prediction: # single_column version done inside loop
            predict_data = self.coeff_table[start:end]
            # predict_data = np.nan_to_num(predict_data)
            train_data = self.coeff_table[train_start:train_end]
            results = self.coeff_table[results_start:results_end]


        # print(f'start:{start} end:{end} train_start:{train_start} train_end:{train_end} train_len:{self.train_len}')

        # train/predict for each coefficient individually
        for i in range(self.coeff_start_col, ncols):

            # get the data buffers from self.coeff_table
            # if single column, then just use a single coefficient
            if self.single_col_prediction:
                predict_data = self.coeff_table[start:end, i].reshape(-1,1)
                # predict_data = np.nan_to_num(predict_data)
                train_data = self.coeff_table[train_start:train_end, i].reshape(-1,1)

            results = self.coeff_table[results_start:results_end, i]

            # print(f'predict_data: {predict_data}')
            # print(f'train_data: {train_data}')
            # print(f'results: {results}')

            # train the forecaster
            self.forecaster.train(train_data, results)

            # get a prediction
            preds = self.forecaster.forecast(predict_data, self.lookahead).squeeze()
            coeff_arr.append(preds[-1])

        # convert back to gain
        coeffs = self.wavelet.array_to_coeff(np.array(coeff_arr))
        preds = self.wavelet.get_values(coeffs)

        # rescale if necessary
        if self.norm_data:
            preds = self.denorm_array(preds)


        # print(f'preds[{start}:{end}]: {preds}')

        return preds

    # -------------

    # single prediction (for use in rolling calculation)
    def predict(self, gain, df) -> float:
       # Get the start and end index labels of the series
        start = gain.index[0]
        end = gain.index[-1]

        # Get the integer positions of the labels in the dataframe index
        start_row = df.index.get_loc(start)
        end_row = df.index.get_loc(end) + 1 # need to add the 1, don't know why!


        if end_row < (self.train_len + self.min_wavelet_size + self.lookahead):
        # if start_row < (self.min_wavelet_size + self.lookahead): # need buffer for training
            # print(f'    ({start_row}:{end_row}) y_pred[-1]:0.0')
            return 0.0

        # print(f'gain.index:{gain.index} start:{start} end:{end} start_row:{start_row} end_row:{end_row}')

        scale_start = max(0, len(gain)-16)

        # print(f'    coeff_table: {np.shape(self.coeff_table)} start_row: {start_row} end_row: {end_row} ')

        self.update_scaler(np.array(gain)[scale_start:])

        y_pred = self.predict_data(start_row, end_row)
        # print(f'    ({start_row}:{end_row}) y_pred[-1]:{y_pred[-1]}')
        return y_pred[-1]

    #-------------
    def add_rolling_predictions(self, dataframe: DataFrame) -> DataFrame:
        # debug:
        try:
            # convert dataframe into wavelet data for use in rolling function
            self.set_data(dataframe)

            # build the coefficient table for entire range
            self.build_coefficient_table(0, np.shape(dataframe)[0])

            # # build the coefficient table for the entire range
            # self.build_coefficient_table(0, np.shape(dataframe)[0])
            # print(f'    self.wavelet_data:{np.shape(self.wavelet_data)}')

            dataframe['predicted_gain'] = dataframe['gain'].rolling(window=self.model_window).apply(self.predict, args=(dataframe,))

            # slightly smooth, to get rid of spikes
            dataframe['predicted_gain'] = self.smooth(dataframe['predicted_gain'], 2)

        except Exception as e:
            print("*** Exception in add_rolling_predictions()")
            print(e) # prints the error message
            print(traceback.format_exc()) # prints the full traceback

        return dataframe


    # add predictions in a jumping fashion. This is a compromise - the rolling version is very slow
    # Note: make sure the rolling version works properly, then it should be OK to use the 'jumping' version
    def add_jumping_predictions(self, dataframe: DataFrame) -> DataFrame:

        # TEMP: do rolling calculation instead:


        if self.use_rolling:
            dataframe = self.add_rolling_predictions(dataframe)

        else:
            try:
                df = dataframe

                # roll through the close data and predict for each step
                nrows = np.shape(df)[0]


                # build the coefficient table and merge into the dataframe (OK outside the main loop since it's built incrementally anyway)


                # set up training data
                future_gain_data = self.get_future_gain(df)
                self.set_data(dataframe)

                # build the coefficient table for the entire range
                self.build_coefficient_table(0, nrows)

                # initialise the prediction array, using the close data
                pred_array = np.zeros(np.shape(future_gain_data), dtype=float)

                win_size = self.model_window

                # loop until we get to/past the end of the buffer
                start = 0
                end = start + win_size
                scale_start = max(0, end-self.scale_len)

                while end < nrows:

                    # set the (unmodified) gain data for scaling
                    self.update_scaler(np.array(dataframe['gain'].iloc[scale_start:end]))
                    # self.update_scaler(np.array(dataframe['gain'].iloc[start:end]))

                    # rebuild data up to end of current window
                    preds = self.predict_data(start, end)

                    # prediction length doesn't always match data length
                    plen = len(preds)
                    clen = min(plen, (nrows-start))

                    # print(f'    preds:{np.shape(preds)}')
                    # copy the predictions for this window into the main predictions array
                    pred_array[start:start+clen] = preds[:clen].copy()

                    # move the window to the next segment
                    # end = end + win_size - 1
                    # end = end + win_size
                    # start = start + win_size
                    end = end + plen
                    start = start + plen
                    scale_start = max(0, end - self.scale_len)


                # predict for last window

                self.update_scaler(np.array(dataframe['gain'].iloc[-self.scale_len:]))
                # self.update_scaler(np.array(dataframe['gain'].iloc[-win_size:]))
                # self.update_scaler(np.array(dataframe['gain'].iloc[train_start:train_end]))

                # preds = self.predict_data(data, -(self.train_len + win_size + 1), -(win_size + 1), -win_size, -1)
                preds = self.predict_data(nrows-win_size, nrows)
                # preds = self.predict_data(data, 0, -(win_size + 1), -win_size, -1)
                plen = len(preds)
                pred_array[-plen:] = preds.copy()

                palen = len(pred_array)
                dataframe['predicted_gain'][-palen:] = pred_array.copy()

                # slightly smooth, to get rid of spikes
                dataframe['predicted_gain'] = self.smooth(dataframe['predicted_gain'], 2)

            except Exception as e:
                print("*** Exception in add_jumping_predictions()")
                print(e) # prints the error message
                print(traceback.format_exc()) # prints the full traceback

        return dataframe

    #-------------

    # add the latest prediction, and update training periodically
    def add_latest_prediction(self, dataframe: DataFrame) -> DataFrame:

        df = dataframe
        win_size = 128
        nrows = np.shape(df)[0]

        try:
            # set up training data
            #TODO: see if we can do this incrementally instead of rebuilding every time, or just use portion of data
            future_gain_data = self.get_future_gain(df)
            self.set_data(dataframe)


            plen = len(self.custom_trade_info[self.curr_pair]['predictions'])
            dlen = len(dataframe['gain'])
            clen = min(plen, dlen)

            pred_array = np.zeros(clen, dtype=float)

            # print(f"[predictions]:{np.shape(self.custom_trade_info[self.curr_pair]['predictions'])}  pred_array:{np.shape(pred_array)}")

            # copy previous predictions and shift down by 1
            pred_array[-clen:] = self.custom_trade_info[self.curr_pair]['predictions'][-clen:].copy()
            pred_array = np.roll(pred_array, -1)
            pred_array[-1] = 0.0

            # # save original model
            # base_model = copy.deepcopy(self.model)

            # get predictions
            end = np.shape(dataframe)[0]
            start = end - win_size
            scale_start = max(0, end - self.scale_len)
            # train_start = 0
            # self.update_scaler(np.array(dataframe['gain'].iloc[-win_size:]))
            self.update_scaler(np.array(dataframe['gain'].iloc[scale_start:end]))


            # # build the coefficient table for (only) the prediction range)
            # self.build_coefficient_table(start, end)

            preds = self.predict_data(start, end)

            # self.model = base_model # restore original model

            # only replace last prediction (i.e. don't overwrite the historical predictions)
            pred_array[-1] = preds[-1].copy()
            # pred_array[-1] = preds[0].copy()

            # slightly smooth, to get rid of spikes
            pred_array = self.smooth(pred_array, 2)

            dataframe['predicted_gain'] = 0.0
            dataframe['predicted_gain'][-clen:] = pred_array[-clen:].copy()
            self.custom_trade_info[self.curr_pair]['predictions'][-clen:] = pred_array[-clen:].copy()

            ''''''
            pg = preds[-1]
            if pg <= dataframe['target_loss'].iloc[-1]:
                tag = "(*)"
            elif pg >= dataframe['target_profit'].iloc[-1]:
                tag = " * "
            else:
                tag = "   "
            print(f'    {tag} predict {pg:6.2f}% gain for: {self.curr_pair}')

            ''''''

        except Exception as e:
            print("*** Exception in add_latest_prediction()")
            print(e) # prints the error message
            print(traceback.format_exc()) # prints the full traceback

        return dataframe


    #-------------

