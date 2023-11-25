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

#pragma pylint: disable=W0105, C0103, C0114, C0115, C0116, C0301, C0302, C0303, C0325, W1203


import sys
from pathlib import Path

import numpy as np
# Get rid of pandas warnings during backtesting
import pandas as pd
from pandas import DataFrame, Series


import freqtrade.vendor.qtpylib.indicators as qtpylib

pd.options.mode.chained_assignment = None  # default='warn'

# Strategy specific imports, files must reside in same folder as strategy


group_dir = str(Path(__file__).parent)
strat_dir = str(Path(__file__).parent.parent)
sys.path.append(strat_dir)
sys.path.append(group_dir)

# warnings.simplefilter(action='ignore', category=pd.errors.PerformanceWarning)
# warnings.simplefilter(action='ignore', category=FutureWarning)

import talib.abstract as ta

import pywt


from sklearn.linear_model import SGDRegressor

from TSPredict import TSPredict



class TS_Coeff(TSPredict):

    coeff_table = None

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

    #-------------

    # override func to get data for this strategy
    def get_data(self, dataframe):

        df_norm = self.convert_dataframe(dataframe)
        gain_data = df_norm['gain'].to_numpy()
        self.build_coefficient_table(gain_data)
        data = self.merge_coeff_table(self.convert_dataframe(df_norm))
        return data

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

    ###################################

    # DWT functions
 
    def madev(self, d, axis=None):
        """ Mean absolute deviation of a signal """
        return np.mean(np.absolute(d - np.mean(d, axis)), axis)


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

    # Coefficient functions
    
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

        merged_table = np.concatenate([np.array(dataframe), self.coeff_table], axis=1)

        return merged_table
    
    #-------------

