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
import utils.custom_indicators as cta

import utils.Wavelets as Wavelets
import utils.Forecasters as Forecasters

import pywt


from sklearn.linear_model import SGDRegressor

from TSPredict import TSPredict



class TS_Coeff(TSPredict):

    coeff_table = None

    wavelet_type = Wavelets.WaveletType.DWT

    use_rolling = True

    def add_strategy_indicators(self, dataframe):

        # add some extra indicators

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
        gain_data = self.smooth(gain_data, 2)
        self.build_coefficient_table(gain_data)
        data = self.merge_coeff_table(df_norm)
        return data


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

            coeffs = self.wavelet.get_coeffs(dslice)
            features = self.wavelet.coeff_to_array(coeffs)
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

        # apply smoothing to each column, otherwise prediction alogorithms will struggle
        num_cols = np.shape(self.coeff_table)[1]
        for j in range (num_cols):
            feature = self.coeff_table[:,j]
            feature = self.smooth(feature, 2)
            self.coeff_table[:,j] = feature

        num_coeffs = np.shape(self.coeff_table)[1]

        merged_table = np.concatenate([np.array(dataframe), self.coeff_table], axis=1)

        return merged_table
    
    #-------------

