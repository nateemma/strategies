#pragma pylint: disable=W0105, C0103, C0301, W1203

from utils.DataframeUtils import DataframeUtils
from sklearn.preprocessing import RobustScaler
from datetime import datetime
from functools import reduce

import cProfile
import pstats

import numpy as np
# Get rid of pandas warnings during backtesting
import pandas as pd
from pandas import DataFrame, Series



pd.options.mode.chained_assignment = None  # default='warn'

# Strategy specific imports, files must reside in same folder as strategy
import sys
from pathlib import Path

import os
import joblib
group_dir = str(Path(__file__).parent)
strat_dir = str(Path(__file__).parent.parent)
sys.path.append(strat_dir)
sys.path.append(group_dir)

import logging
import warnings

log = logging.getLogger(__name__)
# log.setLevel(logging.DEBUG)
warnings.simplefilter(action='ignore', category=pd.errors.PerformanceWarning)

from sklearn.linear_model import SGDRegressor
from sklearn.preprocessing import RobustScaler

from NNPredict.NNPredictor_LSTM0 import NNPredictor_LSTM0

from TS_Coeff_DWT import TS_Coeff_DWT

"""
####################################################################################
TS_Coeff_DWT_LSTM - subclass of TS_Coeff_DWT that uses an LSTM model

####################################################################################
"""


class TS_Coeff_DWT_LSTM(TS_Coeff_DWT):


    seq_len = 1
    supports_incremental_training = True
    model = None

    ###################################
    
    # Model-related funcs. Override  to use a different type of model

    def get_model_path(self, pair=""):
        path = super().get_model_path("")
        path = path.replace(".sav", ".keras")
        return path

    def create_model(self, df_shape):

        print("    creating LSTM0")
        self.model = NNPredictor_LSTM0(pair="", seq_len=self.seq_len, num_features=df_shape[1])

        path = self.get_model_path("")
        self.model.set_model_path(path)
        # self.model.set_combine_models(self.combine_models)
        self.model.set_combine_models(True)
        
        return
    
    #-------------

    def load_model(self, df_shape):
                
        model_path = self.get_model_path("")

        # load from file or create new model
        if os.path.exists(model_path):
            self.model_trained = True
            self.new_model = False
            self.training_mode = False
            print(f'    Using existing model: {model_path}')
        else:
            self.model_trained = False
            self.new_model = True
            self.training_mode = True

        # just create the model class. It will load itself on first call to train_model()
        self.create_model(df_shape)
        return

    def train_model(self, model, data: np.array, train: np.array, save_model):

        if self.model is None:
            print("***    ERR: no model ***")
            return

        data_size = np.shape(data)[0]

        pad = self.lookahead  # have to allow for future results to be in range
        train_ratio = 0.8
        test_ratio = 1.0 - train_ratio
        train_size = int(train_ratio * (data_size - pad)) - 1
        test_size = int(test_ratio * (data_size - pad)) - 1
        train_start = 0
        test_start = data_size - (test_size + pad) - 1

        data_tensor = self.dataframeUtils.df_to_tensor(data, self.seq_len)
        train_data = data_tensor[train_start:train_start + train_size]
        test_data = data_tensor[test_start:test_start + test_size]

        # extract target from dataframe and convert to tensors
        train_results = train[train_start:train_start + train_size]
        test_results = train[test_start:test_start + test_size]

        model.train(train_data, test_data, train_results, test_results, False, save_model)

        return
    
    #-------------

    def predict_data(self, model, data):
        x = np.nan_to_num(data)
        x_tensor = self.dataframeUtils.df_to_tensor(x, self.seq_len)
        preds = model.predict(x_tensor)

        # de-norm
        scaler = RobustScaler()
        scaler.fit(self.gain_data.reshape(-1, 1))
        denorm_preds = scaler.inverse_transform(preds.reshape(-1, 1)).squeeze()

        denorm_preds = np.clip(denorm_preds, -3.0, 3.0)
        return denorm_preds



