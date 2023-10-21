#pragma pylint: disable=W0105, C0103, C0301, W1203

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
import os
from pathlib import Path

group_dir = str(Path(__file__).parent)
strat_dir = str(Path(__file__).parent.parent)
sys.path.append(strat_dir)
sys.path.append(group_dir)

# this adds  ../utils
sys.path.append("../utils")

import logging
import warnings

log = logging.getLogger(__name__)
# log.setLevel(logging.DEBUG)
warnings.simplefilter(action='ignore', category=pd.errors.PerformanceWarning)

from NNPredict.NNPredictor_Attention import NNPredictor_Attention
from utils.DataframeUtils import DataframeUtils

from TS_Simple import TS_Simple

"""
####################################################################################
TS_Simple_Attention - subclass of TS_Simple that uses an Attention model

####################################################################################
"""


class TS_Simple_Attention(TS_Simple):

    seq_len = 6
    num_features= 0
    model_per_pair = False
    dataframeUtils = DataframeUtils()
    training_mode = False # set to True to train initial model (over long period)
    combine_models = True
    supports_incremental_training = False

    ###################################

    def get_model_path(self, pair):
        category = self.__class__.__name__
        root_dir = group_dir + "/models/" + category
        model_name = category
        if self.model_per_pair:
            model_name = model_name + "_" + pair.split("/")[0]
        path = root_dir + "/" + model_name + ".keras"
        return path
    
    # Model-related funcs. Override in subclass to use a different type of model

    def load_model(self, df_shape):

        if self.model is None:
            self.new_model = False if os.path.exists(self.get_model_path("")) else True
            self.create_model(df_shape)
        
        return

    #-------------

    def create_model(self, df_shape):

        self.num_features = df_shape[1]

        self.model = NNPredictor_Attention(self.curr_pair, self.seq_len, self.num_features)
        self.model.set_model_path(self.get_model_path(self.curr_pair))
        self.model.set_combine_models(self.combine_models)
        
        return

    #-------------

    def train_model(self, model, data: np.array, results: np.array, save_model):

        # print(f'data:{np.shape(data)} train:{np.shape(train)}')

        test_len = int(0.9 * len(data))
        train_data = data[:test_len-1]
        test_data = data[test_len:]
        train_results = results[:test_len-1]
        test_results = results[test_len:]

        # convert to tensors
        tsr_train_data = self.dataframeUtils.df_to_tensor(train_data, self.seq_len)
        tsr_train_results = self.dataframeUtils.df_to_tensor(train_results.reshape(-1, 1), self.seq_len)

        tsr_test_data = self.dataframeUtils.df_to_tensor(test_data, self.seq_len)
        tsr_test_results = self.dataframeUtils.df_to_tensor(test_results.reshape(-1, 1), self.seq_len)

        # print(f'tsr_train_data:{np.shape(tsr_train_data)} tsr_train_results:{np.shape(tsr_train_results)}')
        # print(f'tsr_test_data:{np.shape(tsr_test_data)}   tsr_test_results:{np.shape(tsr_test_results)}')

        force = True if (not save_model) else False

        if force:
            model.set_num_epochs(16)
            # model.set_learning_rate(0.001)
        else:
            model.set_num_epochs()
            model.set_learning_rate()

        model.train(tsr_train_data, tsr_test_data, tsr_train_results,  tsr_test_results, 
                    force_train=force, save_model=save_model)

        return
    
    def predict_data(self, data):
        tsr = self.dataframeUtils.df_to_tensor(data, self.seq_len)
        return self.model.predict(tsr)


