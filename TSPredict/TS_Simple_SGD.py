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
from utils.DataframeUtils import DataframeUtils


from TS_Simple import TS_Simple

"""
####################################################################################
TS_Simple_SGD - subclass of TS_Simple that uses a Stochastic Gradient Decent model

####################################################################################
"""


class TS_Simple_SGD(TS_Simple):

    seq_len = 6
    num_features= 0
    model_per_pair = False
    dataframeUtils = DataframeUtils()
    combine_models = False
    training_mode = False # set to True to train initial model (over long period)
    supports_incremental_training = True

    ###################################
    
    # Model-related funcs. Override in subclass to use a different type of model

    def create_model(self, df_shape):

        print("    creating SGDRegressor")
        self.model = SGDRegressor(loss='huber', shuffle=False)
        
        return
    
    #-------------

    def train_model(self, model, data: np.array, train: np.array, save_model):

        if self.model is None:
            print("***    ERR: no model ***")
            return
        
        # train on the supplied data
        # if self.new_model and (not self.model_trained):
        #     self.model = self.model.fit(data, train)
        # else:
        #     self.model = self.model.partial_fit(data, train)

        model = model.partial_fit(data, train)
        # self.model = self.model.fit(data, train)
        return
    
    #-------------



