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

# this adds  ../utils
sys.path.append("../utils")

import logging
import warnings

log = logging.getLogger(__name__)
# log.setLevel(logging.DEBUG)
warnings.simplefilter(action='ignore', category=pd.errors.PerformanceWarning)

from sklearn.ensemble import RandomForestRegressor
from utils.DataframeUtils import DataframeUtils


from TS_Simple import TS_Simple

"""
####################################################################################
TS_Simple_RF - subclass of TS_Simple that uses a Random Forest model

####################################################################################
"""


class TS_Simple_RF(TS_Simple):

    seq_len = 6
    num_features= 0
    model_per_pair = False
    dataframeUtils = DataframeUtils()
    training_mode = False # set to True to train initial model (over long period)
    supports_incremental_training = False
    combine_models = False

    ###################################
    
    # Model-related funcs. Override in subclass to use a different type of model

    def create_model(self, df_shape):

        print("    creating RandomForestRegressor")
        self.model = RandomForestRegressor(n_estimators=100, random_state=42)
        
        return
    

    #-------------



