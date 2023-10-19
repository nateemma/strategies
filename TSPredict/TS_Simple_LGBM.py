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

from lightgbm.sklearn import LGBMRegressor
from utils.DataframeUtils import DataframeUtils


from TS_Simple import TS_Simple

"""
####################################################################################
TS_Simple_LGBM - subclass of TS_Simple that uses a Light GBM model

####################################################################################
"""


class TS_Simple_LGBM(TS_Simple):

    seq_len = 6
    num_features= 0
    model_per_pair = False
    dataframeUtils = DataframeUtils()
    training_mode = False # set to True to train initial model (over long period) - much faster
    supports_incremental_training = True
    combine_models = True

    ###################################
    
    # Model-related funcs. Override in subclass to use a different type of model

    def create_model(self, df_shape):

        print("    creating LGBMRegressor")
        # LGBMRegressor gives better/faster results, but has issues on some MacOS platforms. 

        params = {'n_estimators': 100, 'max_depth': 4, 'learning_rate': 0.1, 'verbose': -1}
        # self.model = LGBMRegressor(n_estimators=100, random_state=42, verbose=-1)
        self.model = LGBMRegressor(**params)
        
        return
    

    #-------------



