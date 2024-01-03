#pragma pylint: disable=W0105, C0103, C0301, W1203
"""
####################################################################################
TS_Gain_TES - subclass of TS_Gain that uses a Triple Exponential Smoothing model

####################################################################################
"""


from datetime import datetime
from functools import reduce

import cProfile
import pstats

import numpy as np
# Get rid of pandas warnings during backtesting
import pandas as pd
from pandas import DataFrame, Series
from sklearn.preprocessing import RobustScaler



pd.options.mode.chained_assignment = None  # default='warn'

# Strategy specific imports, files must reside in same folder as strategy
import sys
from pathlib import Path

import os
import joblib
import logging
import warnings

group_dir = str(Path(__file__).parent)
strat_dir = str(Path(__file__).parent.parent)
sys.path.append(strat_dir)
sys.path.append(group_dir)



log = logging.getLogger(__name__)
# log.setLevel(logging.DEBUG)
warnings.simplefilter(action='ignore', category=pd.errors.PerformanceWarning)

from statsmodels.tsa.api import ExponentialSmoothing, SimpleExpSmoothing, Holt

from utils.DataframeUtils import DataframeUtils


import utils.Forecasters as Forecasters
from TS_Gain import TS_Gain


class TS_Gain_TES(TS_Gain):

    supports_incremental_training = False
    combine_models = False
    forecaster_type = Forecasters.ForecasterType.EXPONENTAL
