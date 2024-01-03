
"""
####################################################################################
TS_Gain_PA - subclass of TS_Gain that uses a Passive Aggressive Regression model

####################################################################################
"""

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

from sklearn.svm import SVR
from utils.DataframeUtils import DataframeUtils


import utils.Forecasters as Forecasters
from TS_Gain import TS_Gain


class TS_Gain_PA(TS_Gain):

    supports_incremental_training = False
    combine_models = False
    forecaster_type = Forecasters.ForecasterType.PA




