import operator

import numpy as np
from enum import Enum


from freqtrade.exchange import timeframe_to_minutes
from freqtrade.strategy import (IStrategy, merge_informative_pair, stoploss_from_open,
                                IntParameter, DecimalParameter, CategoricalParameter)


# Get rid of pandas warnings during backtesting
import pandas as pd
import pandas_ta as pta

pd.options.mode.chained_assignment = None  # default='warn'

# Strategy specific imports, files must reside in same folder as strategy
import sys
import os
from pathlib import Path

sys.path.append(str(Path(__file__).parent))

import logging
import warnings

log = logging.getLogger(__name__)
# log.setLevel(logging.DEBUG)
warnings.simplefilter(action='ignore', category=pd.errors.PerformanceWarning)


from utils.DataframePopulator import DatasetType

from NNPredict import NNPredict
import utils.NNPredictors as NNPredictors

"""
####################################################################################
Predict_WavenetFull - uses an WavenetFull neural network to try and predict the future stock price
      
      This works by creating a  model that we train on the historical data, then use that model to predict 
      future values

####################################################################################
"""

# this inherits from NNPredict and just replaces the model used for predictions

class NNPredict_WavenetFull(NNPredict):

    predictor_type = NNPredictors.PredictorType.WAVENET_FULL
