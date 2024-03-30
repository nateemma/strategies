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
from NNPredictor_Wavenet2 import NNPredictor_Wavenet2

"""
####################################################################################
Predict_Wavenet2 - uses an Wavenet2 neural network to try and predict the future stock price
      
      This works by creating a  model that we train on the historical data, then use that model to predict 
      future values
      
      Note that this is very slow because we are training and running a neural network. 
      This strategy is likely not viable on a configuration of more than a few pairs, and even then needs
      a fast computer, preferably with a GPU
      
      In addition to the normal freqtrade packages, these strategies also require the installation of:
        finta
        keras
        tensorflow
        tqdm

####################################################################################
"""

# this inherits from NNPredict and just replaces the model used for predictions

class NNPredict_Wavenet2(NNPredict):

    curr_pair = ""
    custom_trade_info = {}

    dataset_type = DatasetType.MINIMAL


    ################################

    def get_classifier(self, pair, seq_len: int, num_features: int):
        return NNPredictor_Wavenet2(pair, seq_len, num_features)

    ################################
