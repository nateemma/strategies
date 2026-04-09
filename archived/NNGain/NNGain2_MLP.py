

import pandas as pd
pd.options.mode.chained_assignment = None  # default='warn'

# Strategy specific imports, files must reside in same folder as strategy
import sys
import os
from pathlib import Path

sys.path.append(str(Path(__file__).parent))
sys.path.append(str(Path(__file__).parent.parent))

import logging
import warnings

log = logging.getLogger(__name__)
# log.setLevel(logging.DEBUG)
warnings.simplefilter(action='ignore', category=pd.errors.PerformanceWarning)

# from utils.DataframePopulator import DatasetType

from NNGain2 import NNGain2
import utils.NNPredictors as NNPredictors

"""
####################################################################################
NNGain2_MLP - uses a MLP neural network to try and predict the future stock price
      
      This works by creating a  model that we train on the historical data, then use that model to predict 
      future values
      

####################################################################################
"""

# this inherits from NNGain2 and just replaces the model used for predictions

class NNGain2_MLP(NNGain2):

    predictor_type = NNPredictors.PredictorType.MLP
