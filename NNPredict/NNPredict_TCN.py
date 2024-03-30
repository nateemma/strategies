import operator

import numpy as np
from enum import Enum

import pywt
import talib.abstract as ta
from scipy.ndimage import gaussian_filter1d
from sklearn.preprocessing import RobustScaler, MinMaxScaler, StandardScaler

import freqtrade.vendor.qtpylib.indicators as qtpylib
import arrow

from freqtrade.exchange import timeframe_to_minutes
from freqtrade.strategy import (IStrategy, merge_informative_pair, stoploss_from_open,
                                IntParameter, DecimalParameter, CategoricalParameter)

from typing import Dict, List, Optional, Tuple, Union
from pandas import DataFrame, Series
from functools import reduce
from datetime import datetime, timedelta
from freqtrade.persistence import Trade

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

# import custom_indicators as cta
# from finta import TA as fta

#import keras
from keras import layers
from tqdm import tqdm
from tqdm.keras import TqdmCallback

import random

from utils.DataframePopulator import DatasetType

from NNPredict import NNPredict
from NNPredictor_TCN import NNPredictor_TCN


# this inherits from NNPredict and just replaces the model used for predictions

class NNPredict_TCN(NNPredict):

    curr_pair = ""
    custom_trade_info = {}

    # dataset_type = DatasetType.MINIMAL

    ################################

    def get_classifier(self, pair, seq_len: int, num_features: int):
        return NNPredictor_TCN(pair, seq_len, num_features)

    ################################
