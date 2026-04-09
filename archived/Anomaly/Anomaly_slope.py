import numpy as np
from enum import Enum

import pywt
import scipy
import talib.abstract as ta
from scipy.ndimage import gaussian_filter1d
from statsmodels.discrete.discrete_model import Probit

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
from pathlib import Path

sys.path.append(str(Path(__file__).parent))
sys.path.append(str(Path(__file__).parent.parent))

import logging
import warnings

log = logging.getLogger(__name__)
# log.setLevel(logging.DEBUG)
warnings.simplefilter(action='ignore', category=pd.errors.PerformanceWarning)

from utils.DataframePopulator import DatasetType
import utils.TrainingSignals as TrainingSignals

from Anomaly import Anomaly

"""
####################################################################################
Anomaly_slope:
    This is a subclass of Anomaly, which provides a framework for deriving a dimensionally-reduced model combined
    with anomaly detection
    This class trains the model based on the slope training signals

####################################################################################
"""


class Anomaly_slope(Anomaly):
       
       signal_type = TrainingSignals.SignalType.Slope