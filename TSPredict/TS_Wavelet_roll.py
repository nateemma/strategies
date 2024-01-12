# pragma pylint: disable=W0105, C0103, C0115, C0116, C0301, C0411, C0413,  W1203



from xgboost import XGBRegressor
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))
sys.path.append(str(Path(__file__)))

import numpy as np
import pandas as pd
from pandas import DataFrame, Series
import pywt


from sklearn.preprocessing import RobustScaler

from xgboost import XGBRegressor

import utils.Wavelets as Wavelets
import utils.Forecasters as Forecasters
from TS_Wavelet_DWTA import TS_Wavelet_DWTA


# this class is intended to experiment with global settings (without needing to change the base classes)

class TS_Wavelet_roll(TS_Wavelet_DWTA):


    use_rolling = True
    single_col_prediction = True

