#pragma pylint: disable=W0105, C0103, C0301, W1203


"""
####################################################################################
TS_Gain_DWTA - use a Discreet Wavelet Transform model (approximate) and SGD forecaster

####################################################################################
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))
sys.path.append(str(Path(__file__)))

import utils.Forecasters as Forecasters
import utils.Wavelets as Wavelets
from TS_Gain import TS_Gain

class TS_Gain_SGD(TS_Gain):

    forecaster_type = Forecasters.ForecasterType.SGD
    single_col_prediction = True
    detrend_data = True
    combine_models = True
    wavelet_size = 64
