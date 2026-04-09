#pragma pylint: disable=W0105, C0103, C0301, W1203


"""
####################################################################################
TS_Coeff_DWTA - use a Discreet Wavelet Transform model (approximate) and SGD forecaster

####################################################################################
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))
sys.path.append(str(Path(__file__)))

import utils.Forecasters as Forecasters
import utils.Wavelets as Wavelets
from TS_Coeff import TS_Coeff

class TS_Coeff_SGD(TS_Coeff):

    wavelet_type = Wavelets.WaveletType.MODWT
    forecaster_type = Forecasters.ForecasterType.SGD
    single_col_prediction = False
    norm_data = False
    combine_models = False
    detrend_data = True
