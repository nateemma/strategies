#pragma pylint: disable=W0105, C0103, C0301, W1203


"""
####################################################################################
TS_Coeff_DWTA - use a Discreet Wavelet Transform model (approximate) and MLP forecaster

####################################################################################
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))
sys.path.append(str(Path(__file__)))

import utils.Forecasters as Forecasters
import utils.Wavelets as Wavelets
from TS_Coeff import TS_Coeff

class TS_Coeff_MLP(TS_Coeff):

    wavelet_type = Wavelets.WaveletType.MODWT
    forecaster_type = Forecasters.ForecasterType.MLP
    single_col_prediction = False
    wavelet_size = 64
    norm_data = False
    combine_models = False
    detrend_data = True
