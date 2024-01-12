# pragma pylint: disable=W0105, C0103, C0115, C0116, C0301, C0411, C0413,  W1203


"""
####################################################################################
TS_Wavelet_DWTA_min - use a Digital Wave Transform model (Approximate Coefficients)

                This is a 'stripped down' version of TS_Wavelet_DWT where we use
                only the 'approximate' coefficients of the transform.
                This greatly reduces the number of coefficients to fit, so it runs
                way faster than the normal version

                This is an even more minimalist version where all (added) indicators except 
                gain are removed. This is mostly to serve as a benchmark for the other TS_Wavelet
                variants, since there should not be any lookahead possible)

####################################################################################
"""

from xgboost import XGBRegressor
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))
sys.path.append(str(Path(__file__)))


import utils.Wavelets as Wavelets
import utils.Forecasters as Forecasters

from TS_Wavelet_DWTA import TS_Wavelet_DWTA


# this class is intended to experiment with global settings (without needing to change the base classes)

class TS_Wavelet_DWTA_roll(TS_Wavelet_DWTA):

    use_rolling = True
    # forecaster_type = Forecasters.ForecasterType.SGD


