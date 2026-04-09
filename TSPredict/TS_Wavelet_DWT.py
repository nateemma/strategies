# pragma pylint: disable=W0105, C0103, C0114, C0115, C0116, C0301, C0302, C0303, C0411, C0413,  W1203


"""
####################################################################################
TS_Wavelet_DWT - use a Digital Wave Transform model

####################################################################################
"""

from xgboost import XGBRegressor
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))
sys.path.append(str(Path(__file__)))


import utils.Wavelets as Wavelets
import utils.Forecasters as Forecasters

from TS_Wavelet import TS_Wavelet

class TS_Wavelet_DWT(TS_Wavelet):

    wavelet_type = Wavelets.WaveletType.DWT