#pragma pylint: disable=W0105, C0103, C0301, W1203


"""
####################################################################################
TS_Wavelet_FFT2 - use a Fast Fourier Transform model and an FFT Extrapolation forecaster

####################################################################################
"""


import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))
sys.path.append(str(Path(__file__)))


import utils.Wavelets as Wavelets
import utils.Forecasters as Forecasters

from TS_Wavelet import TS_Wavelet

class TS_Wavelet_FFT2(TS_Wavelet):

    wavelet_type = Wavelets.WaveletType.FFT
    forecaster_type = Forecasters.ForecasterType.FFT_EXTRAPOLATION
    single_col_prediction = True
    use_rolling = True
