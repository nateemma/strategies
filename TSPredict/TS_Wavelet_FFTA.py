#pragma pylint: disable=W0105, C0103, C0301, W1203


"""
####################################################################################
TS_Wavelet_FFT - use a Fast Fourier Transform model (approximation)

####################################################################################
"""


import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))
sys.path.append(str(Path(__file__)))


import utils.Wavelets as Wavelets
import utils.Forecasters as Forecasters

from TS_Wavelet import TS_Wavelet

class TS_Wavelet_FFTA(TS_Wavelet):

    wavelet_type = Wavelets.WaveletType.FFTA
