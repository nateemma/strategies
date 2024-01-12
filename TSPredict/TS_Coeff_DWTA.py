#pragma pylint: disable=W0105, C0103, C0301, W1203


"""
####################################################################################
TS_Coeff_DWTA - use a Discreet Wavelet Transform model (approximate coefficients only)

####################################################################################
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))
sys.path.append(str(Path(__file__)))

import utils.Wavelets as Wavelets
from TS_Coeff import TS_Coeff

class TS_Coeff_DWTA(TS_Coeff):

    wavelet_type = Wavelets.WaveletType.DWTA
