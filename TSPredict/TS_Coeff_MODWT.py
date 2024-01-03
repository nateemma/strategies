#pragma pylint: disable=W0105, C0103, C0301, W1203


"""
####################################################################################
TS_Coeff_MODWT - use a Maximal Overlap Discrete Wavelet Transform model

####################################################################################
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))
sys.path.append(str(Path(__file__)))

import numpy as np
import pandas as pd
from pandas import DataFrame, Series
import pywt

import utils.Wavelets as Wavelets
from TS_Coeff import TS_Coeff

class TS_Coeff_MODWT(TS_Coeff):

    wavelet_type = Wavelets.WaveletType.MODWT

