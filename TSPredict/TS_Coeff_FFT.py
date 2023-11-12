#pragma pylint: disable=W0105, C0103, C0301, W1203


"""
####################################################################################
TS_Coeff_FFT - use a Fast Fourier Transform model

####################################################################################
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))
sys.path.append(str(Path(__file__)))

import numpy as np
import pandas as pd
from pandas import DataFrame, Series

from scipy.fft import fft, rfft, irfft

from TS_Coeff import TS_Coeff

class TS_Coeff_FFT(TS_Coeff):

    ###################################

 
    # function to get dwt coefficients
    def get_coeffs(self, data: np.array) -> np.array:

        n = len(data)
                
        # # de-trend the data
        # w_mean = data.mean()
        # w_std = data.std()
        # x = (data - w_mean) / w_std

        x = data

        # compute the fft
        features = rfft(x, n)
        # features = fft(x, n)

        # print(f'features: {features}')

        return features

    #-------------

