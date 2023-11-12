#pragma pylint: disable=W0105, C0103, C0301, W1203


"""
####################################################################################
TS_Coeff_SWT - use a Stabding Wave Transform model

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

from TS_Coeff import TS_Coeff

class TS_Coeff_SWT(TS_Coeff):

    ###################################

 
    # function to get dwt coefficients
    def get_coeffs(self, data: np.array) -> np.array:

        length = len(data)

        # # de-trend the data
        # w_mean = data.mean()
        # w_std = data.std()
        # x = (data - w_mean) / w_std

        x = data

        # data must be of even length, so trim if necessary
        if (len(x) % 2) != 0:
            x = x[1:]

        wavelet = 'db4'

        # (cA2, cD2), (cA1, cD1) = pywt.swt(data, wavelet, level=2)
        
        # swt returns an array, with each element being 2 arrays - cA_n and cD_n, whre n is the level
        levels = min(2, pywt.swt_max_level(len(x)))
        # levels = 1 #TMP DEBUG
        coeffs = pywt.swt(x, wavelet, level=levels)
        num_levels = np.shape(coeffs)[0] # varies depending upon the wavelet used

        # coeff_list = np.array([data[-1]]) # always include the last input data point
        # coeff_list = [data[-1]] # always include the last input data point
        coeff_list = [] 
        if num_levels > 0:
            # add the approximation coefficients, then the detailed
            for i in range(num_levels):
                cA_n = coeffs[i][0]
                cD_n = coeffs[i][1]
                coeff_list.extend(cA_n)
                coeff_list.extend(cD_n)

        # print(f'coeff_list:{np.shape(coeff_list)}')
        features = np.array(coeff_list, dtype=float)

        return features

    #-------------

