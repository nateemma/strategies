#pragma pylint: disable=W0105, C0103, C0301, W1203


"""
####################################################################################
TS_Wavelet_SWT - use a Standing Wave Transform model

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


from sklearn.preprocessing import RobustScaler

from TS_Wavelet import TS_Wavelet

class TS_Wavelet_SWT(TS_Wavelet):

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

        # self.wavelet = 'db4'

        self.wavelet = 'bior3.9'

        self.coeff_format = "wavedec"

        # (cA2, cD2), (cA1, cD1) = pywt.swt(data, wavelet, level=2)
        
        # swt returns an array, with each element being 2 arrays - cA_n and cD_n, whre n is the level
        levels = min(2, pywt.swt_max_level(len(x)))
        # levels = 1 #TMP DEBUG
        coeffs = pywt.swt(x, self.wavelet, level=levels, trim_approx=True)

        return self.coeff_to_array(coeffs)

    #-------------

    def get_value(self, coeffs):
        # series = pywt.waverec(coeffs, self.wavelet)

        series = pywt.iswt(coeffs, wavelet=self.wavelet)
        # print(f'    coeff_slices:{self.coeff_slices}, coeff_shapes:{self.coeff_shapes} series:{np.shape(series)}')

        return series
    
    # -------------
