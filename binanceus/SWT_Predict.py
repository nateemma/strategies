#pragma pylint: disable=W0105, C0103, C0301, W1203


'''

####################################################################################
SWT_Predict - use a Discreet Wavelet Transform to model the price, and a
              regression algorithm trained on the SWT coefficients, which is then used
              to predict future prices.
              Unfortunately, this must all be done in a rolling fashion to avoid lookahead
              bias - so it is pretty slow

              This variant uses the Standing Wave Transform (part of DWT library). Apparently
              this can be better for signals with sudden changes

####################################################################################
'''

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))
sys.path.append(str(Path(__file__)))

import numpy as np
import pandas as pd
from pandas import DataFrame, Series
import pywt

from TS_Predict import TS_Predict

class SWT_Predict(TS_Predict):
    ###################################

 
    # function to get swt coefficients
    def get_coeffs(self, data: np.array) -> np.array:


        # print(pywt.wavelist(kind='discrete'))

        # print(f"data: {np.shape(data)}")

        retrend = False

        if retrend:
            # de-trend the data
            w_mean = data.mean()
            w_std = data.std()
            x = (data - w_mean) / w_std
        else:
            x = data

        # data must be of even length, so trim if necessary
        if (len(x) % 2) != 0:
            x = x[1:]

        # get the SWT coefficients
        # wavelet = 'haar'
        # wavelet = 'db1'
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
                if retrend:
                    cA_n = (cA_n * w_std) + w_mean
                cD_n = coeffs[i][1]
                coeff_list.extend(cA_n)
                coeff_list.extend(cD_n)

        # print(f'coeff_list:{np.shape(coeff_list)}')
        features = np.array(coeff_list, dtype=float)

        # if retrend:
        #     # re-trend
        #     features = (features * w_std) + w_mean

        # # trim down to max 128 entries
        # if len(features) > 128:
        #     features = features[:128]

        return features
    
