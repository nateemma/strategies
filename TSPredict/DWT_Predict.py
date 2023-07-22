#pragma pylint: disable=W0105, C0103, C0301, W1203


"""
####################################################################################
DWT_Predict - use a Discreet Wavelet Transform to model the price, and a
              regression algorithm trained on the DWT coefficients, which is then used
              to predict future prices.
              Unfortunately, this must all be done in a rolling fashion to avoid lookahead
              bias - so it is pretty slow

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

from TS_Predict import TS_Predict

class DWT_Predict(TS_Predict):

    ###################################

 
    # function to get dwt coefficients
    def get_coeffs(self, data: np.array) -> np.array:

        length = len(data)

        x = data

        # data must be of even length, so trim if necessary
        if (len(x) % 2) != 0:
            x = x[1:]

        # print(pywt.wavelist(kind='discrete'))

        # get the DWT coefficients
        # wavelet = 'db12'
        wavelet = 'db8'
        # wavelet = 'haar'
        level = 2
        coeffs = pywt.wavedec(x, wavelet, mode='smooth', level=level)

                # remove higher harmonics
        std = np.std(coeffs[level])
        sigma = (1 / 0.6745) * self.madev(coeffs[-level])
        # sigma = madev(coeff[-level])
        uthresh = sigma * np.sqrt(2 * np.log(length))

        coeffs[1:] = (pywt.threshold(i, value=uthresh, mode='hard') for i in coeffs[1:])

        # flatten the coefficient arrays
        features = np.concatenate(np.array(coeffs, dtype=object))

        return features

    #-------------

