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

from modwt import modwt, modwtmra

from TS_Coeff import TS_Coeff

class TS_Coeff_MODWT(TS_Coeff):

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
        # wavelet = 'db8'
        wavelet = 'bior3.9'
        level = 5
        coeffs = modwt(x, wavelet, level)

        '''
        # remove higher harmonics
        std = np.std(coeffs[level])
        sigma = (1 / 0.6745) * self.madev(coeffs[-level])
        # sigma = madev(coeff[-level])
        uthresh = sigma * np.sqrt(2 * np.log(length))

        coeffs[1:] = (pywt.threshold(i, value=uthresh, mode='hard') for i in coeffs[1:])
        '''


        # flatten the coefficient arrays
        features = np.concatenate(np.array(coeffs, dtype=object))

        return features

    #-------------

