#pragma pylint: disable=W0105, C0103, C0301, W1203


"""
####################################################################################
TS_Wavelet_DWTA - use a Digital Wave Transform model (Approximate Coefficients)

                This is a 'stripped down' version of TS_Wavelet_DWT where we use
                only the 'approximate' coefficients of the transform.
                This greatly reduces the number of coefficients to fit, so it runs
                way faster than the normal version

####################################################################################
"""

from xgboost import XGBRegressor
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))
sys.path.append(str(Path(__file__)))

import numpy as np
import pandas as pd
from pandas import DataFrame, Series
import pywt


from sklearn.preprocessing import RobustScaler

from xgboost import XGBRegressor

from TS_Wavelet import TS_Wavelet

class TS_Wavelet_DWTA(TS_Wavelet):

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
        self.mode = 'zero'
        self.coeff_format = "wavedec"
        level = 2

        coeffs = pywt.wavedec(x, self.wavelet, mode=self.mode, level=level)

        # set detailed coeffs to zero (they still need to be there though)
        threshold = 0.0
        coeffs[1:] = [pywt.threshold(c, value=threshold, mode='hard') for c in coeffs[1:]]
        
        return self.coeff_to_array(coeffs)

    #-------------

    def get_value(self, coeffs):
        # series = pywt.waverec(coeffs, self.wavelet)

        series = pywt.waverec(coeffs, wavelet=self.wavelet, mode=self.mode)
        # print(f'    coeff_slices:{self.coeff_slices}, coeff_shapes:{self.coeff_shapes} series:{np.shape(series)}')

        # de-norm
        scaler = RobustScaler()
        scaler.fit(self.gain_data.reshape(-1, 1))
        denorm_series = scaler.inverse_transform(series.reshape(-1, 1)).squeeze()
        return denorm_series
    
    # -------------

    save_coeffs = None

    def coeff_to_array(self, coeffs):
        # flatten the coefficient arrays

        # this version is specific to this strat, since we remove the detailed coeficients, and add them back later

        # for i, ci in enumerate(coeffs):
        #     print(f"    Shape of coeffs[{i}] = {ci.shape}")
        features = np.array(coeffs[0])

        self.save_coeffs = coeffs

        return np.array(features)

    # -------------

    def array_to_coeff(self, array):

        coeffs = self.save_coeffs
        coeffs[0] = array
        return coeffs

    # -------------

