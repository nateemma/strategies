#pragma pylint: disable=W0105, C0103, C0301, W1203


"""
####################################################################################
TS_Wavelet_WPT - use a Wavelet Packet Transform model

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

class TS_Wavelet_WPT(TS_Wavelet):

    ###################################

    wp = None # have to keep track of Wavelet Packet object
 
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

        self.wp = pywt.WaveletPacket(data, wavelet=self.wavelet, mode=self.mode)

        # get the coefficients at level 2
        nodes = self.wp.get_level(2, order='freq')
        coeffs = [n.data for n in nodes]

        return self.coeff_to_array(coeffs)

    #-------------

    def get_value(self, coeffs):
        # series = pywt.waverec(coeffs, self.wavelet)

        series = self.wp.reconstruct(coeffs)
        # print(f'    coeff_slices:{self.coeff_slices}, coeff_shapes:{self.coeff_shapes} series:{np.shape(series)}')

        # de-norm
        scaler = RobustScaler()
        scaler.fit(self.gain_data.reshape(-1, 1))
        denorm_series = scaler.inverse_transform(series.reshape(-1, 1)).squeeze()
        return denorm_series
    
    # -------------
