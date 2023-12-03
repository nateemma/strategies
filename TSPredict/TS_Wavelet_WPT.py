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

    wp = None
    nodes = None
 
    # function to get dwt coefficients
    def get_coeffs(self, data: np.array) -> np.array:

        length = len(data)

        x = data

        # data must be of even length, so trim if necessary
        if (len(x) % 2) != 0:
            x = x[1:]

        # self.wavelet = 'db4'

        self.wavelet = 'bior3.9'
        self.coeff_format = "wavedec"

        level = 2

        self.wp = pywt.WaveletPacket(data=x, wavelet=self.wavelet,  maxlevel=level, mode='zero')

        # get the nodes
        self.nodes = self.wp.get_level(level, 'natural')

        # concatenate the coefficients of the nodes into a 1D array
        coeffs = np.concatenate([node.data for node in self.nodes])

        '''
        # apply a low-pass filter
        from scipy.signal import butter, lfilter
        b, a = butter(4, 0.5)
        coeffs = lfilter(b, a, coeffs)

        '''
        
        return coeffs

    #-------------

    def get_value(self, coeffs):

        # set the coefficients of the nodes at level 2
        for node, chunk in zip(self.nodes, coeffs):
            node.data = chunk

        # reconstruct the data from level 2
        series = self.wp.reconstruct(update=True)

        # de-norm
        scaler = RobustScaler()
        scaler.fit(self.gain_data.reshape(-1, 1))
        denorm_series = scaler.inverse_transform(series.reshape(-1, 1)).squeeze()
        return denorm_series
    
    # -------------

    def array_to_coeff(self, array):
        coeffs = np.array_split(array, len(self.nodes))
        return coeffs
