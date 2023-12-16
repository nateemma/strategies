#pragma pylint: disable=W0105, C0103, C0301, W1203


"""
####################################################################################
TS_Wavelet_FFT - use a Fast Fourier Transform model

####################################################################################
"""

from scipy.fft import fft, rfft, irfft, fftfreq
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))
sys.path.append(str(Path(__file__)))

import numpy as np
import pandas as pd
from pandas import DataFrame, Series

from scipy.fft import fft, ifft, rfft, irfft, fftfreq
from scipy.signal import butter, freqz, iirdesign, filtfilt


from sklearn.preprocessing import RobustScaler

from TS_Wavelet import TS_Wavelet

class TS_Wavelet_FFT(TS_Wavelet):

    ###################################

    data_shape = None
 
    # function to get dwt coefficients
    def get_coeffs(self, data: np.array) -> np.array:

        length = len(data)

        x = data

        # data must be of even length, so trim if necessary
        if (len(x) % 2) != 0:
            x = x[1:]

        coeffs = fft(x)

        self.data_shape = np.shape(coeffs)

        '''
        # compute power spectrum density
        # squared magnitude of each fft coefficient
        psd = coeffs * np.conj(coeffs) / len(x)
        threshold = 20
        coeffs = np.where(psd < threshold, 0.0, coeffs)

        '''

        return np.ravel(coeffs)

    #-------------

    def get_value(self, coeffs):


        # reconstruct the data
        series = ifft(coeffs)

        series = series.real

        return series
    
    # -------------

    def array_to_coeff(self, array:np.ndarray):
        coeffs = np.reshape(array, self.data_shape)
        return coeffs
