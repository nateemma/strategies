'''
This class is factory for creating various kinds of Wavelet Transforms with a consistent interface
'''



# -----------------------------------

# base class - to allow generic treatment of all transforms

from abc import ABC, abstractmethod
from enum import Enum

import numpy as np
import pywt
from scipy.fft import fft, hfft, ifft, ihfft, rfft, irfft, fftfreq
from scipy.fft import fht, ifht
from modwt import modwt, imodwt, modwtmra

# all actual instantiations follow this base class

class base_wavelet(ABC):

    # the following are used to store info needed across different calls
    wavelet_type = "db2"
    wavelet = None
    mode = "smooth"
    coeff_slices = None
    coeff_shapes = None
    coeff_format = "wavedec"
    data_shape = None
    detrend = True
    lookahead = 0


    def __init__(self):
        super().__init__()

    # function to get transform coefficients  (different for each wavelet type)
    @abstractmethod
    def get_coeffs(self, data: np.array) -> np.array:
        # this is a DWT transform, for reference:
        coeffs = pywt.wavedec(data, self.wavelet, mode=self.mode, level=2)
        return coeffs

    # function to convert coefficients back into a waveform (inverse of get_coeffs)
    @abstractmethod
    def get_values(self, coeffs):
        # series = pywt.waverec(coeffs, self.wavelet)

        series = pywt.waverec(coeffs, wavelet=self.wavelet, mode=self.mode)
        # print(f'    coeff_slices:{self.coeff_slices}, coeff_shapes:{self.coeff_shapes} series:{np.shape(series)}')

        return series

    # convert wavelet coefficints to 1d numpy array. This works for several variants
    def coeff_to_array(self, coeffs):
        # flatten the coefficient arrays

        # more general purpose (can use with many waveforms)
        array, self.coeff_slices = pywt.coeffs_to_array(coeffs)

        # print(f'    coeff_to_array: coeffs:{dir(coeffs)}, arr:{np.shape(arr)}')
        return np.array(array)

    # convert 1d numpy array back to wavelet coefficient format. This works for several variants
    def array_to_coeff(self, array):

        # more general purpose (can use with many waveforms)
        coeffs = pywt.array_to_coeffs(array, self.coeff_slices, output_format=self.coeff_format)

        # print(f'    coeff_slices:{self.coeff_slices}, coeff_shapes:{self.coeff_shapes} array:{np.shape(array)}')
        # print(f'    array_to_coeff: coeffs:{dir(coeffs)}, array:{np.shape(array)}')
        return coeffs

    # set whether to detrend/retrend data or not. Set to False if you are doing a 1-way transform
    def set_detrend(self, trend:bool):
        self.detrend = trend

    # set lookahead value (for detrending). Only need to do this if you are projecting ahead
    def set_lookahead(self, lookahead):
        self.lookahead = lookahead
        return

# -----------------------------------
    # utilities to de-/re-trend an array
    poly = None
    xhat = None

    def detrend_array(self, x):
        n = x.size
        t = np.arange(0, n)
        self.poly = np.polyfit(t, x, 1)         # find linear trend in x

        self.xhat = np.polyval(self.poly, t)
        x_notrend = x - self.xhat

        return x_notrend

    def retrend_array(self, x_notrend):
        n = x_notrend.size
        t = np.arange(self.lookahead, self.lookahead+n)
        x = x_notrend + self.xhat
        return x

# -----------------------------------

# DWT - Discrete Wavelet Transform

class dwt_wavelet(base_wavelet):

    def get_coeffs(self, data: np.array) -> np.array:

        if self.detrend:
            x = self.detrend_array(data)
        else:
            x = data


        # data must be of even length, so trim if necessary
        if (len(x) % 2) != 0:
            x = x[1:]

        # get the DWT coefficients
        self.wavelet_type = 'bior3.9'
        self.wavelet = pywt.Wavelet(self.wavelet_type)
        self.mode = 'symmetric'
        self.coeff_format = "wavedec"
        level = 3
        coeffs = pywt.wavedec(x, self.wavelet, mode=self.mode, level=level)

        # print(f'    wavelet.dec_len:{self.wavelet.dec_len}  wavelet.rec_len:{self.wavelet.rec_len}')

        '''
        # remove higher harmonics
        std = np.std(coeffs[level])
        sigma = (1 / 0.6745) * self.madev(coeffs[-level])
        # sigma = madev(coeff[-level])
        uthresh = sigma * np.sqrt(2 * np.log(length))

        coeffs[1:] = (pywt.threshold(i, value=uthresh, mode='hard') for i in coeffs[1:])

        '''

        # print(f'  get_coeffs() coeffs:')
        # for arr in coeffs:
        #     print(f'  {np.shape(arr)}')

        return coeffs

    def get_values(self, coeffs):

        series = pywt.waverec(coeffs, wavelet=self.wavelet, mode=self.mode)

        if self.detrend:
            series = self.retrend_array(series)

        # print(f'    coeffs:{np.shape(coeffs)}, series:{np.shape(series)}')

        return series

    def coeff_to_array(self, coeffs):
        array = super().coeff_to_array(coeffs)
        # print(f'  coeff_to_array() {np.shape(array)}')
        return array

    def array_to_coeff(self, array):

        # print(f'  array_to_coeff() {np.shape(array)}')
        coeffs = super().array_to_coeff(array)
        return coeffs

# -----------------------------------

# DWT - Discrete Wavelet Transform, Approximate Coefficients only

class dwta_wavelet(base_wavelet):

    def get_coeffs(self, data: np.array) -> np.array:

        if self.detrend:
            x = self.detrend_array(data)
        else:
            x = data


        # data must be of even length, so trim if necessary
        if (len(x) % 2) != 0:
            x = x[1:]

        # get the DWT coefficients
        self.wavelet_type = 'bior3.9'
        self.wavelet = pywt.Wavelet(self.wavelet_type)
        self.mode = 'symmetric'
        self.coeff_format = "wavedec"
        level = 2
        coeffs = pywt.wavedec(x, self.wavelet, mode=self.mode, level=level)

        # print(f'    wavelet.dec_len:{self.wavelet.dec_len}  wavelet.rec_len:{self.wavelet.rec_len}')


        # set detailed coeffs to zero (they still need to be there though)
        threshold = 0.0
        coeffs[1:] = [pywt.threshold(c, value=threshold, mode='hard') for c in coeffs[1:]]

        return coeffs

    def get_values(self, coeffs):
        # series = pywt.waverec(coeffs, self.wavelet)

        # print(f'  get_values() coeffs:')
        # for arr in coeffs:
        #     print(f'  {np.shape(arr)}')
        series = pywt.waverec(coeffs, wavelet=self.wavelet, mode=self.mode)

        if self.detrend:
            series = self.retrend_array(series)

        # print(f'    coeffs:{np.shape(coeffs)}, series:{np.shape(series)}')

        return series

    def coeff_to_array(self, coeffs):
        array = np.array(coeffs[0])
        self.save_coeffs = coeffs
        return array

    def array_to_coeff(self, array):
        coeffs = self.save_coeffs
        coeffs[0] = array
        return coeffs

# -----------------------------------

# FFT - Fast Fourier Transform

class fft_wavelet(base_wavelet):
    def get_coeffs(self, data: np.array) -> np.array:

        if self.detrend:
            x = self.detrend_array(data)
        else:
            x = data

        freqs = rfft(x)

        # # filter out higher harmonics
        # n_param = int(len(freqs) / 2) + 1
        # h=np.sort(freqs)[-n_param]
        # freqs=[ freqs[i] if np.absolute(freqs[i])>=h else 0 for i in range(len(freqs)) ]

        # deal with complex results
        r_coeffs = np.real(freqs)
        i_coeffs = np.imag(freqs)
        coeffs = np.concatenate([r_coeffs, i_coeffs])

        self.data_shape = np.shape(coeffs)

        return coeffs

    def get_values(self, coeffs):
        # reconstruct the data
        s_notrend = irfft(coeffs)
        if self.detrend:
            series = self.retrend_array(s_notrend)
        else:
            series = s_notrend

        return series

    def coeff_to_array(self, coeffs):
        return np.ravel(coeffs)

    def array_to_coeff(self, array):
        coeffs = np.reshape(array, self.data_shape)
        # convert back to complex numbers
        split = int(len(array) / 2)
        r_coeffs = np.array(array[:split])
        i_coeffs = np.array(array[split:])
        coeffs = np.vectorize(complex)(r_coeffs, i_coeffs)
        return coeffs

# -----------------------------------

# HFFT - Hermitian Fast Fourier Transform

class hfft_wavelet(base_wavelet):
    def get_coeffs(self, data: np.array) -> np.array:

        if self.detrend:
            x = self.detrend_array(data)
        else:
            x = data

        coeffs = hfft(x)

        self.data_shape = np.shape(coeffs)

        return coeffs

    def get_values(self, coeffs):
        # reconstruct the data
        series = ihfft(coeffs)

        if self.detrend:
            series = self.retrend_array(series)

        series = np.abs(series)

        return series

    def coeff_to_array(self, coeffs):
        return np.ravel(coeffs)

    def array_to_coeff(self, array):
        coeffs = np.reshape(array, self.data_shape)
        return coeffs

# -----------------------------------

# FHT - Fast Hankel Transform

dln = None
class fht_wavelet(base_wavelet):
    def get_coeffs(self, data: np.array) -> np.array:

        if self.detrend:
            x = self.detrend_array(data)
        else:
            x = data


        # compute the fht
        self.dln = np.log(x[1]/x[0]) 
        coeffs = fht(x, mu=0.0, dln=self.dln)

        return coeffs

    def get_values(self, coeffs):
        series = ifht(coeffs, dln=self.dln, mu=0)

        if self.detrend:
            series = self.retrend_array(series)

        return series

    def coeff_to_array(self, coeffs):
        self.data_shape = np.shape(coeffs)
        array = coeffs.flatten()
        return array

    def array_to_coeff(self, array):
        series = array.reshape(self.data_shape)
        return series

# -----------------------------------

# MODWT - Maximal Overlap Discrete Wavelet Transform

class modwt_wavelet(base_wavelet):
    def get_coeffs(self, data: np.array) -> np.array:

        if self.detrend:
            x = self.detrend_array(data)
        else:
            x = data

        # data must be of even length, so trim if necessary
        if (len(x) % 2) != 0:
            x = x[1:]

        # get the coefficients
        # self.wavelet = 'db8'
        self.wavelet = 'haar'
        # self.wavelet = 'bior3.9' # does not work well with MODWT
        level = 5
        coeffs = modwt(x, self.wavelet, level)
        return coeffs

    def get_values(self, coeffs):
        series = imodwt(coeffs, self.wavelet)

        if self.detrend:
            series = self.retrend_array(series)

        return series

    def coeff_to_array(self, coeffs):
        array = coeffs.flatten()
        self.data_shape = np.shape(coeffs)
        return array

    def array_to_coeff(self, array):
        series = array.reshape(self.data_shape)
        return series

# -----------------------------------

# SWT - Standing Wave Transform

class swt_wavelet(base_wavelet):
    def get_coeffs(self, data: np.array) -> np.array:

        if self.detrend:
            x = self.detrend_array(data)
        else:
            x = data

        # data must be of even length, so trim if necessary
        if (len(x) % 2) != 0:
            x = x[1:]

        self.wavelet = 'bior3.9'
        self.coeff_format = "wavedec"

        # (cA2, cD2), (cA1, cD1) = pywt.swt(data, wavelet, level=2)
        # swt returns an array, with each element being 2 arrays - cA_n and cD_n, where n is the level
        levels = min(2, pywt.swt_max_level(len(x)))
        coeffs = pywt.swt(x, self.wavelet, level=levels, trim_approx=True)
        return coeffs

    def get_values(self, coeffs):
        series = pywt.iswt(coeffs, wavelet=self.wavelet)

        if self.detrend:
            series = self.retrend_array(series)

        return series

    def coeff_to_array(self, coeffs):
        return super().coeff_to_array(coeffs)

    def array_to_coeff(self, array):
        return super().array_to_coeff(array)
# -----------------------------------

# WPT - Wavelet Packet Transform

wp = None
nodes = None

class wpt_wavelet(base_wavelet):
    def get_coeffs(self, data: np.array) -> np.array:

        if self.detrend:
            x = self.detrend_array(data)
        else:
            x = data

        # # data must be of even length, so trim if necessary
        # if (len(x) % 2) != 0:
        #     x = x[1:]

        self.wavelet = 'bior3.9'
        self.coeff_format = "wavedec"

        level = 1

        self.wp = pywt.WaveletPacket(data=x, wavelet=self.wavelet,  maxlevel=level, mode='symmetric')

        self.nodes = self.wp.get_level(level, 'natural')
        coeffs = self.nodes
        return coeffs

    def get_values(self, coeffs):
        # set the coefficients of the nodes at level 2
        for node, chunk in zip(self.nodes, coeffs):
            node.data = chunk

        series = self.wp.reconstruct(update=True)

        if self.detrend:
            series = self.retrend_array(series)

        return series

    def coeff_to_array(self, coeffs):
        array = np.concatenate([coeff.data for coeff in coeffs])
        return array

    def array_to_coeff(self, array):
        coeffs = np.array_split(array, len(self.nodes))
        return coeffs

# -----------------------------------


# enum of all available wavelet types

class WaveletType(Enum):
    DWT = dwt_wavelet
    DWTA = dwta_wavelet
    FFT = fft_wavelet
    HFFT = hfft_wavelet
    FHT = fht_wavelet
    MODWT = modwt_wavelet
    SWT = swt_wavelet
    WPT = wpt_wavelet


def make_wavelet(wavelet_type: WaveletType) -> base_wavelet:
    return wavelet_type.value()

# -----------------------------------
