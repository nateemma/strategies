# Utilities for de-/re-trending a signal

# Note that there are several ways to detrend a signal. However, the best ways are also compute intensive, or cause a delay
# in the signal. There are several techniques implemented here so that performance can be compared

# Remember that the detrend and retrend calls are paired - i.e. you must call detrend before calling retrend, and you cannot repeatedly 
# call detrend and then expect to retrend (only the last detrend will apply)

from enum import Enum, auto
from abc import ABC, abstractmethod

import numpy as np
import scipy.signal as signal
import pywt
from sklearn.preprocessing import RobustScaler
from scipy.signal import savgol_filter

import statsmodels.tsa.api as tsa
from statsmodels.tsa.ar_model import ar_select_order

class base_detrender(ABC):

    poly = None

    def __init__(self):
        super().__init__()

    # function to detrend the supplied 1d signal
    @abstractmethod
    def detrend_1d(self, data: np.array) -> np.array:
        # base implementation is to just returns the original
        return data

    # function to retrend the supplied 1d signal
    @abstractmethod
    def retrend_1d(self, data: np.array) -> np.array:
        # base implementation is to just returns the original
        return data    # function to detrend the supplied 1d signal

    # detrend 1d or 2d array
    def detrend(self, data: np.array) -> np.array:

        if data.ndim == 1:
            return self.detrend_1d(data)
        elif data.ndim == 2:
            ncols = np.shape(data)[1]
            x_detrend = np.zeros(np.shape(data), dtype=float)
            for i in range(ncols):
                col = np.array(data[:, i])
                x_detrend[:,i] = self.detrend_1d(col)
            return x_detrend
        else:
            print(f'    *** ERR: too many dimensions: {np.shape(data)}')
        return data

    # retrend 1d or 2d array
    def retrend(self, data: np.array) -> np.array:

        if data.ndim == 1:
            return self.retrend_1d(data)
        elif data.ndim == 2:
            ncols = np.shape(data)[1]
            # print(f'    re-trending {ncols} cols')
            x_trend = np.zeros(np.shape(data), dtype=float)
            for i in range(ncols):
                col = np.array(data[:, i])
                x_trend[:,i] = self.retrend_1d(col)
            return x_trend
        else:
            print(f'    *** ERR: too many dimensions: {np.shape(data)}')
        return data

    # return the 'trend' signal. Note that this can be None in some cases
    def get_trend(self) -> np.array:
        return self.poly



    # 'extend' the trend polynomial to support predicted values

    results_forecaster = None

    def extend_trend(self, steps):


        N = len(self.poly)


        # # Create an instance of the AutoReg class with the selected lag order
        # mod = ar_select_order(self.poly, maxlag=6)
        # self.model = tsa.AutoReg(self.poly, lags=mod.ar_lags).fit()
        self.model = tsa.AutoReg(self.poly, lags=8).fit()
        self.model = tsa.AutoReg(self.poly, lags=steps).fit()

        # self.model = tsa.SimpleExpSmoothing(self.poly).fit()

        # Predict the time series forward N steps
        preds = self.model.predict(N, N + steps - 1)[-steps:]

        # t = np.arange(0, len(self.poly))
        # coeff = np.polyfit(t, self.poly, 2)
        # poly = np.poly1d(coeff)
        # t2 = np.arange(N+1, N+steps)
        # preds = poly(t2)

        poly = np.concatenate((self.poly, preds), dtype=float)
        self.poly = poly[-N:]
        return


#---------------------------------------

# 'null' detrender - useful for use as a baseline while testing other detrenders

class null_detrender(base_detrender):
    def detrend_1d(self, data: np.array) -> np.array:
        # just returns the original
        return data

    # function to retrend the supplied signal
    def retrend_1d(self, data: np.array) -> np.array:
        # just returns the original
        return data

#---------------------------------------

class differencing_detrender(base_detrender):

    x_orig = 0.0

    def detrend_1d(self, data: np.array) -> np.array:
        x_detrend = np.zeros(len(data), dtype=float)
        self.x_orig = data[0]
        for i in range(1, len(data)):
            x_detrend[i] = data[i] - data[i - 1]

        self.poly = data - x_detrend
        return x_detrend

    def retrend_1d(self, data: np.array) -> np.array:
        x_trend = np.zeros(len(data), dtype=float)
        x_trend[0] = self.x_orig
        for i in range(1, len(data)):
            x_trend[i] = data[i] + x_trend[i - 1]
        return x_trend

#---------------------------------------


class linear_detrender(base_detrender):

    def detrend_1d(self, data: np.array) -> np.array:
        t = np.arange(0, len(data))
        coeff = np.polyfit(t, data, 1)
        # self.poly = np.poly1d(coeff)
        self.poly = np.polyval(coeff, data)
        x_detrend = data - self.poly
        return x_detrend

    def retrend_1d(self, data: np.array) -> np.array:
        # polynomial can be different length because training data is usually larger than prediction data. So, use last portion
        dlen = min(len(data), len(self.poly))
        x_trend = data
        x_trend[-dlen:] = data[-dlen:] + self.poly[-dlen:]
        return x_trend

#---------------------------------------


class quadratic_detrender(base_detrender):

    def detrend_1d(self, data: np.array) -> np.array:
        # t = np.linspace(0, 1, len(data))
        # coeff = np.polyfit(data, t, 4)
        # self.poly = np.polyval(coeff, data)

        window = max(8, len(data) // 2)
        # window = 8
        self.poly = savgol_filter(data, window, 2)

        x_detrend = data - self.poly
        return x_detrend

    def retrend_1d(self, data: np.array) -> np.array:
        dlen = min(len(data), len(self.poly))
        x_trend = data
        x_trend[-dlen:] = data[-dlen:] + self.poly[-dlen:]
        return x_trend

#---------------------------------------


class smooth_detrender(base_detrender):

    def detrend_1d(self, data: np.array) -> np.array:
        # window = max(8, len(data) // 4)
        window = 12
        box = np.ones(window) / window
        self.poly = np.convolve(data, box, mode="same")
        x_detrend = data - self.poly
        # print(f'data:{np.shape(data)} poly:{np.shape(self.poly)} x_detrend:{np.shape(x_detrend)} ')
        # np.set_printoptions(precision=3)
        # print('----------------------')
        # print(f'data:      {data}')
        # print(f'poly:      {self.poly}')
        # print(f'x_detrend: {x_detrend}')
        return x_detrend

    def retrend_1d(self, data: np.array) -> np.array:
        dlen = min(len(data), len(self.poly))
        x_trend = data
        x_trend[-dlen:] = data[-dlen:] + self.poly[-dlen:]
        # print(f'data:    {data}')
        # print(f'poly:    {self.poly}')
        # print(f'x_trend: {x_trend}')
        return x_trend

#---------------------------------------


class scaler_detrender(base_detrender):

    def detrend_1d(self, data: np.array) -> np.array:
        x = np.array(data)
        if x.ndim == 1:
            x = x.reshape(-1,1)
        x_trend = np.nan_to_num(x)


        if self.scaler is None:
            self.scaler = RobustScaler()

        self.scaler = self.scaler.fit(x_trend)
        x_detrend = self.scaler.transform(x_trend)
        if data.ndim == 1:
            x_detrend = x_detrend.reshape(-1).squeeze()

        self.poly = data - x_detrend

        return x_detrend

    # function to retrend the supplied signal
    def retrend_1d(self, data: np.array) -> np.array:

        x_trend = self.scaler.inverse_transform(data.reshape(-1,1))
        if data.ndim == 1:
            x_trend = x_trend.reshape(-1).squeeze()
        return x_trend

#---------------------------------------


class fft_detrender(base_detrender):

    def detrend_1d(self, data: np.array) -> np.array:

        xf = np.fft.fft(data) # FFT of signal
        xf[4:] = 0.0
        self.poly = np.fft.ifft(xf).real
        x_detrend = data - self.poly

        return x_detrend

    # function to retrend the supplied signal
    def retrend_1d(self, data: np.array) -> np.array:
        dlen = min(len(data), len(self.poly))
        x_trend = data
        x_trend[-dlen:] = data[-dlen:] + self.poly[-dlen:]
        return x_trend

#---------------------------------------


class dwt_detrender(base_detrender):

    wavelet = 'db4'
    # wavelet = 'haar'
    # level = 8
    level = 2
    coeffs = None

    def detrend_1d(self, data: np.array) -> np.array:
        # Choose a wavelet function and a level of decomposition


        # Perform the multilevel DWT
        coeffs = pywt.wavedec(data, self.wavelet, level=self.level, mode='per')

        # Save a copy of the original coefficients
        self.coeffs = coeffs.copy()

        # Apply soft thresholding to the coefficients
        threshold = 0.99 # adjust this value to control the amount of denoising
        thresh = threshold*np.nanmax(data)
        coeffs[1:] = [pywt.threshold(c, thresh, 'soft') for c in coeffs[1:]]

        # Perform the multilevel IDWT on the modified coefficients
        self.poly = pywt.waverec(coeffs, self.wavelet, mode='per')

        if (len(data) != len(self.poly)):
            dlen = min(len(data), len(self.poly))
            self.poly = self.poly[-dlen:]

        x_detrend = data - self.poly

        return x_detrend

    # function to retrend the supplied signal
    def retrend_1d(self, data: np.array) -> np.array:
        # Add the trend component to the supplied values
        # x_trend = data + pywt.waverec([self.coeffs[0], *[np.zeros_like(c) for c in self.coeffs[1:]]], self.wavelet)[len(data):]

        dlen = min(len(data), len(self.poly))
        x_trend = data
        x_trend[-dlen:] = data[-dlen:] + self.poly[-dlen:]
        return x_trend

#---------------------------------------

class DetrenderType(Enum):
    NULL         = null_detrender
    DIFFERENCING = differencing_detrender
    LINEAR       = linear_detrender
    QUADRATIC    = quadratic_detrender
    SMOOTH       = smooth_detrender
    SCALER       = scaler_detrender
    FFT          = fft_detrender
    DWT          = dwt_detrender


# (static) function to create a detrender of the specified type
def make_detrender(detrender_type: DetrenderType) -> base_detrender:
    return detrender_type.value()