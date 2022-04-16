from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np

import pywt

class RollingDWT(BaseEstimator, TransformerMixin):
    """Rolling sDWT

    Calculate a DWT transform of data
    commputed in rolling or expanding mode.

    Parameters
    ----------
    window : int
        Number of periods to compute the mean and std.
    mode : str, optional, default: 'rolling' Mode

    Attributes
    ----------
    pd_object : pandas.Rolling
        Pandas window object.
    w_mean : pandas.Series
        Series of mean values.
    w_std : pandas.Series
        Series of std. values.
    """

    def __init__(self, window, mode='rolling'):
        self.window = window
        self.mode = mode

        # to fill in code
        self.pd_object = None
        self.w_mean = None
        self.w_std = None
        self.__fitted__ = False

    def __repr__(self):
        return f"RollingDWT(window={self.window}, mode={self.mode})"

    def fit(self, X, y=None):
        """Fits.

        Computes the mean and std to be used for later scaling.

        Parameters
        ----------
        X : array-like of shape (n_shape, n_features)
            The data used to compute the per-feature mean and std. Used for
            later scaling along the feature axis.
        y
            Ignored.
        """
        self.pd_object = getattr(X, self.mode)(self.window)
        self.w_mean = self.pd_object.mean()
        self.w_std = self.pd_object.std()
        self.__fitted__ = True

        return self

    def scaledModel(self, X):
        """Models the data using scaling and DWT.

        Scale features of X according to the window mean and standard
        deviation.

        Paramaters
        ----------
        X : array-like of shape (n_shape, n_features)
            Input data that will be transformed.

        Returns
        -------
        standardized : array-like of shape (n_shape, n_features)
            Transformed data.
        """
        self.fit(X)

        # scale the input data
        standardized = X.copy()
        scaled = (standardized - self.w_mean) / self.w_std
        scaled.fillna(0, inplace=True)

        # model using a DWT
        dwt_model = self.dwtModel(scaled)

        ldiff = len(dwt_model) - len(self.w_std)

        return (dwt_model[ldiff:])

    def model(self, X):
        """Models the data using scaling and DWT.

        Scale features of X according to the window mean and standard
        deviation.

        Paramaters
        ----------
        X : array-like of shape (n_shape, n_features)
            Input data that will be transformed.

        Returns
        -------
        standardized : array-like of shape (n_shape, n_features)
            Transformed data.
        """
        self.fit(X)

        # scale the input data
        standardized = X.copy()
        scaled = (standardized - self.w_mean) / self.w_std
        scaled.fillna(0, inplace=True)

        # model using a DWT
        dwt_model = self.dwtModel(scaled)

        ldiff = len(dwt_model) - len(self.w_std)

        # unscale
        if ldiff != 0:
            # print("len(dwt_model):", len(dwt_model), " len(w_std):", len(self.w_std))
            return (dwt_model[ldiff:] * self.w_std) + self.w_mean
        else:
            return (dwt_model * self.w_std) + self.w_mean



    def madev(self, d, axis=None):
        """ Mean absolute deviation of a signal """
        return np.mean(np.absolute(d - np.mean(d, axis)), axis)


    def dwtModel(self, data):

        # the choice of wavelet makes a big difference
        # for an overview, check out: https://www.kaggle.com/theoviel/denoising-with-direct-wavelet-transform
        # wavelet = 'db1'
        # wavelet = 'bior1.1'
        wavelet = 'haar' # deals well with harsh transitions
        level = 1
        wmode = "smooth"
        length = len(data)

        # # de-trend the data
        # n = data.size
        # t = np.arange(0, n)
        # p = np.polyfit(t, data, 1)  # find linear trend in data
        # x_notrend = data - p[0] * t  # detrended data
        #
        # coeff = pywt.wavedec(x_notrend, wavelet, mode=wmode)

        # coeff = pywt.wavedec(data, wavelet, mode=wmode)
        #
        # # remove higher harmonics
        # sigma = (1 / 0.6745) * self.madev(coeff[-level])
        # uthresh = sigma * np.sqrt(2 * np.log(length))
        # coeff[1:] = (pywt.threshold(i, value=uthresh, mode='hard') for i in coeff[1:])
        #
        # # inverse transform
        # restored_sig = pywt.waverec(coeff, wavelet, mode=wmode)

        (ca, cd) = pywt.dwt(data, wavelet)

        # siga = 0.6745 * self.madev(ca)
        # sigd = 0.6745 * self.madev(cd)
        # scale = np.sqrt(2 * np.log(length)) / 2.0
        # tha = siga * scale
        # thd = sigd * scale

        tha2 = np.std(ca)/2.0
        thd2 = np.std(cd)/2.0

        # print ("tha:", tha, "thd:", thd, " tha2:", tha2, "thd2:", thd2)

        cat = pywt.threshold(ca, tha2, mode='soft')
        cdt = pywt.threshold(cd, thd2, mode='soft')

        restored_sig = pywt.idwt(cat, cdt, wavelet)

        # print("l1:", len(data), " l2:", len(restored_sig))

        # for some reason, returns longer array (by +1)
        # # re-trend the data
        # model = restored_sig[1:] + p[0] * t

        ldiff = len(restored_sig) - len(data)
        model = restored_sig[ldiff:]

        return model