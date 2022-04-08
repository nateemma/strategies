from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np
import scipy.fft


class RollingFFT(BaseEstimator, TransformerMixin):
    """Rolling FFT

    Calculate a FFT transform of data
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
        return f"RollingFFT(window={self.window}, mode={self.mode})"

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

    def model(self, X):
        """Models the data using scaling and FFT.

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
        # if not self.__fitted__:
        #     self.fit(X)
        self.fit(X)

        # scale the input data
        standardized = X.copy()
        # standardized = X.fillna(self.w_mean)
        scaled = (standardized - self.w_mean) / self.w_std
        scaled.fillna(0, inplace=True)

        # model using a FFT
        model = self.FFTModel(scaled)

        ldiff = len(model) - len(self.w_std)

        # unscale
        if ldiff != 0:
            # print("len(model):", len(model), " len(w_std):", len(self.w_std))
            return (model[ldiff:] * self.w_std) + self.w_mean
        else:
            return (model * self.w_std) + self.w_mean


    def FFTModel(self, data):

        n = len(data)
        x = np.array(data)

        # yf = scipy.fft.rfft(x)
        #
        # # zero out frequencies beyond 'cutoff'
        # fft_cutoff = 0.5 # TODO: set to stddev?
        # cutoff: int = int(len(yf) * fft_cutoff)
        # yf[(cutoff - 1):] = 0
        #
        # # inverse transform
        # restored_sig = scipy.fft.irfft(yf)


        # compute the fft
        fft = scipy.fft.fft(x, n)

        # compute power spectrum density
        # squared magnitude of each fft coefficient
        psd = fft * np.conj(fft) / n
        threshold = 20
        fft = np.where(psd<threshold, 0, fft)

        # inverse fourier transform
        restored_sig = scipy.fft.ifft(fft)

        restored_sig = restored_sig.real

        ldiff = len(restored_sig) - len(x)
        model = restored_sig[ldiff:]

        return model