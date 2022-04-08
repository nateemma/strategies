from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np
from simdkalman import KalmanFilter


class RollingKalmanSIMD(BaseEstimator, TransformerMixin):
    """Rolling Kalman

    Calculate a Kalman filter (SIMD version)  of data
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
        return f"RollingKalmanSIMD(window={self.window}, mode={self.mode})"

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

    def model(self, X, kfilter: KalmanFilter):
        """Models the data using scaling and Kalman.

        Scale features of X according to the window mean and standard
        deviation.

        Paramaters
        ----------
        X : array-like of shape (n_shape, n_features)
            Input data that will be transformed.

        kfilter: kalman filter object.
                 Passed as an arg because it should be re-used for each window, and a different filter
                 should be used for each data set (not info to do that here)

        Returns
        -------
        standardized : array-like of shape (n_shape, n_features)
            Transformed data.
        """
        # re-fit for this window
        self.fit(X)

        # scale the input data
        standardized = X.copy()
        # standardized = X.fillna(self.w_mean)
        scaled = (standardized - self.w_mean) / self.w_std
        scaled.fillna(0, inplace=True)

        # model using a Kalman
        model = self.KalmanModel(scaled, kfilter=kfilter)

        ldiff = len(model) - len(self.w_std)

        # unscale
        if ldiff != 0:
            # print("len(model):", len(model), " len(w_std):", len(self.w_std))
            return (model[ldiff:] * self.w_std) + self.w_mean
        else:
            return (model * self.w_std) + self.w_mean


    def KalmanModel(self, data, kfilter: KalmanFilter):

        n = len(data)
        x = np.array(data)

        # kfilter = kfilter.em(x, n_iter=6)

        r = kfilter.compute(data, 1);
        smoothed = r.smoothed.states.mean[:,0];
        pred = r.predicted.observations.mean

        # add predictions to the end of the model
        # restored_sig = np.append(smoothed,  pred)
        restored_sig = smoothed

        ldiff = len(restored_sig) - len(x)
        model = restored_sig[ldiff:]

        return model