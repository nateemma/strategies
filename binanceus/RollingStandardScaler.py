from sklearn.base import BaseEstimator, TransformerMixin


class RollingStandardScaler(BaseEstimator, TransformerMixin):
    """Rolling standard Scaler

    Standardized the given data series using the mean and std
    commputed in rolling or expanding mode.

    Parameters
    ----------
    window : int
        Number of periods to compute the mean and std.
    mode : str, optional, default: 'rolling'
        Mode

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
        return f"RollingStandardScaler(window={self.window}, mode={self.mode})"

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

    def transform(self, X):
        """Transforms.

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
        self._check_fitted()

        standardized = X.copy()
        return (standardized - self.w_mean) / self.w_std

    def inverse_transform(self, X):
        """Inverse transform

        Undo the transform operation

        Paramaters
        ----------
        X : array-like of shape (n_shape, n_features)
            Input data that will be transformed.

        Returns
        -------
        standardized : array-like of shape (n_shape, n_features)
            Transformed (original) data.
        """
        self._check_fitted()

        unstandardized = X.copy()
        return (unstandardized * self.w_std) + self.w_mean

    def _check_fitted(self):
        """ Checks if the algorithm is fitted. """
        if not self.__fitted__:
            raise ValueError("Please, fit the algorithm first.")