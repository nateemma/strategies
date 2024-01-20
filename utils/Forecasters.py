"""
This class is factory for creating various kinds of Forecasters with a consistent interface

Note that these are based on both statsmodel and sklearn packages.
     The statsmodel algorithms only support 1d data and are one-shot estimators, there is no incremental training
     The sklearn algorithms typically do support multi-dimensional data and some support incremental training
"""


# -----------------------------------


# base class - to allow generic treatment of all forecasters

from abc import ABC, abstractmethod
from enum import Enum

import numpy as np
import pandas as pd
from sklearn.model_selection import GridSearchCV
import statsmodels.tsa.api as tsa
from statsmodels.tsa.forecasting.theta import ThetaModel
from statsmodels.tsa.ar_model import ar_select_order
from xgboost import XGBRegressor

import lightgbm as lgbm
from sklearn.linear_model import PassiveAggressiveRegressor
from sklearn.linear_model import SGDRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.cluster import MiniBatchKMeans
from sklearn.neural_network import MLPRegressor
# from scipy.fft import fft, hfft, ifft, ihfft, rfft, irfft, fftfreq
# -----------------------------------

import time


# Define a timer decorator function
def timer(func):
    # Define a wrapper function
    def wrapper(*args, **kwargs):
        # Record the start time
        start = time.time()
        # Call the original function
        result = func(*args, **kwargs)
        # Record the end time
        end = time.time()
        # Calculate the duration
        duration = end - start
        # Print the duration
        print(f"{func.__name__} took {duration} seconds to run.")
        # Return the result
        return result

    # Return the wrapper function
    return wrapper

from sklearn.base import BaseEstimator

# Estimator that does nothing. For use where forecaster does not have an underlying model
class NullRegressor(BaseEstimator):
    def __init__(self):
        pass

    def fit(self, X, y=None):
        # Do nothing
        return self

    def predict(self, X):
        # Return the input
        return X

# -----------------------------------

# all actual instantiations follow this base class


class base_forecaster(ABC):
    # the following are used to store info needed across different calls
    model = None
    train_data: np.array = None
    fitted_data: np.array = None
    results = None
    support_multiple_columns = False
    support_retrain = False
    requires_training = False
    detrend_data = False
    smooth_data = False
    smooth_window = 4

    def __init__(self):
        super().__init__()
        self.model = None

    # function to get the name of the forecaster (for debug/display)
    @abstractmethod
    def get_name(self):
        return "ABC"

    # function to get the underlying model (e.g. to save)
    def get_model(self):
        self.create_model()
        return self.model

    # function to specify whether or not to detrend data before training/forecasting
    def set_detrend(self, detrend_flag=True):
        self.detrend_data = detrend_flag
        return

    # function to specify whether or not to smooth data before training/forecasting
    def set_smooth(self, smooth_flag=True, smooth_window=4):
        self.smooth_data = smooth_flag
        self.smooth_window = smooth_window
        return

    # function to set the underlying model (e.g. if loaded from a file)
    def set_model(self, model):
        self.model = model
        return

    # function to create the underlying model (only applies to sklearn-based forecasters)
    def create_model(self):
        return

    # specifies whether the algorithm supports multidiemnsional data (default is False)
    def supports_multiple_columns(self) -> bool:
        return self.support_multiple_columns

    # specifies whether the algorithm supports multidiemnsional data (default is False)
    def supports_retrain(self) -> bool:
        return self.support_retrain

    # sepcifies whether forecaster requires pre-treaining
    def requires_pretraining(self):
        return self.requires_training

    # function to train based on known results. Not all forecasters support this.
    def train(self, train_data: np.array, results: np.array, incremental=True):
        return

    # function to forecast the supplied data N steps into the future
    @abstractmethod
    def forecast(self, data: np.array, steps) -> np.array:
        # base implementation is to just return zeros
        return np.zeros(steps, dtype=float)


# -----------------------------------

# convert an array to 1d, if needed
def check_1d(array: np.array) -> np.array:
    if array.ndim > 1:
        return array[:, 0]
    else:
        return array

# -----------------------------------
# de-trend and re-trend data
# currently only works for 1d

trend_line = None


def detrend(x, axis=0):
    global trend_line
    x = np.nan_to_num(x)

    t = np.arange(0, len(x))
    trend_poly = np.polyfit(t, x, 1)
    trend_line = np.polyval(trend_poly, x)
    x_notrend = x - trend_line

    return x_notrend


def retrend(x_trend, axis=0):
    global trend_line
    return x_trend + trend_line

# -----------------------------------

def smooth(y, window):
    box = np.ones(window) / window
    y_smooth = np.convolve(y, box, mode="same")
    # Hack: constrain to 3 decimal places (should be elsewhere, but convenient here)
    y_smooth = np.round(y_smooth, decimals=3)
    return np.nan_to_num(y_smooth)


#------------------------------

# Null forecaster - doesn't do anything. Intended for verifying general logic


class null_forecaster(base_forecaster):
    def get_name(self):
        return "Null"

    def forecast(self, data: np.array, steps) -> np.array:
        x = check_1d(data)

        predictions = x

        self.model = NullRegressor() # just have something not None that can be called

        return predictions


# -----------------------------------

# Simple Linear Regression. Probably not a good approach but useful for comparisons


class linear_forecaster(base_forecaster):
    def get_name(self):
        return "Linear"

    def forecast(self, data: np.array, steps) -> np.array:

        x = check_1d(data)

        N = len(x)

        # smooth
        if self.smooth_data:
            x = smooth(x, self.smooth_window)

        # de-trend
        if self.detrend_data:
            x = detrend(x)

        # fit the supplied data
        t = np.arange(N)
        coeffs = np.polyfit(t, x, 1)

        # predict forward N steps
        t = np.arange(N, N+steps)
        x_pred = np.polyval(coeffs, t)

        # resize array to initial size
        predictions = np.concatenate((x, x_pred))[-N:]
        # x_pred = x_pred[-N:]

        # re-trend
        if self.detrend_data:
            predictions = retrend(predictions)

        # print(f'    data: {data.squeeze()}')
        # print(f'    preds:{predictions}')

        self.model = NullRegressor() # just have something not None that can be called

        return predictions


# -----------------------------------


class quadratic_forecaster(base_forecaster):
    def get_name(self):
        return "Quadratic"

    def forecast(self, data: np.array, steps) -> np.array:

        x = check_1d(data)

        N = len(x)

        # smooth
        if self.smooth_data:
            x = smooth(x, self.smooth_window)

        # de-trend
        if self.detrend_data:
            x = detrend(x)

        # fit the supplied data
        t = np.arange(N)
        coeffs = np.polyfit(t, x, 3)

        # predict forward N steps
        t = np.arange(N, N+steps)
        # t = np.arange(0, N+steps)
        x_pred = np.polyval(coeffs, t)

        # build array of initial size
        predictions = np.concatenate((x, x_pred), dtype=float)[-N:]

        # re-trend
        if self.detrend_data:
            predictions = retrend(predictions)

        self.model = NullRegressor() # just have something not None that can be called

        return predictions




# -----------------------------------


# performs an FFT transform, moves each frequency bin forward N steps and does the inverse transform
class fft_extrapolation_forecaster(base_forecaster):

    filter_type = 4
    predict_type = 2

    def get_name(self):
        return "FFT Extrapolation"

    def forecast(self, data: np.array, steps) -> np.array:

        y = check_1d(data)

        # print(f'y:{np.shape(y)}')

        N = y.size # number of data points

        # smooth
        if self.smooth_data:
            y = smooth(y, self.smooth_window)

        # de-trend
        if self.detrend_data:
            y = detrend(y)

        # apply FFT
        yf = np.fft.fft(y) # FFT of signal
        # yf = np.concatenate((yf[:(N+1)//2], np.zeros(N), yf[(N+1)//2:])) # zero-padding for higher resolution

        yf_filt = self.filter_freqs(yf, filter_type=self.filter_type)

        y_pred = self.predict_freqs(yf_filt, steps, predict_type=self.predict_type)

        # apply IFFT
        y_pred_ifft = np.fft.ifft(y_pred) # IFFT of predicted values
        y_pred_ifft = np.real(y_pred_ifft) # real part of IFFT values

        predictions = np.concatenate((y, y_pred_ifft))[-len(y):]

        # re-trend
        if self.detrend_data:
            predictions = retrend(predictions)

        self.model = NullRegressor() # just have something not None that can be called

        return predictions[-N:]

    # utility to filter out frequencies (various methods)
    def filter_freqs(self, yf, filter_type=0):

        N = yf.size # number of data points

        if filter_type == 1:
            # simply remove higher frequencies
            yf_filt = yf
            index = max(4, int(N/2))
            yf_filt[index:-index] = 0.0

            # print(f'N:{N} yf_filt2:{yf_filt}')
        elif filter_type == 2:
            # bandpass filter

            # select frequencies and coefficients
            # f1, f2 = 5, 20 # frequencies in Hz
            f1, f2 = 0, 0.08 # cutoff frequencies in Hz
            # fr = np.fft.fftfreq(2*N, 1) # frequency array
            fr = np.fft.fftfreq(N, 1) # frequency array
            fr_sort = np.sort(np.abs(fr))
            f1 = 0.0
            f2 = fr_sort[-N//4]
            # print(f'mean:{fr_mean} median:{fr_med}')
            df = 0.08 # bandwidth of bandpass filter
            gpl = np.exp(-((fr-f1)/(2*df))**2) + np.exp(-((fr-f2)/(2*df))**2) # positive frequencies
            gmn = np.exp(-((fr+f1)/(2*df))**2) + np.exp(-((fr+f2)/(2*df))**2) # negative frequencies
            g = gpl + gmn # bandpass filter
            yf_filt = yf * g # filtered FFT coefficients

        elif filter_type == 3:
            # power spectrum filter
            # Compute the power spectrum
            ps = np.abs (yf)**2

            # Define a threshold for filtering
            # threshold = 100
            # threshold = np.mean(ps)
            threshold = np.sort(ps)[-N//2]

            # Apply a mask to the fft coefficients
            mask = ps > threshold
            yf_filt = yf * mask
            # print(f'yf_filt:{yf_filt}')

        elif filter_type == 4:
            # phase filter
            # Compute the phase spectrum
            phase = np.angle (yf)

            # Define a threshold for filtering
            threshold = np.pi / 2.0
            threshold = np.mean(np.abs(phase))

            # # sort phases, set threshold from end
            # threshold = np.sort(np.abs(phase))[-N//8]

            # Apply a mask to the phase spectrum
            mask = np.abs (phase) < threshold
            yf_filt = yf * mask
            # print(f'phase:{phase}')
            # print(f'yf_filt:{yf_filt}')

        else:
            # default is no filter at all
            yf_filt = yf

        return yf_filt


    # utility to predict frequency data (various methods)
    def predict_freqs(self, yf, steps, predict_type=0):

        N = yf.size # number of data points
        dt = 1.0 / 12.0
        fr = np.fft.fftfreq(2*N, dt) # frequency array

        if predict_type == 1:
            # extend sines and cosines
            t_pred = np.arange(N*dt, (N+steps)) # time array for prediction
            yf_pred = np.zeros(len(t_pred)) # array for predicted values
            for i in range(steps):
                # sum of complex exponentials
                yf_pred_new = yf_pred + yf[i] * (np.cos(2*np.pi*fr[i]*t_pred) + 1j*np.sin(2*np.pi*fr[i]*t_pred))
                yf_pred = yf_pred_new # work around complex casting
        elif predict_type == 2:
            # calculate conjugates

            # Extend the FFT coefficients by steps
            # The new coefficients are the complex conjugates of the previous ones
            yf_pred = np.concatenate((yf, yf[-2:0:-1].conj()))

        elif predict_type == 3:
            # AutoRegressor
            ar = make_forecaster(ForecasterType.AR)
            yf_pred_r = ar.forecast(yf.real, steps)
            # yf_pred_i= ar.forecast(yf.imag, steps)
            # yf_pred = yf_pred_r + yf_pred_i * 1j
            # yf_pred = ar.forecast(abs(yf), steps)
            yf_pred = yf_pred_r

        else:
            # default is no prediction at all
            yf_pred = yf

        return yf_pred
# -----------------------------------


class exponential_forecaster(base_forecaster):
    def get_name(self):
        return "Exponential"

    def forecast(self, data: np.array, steps) -> np.array:
 
        x = check_1d(data)

        # smooth
        if self.smooth_data:
            x = smooth(x, self.smooth_window)

        # detrend
        if self.detrend_data:
            x = detrend(x)

        self.model = tsa.ExponentialSmoothing(x).fit()
        predictions = self.model.predict(0, len(x) + steps)

        predictions = np.concatenate((x, predictions), dtype=float)[-len(x):]

        if self.detrend_data:
            predictions = retrend(predictions)

        return predictions
# -----------------------------------


class statespace_exponential_forecaster(base_forecaster):
    def get_name(self):
        return "State Space Exponential"

    def forecast(self, data: np.array, steps) -> np.array:

        x = check_1d(data)

        # smooth
        if self.smooth_data:
            x = smooth(x, self.smooth_window)

        # detrend
        if self.detrend_data:
            x = detrend(x)

        self.model = tsa.statespace.ExponentialSmoothing(x).fit(disp=False)
        predictions = self.model.predict(0, len(x) + steps)

        predictions = np.concatenate((x, predictions), dtype=float)[-len(x):]

        if self.detrend_data:
            predictions = retrend(predictions)

        return predictions


# -----------------------------------


class holt_forecaster(base_forecaster):
    def get_name(self):
        return "Holt"

    def forecast(self, data: np.array, steps) -> np.array:

        x = check_1d(data)


        # smooth
        if self.smooth_data:
            x = smooth(x, self.smooth_window)

        # detrend
        if self.detrend_data:
            x = detrend(x)

        N = len(x)
        self.model = tsa.Holt(x, damped_trend=True, initialization_method="estimated").fit()
        # predictions = self.model.predict(0, len(x) + steps)
        # predictions = self.model.predict(N, N + steps - 1)

        predictions = self.model.predict(start=N, end=N + steps - 1)

        predictions = np.concatenate((x, predictions), dtype=float)[-len(x):]

        if self.detrend_data:
            predictions = retrend(predictions)

        return predictions[-len(x):]


# -----------------------------------


class simple_exponential_forecaster(base_forecaster):
    def get_name(self):
        return "SimpleExponential"

    def forecast(self, data: np.array, steps) -> np.array:
 
        x = check_1d(data)


        # smooth
        if self.smooth_data:
            x = smooth(x, self.smooth_window)

        # detrend
        if self.detrend_data:
            x = detrend(x)

        self.model = tsa.SimpleExpSmoothing(x).fit()
        predictions = self.model.predict(0, len(x) + steps)

        predictions = np.concatenate((x, predictions), dtype=float)[-len(x):]

        if self.detrend_data:
            predictions = retrend(predictions)

        return predictions


# -----------------------------------


class ets_forecaster(base_forecaster):
    def get_name(self):
        return "ETS"

    def forecast(self, data: np.array, steps) -> np.array:

        x = check_1d(data)

        # smooth
        if self.smooth_data:
            x = smooth(x, self.smooth_window)

        # detrend
        if self.detrend_data:
            x = detrend(x)

        self.model = tsa.ETSModel(x).fit(disp=False)
        predictions = self.model.predict(0, len(x) + steps)

        predictions = np.concatenate((x, predictions), dtype=float)[-len(x):]

        if self.detrend_data:
            predictions = retrend(predictions)

        return predictions


# -----------------------------------


class arima_forecaster(base_forecaster):
    def get_name(self):
        return "ARIMA"

    def forecast(self, data: np.array, steps) -> np.array:
        try:  # ARIMA is very sensitive to zeroes in data, which happens a lot at startup

            y = check_1d(data)

            # smooth
            if self.smooth_data:
                y = smooth(y, self.smooth_window)

            # detrend
            if self.detrend_data:
                y = detrend(y)

            if self.model is None:
                 # Choose the order of the ARIMA model
                order = (1, 1, 1)  # AR(1), I(1), MA(1)
                # order = (0, 0, 1)  # zero order model for 1d
                # order = (2, 1, 2) 

                 # create the model and fit
                self.model = tsa.ARIMA(y, order=order)

            res = self.model.fit()
            start = len(data)

            # predictions = res.forecast(start, start + steps)
            # predictions = res.predict(0, start + steps, typ='levels')
            predictions = res.predict(0, start + steps)


            predictions = np.concatenate((y, predictions), dtype=float)[-len(y):]

            if self.detrend_data:
                predictions = retrend(predictions)

        except Exception as e:
            predictions = np.zeros(len(data), dtype=float)

        return np.array(predictions)

# -----------------------------------


class ar_forecaster(base_forecaster):
    def get_name(self):
        return "AutoReg"

    def forecast(self, data: np.array, steps) -> np.array:

        x = check_1d(data)

        # smooth
        if self.smooth_data:
            x = smooth(x, self.smooth_window)

        # detrend
        if self.detrend_data:
            x = detrend(x)

        N = len(x)

        # Use the ar_select_order function to choose the optimal lag order
        mod = ar_select_order(x, maxlag=15)
        # print(mod.ar_lags) # print the selected lag order

        # Create an instance of the AutoReg class with the selected lag order
        self.model = tsa.AutoReg(x, lags=mod.ar_lags).fit()
        # print(self.model.summary())

        # Predict the time series forward N steps
        preds = self.model.predict(N, N + steps - 1)

        # self.model = tsa.AutoReg(x, lags=steps).fit()
        # print(self.model.summary())
        # preds = self.model.predict(N, N + steps - 1)

        predictions = np.concatenate((x, preds), dtype=float)[-N:]
        if self.detrend_data:
            predictions = retrend(predictions)
        return predictions


# -----------------------------------


class theta_forecaster(base_forecaster):
    res = None

    def get_name(self):
        return "Theta"

    def forecast(self, data: np.array, steps) -> np.array:

        x = check_1d(data)

        # smooth
        if self.smooth_data:
            x = smooth(x, self.smooth_window)

        # detrend
        if self.detrend_data:
            x = detrend(x)

        if self.model is None:
            # Create a pandas date range with 5-minute frequency and the same length as the array
            # TODO: pass in pd.Series?
            dates = pd.date_range(start="2023-01-01", periods=len(x), freq="5m")
            y = pd.Series(x, index=dates)
            # y = pd.Series(data)

            # create the model and fit
            self.model = ThetaModel(y)
            # print(model.summary())

        # get a prediction
        res = self.model.fit()

        preds = res.forecast(steps, theta=1.2)
        predictions = np.concatenate((x, preds), dtype=float)[-len(x):]
        if self.detrend_data:
            predictions = retrend(predictions)
        return np.array(predictions)


# -----------------------------------

#############
## sklearn-based forecasters

class gb_forecaster(base_forecaster):
    reuse_model = True
    n_estimators = 100
    support_multiple_columns = True
    support_retrain = True
    requires_training = True

    def get_name(self):
        return "GradientBoost"

    def create_model(self):
        if self.model is None:
            params = {
                "n_estimators": 100,
                "max_depth": 4,
                "min_samples_split": 2,
                "learning_rate": 0.1,
                "loss": "squared_error",
                "warm_start": self.reuse_model,
            }
            # self.model = GradientBoostingRegressor(**params)
            self.n_estimators = 100
            self.model = GradientBoostingRegressor(n_estimators=self.n_estimators, warm_start=self.reuse_model)
        return

    def train(self, train_data: np.array, results: np.array, incremental=True):
        self.create_model()

        if self.detrend_data:
            train_data = detrend(train_data)
            # results = detrend(results)

        self.model.fit(train_data, results)
        # print(f'n_estimators_:{self.model.n_estimators_}')
        if self.reuse_model:
            self.n_estimators += 1
            self.model.set_params(warm_start=True, n_estimators=self.n_estimators)
        return

    def forecast(self, data: np.array, steps) -> np.array:
        if self.detrend_data:
            x = detrend(np.array(data))
        else:
            x = np.nan_to_num(data)
        predictions = self.model.predict(x)
        # if self.detrend_data:
        #     predictions = retrend(predictions.reshape(-1,1)).squeeze()
        return predictions


# -----------------------------------


class hgb_forecaster(base_forecaster):
    reuse_model = False
    support_multiple_columns = True
    support_retrain = False
    requires_training = True

    def get_name(self):
        return "HistogramGB"

    def create_model(self):
        if self.model is None:
            self.model = HistGradientBoostingRegressor(warm_start=self.reuse_model, n_iter_no_change=5)
        return

    def train(self, train_data: np.array, results: np.array, incremental=True):
        self.create_model()

        if self.detrend_data:
            train_data = detrend(train_data)
            # results = detrend(results)

        self.model.fit(train_data, results)

        return

    def forecast(self, data: np.array, steps) -> np.array:
        if self.detrend_data:
            x = detrend(np.array(data))
        else:
            x = np.nan_to_num(data)
        predictions = self.model.predict(x)
        # if self.detrend_data:
        #     predictions = retrend(predictions.reshape(-1,1)).squeeze()
        return predictions


# -----------------------------------


class kmeans_forecaster(base_forecaster):
    reuse_model = False
    support_multiple_columns = True
    support_retrain = True
    requires_training = True

    def get_name(self):
        return "Mini Batch KMeans"

    def create_model(self):
        if self.model is None:
            self.model = MiniBatchKMeans()
        return

    def train(self, train_data: np.array, results: np.array, incremental=True):
        self.create_model()
        if self.detrend_data:
            train_data = detrend(train_data)
            # results = detrend(results)

        # if you know that the data source is changing, set incremental to False
        if incremental:
            self.model.partial_fit(train_data, results)
        else:
            self.model.fit(train_data, results)

        return

    def forecast(self, data: np.array, steps) -> np.array:
        if self.detrend_data:
            x = detrend(np.array(data))
        else:
            x = np.nan_to_num(data)
        predictions = self.model.predict(x)
        # if self.detrend_data:
        #     predictions = retrend(predictions.reshape(-1,1)).squeeze()
        return predictions


# -----------------------------------


class lgbm_forecaster(base_forecaster):
    reuse_model = False
    support_multiple_columns = True
    support_retrain = False
    requires_training = True

    def get_name(self):
        return "LightGBM"


    def create_model(self):
        if self.model is None:
            self.model = lgbm.LGBMRegressor(objective="regression", metric="rmse", verbose=-1)
        return

    def train(self, train_data: np.array, results: np.array, incremental=True):


        """"""
        #Suggested by ChatGPT:
        params = {
            "objective": "regression",
            "metric": "rmse",
            "num_leaves": 31,
            "max_depth": -1,
            "learning_rate": 0.1,
            "min_child_samples": 1,
            "colsample_bytree": 0.9,
            "subsample": 0.9,
            "force_col_wise": True,
            "verbose": -1,
        }

        """
        # results from GridSearchCV:
        params = {
            "objective": "regression",
            "metric": "rmse",
            "learning_rate": 0.1,
            "colsample_bytree": 0.9,
            "max_depth": 3,
            "min_child_samples": 1,
            "min_data_in_bin": 1,
            "min_data_in_leaf": 1,
            "n_estimators": 500,
            "num_leaves": 7,
            "subsample": 0.5,
            "verbose": -1,
        }
        """

        lgbm_data = lgbm.Dataset(data=train_data, label=results, free_raw_data=True)
        # if self.model is None:
        #     # self.create_model()
        #     self.model = lgbm.train(params, train_set=lgbm_data, valid_sets=[lgbm_data], keep_training_booster=True)
        # else:
        #     print(f'type(self.model): {type(self.model)}')
        #     self.model.update(lgbm_data)

        self.model = lgbm.train(params, lgbm_data, valid_sets=[lgbm_data])

        return

    def find_params(self, train_data: np.array, results: np.array):
        # Define parameter grid
        param_grid = {
            "num_leaves": [7, 14, 21, 28, 31, 50],
            "learning_rate": [0.1],
            "max_depth": [-1, 3, 5, 7, 9],
            "n_estimators": [50, 100, 200, 500],
            "min_child_samples": [1, 10, 20, 50, 100],
            "colsample_bytree": [0.5, 0.7, 0.9, 1.0],
            "subsample": [0.5, 0.7, 0.9, 1.0],
            "min_data_in_bin": [1, 10, 20, 50, 100],
            "min_data_in_leaf": [1, 10, 20, 50, 100],
        }

        # Initialize GridSearchCV object
        grid = GridSearchCV(
            estimator=self.model, param_grid=param_grid, scoring="neg_root_mean_squared_error", cv=5, verbose=1
        )

        # Fit GridSearchCV object with training data
        grid.fit(train_data, results)

        # Print best parameters and best score
        print("***")
        print(grid.best_params_)
        print(grid.best_score_)
        print("***")
        return grid.best_params_

    def forecast(self, data: np.array, steps) -> np.array:
        predictions = self.model.predict(np.array(data))
        return predictions

# -----------------------------------


class mlp_forecaster(base_forecaster):
    reuse_model = True
    support_multiple_columns = True
    support_retrain = True
    requires_training = True

    def get_name(self):
        return "MLP"

    def create_model(self):
        if self.model is None:
            # layer_sizes=(32)
            # layer_sizes=(64)
            # layer_sizes=(128)
            layer_sizes=(64, 32)
            # layer_sizes=(64, 32, 16)
            self.model = MLPRegressor(hidden_layer_sizes=layer_sizes)
        return

    def train(self, train_data: np.array, results: np.array, incremental=True):
        self.create_model()
        if self.detrend_data:
            train_data = detrend(train_data)
            # results = detrend(results)

        # if you know that the data source is changing, set incremental to False
        if incremental:
            self.model.partial_fit(train_data, results)
        else:
            self.model.fit(train_data, results)

        return

    def forecast(self, data: np.array, steps) -> np.array:
        if self.detrend_data:
            x = detrend(np.array(data))
        else:
            x = np.nan_to_num(data)
        predictions = self.model.predict(x)
        # if self.detrend_data:
        #     predictions = retrend(predictions.reshape(-1,1)).squeeze()
        return predictions


# -----------------------------------


class pa_forecaster(base_forecaster):
    reuse_model = True
    support_multiple_columns = True
    support_retrain = True
    requires_training = True

    def get_name(self):
        return "PassiveAggressive"

    def create_model(self):
        if self.model is None:
            self.model = PassiveAggressiveRegressor()
            # self.model = PassiveAggressiveRegressor(shuffle=False)
            # self.model = PassiveAggressiveRegressor(warm_start=self.reuse_model, shuffle=False)
            # self.model = PassiveAggressiveRegressor(warm_start=self.reuse_model, shuffle=False, C=0.8, epsilon=0.001)

            # self.model = PassiveAggressiveRegressor(shuffle=False)
            # self.model = PassiveAggressiveRegressor(C=1.2, epsilon=0.001, loss='epsilon_insensitive', shuffle=False)
            # self.model = PassiveAggressiveRegressor(C=1.0, epsilon=0.01, loss='epsilon_insensitive', 
            #                                         shuffle=False, warm_start=self.reuse_model)

            # self.model = PassiveAggressiveRegressor(C=1.0, epsilon=0.01, loss='epsilon_insensitive', shuffle=False)
        return

    def train(self, train_data: np.array, results: np.array, incremental=True):
        self.create_model()
        if self.detrend_data:
            train_data = detrend(train_data)
            # results = detrend(results)

        # if you know that the data source is changing, set incremental to False
        if incremental:
            self.model.partial_fit(train_data, results)
        else:
            self.model.fit(train_data, results)

        # self.find_params(train_data, results) # only use while testing

        return

    def forecast(self, data: np.array, steps) -> np.array:
        if self.detrend_data:
            x = detrend(np.array(data))
        else:
            x = np.nan_to_num(data)

        predictions = self.model.predict(x)

        # if self.detrend_data:
        #     # predictions = retrend(predictions.reshape(-1,1)).squeeze()
        #     predictions = retrend(predictions.reshape(-1,1))

        plen = np.shape(predictions)[0]
        dlen  = np.shape(data)[0]
        if plen < dlen:
            print(f'    WARNING: len(predictions) < len(data) {plen} vs {dlen}')
        return predictions


    def find_params(self, train_data: np.array, results: np.array):
        # Define parameter grid
        param_grid = {
        # 'C': [0.2, 0.4, 0.6, 0.8, 1.0],
        'C': [0.8, 1.0, 1.2, 1.4],
        'epsilon': [0.005, 0.01, 0.01, 0.05, 0.1, 0.15]
        }

        # Initialize GridSearchCV object
        grid = GridSearchCV(
            estimator=self.model, param_grid=param_grid, scoring="neg_root_mean_squared_error", cv=5, verbose=1
        )

        # Fit GridSearchCV object with training data
        grid.fit(train_data, results)

        # Print best parameters and best score
        print("***")
        print(grid.best_params_)
        print(grid.best_score_)
        print("***")
        return grid.best_params_

# -----------------------------------


class sgd_forecaster(base_forecaster):
    reuse_model = True
    support_multiple_columns = True
    support_retrain = True
    requires_training = True

    def get_name(self):
        return "SGD"

    def create_model(self):
        if self.model is None:
            # self.model = SGDRegressor(loss="huber", warm_start=self.reuse_model)
            self.model = SGDRegressor(loss="epsilon_insensitive", warm_start=self.reuse_model, shuffle=False)
            # print('Creating SGDRegressor')
        return

    def train(self, train_data: np.array, results: np.array, incremental=True):
        self.create_model()

        # print(f'Train SGDRegressor train_data:{np.shape(train_data)} results:{np.shape(results)} incremental:{incremental}')

        if self.detrend_data:
            train_data = detrend(train_data)
            # results = detrend(results)

        # if you know that the data source is changing, set incremental to False
        if incremental:
            self.model.partial_fit(train_data, results)
        else:
            self.model.fit(train_data, results)

        return

    def forecast(self, data: np.array, steps) -> np.array:
        if self.detrend_data:
            x = detrend(np.array(data))
        else:
            x = np.nan_to_num(data)
        predictions = self.model.predict(x)

        # print(f'forecast SGDRegressor data:{np.shape(data)} predictions:{np.shape(predictions)}')


        # if self.detrend_data:
        #     predictions = retrend(predictions.reshape(-1,1)).squeeze()

        # return predictions
        return predictions


# -----------------------------------


class svr_forecaster(base_forecaster):
    support_multiple_columns = True
    support_retrain = False
    requires_training = True

    def get_name(self):
        return "SVR"

    def create_model(self):
        if self.model is None:
            self.model = SVR(C=2.0, epsilon=0.001)
        return

    def train(self, train_data: np.array, results: np.array, incremental=True):
        self.create_model()

        if self.detrend_data:
            train_data = detrend(train_data)
            # results = detrend(results)

        self.model.fit(train_data, results)
        return

    def forecast(self, data: np.array, steps) -> np.array:
        if self.detrend_data:
            x = detrend(np.array(data))
        else:
            x = np.nan_to_num(data)
        predictions = self.model.predict(x)
        # if self.detrend_data:
        #     predictions = retrend(predictions.reshape(-1,1)).squeeze()
        return predictions


# -----------------------------------


class xgb_forecaster(base_forecaster):

    support_multiple_columns = True
    support_retrain = True
    requires_training = True

    def get_name(self):
        return "XGB"


    def train(self, train_data: np.array, results: np.array, incremental=True):
        if self.model is None:
            # params = {"n_estimators": 50, "max_depth": 0, "learning_rate": 0.01}
            params = {"n_estimators": 100, "max_depth": 4, "learning_rate": 0.1}
            self.model = XGBRegressor(**params)

            if self.detrend_data:
                train_data = detrend(train_data)
                # results = detrend(results)

            self.model.fit(train_data, results)
            # self.find_params(train_data, results) # only use while testing
        else:

            if self.detrend_data:
                train_data = detrend(train_data)
                results = detrend(results)

            if incremental:
                self.model.fit(train_data, results, xgb_model=self.model)
            else:
                self.model.fit(train_data, results)

        return

    def forecast(self, data: np.array, steps) -> np.array:
        if self.detrend_data:
            x = detrend(np.array(data))
        else:
            x = np.nan_to_num(data)
        predictions = self.model.predict(x)
        # if self.detrend_data:
        #     predictions = retrend(predictions.reshape(-1,1)).squeeze()
        return predictions

    def find_params(self, train_data: np.array, results: np.array):
        # Define parameter grid
        param_grid = {
            "learning_rate": [0.01, 0.1, 0.2],
            "max_depth": [2, 4, 6, 8],
            "n_estimators": [50, 100, 200, 500],
        }

        # Initialize GridSearchCV object
        grid = GridSearchCV(
            estimator=self.model, param_grid=param_grid, scoring="neg_root_mean_squared_error", cv=5, verbose=1
        )

        # Fit GridSearchCV object with training data
        grid.fit(train_data, results)

        # Print best parameters and best score
        print("***")
        print(grid.best_params_)
        print(grid.best_score_)
        print("***")
        return grid.best_params_


# -----------------------------------


# enum of all available forecaster types


class ForecasterType(Enum):
    NULL = null_forecaster
    LINEAR = linear_forecaster
    QUADRATIC = quadratic_forecaster
    FFT_EXTRAPOLATION = fft_extrapolation_forecaster
    SS_EXPONENTAL = statespace_exponential_forecaster
    EXPONENTAL = exponential_forecaster
    HOLT = holt_forecaster
    SIMPLE_EXPONENTAL = simple_exponential_forecaster
    ETS = ets_forecaster
    AR = ar_forecaster
    ARIMA = arima_forecaster
    THETA = theta_forecaster
    GB = gb_forecaster
    HGB = hgb_forecaster
    KMEANS = kmeans_forecaster
    LGBM = lgbm_forecaster
    MLP = mlp_forecaster
    PA = pa_forecaster
    SGD = sgd_forecaster
    SVR = svr_forecaster
    XGB = xgb_forecaster


# (static) function to create a forecaster of the specified type
def make_forecaster(forecaster_type: ForecasterType) -> base_forecaster:
    return forecaster_type.value()


# -----------------------------------
