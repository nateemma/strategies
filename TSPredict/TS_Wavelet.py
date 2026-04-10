# pragma pylint: disable=W0105, C0103, C0114, C0115, C0116, C0301, C0302, C0303, C0411, C0413,  W1203

"""
####################################################################################
TS_Wavelet - predicts each component/coefficient of a wavelet and reconstructs the signal to produce a prediction
            NOTE: this is *very* compute intensive

####################################################################################
"""


from datetime import datetime
from functools import reduce

import cProfile
import pstats
import traceback

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from sklearn.multioutput import MultiOutputRegressor
from lightgbm import LGBMRegressor

# Get rid of pandas warnings during backtesting
import pandas as pd
from pandas import DataFrame, Series

import pywt

# from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import RobustScaler


# import freqtrade.vendor.qtpylib.indicators as qtpylib

from freqtrade.strategy import IStrategy, DecimalParameter, CategoricalParameter
from freqtrade.persistence import Trade
import freqtrade.vendor.qtpylib.indicators as qtpylib
from typing import Any, List, Optional

pd.options.mode.chained_assignment = None  # default='warn'

# Strategy specific imports, files must reside in same folder as strategy
import sys
from pathlib import Path

import os
import joblib
import copy

group_dir = str(Path(__file__).parent)
strat_dir = str(Path(__file__).parent.parent)
sys.path.append(strat_dir)
sys.path.append(group_dir)

import logging
import warnings

log = logging.getLogger(__name__)
log.setLevel(logging.DEBUG)
warnings.simplefilter(action="ignore", category=pd.errors.PerformanceWarning)
warnings.simplefilter(action="ignore", category=FutureWarning)

from xgboost import XGBRegressor

# from lightgbm import LGBMRegressor
from sklearn.linear_model import PassiveAggressiveRegressor
from sklearn.linear_model import SGDRegressor
from sklearn.svm import SVR

from utils.DataframeUtils import DataframeUtils, ScalerType  # pylint: disable=E0401

# import pywt
import talib.abstract as ta
import utils.custom_indicators as cta

import utils.Wavelets as Wavelets
import utils.Forecasters as Forecasters

from TSPredict import TSPredict


# -----------------------------------------------------------------------
class MultiOutputMLP(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        # Robust MLP for multi-coefficient prediction
        self.layers = [
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, output_dim),
        ]

    def __call__(self, x: mx.array) -> mx.array:
        for layer in self.layers:
            x = layer(x)
        return x


def train_mlx(X, y, epochs=100, seed=42):
    # Set seed for reproducibility
    mx.random.seed(seed)

    X_mx = mx.array(X)
    y_mx = mx.array(y)
    model = MultiOutputMLP(X_mx.shape[1], y_mx.shape[1])
    mx.eval(model.parameters())

    def loss_fn(model, X, y):
        return mx.mean(mx.square(model(X) - y))

    loss_and_grad = nn.value_and_grad(model, loss_fn)
    optimizer = optim.Adam(learning_rate=1e-3)

    for _ in range(epochs):
        loss, grads = loss_and_grad(model, X_mx, y_mx)
        optimizer.update(model, grads)
        mx.eval(model.parameters(), optimizer.state)

    return model


class TS_Wavelet(TSPredict):

    ###################################

    # Strategy Specific Variable Storage

    lookahead = 6
    # lookahead = 9

    coeff_table = None
    coeff_table_offset = 0
    coeff_array = None
    coeff_start_col = 0
    col_forecasters = None
    gain_data = None
    data = None
    # curr_dataframe = None

    norm_data = False  # must be false for these strategies
    merge_indicators = False
    training_required = True
    expanding_window = False
    use_rolling = False  # if True, also set single_col_prediction = True
    detrend_data = True  # if True, also set single_col_prediction = True
    single_col_prediction = True
    use_mlx = hasattr(mx, "metal") and mx.metal.is_available()

    # NOTE: can only use longer lengths with FFT, too slow otherwise
    wavelet_size = 64  # Windowing should match this. Longer = better but slower with edge effects. Should be even
    model_window = wavelet_size  # longer = slower
    # train_min_len = wavelet_size // 2 # longer = slower
    train_min_len = wavelet_size  # longer = slower
    train_max_len = min(128, wavelet_size * 4)  # longer = slower
    # train_max_len = wavelet_size * 2 # longer = slower
    # scale_len = wavelet_size // 4 # no. recent candles to use when scaling
    scale_len = min(16, wavelet_size // 2)  # no. recent candles to use when scaling
    win_size = max(32, wavelet_size)

    wavelet_type: Wavelets.WaveletType = Wavelets.WaveletType.SWT
    wavelet = None

    # forecaster_type:Forecasters.ForecasterType = Forecasters.ForecasterType.PA
    forecaster_type: Forecasters.ForecasterType = Forecasters.ForecasterType.SGD
    # forecaster_type:Forecasters.ForecasterType = Forecasters.ForecasterType.FFT_EXTRAPOLATION
    forecaster = None

    ###################################

    # Model-related functions - note that this family of strategies does not save/reload models

    # the following are here just to maintain compatibilkity with TSPredict

    def create_model(self, df_shape):
        pass

    def init_model(self, dataframe: DataFrame):
        pass

    def train_model(self, model, data: np.array, results: np.array, save_model):
        pass

    def save_model(self):
        pass

    def load_model(self, df_shape):
        self.model_trained = True
        self.new_model = False
        self.training_mode = False
        return

    #####################################

    # Prediction-related functions

    # -------------

    # saved = False

    # override func to get data for this strategy
    def set_data(self, dataframe):

        df_norm = self.convert_dataframe(dataframe)
        self.gain_data = df_norm["gain"].to_numpy()

        # data for building coeff_table
        self.data = self.gain_data.copy()  # copy becaise gain_data changes

        # if not self.saved:
        #     np.save('test_data.npy', self.data)
        #     self.saved = True

        # self.data = self.smooth(self.data, 2)

        # copy of dataframe
        self.curr_dataframe = dataframe

        return

    # builds a numpy array of coefficients
    def build_coefficient_table(self, start, end):

        # print(f'start:{start} end:{end} self.win_size:{self.win_size}')

        # lazy initialisation of vars (so thatthey can be changed in subclasses)
        if self.wavelet is None:
            self.wavelet = Wavelets.make_wavelet(self.wavelet_type)

        self.wavelet.set_lookahead(self.lookahead)

        if self.forecaster is None:
            self.forecaster = Forecasters.make_forecaster(self.forecaster_type)

        # if forecaster does not require pre-training, then just set training length to 0
        if not self.forecaster.requires_pretraining():
            print("    INFO: Training not required. Setting train_max_len=0")
            self.train_max_len = 0
            self.train_min_len = 0
            self.training_required = False

        if self.wavelet is None:
            print("    **** ERR: wavelet not specified")
            return

        # print(f'    Wavelet:{self.wavelet_type.name} Forecaster:{self.forecaster_type.name}')

        # double check forecaster/multicolumn combo
        if (not self.single_col_prediction) and (
            not self.forecaster.supports_multiple_columns()
        ):
            print("    **** WARN: forecaster does not support multiple columns")
            print("               Reverting to single column predictionss")
            self.single_col_prediction = True

        # Vectorized implementation using sliding_window_view
        if end < self.wavelet_size:
            self.coeff_table = np.zeros((end, 1))
            return

        # Prepare data for sliding window
        # We need data from start (potentially 0) up to end
        # But we need at least wavelet_size to produce the first feature set
        win_data = self.data[0:end]

        # Create sliding windows of size wavelet_size
        windows = sliding_window_view(win_data, self.wavelet_size)
        # windows shape: (len(win_data) - wavelet_size + 1, wavelet_size)

        # Calculate coefficients in batch
        # Currently optimized for SWT (most common for this strategy)
        if self.wavelet_type == Wavelets.WaveletType.SWT:
            # Seed the wavelet object state (coeff_slices) for later use in array_to_coeff
            dummy_coeffs = self.wavelet.get_coeffs(windows[0])
            self.wavelet.coeff_to_array(dummy_coeffs)

            # Ensure wavelet name is set (defaults to 'bior3.9' in Wavelets.py but only inside get_coeffs)
            wname = getattr(self.wavelet, "wavelet", None) or "bior3.9"
            levels = min(2, pywt.swt_max_level(self.wavelet_size))
            # trim_approx=True returns [cA_n, cD_n, cD_{n-1}, ..., cD_1]
            coeffs = pywt.swt(windows, wname, level=levels, axis=-1, trim_approx=True)
            # coeffs is a list of arrays (nrows, wavelet_size)
            result = np.concatenate(coeffs, axis=-1)
        else:
            # Fallback for other wavelets that might not support batching as easily
            # or require different formatting
            c_table = []
            for i in range(len(windows)):
                coeffs = self.wavelet.get_coeffs(windows[i].copy())
                features = self.wavelet.coeff_to_array(coeffs)
                c_table.append(features)
            result = np.array(c_table)

        # Initialize full table with zeros and place result at correct offset
        # Result corresponds to indices starting from wavelet_size - 1
        max_features = result.shape[1]
        self.coeff_table = np.zeros((end, max_features), dtype=float)
        offset = self.wavelet_size - 1
        self.coeff_table[offset : offset + result.shape[0]] = result

        # merge data from main dataframe
        self.merge_coeff_table(start, end)

        # print(f"coeff_table:{np.shape(self.coeff_table)}")
        # print(self.coeff_table[15:48])

        return

    # -------------

    # merge the supplied dataframe with the coefficient table. Number of rows must match
    def merge_coeff_table(self, start, end):

        # print(f'merge_coeff_table() self.coeff_table: {np.shape(self.coeff_table)}')

        self.coeff_num_cols = np.shape(self.coeff_table)[1]

        # apply smoothing to each column, otherwise prediction algorithms will struggle
        # num_cols = np.shape(self.coeff_table)[1]
        # for j in range (num_cols):
        #     feature = self.coeff_table[:,j]
        #     feature = self.smooth(feature, 1)
        #     self.coeff_table[:,j] = feature

        # if using single column prediction, no need to merge in dataframe column because they won't be used
        if self.single_col_prediction or (not self.merge_indicators):
            merged_table = self.coeff_table
            self.coeff_start_col = 0
        else:
            self.coeff_start_col = np.shape(self.curr_dataframe)[1]
            df = self.curr_dataframe.iloc[start:end]
            df_norm = self.convert_dataframe(df)
            merged_table = np.concatenate([np.array(df_norm), self.coeff_table], axis=1)

        self.coeff_table = np.nan_to_num(merged_table)

        return

    # -------------

    # -------------

    def predict_data(self, X):
        # determine which model to use
        if self.use_mlx:
            # MLX prediction
            X_mx = mx.array(X)
            preds_mx = self.model(X_mx)
            mx.eval(preds_mx)
            c_array = np.array(preds_mx)
        else:
            # Fallback (LightGBM)
            c_array = self.model.predict(X)

        # c_array is shape (N, C)
        # We only return the prediction for the requested points
        # But for Wavelet reconstruction, we need to apply the inverse to each row
        results = []
        for i in range(len(c_array)):
            coeffs = self.wavelet.array_to_coeff(c_array[i])
            val = self.wavelet.get_values(coeffs)
            results.append(val[-1])

        return np.array(results)

    # -------------

    # -------------

    # single prediction (for use in rolling calculation)
    def predict(self, gain, df) -> float:
        # Get the start and end index labels of the series
        start = gain.index[0]
        end = gain.index[-1]

        # Get the integer positions of the labels in the dataframe index
        start_row = df.index.get_loc(start)
        end_row = df.index.get_loc(end) + 1  # need to add the 1, don't know why!

        # if end_row < (self.train_len + self.wavelet_size + self.lookahead):
        # # if start_row < (self.wavelet_size + self.lookahead): # need buffer for training
        #     # print(f'    ({start_row}:{end_row}) y_pred[-1]:0.0')
        #     return 0.0

        # print(f'gain.index:{gain.index} start:{start} end:{end} start_row:{start_row} end_row:{end_row}')

        scale_start = max(0, len(gain) - 16)

        # print(f'    coeff_table: {np.shape(self.coeff_table)} start_row: {start_row} end_row: {end_row} ')

        self.update_scaler(np.array(gain)[scale_start:])

        y_pred = self.predict_data(start_row, end_row)
        # print(f'    ({start_row}:{end_row}) y_pred[-1]:{y_pred[-1]}')
        return y_pred[-1]

    # -------------

    def rolling_predict(self, data):

        if self.forecaster.requires_pretraining():
            min_data = self.train_min_len + self.wavelet_size + self.lookahead
        else:
            min_data = self.wavelet_size

        end = min_data - 1
        start = max(0, end - self.wavelet_size)

        x = np.nan_to_num(np.array(data))
        nrows = len(x)
        preds = np.zeros(nrows, dtype=float)

        # if self.col_forecasters is  None:
        #     # create an array of forecasters (1 for each column)
        #     self.col_forecasters = np.full(self.coeff_num_cols, self.forecaster)
        self.col_forecasters = np.full(self.coeff_num_cols, self.forecaster)

        while end <= nrows:

            # print(f'    start:{start} end:{end} train_max_len:{self.train_max_len} model_window:{self.model_window} min_data:{min_data}')
            if end < (min_data - 1):
                preds[end] = 0.0
                end = end + 1
                start = max(0, end - self.wavelet_size)
                continue

            scale_start = max(0, start - self.scale_len)
            scale_end = max(scale_start + self.scale_len, start)
            self.update_scaler(np.array(data)[scale_start:scale_end])

            forecast = self.predict_data(start, end)
            preds[end - 1] = forecast[-1]

            # if end == len(x):
            #     print(f'    preds[-1]:{preds[-1]} forecast[-1]: {forecast[-1]}')

            end = end + 1
            start = max(0, end - self.wavelet_size)

        # predict for last window
        self.update_scaler(np.array(data[-self.scale_len :]))
        plast = self.predict_data(nrows - self.wavelet_size, nrows - 1)
        preds[-1] = plast[-1]

        return preds

    # -------------
    def add_rolling_predictions(self, dataframe: DataFrame) -> DataFrame:
        try:
            nrows = dataframe.shape[0]
            self.set_data(dataframe)
            self.build_coefficient_table(0, nrows)

            print(
                f"    Training Causal Model ({'MLX' if self.use_mlx else 'LightGBM'})..."
            )

            # Walk-forward training to avoid lookahead bias
            # We train on past data and predict the next 'chunk'
            chunk_size = 2500  # Re-train every 2500 candles for speed
            results_all = np.zeros(nrows)

            # Start predicting after we have enough training data
            initial_train_size = max(self.train_min_len, 1000)

            for win_end in range(initial_train_size, nrows, chunk_size):
                train_end = win_end - self.lookahead
                predict_end = min(win_end + chunk_size, nrows)

                # Training set: strictly before the current prediction window
                X_train = self.coeff_table[:train_end]
                y_train = self.coeff_table[self.lookahead : train_end + self.lookahead]

                # Prediction set: the next chunk
                X_predict = self.coeff_table[win_end:predict_end]

                if len(X_predict) == 0:
                    continue

                # Scale features causally
                scaler = RobustScaler()
                X_train_scaled = scaler.fit_transform(X_train)
                X_predict_scaled = scaler.transform(X_predict)

                if self.use_mlx:
                    # MLX Path
                    # Increased epochs for better convergence
                    model = train_mlx(X_train_scaled, y_train, epochs=80)
                    X_mx = mx.array(X_predict_scaled)
                    preds_mx = model(X_mx)
                    mx.eval(preds_mx)
                    chunk_pred_coeffs = np.array(preds_mx)
                else:
                    # LightGBM Path
                    model = MultiOutputRegressor(
                        LGBMRegressor(n_estimators=100, random_state=42)
                    )
                    model.fit(X_train_scaled, y_train)
                    chunk_pred_coeffs = model.predict(X_predict_scaled)

                # Inverse Wavelet Transform for this chunk
                for i in range(len(chunk_pred_coeffs)):
                    coeffs = self.wavelet.array_to_coeff(chunk_pred_coeffs[i])
                    val = self.wavelet.get_values(coeffs)
                    # result is in gain range. Apply scaling if necessary
                    res = val[-1]
                    if self.scale_results:
                        # simple local scaling if enabled
                        win = self.data[max(0, win_end + i - 32) : win_end + i]
                        res = self.scale_array(win, np.array([res]))[0]

                    results_all[win_end + i] = res

            dataframe.iloc[:, dataframe.columns.get_loc("predicted_gain")] = results_all

        except Exception as e:
            print("*** Exception in add_rolling_predictions()")
            print(e)
            print(traceback.format_exc())

        return dataframe

    # add predictions in a jumping fashion. This is a compromise - the rolling version is very slow
    # Note: make sure the rolling version works properly, then it should be OK to use the 'jumping' version
    def add_jumping_predictions(self, dataframe: DataFrame) -> DataFrame:
        # For the new vectorized architecture, jumping is no longer needed for speed
        # so we just redirect to the optimized rolling (global) path.
        return self.add_rolling_predictions(dataframe)

    # -------------

    # add the latest prediction, and update training periodically
    def add_latest_prediction(self, dataframe: DataFrame) -> DataFrame:
        # For live/dry runs, we can still use the global model or re-train.
        # For now, redirect to ensure consistency.
        return self.add_rolling_predictions(dataframe)

    # -------------
