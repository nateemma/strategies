"""
Test program for verifying wavelet prediction.
Data is encoded using a wavelet transform, each coefficient is then predicted N steps into the future and the resulting signal is 
re-encoded into a data series

The function names and approach mimic those used in the Time Series Prediction strategies
"""

# Import libraries
import numpy as np
import pandas as pd
import pywt
import matplotlib.pyplot as plt
from regex import D, S
from sklearn.feature_selection import SelectFdr
from sklearn.preprocessing import RobustScaler

import Wavelets
import Forecasters

from sklearn.metrics import mean_squared_error

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


# -----------------------------------


class WaveletPredictor:
    wavelet = None
    forecaster = None
    lookahead = 6

    coeff_table = None
    coeff_table_offset = 0
    coeff_array = None
    coeff_start_col = 0
    coeff_num_cols = 0
    gain_data = None
    data = None

    norm_data = False
    single_col_prediction = True

    model_window = 32 # longer = slower
    train_len = 16 # longer = slower
    scale_len = 16 # no. recent candles to use when scaling
    min_wavelet_size = 16  # needed for consistently-sized transforms
    win_size = 16  # this can vary

    # --------------------------------

    def set_data(self, data: np.array):
        self.data = data.copy()
        self.build_coefficient_table(0, np.shape(data)[0])
        return

    # --------------------------------

    def set_wavelet_type(self, wavelet_type: Wavelets.WaveletType):
        self.wavelet = Wavelets.make_wavelet(wavelet_type)
        return

    # --------------------------------

    def set_forecaster_type(self, forecaster_type: Forecasters.ForecasterType):
        self.forecaster = Forecasters.make_forecaster(forecaster_type)
        return

    # --------------------------------

    def set_lookahead(self, lookahead):
        self.lookahead = lookahead
        self.wavelet.set_lookahead(lookahead)
        return

    # --------------------------------

    # -------------
    # Normalisation

    scaler = RobustScaler()

    def update_scaler(self, data):
        if not self.scaler:
            self.scaler = RobustScaler()

        self.scaler.fit(data.reshape(-1, 1))

    def norm_array(self, a):
        return self.scaler.transform(a.reshape(-1, 1))

    def denorm_array(self, a):
        return self.scaler.inverse_transform(a.reshape(-1, 1)).squeeze()

    # -------------

    # builds a numpy array of coefficients
    @timer
    def build_coefficient_table(self, start, end):

        # print(f'start:{start} end:{end} self.win_size:{self.win_size}')


        # lazy initialisation of vars (so thatthey can be changed in subclasses)
        if self.wavelet is None:
           self.wavelet = Wavelets.make_wavelet(self.wavelet_type)

        if self.forecaster is None:
            self.forecaster = Forecasters.make_forecaster(self.forecaster_type)

        if not self.forecaster.requires_pretraining():
            self.train_len = 0

        if self.wavelet is None:
            print('    **** ERR: wavelet not specified')
            return

        # print(f'    Wavelet:{self.wavelet_type.name} Forecaster:{self.forecaster_type.name}')

        # double check forecaster/multicolumn combo
        if (not self.single_col_prediction) and (not self.forecaster.supports_multiple_columns()):
            print('    **** ERR: forecaster does not support multiple columns')
            print('              Reverting to single column predictionss')
            self.single_col_prediction = True

        self.coeff_table = None
        self.coeff_table_offset = start
        num_coeffs = 0
        init_done = False

        features = None
        nrows = end - start
        row_start = max(self.min_wavelet_size, start) # don't run until we have enough data for the transform

        for row in range(row_start, end):
            # dslice = data[start:end].copy()
            win_start = max(0, row-self.min_wavelet_size)
            dslice = self.data[win_start:row]
            # print(f'data[{win_start}:{row}]: {dslice}')

            coeffs = self.wavelet.get_coeffs(dslice)
            features = self.wavelet.coeff_to_array(coeffs)
            # print(f'features: {np.shape(features)}')

            # initialise the np.array (need features first to know size)
            if not init_done:
                init_done = True
                num_coeffs = len(features)
                self.coeff_table = np.zeros((nrows, num_coeffs), dtype=float)
                # print(f"coeff_table:{np.shape(self.coeff_table)}")

            # copy the features to the appropriate row of the coefficient array 
            # print(f'row: {row} start:{start}')
            self.coeff_table[row-start] = features

        return

    # -------------

    # generate predictions for an np array
    # data should be the entire data array, not a slice
    # since we both train and predict, supply indices to allow trining and predicting in different regions
    @timer
    def predict_data(self, predict_start, predict_end):

        # a little different than other strats, since we train a model for each column

        # check that we have enough data to run a prediction, if not return zeros
        # if predict_start < (self.train_len + self.lookahead):
        #     return np.zeros(predict_end-predict_start, dtype=float)

        ncols = np.shape(self.coeff_table)[1]
        coeff_arr: np.array = []

        # train on previous data (*not* current data!)
        train_start = max(0, predict_start-self.train_len)
        train_end = train_start + self.train_len
        results_start = train_start + self.lookahead
        results_end = train_end + self.lookahead

        # coefficient table may only be partial, so adjust start/end positions
        start = predict_start - self.coeff_table_offset
        end = predict_end - self.coeff_table_offset


        # print(f'self.coeff_table_offset:{self.coeff_table_offset} start:{start} end:{end}')

        # get the data buffers from self.coeff_table
        if not self.single_col_prediction: # single_column version done inside loop
            predict_data = self.coeff_table[start:end]
            # predict_data = np.nan_to_num(predict_data)
            train_data = self.coeff_table[train_start:train_end]
            results = self.coeff_table[results_start:results_end]


        print(f'start:{start} end:{end} train_start:{train_start} train_end:{train_end} train_len:{self.train_len}')

        # train/predict for each coefficient individually
        for i in range(self.coeff_start_col, ncols):

            # get the data buffers from self.coeff_table
            # if single column, then just use a single coefficient
            if self.single_col_prediction:
                predict_data = self.coeff_table[start:end, i].reshape(-1,1)
                # predict_data = np.nan_to_num(predict_data)
                train_data = self.coeff_table[train_start:train_end, i].reshape(-1,1)

            results = self.coeff_table[results_start:results_end, i]

            # print(f'predict_data: {predict_data}')
            # print(f'train_data: {train_data}')
            # print(f'results: {results}')

            # train the forecaster
            self.forecaster.train(train_data, results)

            # get a prediction
            preds = self.forecaster.forecast(predict_data, self.lookahead).squeeze()
            coeff_arr.append(preds[-1])

        # convert back to gain
        coeffs = self.wavelet.array_to_coeff(np.array(coeff_arr))
        preds = self.wavelet.get_values(coeffs)

        # rescale if necessary
        if self.norm_data:
            preds = self.denorm_array(preds)


        # print(f'preds[{start}:{end}]: {preds}')

        return preds

    # -------------

    # single prediction (for use in rolling calculation)
    @timer
    def predict(self, gain, df) -> float:
        # Get the start and end index labels of the series
        start = gain.index[0]
        end = gain.index[-1]

        # Get the integer positions of the labels in the dataframe index
        start_row = df.index.get_loc(start)
        end_row = df.index.get_loc(end) + 1 # need to add the 1, don't know why!


        if end_row < (self.train_len + self.min_wavelet_size + self.lookahead):
        # if start_row < (self.min_wavelet_size + self.lookahead): # need buffer for training

            print(f'    train_len:{self.train_len} min_wavelet_size:{self.min_wavelet_size} lookahead:{self.lookahead}')
            print(f'    ({start_row}:{end_row}) y_pred[-1]:0.0')
            return 0.0

        # print(f'gain.index:{gain.index} start:{start} end:{end} start_row:{start_row} end_row:{end_row}')

        scale_start = max(0, len(gain)-16)

        print(f'    coeff_table: {np.shape(self.coeff_table)} start_row: {start_row} end_row: {end_row} ')

        self.update_scaler(np.array(gain)[scale_start:])

        y_pred = self.predict_data(start_row, end_row)
        print(f'    ({start_row}:{end_row}) y_pred[-1]:{y_pred[-1]}')
        return y_pred[-1]

    # -------------


# --------------------------------

# Main code

# Create some random data

num_samples = 128
np.random.seed(42)  # for reproducibility
X = np.arange(num_samples)  # 100 data points
data = np.sin(X) + np.random.randn(num_samples) * 0.3
# data = np.sin(X)

# data = np.random.normal (0, 0.1, size=num_samples)

# put the data into a dataframe
dataframe = pd.DataFrame(data, columns=["gain"])

lookahead = 2


# Plot the original data
plt.plot(dataframe["gain"], label="Original", marker="o")

wlist = [Wavelets.WaveletType.DWT]
flist = [
    # Forecasters.ForecasterType.EXPONENTAL,
    # Forecasters.ForecasterType.SIMPLE_EXPONENTAL,
    # Forecasters.ForecasterType.HOLT,
    # Forecasters.ForecasterType.ARIMA,
    # Forecasters.ForecasterType.THETA,
    Forecasters.ForecasterType.PA,
    # Forecasters.ForecasterType.FFT_EXTRAPOLATION,
    # Forecasters.ForecasterType.LGBM,
    # Forecasters.ForecasterType.XGB
]
# flist = [ Forecasters.ForecasterType.PA, Forecasters.ForecasterType.GB, Forecasters.ForecasterType.ARIMA ]
# flist = [ Forecasters.ForecasterType.PA, Forecasters.ForecasterType.GB, Forecasters.ForecasterType.SGD, Forecasters.ForecasterType.HGB ]

for wavelet_type in wlist:
    for forecaster_type in flist:
        label = wavelet_type.name + "/" + forecaster_type.name
        print(label)

        predictor = WaveletPredictor()
        predictor.set_wavelet_type(wavelet_type)
        predictor.set_forecaster_type(forecaster_type)
        predictor.set_data(dataframe["gain"].to_numpy())
        predictor.set_lookahead(lookahead)

        dataframe["predicted_gain"] = dataframe["gain"].rolling(window=32).apply(predictor.predict, args=(dataframe,))

        plt.plot(dataframe["predicted_gain"], label=label, linestyle="dashed", marker="o")

plt.legend()
plt.show()
