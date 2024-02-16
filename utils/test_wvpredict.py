"""
Test program for verifying wavelet prediction.
Data is encoded using a wavelet transform, each coefficient is then predicted N self.lookahead into the future and the resulting signal is 
re-encoded into a data series

The function names and approach mimic those used in the Time Series Prediction strategies
"""

# Import libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler, RobustScaler
from sklearn.discriminant_analysis import StandardScaler
from pandas import DataFrame

import sys
import traceback
from pathlib import Path
sys.path.append(str(Path(__file__).parent))

import Wavelets
import Forecasters


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
model_window = 32

class WaveletPredictor:
    wavelet = None
    forecaster = None
    lookahead = 6

    coeff_table = None
    coeff_table_offset = 0
    coeff_array = None
    coeff_start_col = 0
    col_forecasters = None
    gain_data = None
    data = None
    curr_dataframe = None

    norm_data = False
    scale_results = False
    single_col_prediction = False
    merge_indicators = False
    training_required = True
    expanding_window = False
    detrend_data = False

    wavelet_size = 32 # Windowing should match this. Longer = better but slower with edge effects. Should be even
    model_window = wavelet_size # longer = slower
    train_min_len = wavelet_size // 2 # longer = slower
    train_max_len = wavelet_size * 4 # longer = slower
    scale_len = wavelet_size // 2 # no. recent candles to use when scaling
    win_size = wavelet_size


    # --------------------------------

    def set_data(self, dataframe:DataFrame):
        self.curr_dataframe = dataframe
        self.data = np.array(dataframe["gain"])
        # self.data = self.smooth(self.data, 1)
        self.data = np.nan_to_num(self.data)
        self.build_coefficient_table(0, np.shape(self.data)[0])
        return

    # --------------------------------

    def set_wavelet_type(self, wavelet_type: Wavelets.WaveletType):
        self.wavelet = Wavelets.make_wavelet(wavelet_type)
        return

    # --------------------------------

    def set_wavelet_len(self, wavelet_len):
        self.wavelet_size = wavelet_len
        self.model_window = self.wavelet_size
        self.train_min_len = self.wavelet_size // 2 
        self.train_max_len = self.wavelet_size * 4
        self.scale_len = max(8, self.wavelet_size // 2)
        self.win_size = self.wavelet_size
        return

    # --------------------------------

    def set_forecaster_type(self, forecaster_type: Forecasters.ForecasterType):
        self.forecaster = Forecasters.make_forecaster(forecaster_type)
        self.forecaster.set_detrend(self.detrend_data)
        return

    # --------------------------------

    def set_lookahead(self, lookahead):
        self.lookahead = lookahead
        self.wavelet.set_lookahead(lookahead)
        return

    # --------------------------------

    # -------------
    # Normalisation

    array_scaler = RobustScaler()
    scaler = RobustScaler()

    def update_scaler(self, data):
        if not self.array_scaler:
            self.array_scaler = RobustScaler()

        self.array_scaler.fit(data.reshape(-1, 1))

    def norm_array(self, a):
        return self.array_scaler.transform(a.reshape(-1, 1))

    def denorm_array(self, a):
        return self.array_scaler.inverse_transform(a.reshape(-1, 1)).squeeze()

    # def smooth(self, y, window):
    def smooth(self, y, window, axis=-1):
        # Apply a uniform 1d filter along the given axis
        # y_smooth = scipy.ndimage.uniform_filter1d(y, window, axis=axis, mode="nearest")
        box = np.ones(window) / window
        y_smooth = np.convolve(y, box, mode="same")
        # Hack: constrain to 3 decimal places (should be elsewhere, but convenient here)
        y_smooth = np.round(y_smooth, decimals=3)
        return np.nan_to_num(y_smooth)

    # scales array y, based on array x
    def scale_array(self, target, data):

        # detrend the input arrays
        t = np.arange(0, len(target))
        t_poly = np.polyfit(t, target, 1)
        t_line = np.polyval(t_poly, target)
        x = target - t_line

        t = np.arange(0, len(data))
        d_poly = np.polyfit(t, data, 1)
        d_line = np.polyval(d_poly, data)
        y = data - d_line

        # scale untrended data
        self.update_scaler(x)
        y_scaled = self.denorm_array(y)

        # retrend
        y_scaled = y_scaled + d_line

        return y_scaled


    def convert_dataframe(self, dataframe: DataFrame) -> DataFrame:
        df = dataframe.copy()

        '''
        # convert date column so that it can be scaled.
        if "date" in df.columns:
            dates = pd.to_datetime(df["date"], utc=True)
            df["date"] = dates.astype("int64")

        df.fillna(0.0, inplace=True)

        if "date" in df.columns:
            df.set_index("date")
            df.reindex()

        '''
        # print(f'    norm_data:{self.norm_data}')
        if self.norm_data:
            # scale the dataframe
            self.scaler.fit(df)
            df = pd.DataFrame(self.scaler.transform(df), columns=df.columns)


        df.fillna(0.0, inplace=True)

        return df
    # -------------

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
            print('    **** ERR: wavelet not specified')
            return

        # print(f'    Wavelet:{self.wavelet_type.name} Forecaster:{self.forecaster_type.name}')

        # double check forecaster/multicolumn combo
        if (not self.single_col_prediction) and (not self.forecaster.supports_multiple_columns()):
            print('    **** WARN: forecaster does not support multiple columns')
            print('               Reverting to single column predictionss')
            self.single_col_prediction = True

        self.coeff_table = None
        self.coeff_table_offset = start

        features = None
        nrows = end - start + 1
        row_start = max(self.wavelet_size, start) - 1 # don't run until we have enough data for the transform

        c_table: np.array = []

        max_features = 0
        for row in range(row_start, end):
            # dslice = data[start:end].copy()
            win_start = max(0, row-self.wavelet_size+1)
            dslice = self.data[win_start:row+1]

            coeffs = self.wavelet.get_coeffs(dslice)
            features = self.wavelet.coeff_to_array(coeffs)
            features = np.array(features)
            flen = len(features)
            max_features = max(max_features, flen)
            c_table.append(features)

        # convert into a zero-padded fixed size array
        nrows = len(c_table)
        self.coeff_table = np.zeros((row_start+nrows, max_features), dtype=float)
        for i in range(0, nrows):
            flen = len(c_table[i]) # feature length
            # print(f'    flen:{flen} c_table[{i}]:{len(c_table[i])}')
            self.coeff_table[i+row_start-1][:flen] = np.array(c_table[i])

        # merge data from main dataframe
        self.merge_coeff_table(start, end)

        # print(f"coeff_table:{np.shape(self.coeff_table)}")
        # print(self.coeff_table[15:48])

        return

    #-------------

    # merge the supplied dataframe with the coefficient table. Number of rows must match
    def merge_coeff_table(self, start, end):

        # print(f'merge_coeff_table() self.coeff_table: {np.shape(self.coeff_table)}')

        # # apply smoothing to each column, otherwise prediction alogorithms will struggle
        # num_cols = np.shape(self.coeff_table)[1]
        # for j in range (num_cols):
        #     feature = self.coeff_table[:,j]
        #     feature = self.smooth(feature, 1)
        #     self.coeff_table[:,j] = feature

        self.coeff_num_cols = np.shape(self.coeff_table)[1]

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

    # generate predictions for an np array 
    def predict_data(self, predict_start, predict_end):

        # a little different than other strats, since we train a model for each column

        # check that we have enough data to run a prediction, if not return zeros
        if self.forecaster.requires_pretraining():
            min_data = self.train_min_len + self.wavelet_size + self.lookahead
        else:
            min_data = self.wavelet_size

        if predict_end < min_data-1:
            # print(f'   {predict_end} < ({self.train_min_len} + {self.wavelet_size} + {self.lookahead})')
            return np.zeros(predict_end-predict_start+1, dtype=float)

        nrows = np.shape(self.coeff_table)[0]
        ncols = np.shape(self.coeff_table)[1]
        coeff_arr: np.array = []


        # train on previous data
        # train_end = max(self.train_min_len, predict_start-1)
        # train_end = min(predict_end - self.lookahead - 1, nrows - self.lookahead - 2)
        train_end = min(predict_end - self.lookahead, nrows - self.lookahead)
        train_start = max(0, train_end-self.train_max_len)
        results_start = train_start + self.lookahead
        results_end = train_end + self.lookahead

        # coefficient table may only be partial, so adjust start/end positions
        start = predict_start - self.coeff_table_offset
        end = predict_end - self.coeff_table_offset
        if (not self.training_required) and (self.expanding_window):
            # don't need training data, so extend prediction buffer instead
            end = predict_end - self.coeff_table_offset
            plen = 2 * self.model_window
            start = max(0, end-plen+1)

        # print(f' predict_start:{predict_start} predict_end:{predict_end} start:{start} end:{end}')

        # print(f'   {predict_end} < ({self.train_min_len} + {self.wavelet_size} + {self.lookahead})')

        # get the data buffers from self.coeff_table
        if not self.single_col_prediction: # single_column version done inside loop
            predict_data = self.coeff_table[start:end]
            # predict_data = np.nan_to_num(predict_data)
            if self.forecaster.requires_pretraining():
                train_data = self.coeff_table[train_start:train_end]
            # results = self.coeff_table[results_start:results_end]


        # print(f'start:{start} end:{end} train_start:{train_start} train_end:{train_end} nrows:{nrows}')

        # train/predict for each coefficient individually
        for i in range(self.coeff_start_col, ncols):

            # get the data buffers from self.coeff_table
            # if single column, then just use a single coefficient
            if self.single_col_prediction:
                predict_data = self.coeff_table[start:end, i].reshape(-1,1)
                predict_data = np.nan_to_num(predict_data)
                train_data = self.coeff_table[train_start:train_end, i].reshape(-1,1)

            results = self.coeff_table[results_start:results_end, i]

            col_forecaster = self.col_forecasters[i-self.coeff_start_col]

            # print(f'predict_data: {np.shape(predict_data)}')
            # print(f'train_data: {np.shape(train_data)}')
            # print(f'results: {np.shape(results)}')

            if col_forecaster.requires_pretraining():
                # since we have a forecaster per column, we can train incrementally
                col_forecaster.train(train_data, results, incremental=True)

            # get a prediction
            preds = col_forecaster.forecast(predict_data, self.lookahead)

            if preds.ndim > 1:
                preds = preds.squeeze()

            # smooth predictions to try and avoid drastic changes
            # preds = self.smooth(preds, 1)

            # append prediction for this column
            coeff_arr.append(preds[-1])

        # convert back to gain
        c_array = np.array(coeff_arr)
        coeffs = self.wavelet.array_to_coeff(c_array)
        preds = self.wavelet.get_values(coeffs)


        # scale results to somewhat match the (recent) input
        # preds = self.denorm_array(preds)

        # if self.scale_results:
        #     preds = self.scale_array(self.data[predict_end-self.scale_len:predict_end], preds)

        # print(f'preds[{start}:{end}] len:{len(preds)}: {preds}')
        # print(f'preds[{end}]: {preds[-1]}')
        # print('===========================')

        return preds

    # -------------

    # single prediction (for use in rolling calculation)
    # @timer
    def predict(self, gain, df) -> float:
        # Get the start and end index labels of the series
        start = gain.index[0]
        end = gain.index[-1]

        # Get the integer positions of the labels in the dataframe index
        start_row = df.index.get_loc(start)
        end_row = df.index.get_loc(end) + 1 # need to add the 1, don't know why!

        y_pred = self.predict_data(start_row, end_row)
        # print(f'    ({start_row}:{end_row}) y_pred[-1]:{y_pred[-1]}')
        return y_pred[-1]

    # -------------

    @timer
    def rolling_predict(self, data):

        if self.forecaster.requires_pretraining():
            min_data = self.train_min_len + self.wavelet_size + self.lookahead
        else:
            min_data = self.wavelet_size

        end = min_data - 1
        start = max(0, end - self.wavelet_size)

        x = np.nan_to_num(np.array(data))
        preds = np.zeros(len(x), dtype=float)

        if self.col_forecasters is  None:
            # create an array of forecasters (1 for each column)
            self.col_forecasters = np.full(self.coeff_num_cols, self.forecaster)
 
        while end <= len(x):

            # print(f'    start:{start} end:{end} train_max_len:{self.train_max_len} model_window:{self.model_window} min_data:{min_data}')
            if end < (min_data-1):
                preds[end-1] = 0.0
                end = end + 1
                start = max(0, end - self.wavelet_size)
                continue

            scale_start = max(0, start-self.scale_len)
            scale_end = max(scale_start+self.scale_len, start)
            self.update_scaler(np.array(data)[scale_start:scale_end])

            forecast = self.predict_data(start, end)
            preds[end-1] = forecast[-1]

            # print(f'forecast: {forecast}')

            end = end + 1
            start = max(0, end - self.wavelet_size)

        return preds



    # add predictions in a jumping fashion. This is a compromise - the rolling version is very slow
    # Note: make sure the rolling version works properly, then it should be OK to use the 'jumping' version
    @timer
    def jumping_predict(self, data: np.array):

        try:

            # roll through the close data and predict for each step
            nrows = len(data)

            # initialise the prediction array
            pred_array = np.zeros(nrows, dtype=float)

            if self.col_forecasters is  None:
                # create an array of forecasters (1 for each column)
                self.col_forecasters = np.full(self.coeff_num_cols, self.forecaster)
 
            win_size = self.model_window

            # loop until we get to/past the end of the buffer
            start = 0
            end = start + win_size
            scale_start = max(0, end-self.scale_len)

            while end < nrows:

                # set the (unmodified) gain data for scaling
                self.update_scaler(np.array(data[scale_start:end]))
                # self.update_scaler(np.array(dataframe['gain'].iloc[start:end]))

                # rebuild data up to end of current window
                preds = self.predict_data(start, end)

                # prediction length doesn't always match data length
                plen = len(preds)
                clen = min(plen, (nrows-start))

                # print(f'    preds:{np.shape(preds)}')
                # copy the predictions for this window into the main predictions array
                pred_array[start:start+clen] = preds[:clen].copy()

                # move the window to the next segment
                # end = end + win_size - 1
                # end = end + win_size
                # start = start + win_size
                end = end + plen
                start = start + plen
                scale_start = max(0, end - self.scale_len)


            # predict for last window

            self.update_scaler(np.array(data[-self.scale_len:]))

            # print(f'    start:{nrows-win_size}, end:{nrows}')
            # print(f'    data:{data[nrows-win_size:nrows]}')
            # print(f'    self.coeff_table:{np.shape(self.coeff_table)}')
            preds = self.predict_data(nrows-win_size, nrows-1)

            # print(f'    preds:{preds}')
            plen = len(preds)
            pred_array[-plen:] = preds.copy()

        except Exception as e:
            print("*** Exception in add_jumping_predictions()")
            print(e) # prints the error message
            print(traceback.format_exc()) # prints the full traceback

        return pred_array

    # rolling predict using pandas dataframe rolling
    @timer
    def pdroll_predict(self, dataframe):
            preds = dataframe['gain'].rolling(window=self.model_window).apply(self.predict, args=(dataframe,))
            return preds

    # -------------

    # convert self.coeff_table back into a waveform (for debug)
    def rolling_coeff_table(self):
        # nrows = np.shape(self.coeff_table)[0]
        nrows = len(self.coeff_table)

        preds = np.zeros(nrows, dtype=float)

        for i in range(nrows):
            row = self.coeff_table[i]
            # print(f'    i:{i} row:{np.shape(row)}')
            # get the coefficient array from this row
            N = int(self.coeff_num_cols) # number of coeffs
            coeff_array = np.zeros(N, dtype=float)
            coeff_array = row[self.coeff_start_col:self.coeff_start_col+N]
            # print(f'    N:{N} dlen:{dlen} clen:{np.shape(coeff_array)} coeff_array:{coeff_array}')

            # convert to coefficients and get the reconstructed data
            coeffs = self.wavelet.array_to_coeff(np.array(coeff_array))
            values = self.wavelet.get_values(coeffs)
            if (i == self.wavelet_size) or (i == self.wavelet_size+1):
                # print(f'    {i}: coeff_array:{np.array(coeff_array)}')
                print(f'   rolling_coeff_table {i}: data[{i}]:{self.data[i]}')
                print(f'   rolling_coeff_table {i}: values[-1]:{values[-1]}')
            preds[i] = float(values[-1])

        return preds
# --------------------------------

# Main code

# Create some random data


num_samples = 512
# np.random.seed(42)
# f1 = np.random.randn()
# np.random.seed(43)
# f2 = np.random.randn()
# np.random.seed(44)
# f3 = np.random.randn(num_samples)

# X = np.arange(num_samples)  # 100 data points
# gen_data = f1 * np.sin(0.5*X) + f2 * np.cos(0.5*X) + f3 * 0.3

# gen_data should be easier to model (use for debug), test_data is realistic
# data = gen_data
# data = np.array(gen_data)
# data = np.concatenate((test_data, test_data, test_data, test_data), dtype=float)

test_data = np.load('test_data.npy')

data = test_data[0:min(1024, len(test_data))]

print(f' {len(data)} test samples')

# data = StandardScaler().fit_transform(data.reshape(-1,1)).reshape(-1)
# data = RobustScaler().fit_transform(data.reshape(-1,1)).reshape(-1)
# data = MinMaxScaler().fit_transform(data.reshape(-1,1)).reshape(-1)


# put the data into a dataframe
# dataframe = pd.DataFrame(data, columns=["gain"])
# dates = pd.date_range(start="2023-01-01", periods=len(data), freq="5m")
# dataframe = pd.DataFrame(data, columns=["gain"], index=dates)
dataframe = pd.DataFrame(data, columns=["gain"])

lookahead = 6

wlist = [
    # Wavelets.WaveletType.MODWT,
    # Wavelets.WaveletType.SWT,
    # Wavelets.WaveletType.SWTA,
    # Wavelets.WaveletType.WPT,
    # Wavelets.WaveletType.FFT,
    Wavelets.WaveletType.FFTA,
    # Wavelets.WaveletType.HFFT,
    # Wavelets.WaveletType.DWT,
    # Wavelets.WaveletType.DWTA,
    ]
flist = [
    # Forecasters.ForecasterType.NULL, # use this to show effect of wavelet alone
    # Forecasters.ForecasterType.EXPONENTAL,
    # Forecasters.ForecasterType.ETS,
    # Forecasters.ForecasterType.SIMPLE_EXPONENTAL,
    # Forecasters.ForecasterType.HOLT,
    # Forecasters.ForecasterType.SS_EXPONENTAL,
    # Forecasters.ForecasterType.AR,
    # Forecasters.ForecasterType.ARIMA,
    # Forecasters.ForecasterType.THETA,
    # Forecasters.ForecasterType.LINEAR,
    # Forecasters.ForecasterType.QUADRATIC,
    # Forecasters.ForecasterType.FFT_EXTRAPOLATION,
    # Forecasters.ForecasterType.MLP,
    # Forecasters.ForecasterType.KMEANS,
    Forecasters.ForecasterType.PA,
    # Forecasters.ForecasterType.SGD,
    # Forecasters.ForecasterType.SVR,
    # Forecasters.ForecasterType.GB,
    # Forecasters.ForecasterType.HGB,
    # Forecasters.ForecasterType.LGBM,
    # Forecasters.ForecasterType.XGB
]

# llist = [ 16, 32, 36, 64 ]
llist = [ 64 ]
marker_list = [ '.', 'o', 'v', '^', '<', '>', 'p', '*', 'h', 'H', 'D', 'd', 'P', 'X' ]
num_markers = len(marker_list)
mkr_idx = 0


# Plot the original data

dataframe['gain_shifted'] = dataframe['gain'].shift(-lookahead)
# ax = dataframe['gain'].plot(label='Original', marker="x", color="black")
ax = dataframe['gain_shifted'].plot(label='Original (shifted)', marker="x", color="black")

gain = np.array(dataframe["gain"])

for wavelet_type in wlist:
    for forecaster_type in flist:
        for length in llist:
            label = wavelet_type.name + "/" + forecaster_type.name + f" ({length})"
            print(label)

            predictor = WaveletPredictor()
            predictor.set_wavelet_type(wavelet_type)
            predictor.set_wavelet_len(length)
            predictor.set_forecaster_type(forecaster_type)
            predictor.set_data(dataframe)
            predictor.set_lookahead(lookahead)


            # # Plot the coeff_table reconstruction
            # dataframe["coeff_table"] = predictor.rolling_coeff_table()
            # dataframe["coeff_table"].plot(ax=ax, label=label+" coeff_table", linestyle="dashed", marker=marker_list[mkr_idx])
            # mkr_idx = (mkr_idx + 1) % num_markers

            # col = "roll_"+label
            # dataframe[col] = predictor.rolling_predict(gain)
            # print(f'forecaster_type: {forecaster_type}')
            # if "NULL" in forecaster_type.name:
            #     print('shifting NULL)')
            #     dataframe[col] = dataframe[col].shift(-lookahead)
            # dataframe[col].plot(ax=ax, label=label + ' (rolling)', linestyle="dashed", marker=marker_list[mkr_idx])
            # mkr_idx = (mkr_idx + 1) % num_markers

            col = "jump_"+label
            dataframe[col] = predictor.jumping_predict(gain)
            print(f'forecaster_type: {forecaster_type} (jumping)')
            if "NULL" in forecaster_type.name:
                print('shifting NULL)')
                dataframe[col] = dataframe[col].shift(-lookahead)
            dataframe[col].plot(ax=ax, label=label + ' (jumping)', linestyle="dashed", marker=marker_list[mkr_idx])
            mkr_idx = (mkr_idx + 1) % num_markers

            # col = "pd_"+label
            # dataframe[col] = predictor.pdroll_predict(dataframe)
            # print(f'forecaster_type: {forecaster_type} (pd roll)')
            # if "NULL" in forecaster_type.name:
            #     print('shifting NULL)')
            #     dataframe[col] = dataframe[col].shift(-lookahead)
            # dataframe[col].plot(ax=ax, label=label + ' (pd roll)', linestyle="dashed", marker=marker_list[mkr_idx])
            # mkr_idx = (mkr_idx + 1) % num_markers

plt.legend()
plt.show()
