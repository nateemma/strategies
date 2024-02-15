
# Import libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

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

# load data from file
data = np.load('test_data.npy')



lookahead = 6
model_window = 64
train_len = model_window * 4
# train_len = model_window * 2

orig = np.array(data)
results = {}

#---------------------------------------

# provides 'baseline' (unmodified) data for comparison in rolling calculation
def baseline(data):

    # just return last item
    return data.iloc[-1]

#---------------------------------------

def smooth( y, window, axis=-1):
    box = np.ones(window) / window
    y_smooth = np.convolve(y, box, mode="same")
    # Hack: constrain to 3 decimal places (should be elsewhere, but convenient here)
    y_smooth = np.round(y_smooth, decimals=3)
    return np.nan_to_num(y_smooth)

#---------------------------------------

@timer
def rolling_predict(data, window_size, norm_data=False):
    global lookahead
    global train_len
    global forecaster


    # train_data = np.array(data)
    train_data = smooth(data, 1)
    train_results = np.roll(train_data, -lookahead)
    train_results[-lookahead:].fill(0)

    # print(f'    data:{np.shape(train_data)}')
    start = 0
    end = window_size

    x = np.nan_to_num(data)
    nrows = len(x)
    preds = np.zeros(len(x), dtype=float)

    while end <= len(x):
        if forecaster.requires_pretraining():
            # min_data = train_len + window_size + lookahead
            min_data = train_len + lookahead + 1
            # min_data = train_len + lookahead
        else:
            min_data = window_size

        # if end < (min_data-1):
        if end < (min_data):
            # print(f'    start:{start} end:{end} train_len:{train_len} window_size:{window_size} min_data:{min_data}')
            preds[end] = 0.0
            start = start + 1
            end = end + 1
            continue

        if forecaster.requires_pretraining():
            # t_end = start - 1
            # t_end = end - lookahead - 1
            # t_end = min(end - lookahead - 2, nrows - lookahead - 2)
            t_end = min(end - lookahead - 1, nrows - lookahead - 1)
            t_start = max(0, t_end-train_len)
            t_data = train_data[t_start:t_end]
            # t_data = smooth(t_data, 8)
            t_results = train_results[t_start:t_end]
            # print(f'     t_start:{t_start} t_end:{t_end}  len:{len(t_data)}')
            forecaster.train(t_data.reshape(-1,1), t_results, incremental=True)
        # else:
            # print(f'    start:{start} end:{end} train_len:{train_len} window_size:{window_size} min_data:{min_data}')

        dslice = x[start:end]
        # dslice = smooth(dslice, 2)

        forecast = forecaster.forecast(dslice.reshape(-1,1), lookahead)
        preds[end-1] = forecast[-1]

        # print(f'     start:{start} end:{end} len(dslice):{len(dslice)} len(forecast):{len(forecast)}')
        # print(f'     dslice[-6:]:   {dslice[-6:]}')
        # print(f'     forecast[-6:]: {forecast[-6:]}')
        # print(f'     x[{end}]: {x[end]} preds[{end}]: {preds[end]}')
        start = start + 1
        end = end + 1

    return preds

#---------------------------------------


# put the data into a dataframe
dataframe = pd.DataFrame(data, columns=["gain"])


flist = [
    # Forecasters.ForecasterType.EXPONENTAL,
    # Forecasters.ForecasterType.SIMPLE_EXPONENTAL,
    # Forecasters.ForecasterType.AR,
    # Forecasters.ForecasterType.HOLT,
    # Forecasters.ForecasterType.ARIMA,
    # Forecasters.ForecasterType.THETA,
    # Forecasters.ForecasterType.ETS,
    # Forecasters.ForecasterType.HGB,
    # Forecasters.ForecasterType.GB,
    # Forecasters.ForecasterType.NULL,
    # Forecasters.ForecasterType.LINEAR,
    # Forecasters.ForecasterType.QUADRATIC,
    Forecasters.ForecasterType.PA,
    # Forecasters.ForecasterType.SGD,
    # Forecasters.ForecasterType.SVR,
    # Forecasters.ForecasterType.FFT_EXTRAPOLATION,
    # Forecasters.ForecasterType.MLP,
    # Forecasters.ForecasterType.LGBM,
    # Forecasters.ForecasterType.XGB
]

marker_list = [ '.', 'o', 'v', '^', '<', '>', 'p', '*', 'h', 'H', 'D', 'd', 'P', 'X' ]
num_markers = len(marker_list)
mkr_idx = 0

# dataframe["baseline"] = dataframe["gain"].rolling(window=model_window).apply(baseline)
# dataframe["baseline"].plot(label='Baseline', marker="X", color="teal")


# Plot the original data and the reconstructed data
# df = pd.DataFrame(orig, index=np.arange(len(orig)))
# ax = df.plot(label='Original', marker="o", color="black")

dataframe['gain_shifted'] = dataframe['gain'].shift(-lookahead)
# ax = dataframe['gain'].plot(label='Original', marker="x", color="black")
# dataframe['gain_shifted'].plot(ax=ax, label='Training Data', marker="o", color="blue")
ax = dataframe['gain_shifted'].plot(label='Training Data', linestyle='dashed', marker="o", color="gray")

# forecaster = Forecasters.make_forecaster(Forecasters.ForecasterType.NULL)
# forecaster.set_detrend(False)
# dataframe["null"] = rolling_predict(dataframe["gain"], model_window, norm_data=False)
# dataframe["null"] = dataframe["null"].shift(-lookahead)
# dataframe["null"].plot(ax=ax, label="Null (shifted)", linestyle='dashed', color='black', marker="x")

for f in flist:
    forecaster = Forecasters.make_forecaster(f)
    forecaster.set_detrend(False)
    id = forecaster.get_name()
    print(id)

    col = "predicted_gain_" + id
    dataframe[col] = rolling_predict(dataframe["gain"], model_window, norm_data=False)
    dataframe[col].plot(ax=ax, label=id, marker=marker_list[mkr_idx])

    mkr_idx = (mkr_idx + 1) % num_markers

    id = id + "(w/ detrend)"
    col = "predicted_gain_" + id 
    forecaster.set_detrend(True)
    dataframe[col] = rolling_predict(dataframe["gain"], model_window, norm_data=False)
    dataframe[col].plot(ax=ax, label=id, marker=marker_list[mkr_idx])

    mkr_idx = (mkr_idx + 1) % num_markers



plt.legend()
plt.show()
