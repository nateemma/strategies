
# Import libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))

import Detrenders


# load data from file
data = np.load('test_data.npy')


norm_data = False

lookahead = 6
model_window = 32

orig = np.array(data)
trend = np.zeros(np.shape(data), dtype=float)
retrend = np.zeros(np.shape(data), dtype=float)

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

#---------------------------------------

@timer
def rolling_detrend(data, window_size, norm_data=False):
    global lookahead
    global trend
    global retrend


    start = 0
    end = window_size

    x = np.nan_to_num(data)
    nrows = len(x)
    preds = np.zeros(len(x), dtype=float)

    while end <= len(x):
        dslice = x[start:end]
        x_detrend = detrender.detrend(dslice)
        if x_detrend.ndim > 1:
            print(f'x_detrend:{np.shape(x_detrend)}')
        preds[end-1] = x_detrend[-1]
        trend[end-1] = detrender.get_trend()[-1] # debug
        retrend[end-1] = detrender.retrend(x_detrend)[-1] # debug
        start = start + 1
        end = end + 1


    return preds

#---------------------------------------


# put the data into a dataframe
dataframe = pd.DataFrame(data, columns=["gain"])


dlist = [
    # Detrenders.DetrenderType.NULL,
    Detrenders.DetrenderType.DIFFERENCING,
    # Detrenders.DetrenderType.LINEAR,
    # Detrenders.DetrenderType.QUADRATIC,
    # Detrenders.DetrenderType.SMOOTH,
    # Detrenders.DetrenderType.SCALER,
    # Detrenders.DetrenderType.FFT,
    # Detrenders.DetrenderType.DWT
]

marker_list = [ 'x', 'o', 'v', '^', '<', '>', 'p', '*', 'h', 'H', 'D', 'd', 'P', 'X' ]
num_markers = len(marker_list)
mkr_idx = 0

ax = dataframe['gain'].plot(label='Original', marker="o", color="blue")

for d in dlist:
    detrender = Detrenders.make_detrender(d)
    id = d.name
    print(id)

    dataframe["detrend"] = rolling_detrend(dataframe["gain"], model_window, norm_data=norm_data)
    dataframe["detrend"].plot(ax=ax, label=id, linestyle='dashed', marker=marker_list[mkr_idx])

    mkr_idx = (mkr_idx + 1) % num_markers

    dataframe["trend"] = trend
    dataframe["trend"].plot(ax=ax, label=id+" (trend)", linestyle='dashed', marker=marker_list[mkr_idx])
    mkr_idx = (mkr_idx + 1) % num_markers

    # dataframe["retrend"] = retrend
    # dataframe["retrend"].plot(ax=ax, label=id+" (retrend)", linestyle='dashed', marker=marker_list[mkr_idx])

    # mkr_idx = (mkr_idx + 1) % num_markers



plt.legend()
plt.show()
