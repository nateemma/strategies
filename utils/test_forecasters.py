
# Import libraries
import numpy as np
import pandas as pd
import pywt
import matplotlib.pyplot as plt
from regex import F
from sklearn.discriminant_analysis import StandardScaler
from sklearn.preprocessing import MinMaxScaler, RobustScaler

import Wavelets
import Forecasters

from sklearn.metrics import mean_squared_error

from freqtrade.freqai import prediction_models

'''
# test data taken from real run
test_data = [  0.02693603,  0.78708102,  0.29854797,  0.27140725, -0.08078632, -0.08078632,
 -0.88864952, -0.56550424, -0.06764984,  0.10826905, -0.24255491, -0.24255491,
 -0.06792555, -1.78740691, -1.23206066, -1.37893741, -1.82358503, -2.90422802,
 -1.98477433, -0.59285813, -0.87731323, -1.27484578, -1.41717116,  0.01391208,
 -0.29126214,  0.13869626,  0.        , -0.15273535,  0.36287509,  0.02782028,
  0.1391014 ,  0.20775623, -0.58083253, -0.61187596, -0.77875122, -0.77875122,
  0.12501736, -0.3731859 ,  0.26429267,  0.85350497,  1.02312544,  1.02312544,
  0.        ,  0.        ,  0.        ,  0.        , -0.15260821, -0.15260821,
  0.16648169,  0.16648169,  0.16648169, -0.84628191, -0.69473392, -0.69473392,
 -0.47091413, -0.47091413, -0.77562327,  0.08395131, -0.30782146, -0.43374843,
 -0.97411634, -0.79320902, -0.48855388, -0.95065008, -0.29473684, -0.16863406,
  0.14052839, -0.04208164,  0.04208164,  0.57868737,  0.30968468, -0.16891892,
 -0.64552344, -0.98231827, -0.75715087, -1.24894752, -1.15071569, -0.535815,
 -0.36723164, -0.02834467,  0.25430913,  2.23106437,  2.82509938,  1.57357528,
  1.57357528,  1.31840091,  0.62006764, -0.88963025, -0.86980533, -0.58618283,
 -0.58618283, -0.76955366,  0.09803922, -0.09817672, -0.79387187, -0.02807806,
 -0.02807806,  0.40891145, -0.363789 , -0.02807806, -0.02807806,  0.,
  0.3932032 ,  0.3932032 ,  0.61789075,  0.82853532,  1.33408229,  0.983008,
  0.74136243,  0.74136243,  0.51639916,  0.30640669, -0.1940133 ,  0.91781393,
  1.55512358,  1.11080255,  1.0413774 ,  1.0413774 ,  0.6942516 ,  1.01970511,
 -0.36915505,  1.11233178,  1.2367734 ,  1.26425725,  0.20683949, -0.19096985,
  0.60381501, -0.47534972 ]


test_data = np.concatenate((test_data, test_data, test_data, test_data), dtype=float)

# Create some random data
num_samples = 512
np.random.seed(42)
f1 = np.random.randn()
np.random.seed(43)
f2 = np.random.randn()
np.random.seed(44)
f3 = np.random.randn(num_samples)

X = np.arange(num_samples)  # data points
gen_data = f1 * np.sin(0.5*X) + f2 * np.cos(0.5*X) + f3 * 0.3
# gen_data = f1 * np.sin(0.5*X) + f3 * 0.3

data = np.array(gen_data)
# data = np.array(test_data)

'''

# load data from file
data = np.load('test_data.npy')


norm_data = False

lookahead = 6
model_window = 32
# train_len = model_window * 4
train_len = model_window * 4

train_data = np.array(data)
train_results = np.roll(train_data, -lookahead)
train_results[-lookahead:].fill(0)

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

def forecast_data(data):
    global lookahead
    start = data.index[0]
    end = data.index[-1]

    if forecaster.requires_pretraining():
        min_data = train_len + model_window + lookahead
    else:
        min_data = model_window

    if end < min_data-1:
        print(f'    start:{start} end:{end} min_data:{min_data}')
        return 0.0

    if forecaster.requires_pretraining():
        t_end = end - lookahead - 1
        t_start = max(0, t_end-train_len)
        # print(f'    train_data:{np.shape(train_data)} train_results:{np.shape(train_results)} t_start:{t_start} t_end:{t_end}')
        t_data = train_data[t_start:t_end].reshape(-1,1)
        t_results = train_results[t_start:t_end]
        # forecaster.train(t_data, t_results, incremental=False)
        forecaster.train(t_data, t_results, incremental=True)

    dslice = np.array(data).copy().reshape(-1,1)
    forecast = forecaster.forecast(dslice, lookahead)
    # print(f'    start:{start} end:{end} lookahead:{lookahead} data[-1]:{data.iloc[-1]:.3f} forecast:{forecast[-1]:.3f}')
    return forecast[-1]

#---------------------------------------

def rolling_predict(data, window_size, norm_data=False):
    global lookahead


    start = 0
    end = window_size-1

    x = np.array(data)

    x = smooth(x, 2)

    scaler = None
    if norm_data:
        y = x.reshape(-1,1)
        # scaler = MinMaxScaler().fit(y)
        scaler = RobustScaler().fit(y)
        x = scaler.transform(y).reshape(-1)

    x = np.nan_to_num(x)
    nrows = len(x)
    preds = np.zeros(len(x), dtype=float)

    while end < len(x):
        if forecaster.requires_pretraining():
            min_data = train_len + model_window + lookahead
        else:
            min_data = model_window

        if end < (min_data-1):
            # print(f'    start:{start} end:{end} train_len:{train_len} model_window:{model_window} min_data:{min_data}')
            preds[end] = 0.0
            start = start + 1
            end = end + 1
            continue

        if forecaster.requires_pretraining():
            # t_end = start - 1
            # t_end = end - lookahead - 1
            t_end = min(end - lookahead - 1, nrows - lookahead - 2)
            t_start = max(0, t_end-train_len)
            # print(f'     start:{start} end:{end} start:{t_start} t_end:{t_end} model_window:{model_window} min_data:{min_data}')
            t_data = train_data[t_start:t_end].reshape(-1,1)
            t_results = train_results[t_start:t_end]
            forecaster.train(t_data, t_results, incremental=False)
        # else:
            # print(f'    start:{start} end:{end} train_len:{train_len} model_window:{model_window} min_data:{min_data}')

        dslice = x[start:end].reshape(-1,1)
        forecast = forecaster.forecast(dslice, lookahead).squeeze()
        preds[end] = forecast[-1]
        start = start + 1
        end = end + 1

    if norm_data:
        preds = scaler.transform(preds.reshape(-1,1)).reshape(-1)

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
    Forecasters.ForecasterType.NULL,
    # Forecasters.ForecasterType.LINEAR,
    # Forecasters.ForecasterType.QUADRATIC,
    # Forecasters.ForecasterType.PA,
    Forecasters.ForecasterType.SGD,
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

# dataframe['gain_shifted'] = dataframe['gain'].shift(-lookahead)
dataframe['gain_shifted'] = train_results
# ax = dataframe['gain'].plot(label='Original', marker="x", color="black")
# dataframe['gain_shifted'].plot(ax=ax, label='Training Data', marker="o", color="blue")
ax = dataframe['gain_shifted'].plot(label='Training Data', marker="o", color="blue")

for f in flist:
    forecaster = Forecasters.make_forecaster(f)
    forecaster.set_detrend(False)
    id = forecaster.get_name()
    print(id)



    dataframe["predicted_gain"] = rolling_predict(dataframe["gain"], model_window, norm_data=norm_data)
    dataframe["predicted_gain"].plot(ax=ax, label=id, linestyle='dashed', marker=marker_list[mkr_idx])

    # dataframe["predicted_gain"] = dataframe["gain"].rolling(window=model_window).apply(forecast_data)

    # # DBG: manualyy set first portion of prediction
    # dslice = dataframe['gain'].iloc[0:model_window].to_numpy()
    # preds = forecaster.forecast(dslice, lookahead)
    # print(f'    preds:{np.shape(preds)}')
    # dataframe["predicted_gain"].iloc[model_window-len(preds):model_window] = preds

    # dataframe['shifted_pred'] = dataframe['predicted_gain'].shift(lookahead)
    # dataframe["predicted_gain"].plot(ax=ax, label=id, linestyle='dashed', marker=marker_list[mkr_idx])
    # dataframe["shifted_pred"].plot(ax=ax, label=id+" (shifted)", linestyle='dashed', marker=marker_list[mkr_idx])

    mkr_idx = (mkr_idx + 1) % num_markers



plt.legend()
plt.show()
