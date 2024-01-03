
# Import libraries
import numpy as np
import pandas as pd
import pywt
import matplotlib.pyplot as plt
from regex import F

import Wavelets
import Forecasters

from sklearn.metrics import mean_squared_error


# Create some random data
# data = np.random.randn(500)
num_samples = 128
np.random.seed(42) # for reproducibility
X = np.arange(num_samples) # data points
data = np.sin(X) + np.random.randn(num_samples) * 0.1

steps = 16
train_data = np.array(data)
orig = np.array(data)
results = {}


def forecast_data(data):
    forecast = forecaster.forecast(np.array(data), steps)
    return forecast[-1]

# put the data into a dataframe
dataframe = pd.DataFrame(data, columns=["gain"])

# Plot the original data and the reconstructed data
df = pd.DataFrame(orig, index=np.arange(len(orig)))
ax = df.plot(label='Original', linestyle='dashed', marker="o")

flist = [
    # Forecasters.ForecasterType.EXPONENTAL,
    # Forecasters.ForecasterType.SIMPLE_EXPONENTAL,
    # Forecasters.ForecasterType.HOLT,
    # Forecasters.ForecasterType.ARIMA,
    # Forecasters.ForecasterType.THETA,
    # Forecasters.ForecasterType.PA,
    Forecasters.ForecasterType.FFT_EXTRAPOLATION,
    # Forecasters.ForecasterType.LGBM,
    # Forecasters.ForecasterType.XGB
]

for f in flist:
    forecaster = Forecasters.make_forecaster(f)
    id = forecaster.get_name()
    print(id)

    dataframe["predicted_gain"] = dataframe["gain"].rolling(window=16).apply(forecast_data)

    dataframe["predicted_gain"].plot(ax=ax, label=id, marker="o")


plt.legend()
plt.show()
