# test program to explore using the river package

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from river import compose, time_series, preprocessing, linear_model, metrics, evaluate

x = np.load("test_data.npy")
num_steps = len(x)
dataframe = pd.DataFrame(x, columns=["gain"])
lookahead = 6


# Create a pipeline object with scaling and linear regression steps
pipeline = compose.Pipeline(
    ('scale', preprocessing.StandardScaler()),
    ('learn', linear_model.LinearRegression())
)

# Create a SNARIMAX regressor with the pipeline object as an argument
model = time_series.SNARIMAX(
    p=12,  # autoregressive order
    d=1,   # differencing order
    q=12,  # moving average order
    m=24,  # seasonality period
    sp=12, # seasonal autoregressive order
    sq=12, # seasonal moving average order
    regressor=pipeline  # pipeline with scaling and linear regression
)

# evaluation metric
metric = metrics.MAE()

# Create an empty list to store the predictions
y_preds = []

# Loop over the generator of predictions
for y_pred in evaluate.progressive_val_predict(dataset=x, model=model, delay=lookahead):
    # Append the prediction to the list
    y_preds.append(y_pred)

# Convert the list to a dataframe column
dataframe['preds'] = y_preds

# plot the original and prediction

marker_list = [ '.', 'o', 'v', '^', '<', '>', 'p', '*', 'h', 'H', 'D', 'd', 'P', 'X' ]
num_markers = len(marker_list)
mkr_idx = 0

dataframe['gain_shifted'] = dataframe['gain'].shift(-lookahead)
ax = dataframe['gain_shifted'].plot(label='Original (shifted)', marker="x", color="black")
mkr_idx = (mkr_idx + 1) % num_markers

dataframe['preds'].plot(ax=ax, label="Prediction", linestyle="dashed", marker=marker_list[mkr_idx])