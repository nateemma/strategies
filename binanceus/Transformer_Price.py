import operator

import numpy as np
from enum import Enum

import pywt
import talib.abstract as ta
from scipy.ndimage import gaussian_filter1d
from sklearn.preprocessing import RobustScaler, MinMaxScaler, StandardScaler

import freqtrade.vendor.qtpylib.indicators as qtpylib
import arrow

from freqtrade.exchange import timeframe_to_minutes
from freqtrade.strategy import (IStrategy, merge_informative_pair, stoploss_from_open,
                                IntParameter, DecimalParameter, CategoricalParameter)

from typing import Dict, List, Optional, Tuple, Union
from pandas import DataFrame, Series
from functools import reduce
from datetime import datetime, timedelta
from freqtrade.persistence import Trade

# Get rid of pandas warnings during backtesting
import pandas as pd
import pandas_ta as pta

pd.options.mode.chained_assignment = None  # default='warn'

# Strategy specific imports, files must reside in same folder as strategy
import sys
import os
from pathlib import Path

sys.path.append(str(Path(__file__).parent))

import logging
import warnings

log = logging.getLogger(__name__)
# log.setLevel(logging.DEBUG)
warnings.simplefilter(action='ignore', category=pd.errors.PerformanceWarning)

import custom_indicators as cta
from finta import TA as fta

import keras
from keras import layers
from tqdm import tqdm
from tqdm.keras import TqdmCallback

import random

from Predict_LSTM import Predict_LSTM


"""
####################################################################################
Transformer_Price - uses a Transformer neural network to try and predict the future stock price
      
      This works by creating a Transformer model that we train on the historical data, then use that model to predict 
      future values
      
      Note that this is very slow because we are training and running a neural network. 
      This strategy is likely not viable on a configuration of more than a few pairs, and even then needs
      a fast computer, preferably with a GPU
      
      In addition to the normal freqtrade packages, these strategies also require the installation of:
        finta
        keras
        tensorflow
        tqdm

####################################################################################
"""

# this inherits from Predict_LSTM and just replaces the model used for predictions

class Transformer_Price(Predict_LSTM):
    plot_config = {
        'main_plot': {
            #'close': {'color': 'green'},
            'smooth': {'color': 'teal'},
            'predict': {'color': 'lightpink'},
        },
        'subplots': {
            "Diff": {
                'predict_diff': {'color': 'blue'},
            },
        }
    }


    # These parameters control much of the behaviour because they control the generation of the training data
    # Unfortunately, these cannot be hyperopt params because they are used in populate_indicators, which is only run
    # once during hyperopt
    # lookahead_hours = 1.0
    lookahead_hours = 0.4
    n_profit_stddevs = 0.0
    n_loss_stddevs = 0.0
    min_f1_score = 0.70
    max_train_loss = 0.15

    curr_lookahead = int(12 * lookahead_hours)

    curr_pair = ""
    custom_trade_info = {}


    ###################################

    # Strategy Specific Variable Storage

    ## Hyperopt Variables

    # LSTM hyperparams

    # Custom Sell Profit (formerly Dynamic ROI)
    cexit_roi_type = CategoricalParameter(['static', 'decay', 'step'], default='step', space='sell', load=True,
                                          optimize=True)
    cexit_roi_time = IntParameter(720, 1440, default=720, space='sell', load=True, optimize=True)
    cexit_roi_start = DecimalParameter(0.01, 0.05, default=0.01, space='sell', load=True, optimize=True)
    cexit_roi_end = DecimalParameter(0.0, 0.01, default=0, space='sell', load=True, optimize=True)
    cexit_trend_type = CategoricalParameter(['rmi', 'ssl', 'candle', 'any', 'none'], default='any', space='sell',
                                            load=True, optimize=True)
    cexit_pullback = CategoricalParameter([True, False], default=True, space='sell', load=True, optimize=True)
    cexit_pullback_amount = DecimalParameter(0.005, 0.03, default=0.01, space='sell', load=True, optimize=True)
    cexit_pullback_respect_roi = CategoricalParameter([True, False], default=False, space='sell', load=True,
                                                      optimize=True)
    cexit_endtrend_respect_roi = CategoricalParameter([True, False], default=False, space='sell', load=True,
                                                      optimize=True)

    # Custom Stoploss
    cstop_loss_threshold = DecimalParameter(-0.05, -0.01, default=-0.03, space='sell', load=True, optimize=True)
    cstop_bail_how = CategoricalParameter(['roc', 'time', 'any', 'none'], default='none', space='sell', load=True,
                                          optimize=True)
    cstop_bail_roc = DecimalParameter(-5.0, -1.0, default=-3.0, space='sell', load=True, optimize=True)
    cstop_bail_time = IntParameter(60, 1440, default=720, space='sell', load=True, optimize=True)
    cstop_bail_time_trend = CategoricalParameter([True, False], default=True, space='sell', load=True, optimize=True)
    cstop_max_stoploss = DecimalParameter(-0.30, -0.01, default=-0.10, space='sell', load=True, optimize=True)

    ################################

    # see https://www.kaggle.com/code/fredblair/transformers-for-stocks
    def transformer_encoder(self, inputs, head_size, num_heads, ff_dim, dropout=0):

        # Normalization and Attention
        # "EMBEDDING LAYER"
        x = layers.LayerNormalization(epsilon=1e-6)(inputs)

        # "ATTENTION LAYER"
        x = layers.MultiHeadAttention(
            key_dim=head_size, num_heads=num_heads, dropout=dropout
        )(x, x)
        x = layers.Dropout(dropout)(x)
        res = x + inputs

        # FEED FORWARD Part - you can stick anything here or just delete the whole section - it will still work.
        x = layers.LayerNormalization(epsilon=1e-6)(res)
        x = layers.Conv1D(filters=ff_dim, kernel_size=1, activation="relu")(x)
        x = layers.Dropout(dropout)(x)
        x = layers.Conv1D(filters=inputs.shape[-1], kernel_size=1)(x)
        return x + res

    def build_model(self, input_shape,
                    head_size,
                    num_heads,
                    ff_dim,
                    num_transformer_blocks,
                    mlp_units,
                    dropout=0.0,
                    mlp_dropout=0.0
                    ):
        inputs = keras.Input(shape=input_shape)
        x = inputs

        for _ in range(num_transformer_blocks):  # This is what stacks our transformer blocks
            x = self.transformer_encoder(x, head_size, num_heads, ff_dim, dropout)

        x = layers.GlobalAveragePooling1D(data_format="channels_first")(x)
        for dim in mlp_units:
            x = layers.Dense(dim, activation="elu")(x)
            x = layers.Dropout(mlp_dropout)(x)
        outputs = layers.Dense(1, activation="linear")(x)  # this is a pass-through

        return keras.Model(inputs, outputs)

    ################################

    def get_lstm(self, nfeatures: int, seq_len: int):
        model = keras.Sequential()

        input_shape = (seq_len, nfeatures)

        model = self.build_model(
            input_shape,
            head_size=46,  # Embedding size for attention
            num_heads=nfeatures,  # Number of attention heads
            ff_dim=nfeatures,  # Hidden layer size in feed forward network inside transformer
            num_transformer_blocks=5,
            mlp_units=[256],
            mlp_dropout=0.4,
            dropout=0.14,
        )

        model.summary()  # helps keep track of which model is running, while making changes
        model.compile(optimizer="RMSprop",
                      loss='mean_squared_error',
                      metrics=[keras.metrics.MeanAbsoluteError(), 'accuracy'])
        return model

    ################################
