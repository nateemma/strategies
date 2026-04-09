# class that implements an Anomaly detection autoencoder, based on an LSTM model
# This version uses sequences of data, so do not feed it 'point' data, it must be time ordered


import numpy as np
from pandas import DataFrame, Series
import pandas as pd

pd.options.mode.chained_assignment = None  # default='warn'

# Strategy specific imports, files must reside in same folder as strategy
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent))

import logging
import warnings

log = logging.getLogger(__name__)
# log.setLevel(logging.DEBUG)
warnings.simplefilter(action='ignore', category=pd.errors.PerformanceWarning)

import random

import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
os.environ['TF_DETERMINISTIC_OPS'] = '1'

import tensorflow as tf

seed = 42
os.environ['PYTHONHASHSEED'] = str(seed)
random.seed(seed)
tf.random.set_seed(seed)
np.random.seed(seed)

tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.WARN)

#import keras
from keras import layers
from ClassifierKerasEncoder import ClassifierKerasEncoder

import h5py


class AnomalyDetector_LSTM(ClassifierKerasEncoder):
    is_trained = False
    clean_data_required = True  # training data cannot contain anomalies

    # override the build_model function in subclasses
    def create_model(self, seq_len, num_features):
        outer_dim = 64
        inner_dim = 16

        model = tf.keras.Sequential(name=self.name)

        # NOTE: don't use relu with LSTMs, cannot use GPU if you do (much slower). Use tanh

        # Encoder
        model.add(
            layers.LSTM(outer_dim, return_sequences=True, activation='tanh', input_shape=(seq_len, num_features))
        )
        # model.add(layers.RepeatVector(self.num_features))

        # model.add(layers.Dropout(rate=0.2))
        model.add(
            layers.LSTM(int(outer_dim / 2), return_sequences=True, activation='tanh')
        )
        # model.add(layers.Dropout(rate=0.2))
        model.add(layers.Dense(inner_dim, activation='tanh', name=self.encoder_layer))

        # Decoder
        model.add(
            layers.LSTM(int(outer_dim / 2), return_sequences=True, activation='tanh', input_shape=(seq_len, inner_dim))
        )
        # model.add(layers.Dropout(rate=0.2))
        model.add(
            layers.LSTM(outer_dim, return_sequences=True, activation='tanh')
        )
        # model.add(layers.Dropout(rate=0.2))
        model.add(layers.Dense(self.num_features, activation=None))

        return model
