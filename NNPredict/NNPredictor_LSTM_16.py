# Neural Network Binary Classifier: this subclass uses a simple LSTM model


import numpy as np
from pandas import DataFrame, Series
import pandas as pd

pd.options.mode.chained_assignment = None  # default='warn'

# Strategy specific imports, files must reside in same folder as strategy
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent))
sys.path.append(str(Path(__file__).parent.parent))

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
from utils.ClassifierKerasLinear import ClassifierKerasLinear

import h5py


class NNPredictor_LSTM_16(ClassifierKerasLinear):
    is_trained = False
    clean_data_required = False  # training data can contain anomalies
    model_per_pair = False # separate model per pair

    # override the build_model function in subclasses
    def create_model(self, seq_len, num_features):

        print(f"    Creating model: {self.name}")

        model = tf.keras.Sequential(name=self.name)

        # NOTE: don't use relu with LSTMs, cannot use GPU if you do (much slower). Use tanh

        # simplest possible LSTM :
        # model.add(layers.LSTM(64, return_sequences=True, activation='tanh', input_shape=(seq_len, num_features)))
        # model.add(layers.Dropout(rate=0.1))
        #
        # # last layer is a linear (float) value - do not change
        # model.add(layers.Dense(1, activation='linear'))

        inputs = tf.keras.Input(shape=(seq_len, num_features))
        x = inputs
        # x = tf.keras.layers.Dense(64)(x)

        x = tf.keras.layers.LSTM(16, return_sequences=True, activation='tanh', input_shape=(seq_len, num_features))(x)
        x = tf.keras.layers.Dropout(0.1)(x)
        x = tf.keras.layers.BatchNormalization()(x)

        # remplace sequence column with the average value
        x = tf.keras.layers.GlobalAveragePooling1D()(x)

        # intermediate layer to bring the dimensions
        x = tf.keras.layers.Dense(4)(x)
        x = tf.keras.layers.Dropout(0.1)(x)

        # last layer is a linear (float) value - do not change
        outputs = tf.keras.layers.Dense(1, activation="linear")(x)

        model = tf.keras.Model(inputs, outputs, name=self.name)

        return model
