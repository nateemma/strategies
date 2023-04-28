# Neural Network Trinary Classifier: this subclass implements an ensemble of different techniques


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

import keras
from keras import layers
from ClassifierKerasTrinary import ClassifierKerasTrinary



class NNTClassifier_Ensemble(ClassifierKerasTrinary):
    is_trained = False
    clean_data_required = False  # training data cannot contain anomalies

    # override the build_model function in subclasses
    def create_model(self, seq_len, num_features):

        dropout = 0.2

        inputs = keras.Input(shape=(seq_len, num_features))

        # run inputs through a few different types of model
        x1 = self.get_lstm(inputs, seq_len, num_features)
        x2 = self.get_gru(inputs, seq_len, num_features)
        x3 = self.get_cnn(inputs, seq_len, num_features)
        x4 = self.get_simple_wavenet(inputs, seq_len, num_features)

        # combine the outputs of the models
        x_combined = layers.Concatenate()([x1, x2, x3, x4])

        # run an LSTM to learn from the combined models
        x = layers.LSTM(3, activation='tanh', return_sequences=True)(x_combined)
        # x = layers.Dropout(rate=0.1)(x)

        # last layer is a trinary decision - do not change
        outputs = layers.Dense(3, activation="softmax")(x)

        model = keras.Model(inputs, outputs)

        return model


    def get_lstm(self, inputs, seq_len, num_features):
        x = layers.LSTM(64, activation='tanh', return_sequences=True, input_shape=(seq_len, num_features))(inputs)
        # x = layers.Dropout(rate=0.1)(x)
        x = layers.Dense(3, activation="softmax")(x)
        return x

    def get_gru(self, inputs, seq_len, num_features):
        x = layers.Conv1D(filters=64, kernel_size=2, activation="relu", padding="causal")(inputs)
        x = layers.GRU(32, return_sequences=True)(x)
        # x = layers.Dropout(rate=0.1)(x)
        x = layers.Dense(3, activation="softmax")(x)
        return x

    def get_cnn(self, inputs, seq_len, num_features):
        x = keras.layers.Conv1D(filters=64, kernel_size=2, activation='tanh',  padding="causal")(inputs)
        x = keras.layers.Dropout(0.1)(x)
        x = keras.layers.BatchNormalization()(x)

        # intermediate layer to bring down the dimensions
        x = keras.layers.Dense(16)(x)
        # x = keras.layers.Dropout(0.1)(x)
        x = layers.Dense(3, activation="softmax")(x)
        return x

    def get_simple_wavenet(self, inputs, seq_len, num_features):
        x = inputs
        for rate in (1, 2, 4, 8) * 2:
            x = layers.Conv1D(filters=64, kernel_size=2, padding="causal", activation="relu", dilation_rate=rate)(x)

        x = layers.Dense(3, activation="softmax")(x)
        return x
