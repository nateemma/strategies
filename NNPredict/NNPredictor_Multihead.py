# Neural Network Binary Classifier: this subclass uses a Transformer model


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
# from tensorflow.python.keras import layers
from utils.ClassifierKerasLinear import ClassifierKerasLinear

import h5py


class NNPredictor_Multihead(ClassifierKerasLinear):
    is_trained = False
    clean_data_required = False  # training data can contain anomalies
    model_per_pair = False # separate model per pair

    # override the build_model function in subclasses
    def create_model(self, seq_len, num_features):

        dropout = 0.1

        input_shape = (seq_len, num_features)
        inputs = tf.keras.Input(shape=input_shape)
        x = inputs
        x = tf.keras.layers.LSTM(64, return_sequences=True, activation='tanh', input_shape=input_shape)(x)
        x = tf.keras.layers.Dropout(dropout)(x)
        x = tf.keras.layers.Dense(num_features)(x)
        x = tf.keras.layers.LayerNormalization(epsilon=1e-6)(x)

        # "ATTENTION LAYER"
        # x = tf.keras.layers.MultiHeadAttention(key_dim=num_features, num_heads=3, dropout=dropout)(x, x, x)
        x = tf.keras.layers.MultiHeadAttention(key_dim=num_features, num_heads=3, dropout=dropout)(x, x)
        x = tf.keras.layers.Dropout(0.1)(x)
        res = x + inputs

        # FEED FORWARD Part - you can stick anything here or just delete the whole section - it will still work.
        x = tf.keras.layers.LayerNormalization(epsilon=1e-6)(res)
        x = tf.keras.layers.Conv1D(filters=seq_len, kernel_size=1, activation="relu")(x)
        x = tf.keras.layers.Dropout(dropout)(x)
        x = tf.keras.layers.Conv1D(filters=num_features, kernel_size=1)(x)
        x = x + res


        # remplace sequence column with the average value
        x = tf.keras.layers.GlobalAveragePooling1D()(x)

        x = tf.keras.layers.Dense(16)(x)
        x = tf.keras.layers.Dropout(0.1)(x)

        # last layer is a linear decision - do not change
        outputs = tf.keras.layers.Dense(1, activation="linear")(x)

        model = tf.keras.Model(inputs, outputs, name=self.name)

        return model
