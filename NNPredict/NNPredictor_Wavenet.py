# Neural Network Predictor: this subclass uses a simplified Wavenet model
# Note that this is a very big (and slow to train model)


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
from utils.ClassifierKerasLinear import ClassifierKerasLinear

import h5py


class NNPredictor_Wavenet(ClassifierKerasLinear):
    is_trained = False
    clean_data_required = False  # training data can contain anomalies
    model_per_pair = False # separate model per pair

    def wavenetBlock(self, n_filters, filter_size, rate):
        def f(input_):
            residual = input_
            tanh_out = tf.keras.layers.Convolution1D(n_filters, filter_size, padding="causal", dilation_rate=rate,
                                            activation='tanh')(input_)
            sigmoid_out = tf.keras.layers.Convolution1D(n_filters, filter_size, padding="causal", dilation_rate=rate,
                                               activation='sigmoid')(input_)
            merged = tf.keras.layers.Multiply()([tanh_out, sigmoid_out])

            x = tf.keras.layers.Multiply()([tanh_out, sigmoid_out])

            res_x = tf.keras.layers.Convolution1D(n_filters, 1, padding='same', kernel_regularizer=tf.keras.regularizers.l2(0))(x)
            skip_x = tf.keras.layers.Convolution1D(n_filters, 1, padding='same', kernel_regularizer=tf.keras.regularizers.l2(0))(x)
            res_x = tf.keras.layers.Add()([input_, res_x])
            return res_x, skip_x

        return f

    # override the build_model function in subclasses
    def create_model(self, seq_len, num_features):
        inputs = tf.keras.layers.Input(shape=(seq_len, num_features))

        x = inputs
        # x = tf.keras.layers.Convolution1D(64, 2, padding="causal", dilation_rate=1)(inputs)

        # Wavenet model, which is a series of convolutional layers with increasing dilution rate:
        # This is a greatly simplified version
        for rate in (1, 2, 4, 8) * 2:
            x = tf.keras.layers.Conv1D(filters=64, kernel_size=2, padding="causal", activation="relu", dilation_rate=rate)(x)

        # remplace sequence column with the average value
        x = tf.keras.layers.GlobalAveragePooling1D()(x)

        # intermediate layer to bring the dimensions
        x = tf.keras.layers.Dense(16)(x)
        x = tf.keras.layers.Dropout(0.1)(x)
        
        # last layer is a linear (float) value - do not change
        outputs = tf.keras.layers.Dense(1, activation="linear")(x)

        model = tf.keras.Model(inputs, outputs, name=self.name)

        return model
