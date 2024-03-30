# Neural Network Binary Classifier: this subclass uses a Temporal Convolutional Neural Network (TCN) model


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

from TCN import TCN


class NNPredictor_TCN(ClassifierKerasLinear):

    # override the build_model function in subclasses
    def create_model(self, seq_len, num_features):

        inputs = tf.keras.layers.Input(shape=(seq_len, num_features))


        x = tf.keras.layers.LSTM(32, activation='tanh', return_sequences=True)(inputs)

        nf = x[:, -1, :].shape[-1]
        # print(f'nf: {nf}')

        # x = TCN(nb_filters=num_features, kernel_size=seq_len, return_sequences=False, activation='tanh')(inputs)
        # x = TCN(nb_filters=num_features, return_sequences=False, activation='tanh')(inputs)
        x = TCN(nb_filters=nf, return_sequences=False, kernel_initializer='glorot_uniform',  use_layer_norm=True)(x)

        # reduce dimensions
        # x = tf.keras.layers.LSTM(1, activation='tanh')(x)
        # x = tf.keras.layers.Dense(32)(x)
        # x = tf.keras.layers.Dense(16)(x)
        x = tf.keras.layers.Dense(8)(x)

        # last layer is a linear (float) value - do not change
        outputs = tf.keras.layers.Dense(1, activation="linear")(x)

        model = tf.keras.Model(inputs, outputs, name=self.name)

        return model
