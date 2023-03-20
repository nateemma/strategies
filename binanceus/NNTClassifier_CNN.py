# Neural Network Trinary Classifier: this subclass uses a simple Convolutional model


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

import h5py


class NNTClassifier_CNN(ClassifierKerasTrinary):
    is_trained = False
    clean_data_required = False  # training data cannot contain anomalies

    # override the build_model function in subclasses
    def create_model(self, seq_len, num_features):

        dropout = 0.1
        n_filters = (8, seq_len, seq_len)

        inputs = keras.Input(shape=(seq_len, num_features))
        x = inputs

        x = keras.layers.Dense(64, input_shape=(seq_len, num_features))(x)
        x = keras.layers.BatchNormalization()(x)

        x = keras.layers.Conv1D(filters=64, kernel_size=2, activation='tanh')(x)
        x = keras.layers.MaxPooling1D(pool_size=2)(x)
        x = keras.layers.Flatten()(x)
        x = keras.layers.Dropout(dropout)(x)
        x = keras.layers.BatchNormalization()(x)

        # intermediate layer to bring down the dimensions
        x = keras.layers.Dense(16)(x)
        x = keras.layers.Dropout(0.1)(x)

        # last layer is a linear trinary decision - do not change
        outputs = layers.Dense(3, activation="softmax")(x)

        model = keras.Model(inputs, outputs)

        return model
