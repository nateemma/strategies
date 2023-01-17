# Neural Network Binary Classifier: this subclass uses a simple MLP model


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
from ClassifierKerasLinear import ClassifierKerasLinear

import h5py


class NNPredictor_MLP(ClassifierKerasLinear):
    is_trained = False
    clean_data_required = False  # training data can contain anomalies
    model_per_pair = False # separate model per pair

    # override the build_model function in subclasses
    def create_model(self, seq_len, num_features):

        # model = keras.Sequential(name=self.name)
        #
        # # simple MLP :
        # dropout = 0.1
        # model.add(layers.Dense(156, input_shape=(seq_len, num_features)))
        # model.add(layers.Dropout(rate=dropout))
        # model.add(layers.Dense(16))
        # model.add(layers.Dropout(rate=dropout))
        #
        # # last layer is a linear (float) value - do not change
        # model.add(layers.Dense(1, activation='linear'))

        inputs = keras.Input(shape=(seq_len, num_features))
        x = inputs
        x = keras.layers.Dense(156)(x)
        x = keras.layers.Dropout(0.1)(x)
        x = keras.layers.Dense(16)(x)
        x = keras.layers.Dropout(0.1)(x)

        # last layer is a linear (float) value - do not change
        outputs = layers.Dense(1, activation="linear")(x)

        model = keras.Model(inputs, outputs)

        return model
