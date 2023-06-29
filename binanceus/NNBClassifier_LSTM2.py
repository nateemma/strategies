# Neural Network Binary Classifier: this subclass uses a complex LSTM model


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
from ClassifierKerasBinary import ClassifierKerasBinary

import h5py


class NNBClassifier_LSTM2(ClassifierKerasBinary):
    is_trained = False
    clean_data_required = True  # training data cannot contain anomalies

    # override the build_model function in subclasses
    def create_model(self, seq_len, num_features):

        model = tf.keras.Sequential(name=self.name)

        # NOTE: don't use relu with LSTMs, cannot use GPU if you do (much slower). Use tanh

        model.add(layers.LSTM(64, activation='tanh', return_sequences=True, input_shape=(seq_len, num_features)))
        model.add(layers.Dropout(rate=0.2))
        model.add(layers.LSTM(64, return_sequences=True, activation='tanh'))
        model.add(layers.Dropout(rate=0.2))
        model.add(layers.LSTM(32, return_sequences=True, activation='tanh'))
        model.add(layers.Dropout(rate=0.2))
        model.add(layers.LSTM(16, return_sequences=True, activation='tanh'))
        model.add(layers.Dropout(rate=0.2))

        # last layer is a binary decision - do not change
        model.add(layers.Dense(1, activation='sigmoid'))

        return model
