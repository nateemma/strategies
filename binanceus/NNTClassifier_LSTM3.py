# Neural Network Trinary Classifier: this subclass uses a combo CNN/LSTM model


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



class NNTClassifier_LSTM3(ClassifierKerasTrinary):
    is_trained = False
    clean_data_required = False  # training data cannot contain anomalies

    # override the build_model function in subclasses
    def create_model(self, seq_len, num_features):

        model = keras.Sequential(name=self.name)

        # NOTE: don't use relu with LSTMs, cannot use GPU if you do (much slower). Use tanh

        model.add(layers.Conv1D(64, kernel_size=3, padding='same', activation='relu', input_shape=(seq_len, num_features)))
        model.add(layers.Conv1D(128, kernel_size=3, padding='same', activation='relu'))
        # model.add(layers.MaxPooling1D(pool_size=2))
        model.add(layers.BatchNormalization())
        model.add(layers.LSTM(128, activation='tanh', return_sequences=True))

        # last layer is a trinary decision - do not change
        model.add(layers.Dense(3, activation="softmax"))

        return model

