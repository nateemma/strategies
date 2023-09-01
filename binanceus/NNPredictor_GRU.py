# Neural Network Binary Classifier: this subclass uses a simple GRU model


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
from ClassifierKerasLinear import ClassifierKerasLinear

import h5py


class NNPredictor_GRU(ClassifierKerasLinear):
    is_trained = False
    clean_data_required = False  # training data can contain anomalies
    model_per_pair = False # separate model per pair

    # override the build_model function in subclasses
    def create_model(self, seq_len, num_features):

        model = tf.keras.Sequential(name=self.name)

        inputs = tf.keras.Input(shape=(seq_len, num_features))
        x = inputs
        x = layers.Conv1D(filters=64, kernel_size=2, activation="relu", padding="causal")(x)
        x = layers.GRU(32, return_sequences=True)(x)

        # last layer is a linear (float) value - do not change
        outputs = layers.Dense(1, activation="linear")(x)

        model = tf.keras.Model(inputs, outputs, name=self.name)

        return model
