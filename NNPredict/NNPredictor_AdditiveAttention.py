# Neural Network Binary Classifier: this subclass uses an Additive Attention model


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
# from tensorflow.python.keras import layers, Model
from utils.ClassifierKerasLinear import ClassifierKerasLinear
# from Attention import Attention

import h5py


class NNPredictor_AdditiveAttention(ClassifierKerasLinear):
    is_trained = False
    clean_data_required = False  # training data can contain anomalies
    model_per_pair = False # separate model per pair

    # override the build_model function in subclasses
    def create_model(self, seq_len, num_features):
        # Attention (Single Head)

        inputs = tf.keras.layers.Input(shape=(seq_len, num_features))

        x = tf.keras.layers.LSTM(num_features, activation='tanh', return_sequences=True,
                        input_shape=(seq_len, num_features))(inputs)
        x = tf.keras.layers.Dropout(0.2)(x)
        x = tf.keras.layers.BatchNormalization()(x)

        # x = tf.keras.layers.AdditiveAttention()([x, inputs])
        x = tf.keras.layers.AdditiveAttention()([x, x])

        x = tf.keras.layers.LSTM(1, activation='tanh')(x)

        # remplace sequence column with the average value
        # x = tf.keras.layers.GlobalAveragePooling1D()(x)

        # x = tf.keras.layers.Dense(32)(x)

         # last layer is linear - do not change
        # x = tf.keras.layers.Dropout(0.2)(x)
        outputs = tf.keras.layers.Dense(1, activation="linear")(x)

        model = tf.keras.Model(inputs, outputs, name=self.name)

        return model
