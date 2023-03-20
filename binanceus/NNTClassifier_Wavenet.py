# Neural Network Trinary Classifier: this subclass uses a Wavenet model


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


class NNTClassifier_Wavenet(ClassifierKerasTrinary):
    is_trained = False
    clean_data_required = False  # training data cannot contain anomalies

    # override the build_model function in subclasses
    def create_model(self, seq_len, num_features):

        model = keras.Sequential(name=self.name)
        model.add(layers.Input(shape=(seq_len, num_features)))

        # Wavenet model, which is a series of convolutional layers with increasing dilution rate:
        for rate in (1, 2, 4, 8) * 2:
            model.add(layers.Conv1D(filters=64, kernel_size=2, padding="causal", activation="relu", dilation_rate=rate))

        # last layer is a trinary decision - do not change
        model.add(layers.Dense(3, activation="softmax"))

        return model
