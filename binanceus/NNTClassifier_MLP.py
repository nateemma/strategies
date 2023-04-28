# Neural Network Trinary Classifier: this subclass uses a simple Multi-Layer Perceptron model

# NOTE: recommend using seq_len of 1 with MLPs

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


class NNTClassifier_MLP(ClassifierKerasTrinary):
    is_trained = False
    clean_data_required = False  # training data cannot contain anomalies

    # override the build_model function in subclasses
    def create_model(self, seq_len, num_features):

        model = keras.Sequential(name=self.name)

        # very simple MLP model:
        model.add(layers.Dense(128, input_shape=(seq_len, num_features)))
        model.add(layers.Dropout(rate=0.1))
        model.add(layers.Dense(32))
        model.add(layers.Dropout(rate=0.1))

        # last layer is a trinary decision - do not change
        model.add(layers.Dense(3, activation="softmax"))

        return model
