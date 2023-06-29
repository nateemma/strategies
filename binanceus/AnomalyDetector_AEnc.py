# class that implements an Anomaly detection autoencoder
# The idea is to train the autoencoder on data that does NOT contain the signal you are looking for (buy/sell)
# Then when the autoencoder tries to predict the transform, anything with unusual error is considered to be an 'anomoly'


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

import h5py
from ClassifierKerasEncoder import ClassifierKerasEncoder


class AnomalyDetector_AEnc(ClassifierKerasEncoder):

    is_trained = False
    clean_data_required = True # training data cannot contain anomalies

    # the following affect training of the model.
    seq_len = 8  # 'depth' of training sequence
    num_epochs = 512  # number of iterations for training
    batch_size = 1024  # batch size for training

    # override the build_model function in subclasses
    def create_model(self, seq_len, num_features):

        print("    Creating model: ", self.name)
        model = tf.keras.Sequential(name=self.name)

        outer_dim = 64
        inner_dim = 16
        
        # Encoder
        model.add(layers.Dense(outer_dim, activation='relu', input_shape=(seq_len, num_features)))
        # model.add(layers.Dropout(rate=0.2))
        model.add(layers.Dense(2*outer_dim, activation='relu'))
        # model.add(layers.Dropout(rate=0.2))
        # model.add(layers.Dense(32, activation='elu'))
        # model.add(layers.Dropout(rate=0.2))
        model.add(layers.Dense(inner_dim, activation='relu', name=self.encoder_layer))

        # Decoder
        model.add(layers.Dense(2*outer_dim, activation='relu', input_shape=(seq_len, inner_dim)))
        # model.add(layers.Dropout(rate=0.2))
        model.add(layers.Dense(outer_dim, activation='relu'))
        # model.add(layers.Dropout(rate=0.2))
        # model.add(layers.Dense(128, activation='elu'))
        model.add(layers.Dense(self.num_features, activation=None))

        return model
