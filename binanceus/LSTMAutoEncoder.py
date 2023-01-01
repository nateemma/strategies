# class that implements an Auto-Encoder for dimensional reduction of a panda dataframe
# This can be used as-is, and can also be sub-classed - just override the init function and create
# different encoder and decoder variables


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

import h5py
from AutoEncoder import AutoEncoder

class LSTMAutoEncoder(AutoEncoder):

    encoder: keras.Model =  None
    decoder: keras.Model =  None
    autoencoder: keras.Model =  None

    # these will be overwritten by the specific autoencoder
    name = "LSTMAutoEncoder"
    latent_dim = 32
    num_features = 128

    # override the build_model function in subclasses
    def build_model(self, num_features, num_dims):
        # default autoencoder is a (fairly) simple LSTM layers


        self.encoder = keras.Sequential(name=self.name+"_encoder")
        self.encoder.add(layers.Dense(64, activation='relu', input_shape=(1, num_features)))
        self.encoder.add(layers.Dropout(rate=0.1))
        self.encoder.add(layers.LSTM(32, return_sequences=True, activation='relu'))
        self.encoder.add(layers.Dropout(rate=0.1))
        # self.encoder.add(layers.BatchNormalization(64))
        # self.encoder.add(layers.Activation(64))
        self.encoder.add(layers.Dense(24, activation='relu'))
        self.encoder.add(layers.Dropout(rate=0.1))
        self.encoder.add(layers.Dense(self.latent_dim, activation='relu', name='encoder_output'))

        self.decoder = keras.Sequential(name=self.name+"_decoder")
        self.decoder.add(layers.Dense(24, activation='relu', input_shape=(1, num_dims)))
        self.encoder.add(layers.Dropout(rate=0.1))
        # self.decoder.add(layers.Activation(64))
        # self.decoder.add(layers.BatchNormalization(64))
        self.decoder.add(layers.LSTM(32, return_sequences=True, activation='relu'))
        self.encoder.add(layers.Dropout(rate=0.1))
        self.decoder.add(layers.Dense(64, activation='sigmoid'))
        self.encoder.add(layers.Dropout(rate=0.1))
        self.decoder.add(layers.Dense(num_features, activation=None))

        self.autoencoder = keras.Sequential(name=self.name)
        self.autoencoder.add(self.encoder)
        self.autoencoder.add(self.decoder)

        # self.autoencoder = keras.Model(inputs=self.encoder.input,
        #                                outputs=self.decoder(self.encoder.output),
        #                                name=self.name)
        self.autoencoder.compile(metrics=['accuracy', 'mse'], loss='mse', optimizer='adam')

        self.update_model_weights()

        return
