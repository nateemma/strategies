# Neural Network Binary Classifier: this subclass uses a Transformer model


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
import keras_nlp

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


class NNPredictor_Transformer2(ClassifierKerasLinear):
    is_trained = False
    clean_data_required = False  # training data can contain anomalies
    model_per_pair = False # separate model per pair

    # override the build_model function in subclasses
    def create_model(self, seq_len, num_features):

        head_size = num_features
        num_heads = 4
        # ff_dim = 4
        ff_dim = seq_len
        num_transformer_blocks = 4
        mlp_units = [128]
        mlp_dropout = 0.4
        dropout = 0.2

        # This uses the keras_nlp Transformer.
        # you need to run pip install keras_transformer to get this

        model = keras.Sequential(name=self.name)

        model.add(
            keras_nlp.layers.TransformerEncoder(num_layers=2, hidden_size=64, num_attention_heads=2,
                                      input_shape=(seq_len, num_features))
        )
        # last layer is a linear (float) value - do not change
        model.add(layers.Dense(1, activation='linear'))

        return model

