# Neural Network Trinary Classifier: this subclass uses a Temporal Fusion Transformer model


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
# import tensorflow_probability as tfp
from ClassifierKerasTrinary import ClassifierKerasTrinary



class NNTClassifier_TFT(ClassifierKerasTrinary):
    is_trained = False
    clean_data_required = False  # training data cannot contain anomalies

    # override the builnum_features function in subclasses
    def create_model(self, seq_len, num_features):

        head_size = num_features
        # num_heads = int(num_features / 2)
        num_heads = 3
        ff_dim = 4
        # ff_dim = seq_len
        # num_transformer_blocks = seq_len
        num_transformer_blocks = 4
        mlp_units = [32, 8]
        mlp_dropout = 0.1
        dropout_rate = 0.1
        hidden_layer_size = num_features
        num_outputs = 3
        num_encoder_steps = seq_len
        num_decoder_steps = num_encoder_steps

        # inputs = keras.Input(shape=(seq_len, num_features))

        # Note that this requires 2 inputs, so need to override the train method
        encoder_inputs = keras.Input(shape=(num_encoder_steps, num_features))
        encoder_l1 = layers.LSTM(hidden_layer_size, return_sequences=True)(encoder_inputs)
        encoder_l2 = layers.LSTM(hidden_layer_size, return_sequences=True)(encoder_l1)

        decoder_inputs = keras.Input(shape=(num_decoder_steps, num_features))
        decoder_l1 = layers.LSTM(hidden_layer_size, return_sequences=True)(decoder_inputs)
        decoder_l2 = layers.LSTM(hidden_layer_size, return_sequences=True)(decoder_l1)

        concatenated = layers.Concatenate(axis=1)([encoder_l2, decoder_l2])
        x = layers.Conv1D(filters=num_features, kernel_size=seq_len+1, activation="relu")(concatenated)
        out = layers.Dense(num_outputs)(x)

        # output_distribution = tfp.layers.DistributionLambda(
        #     lambda t: tfp.distributions.Normal(loc=t[..., :num_outputs],
        #                                        scale=1e-3 + tf.math.softplus(0.01 * t[..., num_outputs:])))
        #
        # out2 = output_distribution(out)

        # last layer is a trinary decision - do not change
        outputs = layers.Dense(3, activation="softmax")(out)

        # Define the model
        model = keras.Model(inputs=[encoder_inputs, decoder_inputs], outputs=outputs)

        return model
