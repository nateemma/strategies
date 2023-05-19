# Neural Network Trinary Classifier: collection of neural network classifiers, types and access functions


import numpy as np
from pandas import DataFrame, Series
import pandas as pd

pd.options.mode.chained_assignment = None  # default='warn'

# Strategy specific imports, files must reside in same folder as strategy
import sys
from pathlib import Path
from enum import Enum, auto

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


# --------------------------------------------------------------

# Types of Classifier

class ClassifierType(Enum):
    Attention = auto()  # self-Attention (Transformer Attention)
    AdditiveAttention = auto()  # Additive-Attention
    CNN = auto()  # Convolutional Neural Network
    Ensemble = auto()  # Ensemble/Stack of several Classifiers
    GRU = auto()  # Gated Recurrent Unit
    LSTM = auto()  # Long-Short Term Memory (basic)
    LSTM2 = auto()  # Two-tier LSTM
    LSTM3 = auto()  # Convolutional/LSTM Combo
    MLP = auto()  # Multi-Layer Perceptron
    Multihead = auto()  # Multihead Self-Attention
    TCN = auto()  # Temporal Convolutional Network
    Transformer = auto()  # Transformer
    Wavenet = auto()  # Simplified Wavenet
    Wavenet2 = auto()  # Full Wavenet


# --------------------------------------------------------------

# Additive Attention Classifier

class NNTClassifier_AdditiveAttention(ClassifierKerasTrinary):
    is_trained = False
    clean_data_required = False  # training data cannot contain anomalies

    # override the build_model function in subclasses
    def create_model(self, seq_len, num_features):
        # model = keras.Sequential(name=self.name)
        inputs = layers.Input(shape=(seq_len, num_features))

        x = layers.LSTM(num_features, return_sequences=True, input_shape=(seq_len, num_features))(inputs)
        # x = layers.Dropout(0.2)(x)
        # x = layers.BatchNormalization()(x)

        # x = layers.Attention()([x, inputs])
        x = layers.AdditiveAttention()([x, inputs])

        # Attention produces strange datatypes that cause issues with softmax, so use Dense layer to map/downsize
        x = layers.Dense(32)(x)

        # last layer is a trinary decision - do not change
        outputs = layers.Dense(3, activation="softmax")(x)

        model = keras.Model(inputs, outputs)

        # model.summary()

        return model


# --------------------------------------------------------------

# Self-Attention Classifier

class NNTClassifier_Attention(ClassifierKerasTrinary):
    is_trained = False
    clean_data_required = False  # training data cannot contain anomalies

    # override the build_model function in subclasses
    def create_model(self, seq_len, num_features):
        # model = keras.Sequential(name=self.name)
        inputs = layers.Input(shape=(seq_len, num_features))

        x = layers.LSTM(num_features, return_sequences=True, input_shape=(seq_len, num_features))(inputs)
        x = layers.Dropout(0.2)(x)
        x = layers.BatchNormalization()(x)

        # x = layers.Attention()([x, inputs])
        x = layers.Attention()([x, x])

        # Attention produces strange datatypes that cause issues with softmax, so use Dense layer to map/downsize
        x = layers.Dense(32)(x)

        # last layer is a trinary decision - do not change
        outputs = layers.Dense(3, activation="softmax")(x)

        model = keras.Model(inputs, outputs)

        # model.summary()

        return model


# --------------------------------------------------------------
# Convolutional Neural Network

class NNTClassifier_CNN(ClassifierKerasTrinary):
    is_trained = False
    clean_data_required = False  # training data cannot contain anomalies

    # override the build_model function in subclasses
    def create_model(self, seq_len, num_features):
        dropout = 0.1
        n_filters = (8, seq_len, seq_len)

        inputs = keras.Input(shape=(seq_len, num_features))
        x = inputs

        # x = keras.layers.Dense(64, input_shape=(seq_len, num_features))(x)
        # x = keras.layers.BatchNormalization()(x)

        x = keras.layers.Conv1D(filters=64, kernel_size=2, activation='tanh', padding="causal")(x)
        x = keras.layers.Dropout(dropout)(x)
        x = keras.layers.BatchNormalization()(x)

        # intermediate layer to bring down the dimensions
        x = keras.layers.Dense(16)(x)
        x = keras.layers.Dropout(0.1)(x)

        # last layer is a linear trinary decision - do not change
        outputs = layers.Dense(3, activation="softmax")(x)

        model = keras.Model(inputs, outputs)

        return model


# --------------------------------------------------------------
# Ensemble/Stack of several Classifiers

class NNTClassifier_Ensemble(ClassifierKerasTrinary):
    is_trained = False
    clean_data_required = False  # training data cannot contain anomalies

    # override the build_model function in subclasses
    def create_model(self, seq_len, num_features):
        dropout = 0.2

        inputs = keras.Input(shape=(seq_len, num_features))

        # run inputs through a few different types of model
        x1 = self.get_lstm(inputs, seq_len, num_features)
        x2 = self.get_gru(inputs, seq_len, num_features)
        x3 = self.get_cnn(inputs, seq_len, num_features)
        x4 = self.get_simple_wavenet(inputs, seq_len, num_features)

        # combine the outputs of the models
        x_combined = layers.Concatenate()([x1, x2, x3, x4])

        # run an LSTM to learn from the combined models
        x = layers.LSTM(3, activation='tanh', return_sequences=True)(x_combined)
        # x = layers.Dropout(rate=0.1)(x)

        # last layer is a trinary decision - do not change
        outputs = layers.Dense(3, activation="softmax")(x)

        model = keras.Model(inputs, outputs)

        return model

    def get_lstm(self, inputs, seq_len, num_features):
        x = layers.LSTM(64, activation='tanh', return_sequences=True, input_shape=(seq_len, num_features))(inputs)
        # x = layers.Dropout(rate=0.1)(x)
        x = layers.Dense(3, activation="softmax")(x)
        return x

    def get_gru(self, inputs, seq_len, num_features):
        x = layers.Conv1D(filters=64, kernel_size=2, activation="relu", padding="causal")(inputs)
        x = layers.GRU(32, return_sequences=True)(x)
        # x = layers.Dropout(rate=0.1)(x)
        x = layers.Dense(3, activation="softmax")(x)
        return x

    def get_cnn(self, inputs, seq_len, num_features):
        x = keras.layers.Conv1D(filters=64, kernel_size=2, activation='tanh', padding="causal")(inputs)
        x = keras.layers.Dropout(0.1)(x)
        x = keras.layers.BatchNormalization()(x)

        # intermediate layer to bring down the dimensions
        x = keras.layers.Dense(16)(x)
        # x = keras.layers.Dropout(0.1)(x)
        x = layers.Dense(3, activation="softmax")(x)
        return x

    def get_simple_wavenet(self, inputs, seq_len, num_features):
        x = inputs
        for rate in (1, 2, 4, 8) * 2:
            x = layers.Conv1D(filters=64, kernel_size=2, padding="causal", activation="relu", dilation_rate=rate)(x)

        x = layers.Dense(3, activation="softmax")(x)
        return x


# --------------------------------------------------------------
# Gated Recurrent Unit


class NNTClassifier_GRU(ClassifierKerasTrinary):
    is_trained = False
    clean_data_required = False  # training data cannot contain anomalies

    # override the build_model function in subclasses
    def create_model(self, seq_len, num_features):
        model = keras.Sequential(name=self.name)
        model.add(layers.Input(shape=(seq_len, num_features)))
        # model.add(layers.Conv1D(filters=64, kernel_size=2, strides=2, padding="causal", activation="relu"))
        model.add(layers.Conv1D(filters=64, kernel_size=2, activation="relu", padding="causal"))
        model.add(layers.GRU(32, return_sequences=True))

        # last layer is a trinary decision - do not change
        model.add(layers.Dense(3, activation="softmax"))

        return model


# --------------------------------------------------------------
# Long-Short Term Memory (basic)


class NNTClassifier_LSTM(ClassifierKerasTrinary):
    is_trained = False
    clean_data_required = False  # training data cannot contain anomalies

    # override the build_model function in subclasses
    def create_model(self, seq_len, num_features):
        model = keras.Sequential(name=self.name)

        # NOTE: don't use relu with LSTMs, cannot use GPU if you do (much slower). Use tanh

        model.add(layers.LSTM(128, activation='tanh', return_sequences=True, input_shape=(seq_len, num_features)))
        model.add(layers.Dropout(rate=0.2))

        # last layer is a trinary decision - do not change
        model.add(layers.Dense(3, activation="softmax"))

        return model


# --------------------------------------------------------------
# Two-tier LSTM


class NNTClassifier_LSTM2(ClassifierKerasTrinary):
    is_trained = False
    clean_data_required = False  # training data cannot contain anomalies

    # override the build_model function in subclasses
    def create_model(self, seq_len, num_features):
        model = keras.Sequential(name=self.name)

        # NOTE: don't use relu with LSTMs, cannot use GPU if you do (much slower). Use tanh

        model.add(layers.LSTM(64, activation='tanh', recurrent_dropout=0.25, return_sequences=True,
                              input_shape=(seq_len, num_features)))
        model.add(layers.Dropout(rate=0.5))

        model.add(layers.LSTM(64, activation='tanh', return_sequences=True, recurrent_dropout=0.25))
        model.add(layers.Dropout(rate=0.5))
        #
        # model.add(layers.Dense(16))

        # last layer is a trinary decision - do not change
        model.add(layers.Dense(3, activation="softmax"))

        return model


# --------------------------------------------------------------
# Convolutional/LSTM Combo


class NNTClassifier_LSTM3(ClassifierKerasTrinary):
    is_trained = False
    clean_data_required = False  # training data cannot contain anomalies

    # override the build_model function in subclasses
    def create_model(self, seq_len, num_features):
        model = keras.Sequential(name=self.name)

        # NOTE: don't use relu with LSTMs, cannot use GPU if you do (much slower). Use tanh

        model.add(
            layers.Conv1D(64, kernel_size=3, padding='same', activation='relu', input_shape=(seq_len, num_features)))
        model.add(layers.Conv1D(128, kernel_size=3, padding='same', activation='relu'))
        # model.add(layers.MaxPooling1D(pool_size=2))
        model.add(layers.BatchNormalization())
        model.add(layers.LSTM(128, activation='tanh', return_sequences=True))

        # last layer is a trinary decision - do not change
        model.add(layers.Dense(3, activation="softmax"))

        return model


# --------------------------------------------------------------
# Multi-Layer Perceptron (simple)


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


# --------------------------------------------------------------
# Multihead Self-Attention


class NNTClassifier_Multihead(ClassifierKerasTrinary):
    is_trained = False
    clean_data_required = False  # training data cannot contain anomalies

    # override the build_model function in subclasses
    def create_model(self, seq_len, num_features):
        dropout = 0.1

        input_shape = (seq_len, num_features)
        inputs = keras.Input(shape=input_shape)
        x = inputs
        x = layers.LSTM(num_features, return_sequences=True, activation='tanh', input_shape=input_shape)(x)
        x = layers.Dropout(dropout)(x)
        x = layers.Dense(num_features)(x)
        x = layers.LayerNormalization(epsilon=1e-6)(x)

        # "ATTENTION LAYER"
        x = layers.MultiHeadAttention(key_dim=num_features, num_heads=16, dropout=dropout)(x, x, x)
        # x = layers.MultiHeadAttention(key_dim=num_features, num_heads=16, dropout=dropout)(x, inputs)
        x = layers.Dropout(0.1)(x)
        res = x + inputs

        # FEED FORWARD Part - you can stick anything here or just delete the whole section - it will still work.
        x = layers.LayerNormalization(epsilon=1e-6)(res)
        x = layers.Conv1D(filters=seq_len, kernel_size=1, activation="relu")(x)
        x = layers.Dropout(dropout)(x)
        x = layers.Conv1D(filters=num_features, kernel_size=1)(x)
        x = x + res

        # last layer is a trinary decision - do not change
        outputs = layers.Dense(3, activation="softmax")(x)

        model = keras.Model(inputs, outputs)

        return model


# --------------------------------------------------------------
# Temporal Convolutional Network

from TCN import TCN


class NNTClassifier_TCN(ClassifierKerasTrinary):
    is_trained = False
    clean_data_required = False  # training data cannot contain anomalies

    # override the build_model function in subclasses
    def create_model(self, seq_len, num_features):
        # model = keras.Sequential(name=self.name)
        inputs = layers.Input(shape=(seq_len, num_features))

        x = TCN(nb_filters=num_features, kernel_size=seq_len, return_sequences=True, activation='tanh')(inputs)

        # last layer is a trinary decision - do not change
        outputs = layers.Dense(3, activation="softmax")(x)

        model = keras.Model(inputs, outputs)

        return model

    # implement custom_load() because we use a custom layer (TCN)

    def custom_load(self, path):
        model = keras.models.load_model(path, compile=False, custom_objects={'TCN': TCN})
        return model


# --------------------------------------------------------------
# Transformer


class NNTClassifier_Transformer(ClassifierKerasTrinary):
    is_trained = False
    clean_data_required = False  # training data cannot contain anomalies

    # override the build_model function in subclasses
    def create_model(self, seq_len, num_features):

        head_size = num_features
        # num_heads = int(num_features / 2)
        num_heads = 4
        ff_dim = 4
        # ff_dim = seq_len
        # num_transformer_blocks = seq_len
        num_transformer_blocks = 4
        mlp_units = [32, 8]
        mlp_dropout = 0.4
        dropout = 0.25

        inputs = keras.Input(shape=(seq_len, num_features))
        x = inputs
        for _ in range(num_transformer_blocks):
            x = self.transformer_encoder(x, head_size, num_heads, dropout, ff_dim)
            x = keras.layers.BatchNormalization()(x)

        x = layers.GlobalAveragePooling1D(keepdims=True, data_format="channels_first")(x)

        for dim in mlp_units:
            x = layers.Dense(dim)(x)
            x = layers.Dropout(mlp_dropout)(x)

        # last layer is a trinary decision - do not change
        outputs = layers.Dense(3, activation="softmax")(x)

        model = keras.Model(inputs, outputs)

        return model

    def transformer_encoder(self, inputs, head_size, num_heads, dropout, ff_dim):

        # Normalization and Attention
        x = layers.LayerNormalization(epsilon=1e-6)(inputs)
        x = layers.MultiHeadAttention(key_dim=head_size, num_heads=num_heads, dropout=dropout)(x, x)
        x = layers.Dropout(dropout)(x)

        res = x + inputs

        # Feed Forward Part
        x = layers.LayerNormalization(epsilon=1e-6)(res)
        x = layers.Conv1D(filters=ff_dim, kernel_size=1, activation="relu")(x)
        x = layers.Dropout(dropout)(x)
        x = layers.Conv1D(filters=head_size, kernel_size=1)(x)
        return x + res


# --------------------------------------------------------------
# Simplified Wavenet


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


# --------------------------------------------------------------
# Full Wavenet

# code influenced by: https://github.com/basveeling/wavenet/blob/bf8ef958372692ecb32e8540f7c81f69a186eb8d/wavenet.py#L20


class NNTClassifier_Wavenet2(ClassifierKerasTrinary):
    is_trained = False
    clean_data_required = False  # training data cannot contain anomalies

    def wavenetBlock(self, n_filters, filter_size, rate):
        def f(input_):
            residual = input_
            tanh_out = layers.Convolution1D(n_filters, filter_size, padding="causal", dilation_rate=rate,
                                            activation='tanh')(input_)
            sigmoid_out = layers.Convolution1D(n_filters, filter_size, padding="causal", dilation_rate=rate,
                                               activation='sigmoid')(input_)
            merged = layers.Multiply()([tanh_out, sigmoid_out])

            # skip_x = layers.Convolution1D(nb_filters, 1, padding='same', use_bias=use_bias,
            #                               kernel_regularizer=l2(res_l2))(x)
            # res_x = layers.Add()([residual, res_x])
            #
            # skip_out = layers.Convolution1D(n_filters, 1, padding='same')(merged)
            # out = layers.Add()([skip_out, residual])
            # return out, skip_out

            x = layers.Multiply()([tanh_out, sigmoid_out])

            res_x = layers.Convolution1D(n_filters, 1, padding='same', kernel_regularizer=l2(0))(x)
            skip_x = layers.Convolution1D(n_filters, 1, padding='same', kernel_regularizer=l2(0))(x)
            res_x = layers.Add()([input_, res_x])
            return res_x, skip_x

        return f

    # override the build_model function in subclasses
    def create_model(self, seq_len, num_features):

        n_filters = num_features
        # filter_size = max(int(seq_len / 2), 2)
        filter_size = 2  # anything larger is really slow!

        # model = keras.Sequential(name=self.name)
        inputs = layers.Input(shape=(seq_len, num_features))

        # x = inputs
        x = layers.Convolution1D(n_filters, filter_size, padding="causal", dilation_rate=1)(inputs)

        # A, B = self.wavenetBlock(64, 2, 1)(inputs)

        skip_connections = []
        for i in range(1, 3):
            rate = 1
            for j in range(1, 10):
                x, skip = self.wavenetBlock(n_filters, filter_size, rate)(x)
                skip_connections.append(skip)
                rate = 2 * rate

            x = layers.BatchNormalization()(x)

        x = layers.Add()(skip_connections)
        x = layers.Activation('relu')(x)
        x = layers.Convolution1D(n_filters, 1, padding='same', kernel_regularizer=l2(0))(x)
        x = layers.Activation('relu')(x)
        x = layers.Convolution1D(n_filters, 1, padding='same')(x)

        # last layer is a trinary decision - do not change
        outputs = layers.Dense(3, activation="softmax")(x)

        model = keras.Model(inputs, outputs)

        return model

# --------------------------------------------------------------

# factory to create classifier based on ID. Returns classifier and name
def create_classifier(clf_type: ClassifierType, pair, nfeatures, seq_len, tag=""):

    clf = None
    clf_name = str(clf_type).split(".")[-1]

    if clf_type == ClassifierType.Transformer:
        clf = NNTClassifier_Transformer(pair, seq_len, nfeatures, tag=tag)

    elif clf_type == ClassifierType.LSTM:
        clf = NNTClassifier_LSTM(pair, seq_len, nfeatures, tag=tag)

    elif clf_type == ClassifierType.LSTM2:
        clf = NNTClassifier_LSTM2(pair, seq_len, nfeatures, tag=tag)

    elif clf_type == ClassifierType.LSTM3:
        clf = NNTClassifier_LSTM3(pair, seq_len, nfeatures, tag=tag)

    elif clf_type == ClassifierType.MLP:
        clf = NNTClassifier_MLP(pair, seq_len, nfeatures, tag=tag)

    elif clf_type == ClassifierType.CNN:
        clf = NNTClassifier_CNN(pair, seq_len, nfeatures, tag=tag)

    elif clf_type == ClassifierType.Multihead:
        clf = NNTClassifier_Multihead(pair, seq_len, nfeatures, tag=tag)

    elif clf_type == ClassifierType.Wavenet:
        clf = NNTClassifier_Wavenet(pair, seq_len, nfeatures, tag=tag)

    elif clf_type == ClassifierType.Wavenet2:
        clf = NNTClassifier_Wavenet2(pair, seq_len, nfeatures, tag=tag)

    elif clf_type == ClassifierType.GRU:
        clf = NNTClassifier_GRU(pair, seq_len, nfeatures, tag=tag)

    elif clf_type == ClassifierType.Ensemble:
        clf = NNTClassifier_Ensemble(pair, seq_len, nfeatures, tag=tag)

    elif clf_type == ClassifierType.TCN:
        clf = NNTClassifier_TCN(pair, seq_len, nfeatures, tag=tag)

    elif clf_type == ClassifierType.Attention:
        clf = NNTClassifier_Attention(pair, seq_len, nfeatures, tag=tag)

    elif clf_type == ClassifierType.AdditiveAttention:
        clf = NNTClassifier_AdditiveAttention(pair, seq_len, nfeatures, tag=tag)

    else:
        print("Unknown classifier: ", clf_type)
        clf = None

    return clf, clf_name

# --------------------------------------------------------------
