# Neural Network Trinary Classifier: collection of neural network classifiers, types and access functions

# usage: classifer, name = NNTClassifier.create_classifier(classifier_type, pair, nfeatures, seq_len, tag="")

# NOTE: all models should have a Droput layer to avoid overfitting

import numpy as np
from pandas import DataFrame, Series
import pandas as pd

pd.options.mode.chained_assignment = None  # default='warn'

# Strategy specific imports, files must reside in same folder as strategy
import sys
from pathlib import Path
from enum import Enum, auto

sys.path.append(str(Path(__file__).parent))
sys.path.append(str(Path(__file__).parent.parent))

import logging
import warnings

# log = logging.getLogger(__name__)
# # log.setLevel(logging.DEBUG)
# warnings.simplefilter(action='ignore', category=pd.errors.PerformanceWarning)

import random

import os

# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
# os.environ['TF_DETERMINISTIC_OPS'] = '1'

os.environ['TF_RUN_EAGER_OP_AS_FUNCTION'] = '0'

import tensorflow as tf

# seed = 42
# os.environ['PYTHONHASHSEED'] = str(seed)
# random.seed(seed)
# tf.random.set_seed(seed)
# np.random.seed(seed)

# tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.WARN)

# #import keras
# from keras import layers
# from tf.keras.regularizers import l2
from utils.ClassifierKerasTrinary import ClassifierKerasTrinary


# --------------------------------------------------------------
# Define classes for each type of classifier (have to declare them first)
# --------------------------------------------------------------

# Additive Attention Classifier

class NNTClassifier_AdditiveAttention(ClassifierKerasTrinary):
    is_trained = False
    clean_data_required = False  # training data cannot contain anomalies

    # override the build_model function in subclasses
    def create_model(self, seq_len, num_features):
        # model = tf.keras.Sequential(name=self.name)
        inputs = tf.keras.layers.Input(shape=(seq_len, num_features))

        x = tf.keras.layers.LSTM(num_features, recurrent_dropout=0.25, return_sequences=True,
                        input_shape=(seq_len, num_features))(inputs)
        x = tf.keras.layers.Dropout(0.2)(x)
        x = tf.keras.layers.BatchNormalization()(x)

        # x = tf.keras.layers.Attention()([x, inputs])
        x = tf.keras.layers.AdditiveAttention()([x, inputs])

        # replace sequence column with the average value
        model.add(tf.keras.layers.GlobalAveragePooling1D())

        # Attention produces strange datatypes that cause issues with softmax, so use Dense layer to map/downsize
        x = tf.keras.layers.Dense(32)(x)

        # last layer is a trinary decision - do not change
        x = tf.keras.layers.Dropout(0.2)(x)
        outputs = tf.keras.layers.Dense(3, activation="softmax")(x)

        model = tf.keras.Model(inputs, outputs, name=self.name)

        # model.summary()

        return model


# --------------------------------------------------------------

# Self-Attention Classifier

class NNTClassifier_Attention(ClassifierKerasTrinary):
    is_trained = False
    clean_data_required = False  # training data cannot contain anomalies

    # override the build_model function in subclasses
    def create_model(self, seq_len, num_features):
        # model = tf.keras.Sequential(name=self.name)
        inputs = tf.keras.layers.Input(shape=(seq_len, num_features))

        x = tf.keras.layers.LSTM(num_features, recurrent_dropout=0.25, return_sequences=True,
                        input_shape=(seq_len, num_features))(inputs)
        x = tf.keras.layers.Dropout(0.2)(x)
        x = tf.keras.layers.BatchNormalization()(x)

        # x = tf.keras.layers.Attention()([x, inputs])
        x = tf.keras.layers.Attention()([x, x])

        # replace sequence column with the average value
        model.add(tf.keras.layers.GlobalAveragePooling1D())

        # Attention produces strange datatypes that cause issues with softmax, so use Dense layer to map/downsize
        x = tf.keras.layers.Dense(32)(x)

        # last layer is a trinary decision - do not change
        x = tf.keras.layers.Dropout(0.2)(x)
        outputs = tf.keras.layers.Dense(3, activation="softmax")(x)

        model = tf.keras.Model(inputs, outputs, name=self.name)

        # model.summary()

        return model


# --------------------------------------------------------------
# Convolutional Neural Network

class NNTClassifier_CNN(ClassifierKerasTrinary):
    is_trained = False
    clean_data_required = False  # training data cannot contain anomalies

    # override the build_model function in subclasses
    def create_model(self, seq_len, num_features):
        dropout = 0.4
        n_filters = (8, seq_len, seq_len)

        inputs = tf.keras.Input(shape=(seq_len, num_features))
        x = inputs

        # x = tf.keras.layers.Dense(64, input_shape=(seq_len, num_features))(x)
        # x = tf.keras.layers.BatchNormalization()(x)

        x = tf.keras.layers.Conv1D(filters=64, kernel_size=2, activation='tanh', padding="causal")(x)
        x = tf.keras.layers.Dropout(dropout)(x)
        x = tf.keras.layers.BatchNormalization()(x)

        # replace sequence column with the average value
        model.add(tf.keras.layers.GlobalAveragePooling1D())

        # intermediate layer to bring down the dimensions
        x = tf.keras.layers.Dense(16)(x)

        # last layer is a linear trinary decision - do not change
        x = tf.keras.layers.Dropout(0.2)(x)
        outputs = tf.keras.layers.Dense(3, activation="softmax")(x)

        model = tf.keras.Model(inputs, outputs, name=self.name)

        return model


# --------------------------------------------------------------
# Ensemble/Stack of several Classifiers

class NNTClassifier_Ensemble(ClassifierKerasTrinary):
    is_trained = False
    clean_data_required = False  # training data cannot contain anomalies

    # override the build_model function in subclasses
    def create_model(self, seq_len, num_features):
        dropout = 0.2

        inputs = tf.keras.Input(shape=(seq_len, num_features))

        # run inputs through a few different types of model
        x1 = self.get_lstm(inputs, seq_len, num_features)
        x2 = self.get_gru(inputs, seq_len, num_features)
        x3 = self.get_cnn(inputs, seq_len, num_features)
        # x4 = self.get_simple_wavenet(inputs, seq_len, num_features)
        x4 = self.get_attention(inputs, seq_len, num_features)

        # combine the outputs of the models
        x_combined = tf.keras.layers.Concatenate()([x1, x2, x3, x4])

        # run an LSTM to learn from the combined models
        x = tf.keras.layers.LSTM(3, activation='tanh', recurrent_dropout=0.25, return_sequences=True)(x_combined)

        # replace sequence column with the average value
        model.add(tf.keras.layers.GlobalAveragePooling1D())

        # last layer is a trinary decision - do not change
        x = tf.keras.layers.Dropout(0.2)(x)
        outputs = tf.keras.layers.Dense(3, activation="softmax")(x)

        model = tf.keras.Model(inputs, outputs, name=self.name)

        return model

    def get_lstm(self, inputs, seq_len, num_features):
        x = tf.keras.layers.LSTM(64, activation='tanh', recurrent_dropout=0.25,
                        return_sequences=True, input_shape=(seq_len, num_features))(inputs)
        x = tf.keras.layers.Dropout(rate=0.2)(x)
        x = tf.keras.layers.Dense(3, activation="softmax")(x)
        return x

    def get_gru(self, inputs, seq_len, num_features):
        x = tf.keras.layers.Conv1D(filters=64, kernel_size=2, activation="relu", padding="causal")(inputs)
        x = tf.keras.layers.GRU(32, return_sequences=True)(x)
        x = tf.keras.layers.Dropout(rate=0.2)(x)
        x = tf.keras.layers.Dense(3, activation="softmax")(x)
        return x

    def get_cnn(self, inputs, seq_len, num_features):
        x = tf.keras.layers.Conv1D(filters=64, kernel_size=2, activation='tanh', padding="causal")(inputs)
        x = tf.keras.layers.Dropout(0.2)(x)
        x = tf.keras.layers.BatchNormalization()(x)

        # intermediate layer to bring down the dimensions
        x = tf.keras.layers.Dense(16)(x)
        x = tf.keras.layers.Dropout(0.2)(x)
        x = tf.keras.layers.Dense(3, activation="softmax")(x)
        return x

    def get_simple_wavenet(self, inputs, seq_len, num_features):
        x = inputs
        for rate in (1, 2, 4, 8) * 2:
            x = tf.keras.layers.Conv1D(filters=64, kernel_size=2, padding="causal", activation="relu", dilation_rate=rate)(x)
        x = tf.keras.layers.Dropout(0.2)(x)
        x = tf.keras.layers.Dense(3, activation="softmax")(x)
        return x

    def get_attention(self, inputs, seq_len, num_features):
        x = inputs
        x = tf.keras.layers.LSTM(num_features, recurrent_dropout=0.25, return_sequences=True,
                        input_shape=(seq_len, num_features))(x)
        x = tf.keras.layers.Dropout(0.2)(x)
        x = tf.keras.layers.BatchNormalization()(x)

        x = tf.keras.layers.Attention()([x, x])

        # last layer is a trinary decision - do not change
        x = tf.keras.layers.Dropout(0.2)(x)
        x = tf.keras.layers.Dense(3, activation="softmax")(x)
        return x


# --------------------------------------------------------------
# Gated Recurrent Unit


class NNTClassifier_GRU(ClassifierKerasTrinary):
    is_trained = False
    clean_data_required = False  # training data cannot contain anomalies

    # override the build_model function in subclasses
    def create_model(self, seq_len, num_features):
        model = tf.keras.Sequential(name=self.name)
        model.add(tf.keras.layers.Input(shape=(seq_len, num_features)))
        # model.add(tf.keras.layers.Conv1D(filters=64, kernel_size=2, strides=2, padding="causal", activation="relu"))
        model.add(tf.keras.layers.Conv1D(filters=64, kernel_size=2, activation="relu", padding="causal"))
        model.add(tf.keras.layers.GRU(32, return_sequences=True))

        # replace sequence column with the average value
        model.add(tf.keras.layers.GlobalAveragePooling1D())

        # last layer is a trinary decision - do not change
        model.add(tf.keras.layers.Dropout(0.2))
        model.add(tf.keras.layers.Dense(3, activation="softmax"))

        return model


# --------------------------------------------------------------
# Long-Short Term Memory (basic)

@tf.keras.saving.register_keras_serializable(package="ClassifierKeras")
class NNTClassifier_LSTM(ClassifierKerasTrinary):
    is_trained = False
    clean_data_required = False  # training data cannot contain anomalies

    # override the build_model function in subclasses
    def create_model(self, seq_len, num_features):
        model = tf.keras.Sequential(name=self.name)

        # NOTE: don't use relu with LSTMs, cannot use GPU if you do (much slower). Use tanh

        # model.add(tf.keras.layers.LSTM(128, activation='tanh', recurrent_dropout=0.25,
        #                       return_sequences=True, input_shape=(seq_len, num_features)))
        model.add(tf.keras.layers.LSTM(128, activation='tanh', 
                              return_sequences=True, input_shape=(seq_len, num_features)))

        # replace sequence column with the average value
        model.add(tf.keras.layers.GlobalAveragePooling1D())

        # last layer is a trinary decision - do not change
        model.add(tf.keras.layers.Dropout(0.2))
        model.add(tf.keras.layers.Dense(3, activation="softmax"))

        return model
    
    def get_config(self):
        return super().get_config()


# --------------------------------------------------------------
# Two-tier LSTM


class NNTClassifier_LSTM2(ClassifierKerasTrinary):
    is_trained = False
    clean_data_required = False  # training data cannot contain anomalies

    # override the build_model function in subclasses
    def create_model(self, seq_len, num_features):
        model = tf.keras.Sequential(name=self.name)

        # NOTE: don't use relu with LSTMs, cannot use GPU if you do (much slower). Use tanh

        model.add(tf.keras.layers.LSTM(64, activation='tanh', recurrent_dropout=0.25, return_sequences=True,
                              input_shape=(seq_len, num_features)))
        model.add(tf.keras.layers.Dropout(rate=0.5))

        model.add(tf.keras.layers.LSTM(64, activation='tanh', return_sequences=True, recurrent_dropout=0.25))

        # replace sequence column with the average value
        model.add(tf.keras.layers.GlobalAveragePooling1D())

        #
        # model.add(tf.keras.layers.Dense(16))

        # last layer is a trinary decision - do not change
        model.add(tf.keras.layers.Dropout(0.2))
        model.add(tf.keras.layers.Dense(3, activation="softmax"))

        return model


# --------------------------------------------------------------
# Convolutional/LSTM Combo


class NNTClassifier_LSTM3(ClassifierKerasTrinary):
    is_trained = False
    clean_data_required = False  # training data cannot contain anomalies

    # override the build_model function in subclasses
    def create_model(self, seq_len, num_features):
        model = tf.keras.Sequential(name=self.name)

        # NOTE: don't use relu with LSTMs, cannot use GPU if you do (much slower). Use tanh

        model.add(
            tf.keras.layers.Conv1D(64, kernel_size=3, padding='same', activation='relu', input_shape=(seq_len, num_features)))
        model.add(tf.keras.layers.Conv1D(128, kernel_size=3, padding='same', activation='relu'))
        # model.add(tf.keras.layers.MaxPooling1D(pool_size=2))
        model.add(tf.keras.layers.BatchNormalization())
        model.add(tf.keras.layers.LSTM(128, activation='tanh', recurrent_dropout=0.25, return_sequences=True))

        # replace sequence column with the average value
        model.add(tf.keras.layers.GlobalAveragePooling1D())

        # last layer is a trinary decision - do not change
        model.add(tf.keras.layers.Dropout(0.2))
        model.add(tf.keras.layers.Dense(3, activation="softmax"))

        return model


# --------------------------------------------------------------
# Multi-Layer Perceptron (simple)


class NNTClassifier_MLP(ClassifierKerasTrinary):
    is_trained = False
    clean_data_required = False  # training data cannot contain anomalies

    # override the build_model function in subclasses
    def create_model(self, seq_len, num_features):
        model = tf.keras.Sequential(name=self.name)

        # very simple MLP model:

        # replace sequence column with the average value
        model.add(tf.keras.layers.GlobalAveragePooling1D(), input_shape=(seq_len, num_features))

        model.add(tf.keras.layers.Dense(128))
        model.add(tf.keras.layers.Dropout(rate=0.2))
        model.add(tf.keras.layers.Dense(32))

        # last layer is a trinary decision - do not change
        model.add(tf.keras.layers.Dropout(0.2))
        model.add(tf.keras.layers.Dense(3, activation="softmax"))

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
        inputs = tf.keras.Input(shape=input_shape)
        x = inputs
        x = tf.keras.layers.LSTM(num_features, return_sequences=True, recurrent_dropout=0.25, activation='tanh',
                        input_shape=input_shape)(x)
        x = tf.keras.layers.Dropout(dropout)(x)
        x = tf.keras.layers.Dense(num_features)(x)
        x = tf.keras.layers.LayerNormalization(epsilon=1e-6)(x)

        # "ATTENTION LAYER"
        x = tf.keras.layers.MultiHeadAttention(key_dim=num_features, num_heads=16, dropout=dropout)(x, x, x)
        # x = tf.keras.layers.MultiHeadAttention(key_dim=num_features, num_heads=16, dropout=dropout)(x, inputs)
        x = tf.keras.layers.Dropout(0.1)(x)
        res = x + inputs

        # FEED FORWARD Part - you can stick anything here or just delete the whole section - it will still work.
        x = tf.keras.layers.LayerNormalization(epsilon=1e-6)(res)
        x = tf.keras.layers.Conv1D(filters=seq_len, kernel_size=1, activation="relu")(x)
        x = tf.keras.layers.Dropout(dropout)(x)
        x = tf.keras.layers.Conv1D(filters=num_features, kernel_size=1)(x)
        x = x + res

        # replace sequence column with the average value
        model.add(tf.keras.layers.GlobalAveragePooling1D())

        # last layer is a trinary decision - do not change
        x = tf.keras.layers.Dropout(0.2)(x)
        outputs = tf.keras.layers.Dense(3, activation="softmax")(x)

        model = tf.keras.Model(inputs, outputs, name=self.name)

        return model


# --------------------------------------------------------------
# Temporal Convolutional Network

from TCN import TCN


class NNTClassifier_TCN(ClassifierKerasTrinary):
    is_trained = False
    clean_data_required = False  # training data cannot contain anomalies

    # override the build_model function in subclasses
    def create_model(self, seq_len, num_features):
        # model = tf.keras.Sequential(name=self.name)
        inputs = tf.keras.layers.Input(shape=(seq_len, num_features))

        x = TCN(nb_filters=num_features, kernel_size=seq_len, return_sequences=True, activation='tanh')(inputs)

        # replace sequence column with the average value
        model.add(tf.keras.layers.GlobalAveragePooling1D())

        # last layer is a trinary decision - do not change
        x = tf.keras.layers.Dropout(0.2)(x)
        outputs = tf.keras.layers.Dense(3, activation="softmax")(x)

        model = tf.keras.Model(inputs, outputs, name=self.name)

        return model

    # implement custom_load() because we use a custom layer (TCN)

    def custom_load(self, path):
        model = tf.keras.models.load_model(path, compile=False, custom_objects={'TCN': TCN})
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
        mlp_units = [32]
        mlp_dropout = 0.4
        dropout = 0.25

        inputs = tf.keras.Input(shape=(seq_len, num_features))
        x = inputs
        for _ in range(num_transformer_blocks):
            x = self.transformer_encoder(x, head_size, num_heads, dropout, ff_dim)
            x = tf.keras.layers.BatchNormalization()(x)

        # x = tf.keras.layers.GlobalAveragePooling1D(keepdims=True, data_format="channels_first")(x)
        # x = tf.keras.layers.GlobalMaxPooling1D(keepdims=True, data_format="channels_first")(x)

        # replace sequence column with the average value
        model.add(tf.keras.layers.GlobalAveragePooling1D())

        for dim in mlp_units:
            x = tf.keras.layers.Dense(dim)(x)
            x = tf.keras.layers.Dropout(mlp_dropout)(x)

        # # last layer is a trinary decision - do not change
        # outputs = tf.keras.layers.Dense(3, activation="softmax")(x)

        # last layer is a trinary decision - do not change
        x = tf.keras.layers.Dropout(0.2)(x)
        x = tf.keras.layers.Dense(3, activation="softmax")(x)

        # add timestep dimension back in for compatibility
        # outputs = tf.keras.layers.Reshape((1,3))(x)
        outputs = x

        model = tf.keras.Model(inputs, outputs, name=self.name)

        return model

    def transformer_encoder(self, inputs, head_size, num_heads, dropout, ff_dim):

        # Normalization and Attention
        x = tf.keras.layers.LayerNormalization(epsilon=1e-6)(inputs)
        x = tf.keras.layers.MultiHeadAttention(key_dim=head_size, num_heads=num_heads, dropout=dropout)(x, x)
        x = tf.keras.layers.Dropout(dropout)(x)

        res = x + inputs

        # Feed Forward Part
        x = tf.keras.layers.LayerNormalization(epsilon=1e-6)(res)
        x = tf.keras.layers.Conv1D(filters=ff_dim, kernel_size=2, padding="causal", activation="relu")(x)
        x = tf.keras.layers.Dropout(dropout)(x)
        x = tf.keras.layers.Conv1D(filters=head_size, kernel_size=2, padding="causal")(x)
        return x + res


# --------------------------------------------------------------
# Simplified Wavenet


class NNTClassifier_Wavenet(ClassifierKerasTrinary):
    is_trained = False
    clean_data_required = False  # training data cannot contain anomalies

    # override the build_model function in subclasses
    def create_model(self, seq_len, num_features):
        model = tf.keras.Sequential(name=self.name)
        model.add(tf.keras.layers.Input(shape=(seq_len, num_features)))

        # Wavenet model, which is a series of convolutional layers with increasing dilution rate:
        for rate in (1, 2, 4, 8) * 2:
            model.add(tf.keras.layers.Conv1D(filters=64, kernel_size=2, padding="causal", activation="relu", dilation_rate=rate))

        # replace sequence column with the average value
        model.add(tf.keras.layers.GlobalAveragePooling1D())

        # last layer is a trinary decision - do not change
        model.add(tf.keras.layers.Dropout(0.2))
        model.add(tf.keras.layers.Dense(3, activation="softmax"))

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
            tanh_out = tf.keras.layers.Convolution1D(n_filters, filter_size, padding="causal", dilation_rate=rate,
                                            activation='tanh')(input_)
            sigmoid_out = tf.keras.layers.Convolution1D(n_filters, filter_size, padding="causal", dilation_rate=rate,
                                               activation='sigmoid')(input_)
            # merged = tf.keras.layers.Multiply()([tanh_out, sigmoid_out])

            # skip_x = tf.keras.layers.Convolution1D(nb_filters, 1, padding='same', use_bias=use_bias,
            #                               kernel_regularizer=tf.keras.regularizers.l2(res_l2))(x)
            # res_x = tf.keras.layers.Add()([residual, res_x])
            #
            # skip_out = tf.keras.layers.Convolution1D(n_filters, 1, padding='same')(merged)
            # out = tf.keras.layers.Add()([skip_out, residual])
            # return out, skip_out

            x = tf.keras.layers.Multiply()([tanh_out, sigmoid_out])

            res_x = tf.keras.layers.Convolution1D(n_filters, 1, padding='same', kernel_regularizer=tf.keras.regularizers.l2(0))(x)
            skip_x = tf.keras.layers.Convolution1D(n_filters, 1, padding='same', kernel_regularizer=tf.keras.regularizers.l2(0))(x)
            res_x = tf.keras.layers.Add()([input_, res_x])
            return res_x, skip_x

        return f

    # override the build_model function in subclasses
    def create_model(self, seq_len, num_features):

        n_filters = num_features
        # filter_size = max(int(seq_len / 2), 2)
        filter_size = 2  # anything larger is really slow!

        # model = tf.keras.Sequential(name=self.name)
        inputs = tf.keras.layers.Input(shape=(seq_len, num_features))

        # x = inputs
        x = tf.keras.layers.Convolution1D(n_filters, filter_size, padding="causal", dilation_rate=1)(inputs)

        # A, B = self.wavenetBlock(64, 2, 1)(inputs)

        skip_connections = []
        for i in range(1, 3):
            rate = 1
            for j in range(1, 10):
                x, skip = self.wavenetBlock(n_filters, filter_size, rate)(x)
                skip_connections.append(skip)
                rate = 2 * rate

            x = tf.keras.layers.BatchNormalization()(x)

        x = tf.keras.layers.Add()(skip_connections)
        x = tf.keras.layers.Activation('relu')(x)
        x = tf.keras.layers.Convolution1D(n_filters, 1, padding='same', kernel_regularizer=tf.keras.regularizers.l2(0))(x)
        x = tf.keras.layers.Activation('relu')(x)
        x = tf.keras.layers.Convolution1D(n_filters, 1, padding='same')(x)

        # replace sequence column with the average value
        model.add(tf.keras.layers.GlobalAveragePooling1D())

        # last layer is a trinary decision - do not change
        x = tf.keras.layers.Dropout(0.2)(x)
        outputs = tf.keras.layers.Dense(3, activation="softmax")(x)

        model = tf.keras.Model(inputs, outputs, name=self.name)

        return model


# --------------------------------------------------------------
# Full Wavenet, but with reduced dimensions. Should be much smaller/faster than the full version

# code influenced by: https://github.com/basveeling/wavenet/blob/bf8ef958372692ecb32e8540f7c81f69a186eb8d/wavenet.py#L20


class NNTClassifier_Wavenet3(ClassifierKerasTrinary):
    is_trained = False
    clean_data_required = False  # training data cannot contain anomalies

    def wavenetBlock(self, n_filters, filter_size, rate):
        def f(input_):
            residual = input_
            tanh_out = tf.keras.layers.Convolution1D(n_filters, filter_size, padding="causal", dilation_rate=rate,
                                            activation='tanh')(input_)
            sigmoid_out = tf.keras.layers.Convolution1D(n_filters, filter_size, padding="causal", dilation_rate=rate,
                                               activation='sigmoid')(input_)

            x = tf.keras.layers.Multiply()([tanh_out, sigmoid_out])

            res_x = tf.keras.layers.Convolution1D(n_filters, 1, padding='same', kernel_regularizer=tf.keras.regularizers.l2(0))(x)
            skip_x = tf.keras.layers.Convolution1D(n_filters, 1, padding='same', kernel_regularizer=tf.keras.regularizers.l2(0))(x)
            res_x = tf.keras.layers.Add()([input_, res_x])
            return res_x, skip_x

        return f

    # override the build_model function in subclasses
    def create_model(self, seq_len, num_features):

        # reduced sizes, for improved training speed
        n_filters = 16
        filter_size = 2

        # model = tf.keras.Sequential(name=self.name)
        inputs = tf.keras.layers.Input(shape=(seq_len, num_features))

        # bring down dimensions from num_features to n_filters
        x = tf.keras.layers.GRU(n_filters, activation="tanh", return_sequences=True)(inputs)

        x = tf.keras.layers.Convolution1D(n_filters, filter_size, padding="causal", dilation_rate=1)(x)

        # A, B = self.wavenetBlock(64, 2, 1)(inputs)

        skip_connections = []
        for i in range(1, 3):
            rate = 1
            for j in range(1, 10):
                x, skip = self.wavenetBlock(n_filters, filter_size, rate)(x)
                skip_connections.append(skip)
                rate = 2 * rate

            x = tf.keras.layers.BatchNormalization()(x)

        x = tf.keras.layers.Add()(skip_connections)
        x = tf.keras.layers.Activation('relu')(x)
        x = tf.keras.layers.Convolution1D(n_filters, 1, padding='same', kernel_regularizer=tf.keras.regularizers.l2(0))(x)
        x = tf.keras.layers.Activation('relu')(x)
        x = tf.keras.layers.Convolution1D(n_filters, 1, padding='same')(x)

        # # remove the timesteps axis
        # x = tf.keras.layers.GlobalMaxPooling1D(n_filters)(x)

        # replace sequence column with the average value
        model.add(tf.keras.layers.GlobalAveragePooling1D())

        # last layer is a trinary decision - do not change
        x = tf.keras.layers.Dropout(0.2)(x)
        outputs = tf.keras.layers.Dense(3, activation="softmax")(x)

        model = tf.keras.Model(inputs, outputs, name=self.name)

        return model


# --------------------------------------------------------------

# Types of Classifier

class ClassifierType(Enum):
    AdditiveAttention = NNTClassifier_AdditiveAttention  # Additive-Attention
    Attention = NNTClassifier_Attention  # self-Attention (Transformer Attention)
    CNN = NNTClassifier_CNN  # Convolutional Neural Network
    Ensemble = NNTClassifier_Ensemble  # Ensemble/Stack of several Classifiers
    GRU = NNTClassifier_GRU  # Gated Recurrent Unit
    LSTM = NNTClassifier_LSTM  # Long-Short Term Memory (basic)
    LSTM2 = NNTClassifier_LSTM2  # Two-tier LSTM
    LSTM3 = NNTClassifier_LSTM3  # Convolutional/LSTM Combo
    MLP = NNTClassifier_MLP  # Multi-Layer Perceptron
    Multihead = NNTClassifier_MLP  # Multihead Self-Attention
    TCN = NNTClassifier_TCN  # Temporal Convolutional Network
    Transformer = NNTClassifier_Transformer  # Transformer
    Wavenet = NNTClassifier_Wavenet  # Simplified Wavenet
    Wavenet2 = NNTClassifier_Wavenet2  # Full Wavenet
    Wavenet3 = NNTClassifier_Wavenet3  # Full Wavenet, reduced dimensions


# --------------------------------------------------------------

# factory to create classifier based on ID. Returns classifier and name
def create_classifier(clf_type: ClassifierType, pair, nfeatures, seq_len, tag=""):
    clf_name = str(clf_type).split(".")[-1]
    clf = clf_type.value(pair, seq_len, nfeatures, tag=tag)

    return clf, clf_name

# --------------------------------------------------------------
