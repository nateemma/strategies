# pragma pylint: disable=C0103, C0114, C0115, C0116, C0301, C0302, C0303, C0325, C0411, C0413
# pragma pylint: disable=E1101
# pragma pylint: disable=W0105, W1203, W1309, W1514, W0613, W0621,
# type: ignore
# pylint: disable=import-error

# Neural Network N-ary Classifier: collection of neural network classifiers, types and access functions

# usage: classifer, name = NNNClassifier.create_classifier(classifier_type, pair, nfeatures, seq_len, tag="")

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

warnings.simplefilter(action="ignore", category=FutureWarning)

import random

import os

# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
# os.environ['TF_DETERMINISTIC_OPS'] = '1'

os.environ["TF_RUN_EAGER_OP_AS_FUNCTION"] = "0"

import tensorflow as tf
import keras
from TCN import TCN

# seed = 42
# os.environ['PYTHONHASHSEED'] = str(seed)
# random.seed(seed)
# tf.random.set_seed(seed)
# np.random.seed(seed)

# tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.WARN)

# #import keras
# from keras import layers
# from tf.keras.regularizers import l2
from utils.ClassifierKerasNary import ClassifierKerasNary
from utils.AttentionLayers import AttentionPooling1D

# --------------------------------------------------------------


# --------------------------------------------------------------
# Define classes for each type of classifier (have to declare them first)
# --------------------------------------------------------------

# Additive Attention Classifier


DROPOUT = 0.25  # Increased from 0.2 to reduce overfitting
MIN_FILTER_SIZE = 32
# MIN_FILTER_SIZE = 64
L2_STRENGTH = 2e-3  # Increased from 1e-3 to reduce overfitting


class NNNClassifier_AdditiveAttention(ClassifierKerasNary):
    is_trained = False
    clean_data_required = False  # training data cannot contain anomalies

    # override the build_model function in subclasses
    def create_model(self, seq_len, num_features):
        # model = tf.keras.Sequential(name=self.name)
        inputs = tf.keras.layers.Input(shape=(seq_len, num_features))

        x = inputs

        x = tf.keras.layers.LSTM(
            num_features, return_sequences=True, input_shape=(seq_len, num_features)
        )(x)
        x = tf.keras.layers.Dropout(DROPOUT)(x)
        x = tf.keras.layers.BatchNormalization()(x)

        # x = tf.keras.layers.Attention()([x, inputs])
        x = tf.keras.layers.AdditiveAttention()(
            [x, x]
        )  # Use x instead of inputs for consistency

        if seq_len <= 8:
            # replace sequence column with the average value
            x = tf.keras.layers.GlobalAveragePooling1D()(x)
        else:
            # for larger seq_len, use an LSTM to collapse the sequence axis
            x = tf.keras.layers.LSTM(64, return_sequences=False)(x)

        # Attention produces strange datatypes that cause issues with softmax, so use Dense layer to map/downsize
        x = tf.keras.layers.Dense(32)(x)

        # last layer is a n-ary decision - do not change
        x = tf.keras.layers.Dropout(DROPOUT)(x)
        outputs = tf.keras.layers.Dense(self.num_classes, activation="softmax")(x)

        model = tf.keras.Model(inputs, outputs, name=self.name)

        # model.summary()

        return model


# --------------------------------------------------------------

# Self-Attention Classifier


class NNNClassifier_Attention(ClassifierKerasNary):
    is_trained = False
    clean_data_required = False  # training data cannot contain anomalies

    # override the build_model function in subclasses
    def create_model(self, seq_len, num_features):
        dropout = 0.2  # Increased dropout

        input_shape = (seq_len, num_features)
        inputs = tf.keras.Input(shape=input_shape)

        x = inputs

        # Reduce layer size significantly
        layer_size = min(
            self.nearest_power_of_2(num_features), MIN_FILTER_SIZE
        )  # Reduced from 512 to 64

        # Add Conv1D layer for local feature extraction before attention
        x0 = tf.keras.layers.Conv1D(
            filters=layer_size,  # Reduced from 128 to 64
            kernel_size=3,
            padding="same",
            activation="relu",
        )(x)
        x = tf.keras.layers.BatchNormalization()(x0)
        x = tf.keras.layers.Dropout(dropout)(x)

        # Query from processed features, Key/Value from original input
        query = tf.keras.layers.Dense(layer_size)(x)  # Processed features
        key = tf.keras.layers.Dense(layer_size)(inputs)  # Original input
        value = tf.keras.layers.Dense(layer_size)(inputs)  # Original input

        # MultiHeadAttention with fewer heads
        x = tf.keras.layers.MultiHeadAttention(
            key_dim=layer_size,
            num_heads=8,
            dropout=dropout,  # Reduced from 16 to 8 heads
        )(query, key, value)

        x = tf.keras.layers.Dropout(dropout)(x)
        res = x + x0  # True residual connection with original input

        # Simplified feed forward - remove the complex Conv1D layers
        x = tf.keras.layers.LayerNormalization(epsilon=1e-6)(res)
        x = tf.keras.layers.Dense(layer_size, activation="relu")(x)
        x = tf.keras.layers.Dropout(dropout)(x)
        x = tf.keras.layers.Dense(layer_size)(x)
        x = x + res

        x = tf.keras.layers.GlobalAveragePooling1D()(x)

        # Add a small dense layer before output
        x = tf.keras.layers.Dense(32, activation="relu")(x)  # Small bottleneck
        x = tf.keras.layers.Dropout(0.3)(x)  # Higher dropout before output

        # last layer is a n-ary decision - do not change
        outputs = tf.keras.layers.Dense(self.num_classes, activation="softmax")(x)

        model = tf.keras.Model(inputs, outputs, name=self.name)

        return model


# --------------------------------------------------------------
# Convolutional Neural Network


class NNNClassifier_CNN(ClassifierKerasNary):
    is_trained = False
    clean_data_required = False  # training data cannot contain anomalies

    # override the build_model function in subclasses
    def create_model(self, seq_len, num_features):
        dropout = 0.4

        print(
            f"    Creating CNN model for {self.name} with {seq_len} seq_len and {num_features} num_features"
        )

        inputs = tf.keras.Input(shape=(seq_len, num_features))

        x = tf.keras.layers.Conv1D(
            filters=64, kernel_size=2, activation="tanh", padding="causal"
        )(inputs)
        x = tf.keras.layers.Dropout(dropout)(x)
        x = tf.keras.layers.BatchNormalization()(x)

        # Flatten or pool to collapse the sequence dimension
        x = tf.keras.layers.GlobalAveragePooling1D()(x)

        # intermediate layer to bring down the dimensions
        x = tf.keras.layers.Dense(16)(x)

        # last layer is a linear n-ary decision - do not change
        x = tf.keras.layers.Dropout(DROPOUT)(x)
        outputs = tf.keras.layers.Dense(self.num_classes, activation="softmax")(x)

        model = tf.keras.Model(inputs, outputs, name=self.name)

        return model


# --------------------------------------------------------------
# Ensemble/Stack of several Classifiers


class NNNClassifier_Ensemble(ClassifierKerasNary):
    is_trained = False
    clean_data_required = False  # training data cannot contain anomalies

    # override the build_model function in subclasses
    def create_model(self, seq_len, num_features):
        dropout = 0.2

        lstm_size = max(4 * self.nearest_power_of_2(num_features - 1), MIN_FILTER_SIZE)

        inputs = tf.keras.Input(shape=(seq_len, num_features))

        # run inputs through a few different types of model
        x1 = self.get_lstm(inputs, seq_len, lstm_size)
        x2 = self.get_gru(inputs, seq_len, lstm_size)
        x3 = self.get_cnn(inputs, seq_len, lstm_size)
        # x4 = self.get_simple_wavenet(inputs, seq_len, lstm_size)
        x4 = self.get_attention(inputs, seq_len, lstm_size)

        # combine the outputs of the models
        x_combined = tf.keras.layers.Concatenate()([x1, x2, x3, x4])

        # run an LSTM to learn from the combined models
        x = tf.keras.layers.LSTM(64, activation="tanh", return_sequences=False)(
            x_combined
        )

        # replace sequence column with the average value
        # x = tf.keras.layers.GlobalAveragePooling1D()(x)

        # last layer is a n-ary decision - do not change
        x = tf.keras.layers.Dropout(DROPOUT)(x)
        outputs = tf.keras.layers.Dense(self.num_classes, activation="softmax")(x)

        model = tf.keras.Model(inputs, outputs, name=self.name)

        return model

    def get_lstm(self, inputs, seq_len, num_features):
        x = tf.keras.layers.LSTM(
            64,
            activation="tanh",
            return_sequences=True,
            input_shape=(seq_len, num_features),
        )(inputs)
        x = tf.keras.layers.Dropout(rate=0.2)(x)
        x = tf.keras.layers.Dense(self.num_classes, activation="softmax")(x)
        return x

    def get_gru(self, inputs, seq_len, num_features):
        x = tf.keras.layers.Conv1D(
            filters=64, kernel_size=2, activation="relu", padding="causal"
        )(inputs)
        x = tf.keras.layers.GRU(32, return_sequences=True)(x)
        x = tf.keras.layers.Dropout(rate=0.2)(x)
        x = tf.keras.layers.Dense(self.num_classes, activation="softmax")(x)
        return x

    def get_cnn(self, inputs, seq_len, num_features):
        x = tf.keras.layers.Conv1D(
            filters=64, kernel_size=2, activation="tanh", padding="causal"
        )(inputs)
        x = tf.keras.layers.Dropout(DROPOUT)(x)
        x = tf.keras.layers.BatchNormalization()(x)

        # intermediate layer to bring down the dimensions
        x = tf.keras.layers.Dense(16)(x)
        x = tf.keras.layers.Dropout(DROPOUT)(x)
        x = tf.keras.layers.Dense(self.num_classes, activation="softmax")(x)
        return x

    def get_simple_wavenet(self, inputs, seq_len, num_features):
        x = inputs
        for rate in (1, 2, 4, 8) * 2:
            x = tf.keras.layers.Conv1D(
                filters=64,
                kernel_size=2,
                padding="causal",
                activation="relu",
                dilation_rate=rate,
            )(x)
        x = tf.keras.layers.Dropout(DROPOUT)(x)
        x = tf.keras.layers.Dense(self.num_classes, activation="softmax")(x)
        return x

    def get_attention(self, inputs, seq_len, num_features):
        x = inputs
        x = tf.keras.layers.LSTM(
            num_features, return_sequences=True, input_shape=(seq_len, num_features)
        )(x)
        x = tf.keras.layers.Dropout(DROPOUT)(x)
        x = tf.keras.layers.BatchNormalization()(x)

        x = tf.keras.layers.Attention()([x, x])

        # last layer is a n-ary decision - do not change
        x = tf.keras.layers.Dropout(DROPOUT)(x)
        x = tf.keras.layers.Dense(self.num_classes, activation="softmax")(x)
        return x


# --------------------------------------------------------------
# Gated Recurrent Unit


class NNNClassifier_GRU(ClassifierKerasNary):
    is_trained = False
    clean_data_required = False  # training data cannot contain anomalies

    # override the build_model function in subclasses
    def create_model(self, seq_len, num_features):
        # Use functional API to add positional encoding
        inputs = tf.keras.layers.Input(shape=(seq_len, num_features))

        x = inputs

        # model.add(tf.keras.layers.Conv1D(filters=64, kernel_size=2, strides=2, padding="causal", activation="relu"))
        x = tf.keras.layers.Conv1D(
            filters=1024, kernel_size=2, activation="relu", padding="causal"
        )(x)
        x = tf.keras.layers.GRU(512, return_sequences=True)(x)

        if seq_len <= 8:
            # replace sequence column with the average value
            x = tf.keras.layers.GlobalAveragePooling1D()(x)
        else:
            # for larger seq_len, use an LSTM to collapse the sequence axis
            x = tf.keras.layers.LSTM(64, return_sequences=False)(x)

        # last layer is a n-ary decision - do not change
        x = tf.keras.layers.Dropout(DROPOUT)(x)
        outputs = tf.keras.layers.Dense(self.num_classes, activation="softmax")(x)

        model = tf.keras.Model(inputs, outputs, name=self.name)
        return model


# --------------------------------------------------------------
# Long-Short Term Memory (most basic)


# @tf.keras.saving.register_keras_serializable(package="ClassifierKeras")
class NNNClassifier_LSTM(ClassifierKerasNary):
    is_trained = False
    clean_data_required = False  # training data cannot contain anomalies

    # override the build_model function in subclasses
    def create_model(self, seq_len, num_features):

        self.seq_len = seq_len

        # Input layer
        inputs = tf.keras.layers.Input(
            shape=(seq_len, num_features), name="input_layer"
        )

        x = inputs

        # Calculate appropriate filter size based on num_features
        num_filters = max(
            # MIN_FILTER_SIZE, self.nearest_power_of_2(int(num_features / 2)+1)
            MIN_FILTER_SIZE,
            self.nearest_power_of_2(num_features - 1),
        )

        # Expand Feature Space
        x_resized = tf.keras.layers.Dense(
            num_filters, activation="linear", name="resized_input"
        )(x)

        # CNN to capture patterns
        x_conv1 = tf.keras.Sequential(
            [
                tf.keras.layers.Conv1D(
                    filters=num_filters,
                    kernel_size=2,  # Capture patterns across N consecutive time steps
                    activation="relu",
                    padding="same",
                    # kernel_regularizer=tf.keras.regularizers.l2(L2_STRENGTH),
                    name="cnn_1",
                ),
                tf.keras.layers.LayerNormalization(),
                tf.keras.layers.Dropout(DROPOUT),
            ],
            name="cnn_1",
        )(x_resized)

        # Residual connection: Skip from input to after conv layers
        x_res1 = tf.keras.layers.Add(name="residual_1")([x_resized, x_conv1])
        # x_res1 = tf.keras.layers.Concatenate(name="residual_1")([x_resized, x_conv1])

        x_lstm1 = tf.keras.Sequential(
            [
                tf.keras.layers.LSTM(
                    num_filters,
                    activation="tanh",  # tanh required for GPU (not relu)
                    return_sequences=False,
                    # kernel_regularizer=tf.keras.regularizers.l2(L2_STRENGTH),
                    name="lstm_1",
                ),
                tf.keras.layers.LayerNormalization(),
                tf.keras.layers.Dropout(DROPOUT),
            ],
            name="lstm_1",
        )(x_res1)

        # Bottleneck layer before output (prevents overfitting)
        bottleneck_size = num_filters
        x_dense = tf.keras.Sequential(
            [
                tf.keras.layers.Dense(
                    bottleneck_size,
                    activation="elu",
                    # kernel_regularizer=tf.keras.regularizers.l2(L2_STRENGTH),
                ),
                tf.keras.layers.Dropout(DROPOUT),
            ],
            name="bottleneck",
        )(x_lstm1)

        # Output layer
        outputs = tf.keras.layers.Dense(
            self.num_classes, activation="softmax", name="softmax"
        )(x_dense)

        model = tf.keras.Model(inputs, outputs, name=self.name)

        return model

    def get_config(self):
        return super().get_config()


# --------------------------------------------------------------
# Larger LSTM


class NNNClassifier_LSTM2(ClassifierKerasNary):
    is_trained = False
    clean_data_required = False  # training data cannot contain anomalies

    # override the build_model function in subclasses
    def create_model(self, seq_len, num_features):

        self.seq_len = seq_len
        inputs = tf.keras.layers.Input(
            shape=(seq_len, num_features), name="input_layer"
        )

        # Determine filter size
        num_filters = max(
            2 * MIN_FILTER_SIZE, self.nearest_power_of_2(num_features - 1)
        )

        # 1. Expand Feature Space
        x_resized = tf.keras.layers.Dense(
            num_filters, activation="linear", name="resized_input"
        )(inputs)

        # 2. Multi-Scale CNN Block (Parallel Kernels)
        # Kernel 2 for micro-momentum
        cnn_2 = tf.keras.layers.Conv1D(
            num_filters // 2, kernel_size=2, padding="same", activation="relu"
        )(x_resized)
        # Kernel 5 for trend-following patterns
        cnn_5 = tf.keras.layers.Conv1D(
            num_filters // 2, kernel_size=5, padding="same", activation="relu"
        )(x_resized)

        # Merge the two scales
        x_conv = tf.keras.layers.Concatenate()([cnn_2, cnn_5])
        x_conv = tf.keras.layers.LayerNormalization()(x_conv)
        x_conv = tf.keras.layers.Dropout(DROPOUT)(x_conv)

        # 3. Residual connection
        x_res = tf.keras.layers.Add(name="residual_1")([x_resized, x_conv])

        # 4. Two-Tier LSTM Stack
        # Layer 1 learns temporal sequences (returns sequences)
        x = tf.keras.layers.LSTM(
            num_filters, activation="tanh", return_sequences=True, name="lstm_tier_1"
        )(x_res)
        x = tf.keras.layers.LayerNormalization()(x)
        x = tf.keras.layers.Dropout(DROPOUT)(x)

        # Layer 2 collapses into a single "Decision Vector"
        x = tf.keras.layers.LSTM(
            num_filters, activation="tanh", return_sequences=False, name="lstm_tier_2"
        )(x)
        x = tf.keras.layers.LayerNormalization()(x)
        x = tf.keras.layers.Dropout(DROPOUT)(x)

        # 5. Bottleneck and Decision
        x = tf.keras.layers.Dense(num_filters, activation="elu", name="bottleneck")(x)
        x = tf.keras.layers.Dropout(DROPOUT)(x)
        outputs = tf.keras.layers.Dense(
            self.num_classes, activation="softmax", name="softmax"
        )(x)

        model = tf.keras.Model(inputs, outputs, name=self.name)

        return model

    def calculate_gamma(self, alpha):
        """Force a higher gamma for this specific model to boost Recall on hard samples.
        gamma=2.0 targets difficult trade entry points; gamma=3.0 was tested and
        destroyed recall (SOL: 0.49→0.23, BTC: 0.22→0.12) — reverted."""
        return 2.0

    def get_config(self):
        return super().get_config()


# --------------------------------------------------------------
# Convolutional/LSTM Combo


class NNNClassifier_LSTM3(ClassifierKerasNary):
    is_trained = False
    clean_data_required = False  # training data cannot contain anomalies

    # override the build_model function in subclasses
    def create_model(self, seq_len, num_features):
        # Use functional API instead of Sequential to properly handle input shape
        inputs = tf.keras.layers.Input(shape=(seq_len, num_features))

        x = inputs

        # NOTE: don't use relu with LSTMs, cannot use GPU if you do (much slower). Use tanh
        x = tf.keras.layers.Conv1D(
            64, kernel_size=3, padding="same", activation="relu"
        )(x)
        x = tf.keras.layers.Conv1D(
            128, kernel_size=3, padding="same", activation="relu"
        )(x)
        # x = tf.keras.layers.MaxPooling1D(pool_size=2)(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.LSTM(1024, activation="tanh", return_sequences=False)(x)

        x = tf.keras.layers.Dense(32)(x)

        # replace sequence column with the average value
        # x = tf.keras.layers.GlobalAveragePooling1D()(x)

        # last layer is a n-ary decision - do not change
        x = tf.keras.layers.Dropout(DROPOUT)(x)
        # outputs = tf.keras.layers.Dense(self.num_classes, activation="softmax")(x)
        outputs = tf.keras.layers.Dense(self.num_classes, activation="softmax")(x)

        model = tf.keras.Model(inputs, outputs, name=self.name)

        return model


# --------------------------------------------------------------
# Parallel LSTMs


class NNNClassifier_LSTM4(ClassifierKerasNary):
    is_trained = False
    clean_data_required = False  # training data cannot contain anomalies

    # override the build_model function in subclasses
    def create_model(self, seq_len, num_features):
        # Use functional API instead of Sequential to properly handle input shape
        inputs = tf.keras.layers.Input(shape=(seq_len, num_features))

        x = inputs

        # NOTE: don't use relu with LSTMs, cannot use GPU if you do (much slower). Use tanh

        x1 = tf.keras.layers.LSTM(128, activation="tanh", return_sequences=True)(x)
        x2 = tf.keras.layers.LSTM(256, activation="tanh", return_sequences=True)(x)
        x3 = tf.keras.layers.LSTM(512, activation="tanh", return_sequences=True)(x)

        x_concat = tf.keras.layers.Concatenate()([inputs, x1, x2, x3])

        x = tf.keras.layers.LSTM(64, activation="tanh", return_sequences=False)(
            x_concat
        )

        # last layer is a n-ary decision - do not change
        x = tf.keras.layers.Dropout(DROPOUT)(x)

        x = tf.keras.layers.Dense(16)(x)

        outputs = tf.keras.layers.Dense(self.num_classes, activation="softmax")(x)

        model = tf.keras.Model(inputs, outputs, name=self.name)

        return model


# --------------------------------------------------------------
# Multi-Layer Perceptron (simple)


class NNNClassifier_MLP(ClassifierKerasNary):
    is_trained = False
    clean_data_required = False  # training data cannot contain anomalies

    # override the build_model function in subclasses
    def create_model(self, seq_len, num_features):

        input_shape = (seq_len, num_features)
        inputs = tf.keras.Input(shape=input_shape)
        x = inputs

        # Deep MLP model - pyramid structure for better feature extraction
        x = tf.keras.layers.Flatten()(x)

        # Layer 1: Expansion/High-level features
        x = tf.keras.layers.Dense(
            256,
            activation="relu",
            kernel_regularizer=tf.keras.regularizers.l2(L2_STRENGTH),
        )(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Dropout(DROPOUT)(x)

        # Layer 2: Refinement
        x = tf.keras.layers.Dense(
            128,
            activation="relu",
            kernel_regularizer=tf.keras.regularizers.l2(L2_STRENGTH),
        )(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Dropout(DROPOUT)(x)

        # Layer 3: Compression
        x = tf.keras.layers.Dense(
            64,
            activation="relu",
            kernel_regularizer=tf.keras.regularizers.l2(L2_STRENGTH),
        )(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Dropout(DROPOUT)(x)

        # Output layer
        outputs = tf.keras.layers.Dense(self.num_classes, activation="softmax")(x)

        model = tf.keras.Model(inputs, outputs, name=self.name)

        return model


# --------------------------------------------------------------
# Multihead Self-Attention


class NNNClassifier_Multihead(ClassifierKerasNary):
    is_trained = False
    clean_data_required = False  # training data cannot contain anomalies

    def custom_load(self, path):
        model = keras.models.load_model(
            path,
            compile=False,
            safe_mode=False,
            custom_objects={
                "TransformerPositionalEmbedding": TransformerPositionalEmbedding
            },
        )
        return model

    # override the build_model function in subclasses
    def create_model(self, seq_len, num_features):
        dropout = 0.3

        input_shape = (seq_len, num_features)
        inputs = tf.keras.Input(shape=input_shape)

        # Initial projection and local feature extraction
        layer_size = min(self.nearest_power_of_2(num_features), MIN_FILTER_SIZE * 2)

        x = tf.keras.layers.Conv1D(
            filters=layer_size,
            kernel_size=3,
            padding="same",
            activation="relu",
        )(inputs)
        x = tf.keras.layers.BatchNormalization()(x)

        # Apply Positional Embedding (crucial for time-series)
        x = TransformerPositionalEmbedding(seq_len, layer_size)(x)

        # Transformer Block 1
        # Multi-Head Self-Attention with Pre-LayerNorm
        # Note: key_dim should be layer_size // num_heads for standard multi-head attention
        num_heads = 4
        head_dim = layer_size // num_heads

        res = x
        x = tf.keras.layers.LayerNormalization(epsilon=1e-6)(x)
        x = tf.keras.layers.MultiHeadAttention(
            key_dim=head_dim,
            num_heads=num_heads,
            dropout=dropout,
            kernel_regularizer=tf.keras.regularizers.l2(L2_STRENGTH),
        )(x, x)
        x = tf.keras.layers.Dropout(dropout)(x)
        x = x + res

        # Feed-Forward Network with Pre-LayerNorm
        res = x
        x = tf.keras.layers.LayerNormalization(epsilon=1e-6)(x)
        x = tf.keras.layers.Dense(
            layer_size * 2,
            activation="relu",
            kernel_regularizer=tf.keras.regularizers.l2(L2_STRENGTH),
        )(x)
        x = tf.keras.layers.Dropout(dropout)(x)
        x = tf.keras.layers.Dense(
            layer_size, kernel_regularizer=tf.keras.regularizers.l2(L2_STRENGTH)
        )(x)
        x = x + res

        # Global average pooling to flatten the sequence
        x = tf.keras.layers.GlobalAveragePooling1D()(x)

        # Final classification head
        x = tf.keras.layers.Dense(
            32,
            activation="relu",
            kernel_regularizer=tf.keras.regularizers.l2(L2_STRENGTH),
        )(x)
        x = tf.keras.layers.Dropout(DROPOUT)(x)

        # last layer is a n-ary decision - do not change
        outputs = tf.keras.layers.Dense(self.num_classes, activation="softmax")(x)

        model = tf.keras.Model(inputs, outputs, name=self.name)

        return model


# --------------------------------------------------------------
# --------------------------------------------------------------
# Temporal Convolutional Network


class NNNClassifier_TCN(ClassifierKerasNary):
    is_trained = False
    clean_data_required = False  # training data cannot contain anomalies

    # override the build_model function in subclasses
    def create_model(self, seq_len, num_features):
        # model = tf.keras.Sequential(name=self.name)
        inputs = tf.keras.layers.Input(shape=(seq_len, num_features))

        x = inputs

        x = TCN(
            nb_filters=max(64, num_features),
            kernel_size=2,  # Use small kernel with dilation for better feature capture
            return_sequences=False,  # TCN collapses sequence dimension internally
            activation="relu",
        )(x)

        # last layer is a n-ary decision - do not change
        x = tf.keras.layers.Dropout(DROPOUT)(x)
        outputs = tf.keras.layers.Dense(self.num_classes, activation="softmax")(x)

        model = tf.keras.Model(inputs, outputs, name=self.name)

        return model

    # implement custom_load() because we use a custom layer (TCN)

    def custom_load(self, path):
        model = keras.models.load_model(
            path, compile=False, custom_objects={"TCN": TCN}
        )
        return model


# Transformer Positional Embedding Layer
@keras.saving.register_keras_serializable(package="ClassifierKeras")
class TransformerPositionalEmbedding(tf.keras.layers.Layer):
    def __init__(self, seq_len, embed_dim, **kwargs):
        super().__init__(**kwargs)
        self.pos_emb = tf.keras.layers.Embedding(
            input_dim=seq_len, output_dim=embed_dim
        )
        self.seq_len = seq_len
        self.embed_dim = embed_dim

    def build(self, input_shape):
        # The weights of the Embedding layer are built based on seq_len/embed_dim
        # We can explicitly build it here or let it happen on first call.
        # However, to satisfy Keras 3's "built" check, we should ensure it's built.
        if not self.pos_emb.built:
            self.pos_emb.build((self.seq_len,))
        super().build(input_shape)

    def call(self, inputs):
        # inputs shape: (batch_size, seq_len, embed_dim)
        positions = tf.range(start=0, limit=self.seq_len, delta=1)
        positions = self.pos_emb(positions)  # (seq_len, embed_dim)
        
        # Cast positions to the same dtype as inputs to avoid mixed-precision mismatches in hyperopt
        # Hyperopt often runs with float16 or mixed precision, while embeddings default to float32
        if inputs.dtype != positions.dtype:
            positions = tf.cast(positions, inputs.dtype)
            
        # Broadcasting will handle the batch dimension
        return inputs + positions

    def compute_output_shape(self, input_shape):
        return input_shape

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "seq_len": self.seq_len,
                "embed_dim": self.embed_dim,
            }
        )
        return config


class NNNClassifier_Transformer(ClassifierKerasNary):
    is_trained = False
    clean_data_required = False  # training data cannot contain anomalies

    def custom_load(self, path):
        model = keras.models.load_model(
            path,
            compile=False,
            safe_mode=False,
            custom_objects={
                "TransformerPositionalEmbedding": TransformerPositionalEmbedding
            },
        )
        return model

    # override the build_model function in subclasses
    def create_model(self, seq_len, num_features):

        # head_size = num_features
        head_size = max(32, self.nearest_power_of_2(num_features - 1))
        # num_heads = int(num_features / 2)
        num_heads = 4
        ff_dim = 16  # Increased from 4 to reduce bottleneck
        # ff_dim = seq_len
        # num_transformer_blocks = seq_len
        num_transformer_blocks = 2  # Reduced from 4 to prevent overfitting
        mlp_units = [head_size, head_size // 2]
        mlp_dropout = 0.4  # Increased from 0.3 for better regularization
        dropout = 0.4  # Increased from 0.25 for better regularization

        inputs = tf.keras.Input(shape=(seq_len, num_features))

        # project to higher dimensions and add positional encoding
        x = tf.keras.layers.Dense(head_size, activation="linear")(inputs)
        x = TransformerPositionalEmbedding(seq_len, head_size)(x)

        for _ in range(num_transformer_blocks):
            x = self.transformer_encoder(x, head_size, num_heads, dropout, ff_dim)
            # Removed BatchNormalization - LayerNorm in transformer_encoder is sufficient
            # BatchNorm conflicts with LayerNorm and can cause training instability

        # x = tf.keras.layers.GlobalAveragePooling1D(keepdims=True, data_format="channels_first")(x)
        # x = tf.keras.layers.GlobalMaxPooling1D(keepdims=True, data_format="channels_first")(x)

        # replace sequence column with the average value
        x = tf.keras.layers.GlobalAveragePooling1D()(x)

        for dim in mlp_units:
            x = tf.keras.layers.Dense(dim)(x)
            x = tf.keras.layers.Dropout(mlp_dropout)(x)

        # # last layer is a n-ary decision - do not change
        # outputs = tf.keras.layers.Dense(self.num_classes, activation="softmax")(x)

        # last layer is a n-ary decision - do not change
        x = tf.keras.layers.Dropout(mlp_dropout)(
            x
        )  # Added dropout before final layer for regularization
        x = tf.keras.layers.Dense(self.num_classes, activation="softmax")(x)

        # add timestep dimension back in for compatibility
        # outputs = tf.keras.layers.Reshape((1,3))(x)
        outputs = x

        model = tf.keras.Model(inputs, outputs, name=self.name)

        return model

    def transformer_encoder(self, inputs, head_size, num_heads, dropout, ff_dim):

        # Normalization and Attention
        # Note: key_dim should be head_size // num_heads for standard transformer architecture
        head_dim = head_size // num_heads
        x = tf.keras.layers.LayerNormalization(epsilon=1e-6)(inputs)
        x = tf.keras.layers.MultiHeadAttention(
            key_dim=head_dim,
            num_heads=num_heads,
            dropout=dropout,
            kernel_regularizer=tf.keras.regularizers.l2(L2_STRENGTH),
        )(x, x)
        x = tf.keras.layers.Dropout(dropout)(x)

        res = x + inputs

        # Feed Forward Part
        x = tf.keras.layers.LayerNormalization(epsilon=1e-6)(res)
        x = tf.keras.layers.Conv1D(
            filters=ff_dim, kernel_size=2, padding="causal", activation="relu"
        )(x)
        x = tf.keras.layers.Dropout(dropout)(x)
        x = tf.keras.layers.Conv1D(filters=head_size, kernel_size=2, padding="causal")(
            x
        )
        return x + res


# Add VAE Sampling Layer
@tf.keras.utils.register_keras_serializable(package="ClassifierKeras")
class SamplingLayer(tf.keras.layers.Layer):
    """Custom sampling layer for VAE that is properly serializable"""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def call(self, inputs):
        z_mean, z_log_var = inputs
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        # Cast epsilon to match input dtype (essential for mixed precision float16)
        epsilon = tf.keras.backend.random_normal(
            shape=(batch, dim), dtype=tf.keras.backend.dtype(z_mean)
        )
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon

    def compute_output_shape(self, input_shape):
        # The output shape is the same as the shape of z_mean
        return input_shape[0]

    def get_config(self):
        return super().get_config()


@tf.keras.utils.register_keras_serializable(package="ClassifierKeras")
class VAELossLayer(tf.keras.layers.Layer):
    """Custom layer to add VAE losses in Keras 3 Functional API"""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def call(self, inputs):
        x_flat, reconstruction, z_mean, z_log_var = inputs

        # Reconstruction loss
        rec_loss = keras.ops.mean(
            keras.ops.sum(
                keras.losses.mean_squared_error(x_flat, reconstruction), axis=-1
            )
        )
        self.add_loss(rec_loss)

        # KL divergence loss
        kl_loss = -0.5 * keras.ops.mean(
            keras.ops.sum(
                1 + z_log_var - keras.ops.square(z_mean) - keras.ops.exp(z_log_var),
                axis=-1,
            )
        )
        self.add_loss(kl_loss)

        return reconstruction  # Return something to keep it in the graph

    def get_config(self):
        return super().get_config()


# --------------------------------------------------------------
#


# VAE (Variational Auto Encoder)Classifier
class NNNClassifier_VAE(ClassifierKerasNary):
    is_trained = False
    clean_data_required = False  # training data cannot contain anomalies

    def custom_load(self, path):
        model = keras.models.load_model(
            path,
            compile=False,
            safe_mode=False,
            custom_objects={
                "SamplingLayer": SamplingLayer,
                "VAELossLayer": VAELossLayer,
            },
        )
        return model

    def create_model(self, seq_len, num_features):
        # Calculate total input features (seq_len * num_features for flattened data)
        total_features = seq_len * num_features

        # Encoder
        inputs = tf.keras.layers.Input(shape=(seq_len, num_features))
        x_flat = tf.keras.layers.Flatten()(inputs)

        # Encoder layers
        h = tf.keras.layers.Dense(128, activation="relu")(x_flat)
        h = tf.keras.layers.BatchNormalization()(h)
        h = tf.keras.layers.Dropout(DROPOUT)(h)

        h = tf.keras.layers.Dense(64, activation="relu")(h)
        h = tf.keras.layers.BatchNormalization()(h)
        h = tf.keras.layers.Dropout(DROPOUT)(h)

        # Latent space parameters
        z_mean = tf.keras.layers.Dense(32, name="z_mean")(h)
        z_log_var = tf.keras.layers.Dense(32, name="z_log_var")(h)

        # Sampling layer using custom layer
        z = SamplingLayer(name="z")([z_mean, z_log_var])

        # Decoder (Reconstruction)
        decoder_h = tf.keras.layers.Dense(64, activation="relu")(z)
        decoder_h = tf.keras.layers.Dense(128, activation="relu")(decoder_h)
        reconstruction = tf.keras.layers.Dense(
            total_features, activation="linear", name="reconstruction"
        )(decoder_h)

        # Classifier Branch
        clf_h = tf.keras.layers.Dense(32, activation="relu")(z)
        clf_h = tf.keras.layers.Dropout(DROPOUT)(clf_h)

        # Binary/N-ary classification output
        outputs = tf.keras.layers.Dense(self.num_classes, activation="softmax")(clf_h)

        # Add the losses using custom layer instead of model.add_loss()
        # This keeps the Functional API clean and compatible with Keras 3
        _ = VAELossLayer(name="vae_loss")([x_flat, reconstruction, z_mean, z_log_var])

        model = tf.keras.Model(inputs, outputs, name=self.name)

        return model


# --------------------------------------------------------------
# Simplified Wavenet


class NNNClassifier_Wavenet(ClassifierKerasNary):
    is_trained = False
    clean_data_required = False  # training data cannot contain anomalies

    # override the build_model function in subclasses
    def create_model(self, seq_len, num_features):
        model = tf.keras.Sequential(name=self.name)
        model.add(tf.keras.layers.Input(shape=(seq_len, num_features)))

        # Wavenet model, which is a series of convolutional layers with increasing dilution rate:
        for rate in (1, 2, 4, 8) * 2:
            model.add(
                tf.keras.layers.Conv1D(
                    filters=64,
                    kernel_size=2,
                    padding="causal",
                    activation="relu",
                    dilation_rate=rate,
                )
            )

        # replace sequence column with the average value
        model.add(tf.keras.layers.GlobalAveragePooling1D())
        # model.add(tf.keras.layers.LSTM(16, activation='tanh'))

        # last layer is a n-ary decision - do not change
        model.add(tf.keras.layers.Dropout(DROPOUT))
        model.add(tf.keras.layers.Dense(self.num_classes, activation="softmax"))

        return model


# --------------------------------------------------------------
# Full Wavenet

# code influenced by: https://github.com/basveeling/wavenet/blob/bf8ef958372692ecb32e8540f7c81f69a186eb8d/wavenet.py#L20


class NNNClassifier_Wavenet2(ClassifierKerasNary):
    is_trained = False
    clean_data_required = False  # training data cannot contain anomalies

    def wavenetBlock(self, n_filters, filter_size, rate):
        def f(input_):
            residual = input_
            tanh_out = tf.keras.layers.Convolution1D(
                n_filters,
                filter_size,
                padding="causal",
                dilation_rate=rate,
                activation="tanh",
            )(input_)
            sigmoid_out = tf.keras.layers.Convolution1D(
                n_filters,
                filter_size,
                padding="causal",
                dilation_rate=rate,
                activation="sigmoid",
            )(input_)
            # merged = tf.keras.layers.Multiply()([tanh_out, sigmoid_out])

            # skip_x = tf.keras.layers.Convolution1D(nb_filters, 1, padding='same', use_bias=use_bias,
            #                               kernel_regularizer=tf.keras.regularizers.l2(res_l2))(x)
            # res_x = tf.keras.layers.Add()([residual, res_x])
            #
            # skip_out = tf.keras.layers.Convolution1D(n_filters, 1, padding='same')(merged)
            # out = tf.keras.layers.Add()([skip_out, residual])
            # return out, skip_out

            x = tf.keras.layers.Multiply()([tanh_out, sigmoid_out])

            res_x = tf.keras.layers.Convolution1D(
                n_filters,
                1,
                padding="same",
                kernel_regularizer=tf.keras.regularizers.l2(0),
            )(x)
            skip_x = tf.keras.layers.Convolution1D(
                n_filters,
                1,
                padding="same",
                kernel_regularizer=tf.keras.regularizers.l2(0),
            )(x)
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
        x = tf.keras.layers.Convolution1D(
            n_filters, filter_size, padding="causal", dilation_rate=1
        )(inputs)

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
        x = tf.keras.layers.Activation("relu")(x)
        x = tf.keras.layers.Convolution1D(
            n_filters, 1, padding="same", kernel_regularizer=tf.keras.regularizers.l2(0)
        )(x)
        x = tf.keras.layers.Activation("relu")(x)
        x = tf.keras.layers.Convolution1D(n_filters, 1, padding="same")(x)

        # replace sequence column with the average value
        x = tf.keras.layers.GlobalAveragePooling1D()(x)

        # last layer is a n-ary decision - do not change
        x = tf.keras.layers.Dropout(DROPOUT)(x)
        outputs = tf.keras.layers.Dense(self.num_classes, activation="softmax")(x)

        model = tf.keras.Model(inputs, outputs, name=self.name)

        return model


# --------------------------------------------------------------
# Full Wavenet, but with reduced dimensions. Should be much smaller/faster than the full version

# code influenced by: https://github.com/basveeling/wavenet/blob/bf8ef958372692ecb32e8540f7c81f69a186eb8d/wavenet.py#L20


class NNNClassifier_Wavenet3(ClassifierKerasNary):
    is_trained = False
    clean_data_required = False  # training data cannot contain anomalies

    def wavenetBlock(self, n_filters, filter_size, rate):
        def f(input_):
            residual = input_
            tanh_out = tf.keras.layers.Convolution1D(
                n_filters,
                filter_size,
                padding="causal",
                dilation_rate=rate,
                activation="tanh",
            )(input_)
            sigmoid_out = tf.keras.layers.Convolution1D(
                n_filters,
                filter_size,
                padding="causal",
                dilation_rate=rate,
                activation="sigmoid",
            )(input_)

            x = tf.keras.layers.Multiply()([tanh_out, sigmoid_out])

            res_x = tf.keras.layers.Convolution1D(
                n_filters,
                1,
                padding="same",
                kernel_regularizer=tf.keras.regularizers.l2(0),
            )(x)
            skip_x = tf.keras.layers.Convolution1D(
                n_filters,
                1,
                padding="same",
                kernel_regularizer=tf.keras.regularizers.l2(0),
            )(x)
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
        x = tf.keras.layers.GRU(n_filters, activation="tanh", return_sequences=True)(
            inputs
        )

        x = tf.keras.layers.Convolution1D(
            n_filters, filter_size, padding="causal", dilation_rate=1
        )(x)

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
        x = tf.keras.layers.Activation("relu")(x)
        x = tf.keras.layers.Convolution1D(
            n_filters, 1, padding="same", kernel_regularizer=tf.keras.regularizers.l2(0)
        )(x)
        x = tf.keras.layers.Activation("relu")(x)
        x = tf.keras.layers.Convolution1D(n_filters, 1, padding="same")(x)

        # # remove the timesteps axis
        # x = tf.keras.layers.GlobalMaxPooling1D(n_filters)(x)

        # replace sequence column with the average value
        x = tf.keras.layers.GlobalAveragePooling1D()(x)

        # last layer is a n-ary decision - do not change
        x = tf.keras.layers.Dropout(DROPOUT)(x)
        outputs = tf.keras.layers.Dense(self.num_classes, activation="softmax")(x)

        model = tf.keras.Model(inputs, outputs, name=self.name)

        return model


# --------------------------------------------------------------
# Multi-Task Learning Classifier


# --------------------------------------------------------------

# Types of Classifier


class ClassifierType(Enum):
    AdditiveAttention = NNNClassifier_AdditiveAttention  # Additive-Attention
    Attention = NNNClassifier_Attention  # self-Attention (Transformer Attention)
    CNN = NNNClassifier_CNN  # Convolutional Neural Network
    Ensemble = NNNClassifier_Ensemble  # Ensemble/Stack of several Classifiers
    GRU = NNNClassifier_GRU  # Gated Recurrent Unit
    LSTM = NNNClassifier_LSTM  # Long-Short Term Memory (basic)
    LSTM2 = NNNClassifier_LSTM2  # Two-tier LSTM
    LSTM3 = NNNClassifier_LSTM3  # Convolutional/LSTM Combo
    LSTM4 = NNNClassifier_LSTM4  # Parallel LSTMs
    MLP = NNNClassifier_MLP  # Multi-Layer Perceptron
    Multihead = NNNClassifier_Multihead  # Multihead Self-Attention
    TCN = NNNClassifier_TCN  # Temporal Convolutional Network
    Transformer = NNNClassifier_Transformer  # Transformer
    VAE = NNNClassifier_VAE  # Variational Autoencoder
    Wavenet = NNNClassifier_Wavenet  # Simplified Wavenet
    Wavenet2 = NNNClassifier_Wavenet2  # Full Wavenet
    Wavenet3 = NNNClassifier_Wavenet3  # Full Wavenet, reduced dimensions


# --------------------------------------------------------------


# factory to create classifier based on ID. Returns classifier and name
def create_classifier(
    clf_type: ClassifierType, pair, nfeatures, seq_len, nclasses, tag=""
):
    clf_name = str(clf_type).split(".")[-1]
    clf = clf_type.value(pair, seq_len, nfeatures, tag=tag)

    # Set num_classes as instance attribute so it can be used in create_model
    clf.num_classes = nclasses

    return clf, clf_name


# --------------------------------------------------------------
