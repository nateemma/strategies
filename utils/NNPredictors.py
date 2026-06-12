"""
This class is factory for creating various kinds of Neural Network Predictors with a consistent interface

"""
# pragma pylint: disable=W0105, C0103, C0114, C0115, C0116, C0301, C0302, C0303, C0325, C0411, C0413, W0611, W1203, W291


# -----------------------------------


# base class - to allow generic treatment of all predictors

# from abc import ABC, abstractmethod
from enum import Enum

import numpy as np
import pandas as pd

# -----------------------------------

import time

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))

from sklearn.base import is_classifier
import tensorflow as tf
# from tensorflow import keras
from tensorflow.keras import Input, Model, layers, regularizers # type: ignore

from Predictors.KerasRegressor import KerasRegressor
from TCN import TCN as TCN
from NBeatsNet import NBeatsNet as NBeatsNet
from PositionalEncoding import PositionalEncoding

# -----------------------------------

# all actual instantiations follow this base class


class NNPredictor(KerasRegressor):

    is_trained = False
    clean_data_required = False  # training data can contain anomalies
    model_per_pair = False # separate model per pair
    prescale_dataframe = False

    # default — gives the simplified LSTM room to actually train.
    default_learning_rate = 0.001
    learning_rate = 0.0001

    def __init__(self, pair, seq_len, num_features):
        super().__init__(pair, seq_len, num_features)

        self.pair = pair
        self.seq_len = seq_len
        self.num_features = num_features


    # creates the model and uses the estimation layer from the specific subclass
    def create_model(self, seq_len, num_features):

        inputs = Input(shape=(seq_len, num_features))

        # Normalise the input data (to make sure there aren't any pair-specific values like price)
        x = layers.BatchNormalization()(inputs)

        # # don't normalise:
        # x = inputs

        lstm_size = max(self.nearest_power_of_2(num_features-1), 16)

        # LSTM Layer to extract temporal features
        lstm_output = layers.LSTM(lstm_size, return_sequences=True, activation='tanh')(x)
        # lstm_output = layers.LSTM(lstm_size, return_sequences=True)(inputs)
        lstm_output = layers.Dropout(0.2)(lstm_output)

        # get the estimation layer (typically from a subclass)
        est_output = self.get_estimation_layer(lstm_output)

        # layers.Concatenate attention output with LSTM output
        concat_output = layers.Concatenate()([est_output, lstm_output])

        # out_size = concat_output.shape[-1]

        # print(f'est_output:{est_output.shape}  lstm_output:{lstm_output.shape} concat_output:{concat_output.shape}')

        x = layers.BatchNormalization()(concat_output)

        # Get the latest sequences, reduce down to 2D
        # x = self.get_latest(x)
        x = layers.LSTM(lstm_size, activation='tanh')(x)
        x = layers.Dropout(0.2)(x)

        # last layer is a linear (float) value
        outputs = layers.Dense(1, activation="linear")(x)

        model = Model(inputs, outputs, name=self.name)

        return model

    # get estimation layer. Override this in subclasses (LSTM implementation here as default)
    def get_estimation_layer(self, x_in):
        x_shape = np.shape(x_in)
        lstm_size = max(16, self.nearest_power_of_2(x_shape[-1]-1))

        x = x_in
        x = layers.LSTM(lstm_size, return_sequences=True)(x)
        x = layers.Dropout(0.2)(x)
        return x

    # utilities used by subclasses


    def transformer_encoder(self, inputs, head_size, num_heads, dropout, ff_dim):

        # Attention and Normalization
        x = layers.MultiHeadAttention(key_dim=head_size, num_heads=num_heads, dropout=dropout)(inputs, inputs)
        x = layers.Dropout(dropout)(x)
        x = layers.LayerNormalization(epsilon=1e-6)(x)
        res = x + inputs

        # Feed Forward Part
        x = layers.Conv1D(filters=ff_dim, kernel_size=1, activation="relu")(res)
        x = layers.Dropout(dropout)(x)
        x = layers.Conv1D(filters=inputs.shape[-1], kernel_size=1)(x)
        x = layers.LayerNormalization(epsilon=1e-6)(x)

        return x + res


    def wavenetBlock(self, n_filters, filter_size, rate):
        def f(input_):
            residual = input_
            tanh_out = layers.Convolution1D(n_filters, filter_size, padding="causal", dilation_rate=rate,
                                            activation='tanh')(input_)
            sigmoid_out = layers.Convolution1D(n_filters, filter_size, padding="causal", dilation_rate=rate,
                                               activation='sigmoid')(input_)
            merged = layers.Multiply()([tanh_out, sigmoid_out])

            x = layers.Multiply()([tanh_out, sigmoid_out])

            res_x = layers.Convolution1D(n_filters, 1, padding='same', kernel_regularizer=regularizers.l2(0))(x)
            skip_x = layers.Convolution1D(n_filters, 1, padding='same', kernel_regularizer=regularizers.l2(0))(x)
            res_x = layers.Add()([input_, res_x])
            return res_x, skip_x

        return f

# ------------------------------

class EstimationLayer(tf.keras.layers.Layer): # type: ignore
    def __init__(self, parent, **kwargs):
        super().__init__(**kwargs)
        self.parent = parent

    def call(self, inputs):
        return self.parent.get_estimation_layer(inputs)

    def compute_output_shape(self, input_shape):
        if input_shape is None:
            return None
        
        # Create a dummy input tensor with a valid shape
        dummy_shape = [1 if dim is None else dim for dim in input_shape]
        dummy_input = tf.zeros(dummy_shape)
        
        # Get the output of get_estimation_layer for the dummy input
        dummy_output = self.parent.get_estimation_layer(dummy_input)
        
        # Return the shape of the dummy output, preserving None dimensions
        output_shape = list(dummy_output.shape)
        for i, dim in enumerate(input_shape):
            if dim is None:
                output_shape[i] = None
        return tuple(output_shape)
    

class predictor_additive_attention(NNPredictor):

    def get_estimation_layer(self, x_in):
        x_shape = np.shape(x_in)
        x_size = x_shape[-1]

        x = x_in

        # Query, Value, Key for attention (Same size as input units)
        query = layers.Dense(x_size, name="query")(x)
        value = layers.Dense(x_size, name="value")(x)
        key = layers.Dense(x_size, name="key")(x)

        # Additive Attention Layer
        x = layers.AdditiveAttention()([query, value, key])

        x = layers.Dropout(0.2)(x)
        return x


# ------------------------------

class predictor_attention(NNPredictor):

    def get_estimation_layer(self, x_in):
        x_shape = np.shape(x_in)
        x_size = x_shape[-1]

        x = x_in

        # Query, Value, Key for attention (Same size as input units)
        query = layers.Dense(x_size, name="query")(x)
        value = layers.Dense(x_size, name="value")(x)
        key = layers.Dense(x_size, name="key")(x)

        # Additive Attention Layer
        x = layers.Attention()([query, value, key])

        x = layers.Dropout(0.2)(x)
        return x

# ------------------------------

class predictor_cnn(NNPredictor):

    def get_estimation_layer(self, x_in):
        x_shape = np.shape(x_in)

        x = x_in
        # x = layers.Conv1D(filters=64, kernel_size=2, padding="causal", 
        #                            activation="relu", input_shape=(x_shape[1], x_shape[2]))(x)

        # k_size = min(3, x_shape[1])
        k_size = 3
        num_filters = self.nearest_power_of_2(x_shape[-1]-1)
        x = layers.Conv1D(filters=num_filters, kernel_size=k_size, activation='relu', padding='same', 
                          input_shape=(x_shape[1], x_shape[2]))(x)

        num_filters = num_filters // 2
        while num_filters >= 32:
            x = layers.Conv1D(filters=num_filters, kernel_size=k_size, activation='relu', padding='same')(x)
            num_filters = num_filters // 2

        x = layers.Dropout(0.2)(x)
        return x
# ------------------------------

class predictor_combo(NNPredictor):

    def get_estimation_layer(self, x_in):
        x_shape = np.shape(x_in)

        x = x_in

        # create various types of predictors (avoid heavy duty layers if possible)
        p1 = make_predictor(PredictorType.PCA, self.pair, self.seq_len, self.num_features)
        p2 = make_predictor(PredictorType.MLP, self.pair, self.seq_len, self.num_features)
        p3 = make_predictor(PredictorType.ATTENTION, self.pair, self.seq_len, self.num_features)
        p4 = make_predictor(PredictorType.WAVENET, self.pair, self.seq_len, self.num_features)

        # get the estimation layers
        x1 = p1.get_estimation_layer(x)
        x2 = p2.get_estimation_layer(x)
        x3 = p3.get_estimation_layer(x)
        x4 = p4.get_estimation_layer(x)

        # combine outputs
        x = layers.Concatenate()([x1, x2, x3, x4])

        # # return to original size
        # x = layers.Dense(x_shape[-1], activation='tanh')(x) # type: ignore

        return x

# ------------------------------


class predictor_encoder(NNPredictor):

    def get_estimation_layer(self, x_in):
        x_shape = np.shape(x_in)
        x = x_in

        n_steps = x_shape[1]
        n_features = x_shape[2]

        # layers.Reshape the input from (batch_size, n_steps, n_features) -> (batch_size, n_steps * n_features)
        x = layers.Flatten(input_shape=(n_steps, n_features))(x) # Flatten 3D input to 2D

        # Autoencoder/MLP
        n = self.nearest_power_of_2(n_features-1)
        while n >= 16:
            x = layers.Dense(n)(x)
            x = layers.Dropout(0.1)(x)
            n = n // 2

        # layers.Reshape the output back to (batch_size, n_steps * n_features) -> (batch_size, steps, n_features)
        # x = layers.Dense(n_steps * n_features)(x)  # 1D output
        # x = layers.Reshape((n_steps, n_features))(x)  # layers.Reshape output back to 3D

        x = layers.Dense(n_steps * 16)(x)  # 1D output
        x = layers.Reshape((n_steps, 16))(x)  # layers.Reshape output back to 3D

        # # Add an LSTM to extract patterns
        # x = layers.LSTM(16, return_sequences=True)(x)

        return x

# ------------------------------


class predictor_gru(NNPredictor):

    def get_estimation_layer(self, x_in):
        x_shape = np.shape(x_in)

        x = x_in
        x = layers.Conv1D(filters=64, kernel_size=2, activation="relu", padding="causal")(x)
        x = layers.GRU(32, return_sequences=True)(x)

        x = layers.Dropout(0.2)(x)
        return x


# ------------------------------

class predictor_lstm(NNPredictor):
    """Hybrid LSTM regressor — same blueprint as the prior classifier work:

        inputs                                    (B, T, num_features)
          → Dense(num_filters, tanh)              [resize, in parallel]
          ├─ identity (resize_out)
          └─ Conv1D(kernel=2, padding='same')
                → Dropout(0.2)
          → Concatenate([resize_out, conv_out])    (B, T, 2*num_filters)
          → LSTM (returns last timestep)           (B, num_filters)
          → Dropout(0.2)
          → Dense(1, linear)                       (B, 1)

    Resize gives global feature interaction at each timestep; Conv1D
    sees adjacent timesteps for local temporal patterns. Concatenating
    keeps both signals in separate channels rather than collapsing
    them via add. Mirrors the MLX _LSTMRegressorModel.

    This bypasses the parent NNPredictor.create_model. get_estimation_layer
    is retained for compatibility but unused by predictor_lstm itself.
    """

    def create_model(self, seq_len, num_features):
        inputs = Input(shape=(seq_len, num_features))
        lstm_size = max(self.nearest_power_of_2(num_features - 1), 16)

        x_r = layers.Dense(lstm_size, activation='tanh')(inputs)
        x_c = layers.Conv1D(lstm_size, kernel_size=2, padding='same')(x_r)
        x_c = layers.Dropout(0.2)(x_c)

        concat = layers.Concatenate(axis=-1)([x_r, x_c])

        x = layers.LSTM(lstm_size)(concat)
        x = layers.Dropout(0.2)(x)

        outputs = layers.Dense(1, activation='linear')(x)
        return Model(inputs, outputs, name=self.name)

    def get_estimation_layer(self, x_in):
        # Retained for compatibility with NNPredictor.create_model (called by
        # other subclasses that don't override create_model). Unused by
        # predictor_lstm itself now that this class has its own create_model.
        x_shape = np.shape(x_in)
        lstm_size = max(16, self.nearest_power_of_2(x_shape[-1]-1))

        x = x_in
        x = layers.LSTM(lstm_size, return_sequences=True)(x)
        x = layers.Dropout(0.2)(x)
        return x


# ------------------------------

class predictor_lstm3(NNPredictor):

    def get_estimation_layer(self, x_in):
        x_shape = np.shape(x_in)
        x = x_in

        n = max(16, self.nearest_power_of_2(x_shape[-1]-1))
        while n >= 8:
            for _ in range(3):
                x = layers.LSTM(n, activation='tanh', return_sequences=True)(x)
                x = layers.Dropout(rate=0.1)(x)

            n = n // 2

        x = layers.Dropout(0.2)(x)
        return x

# ------------------------------


class predictor_mlp(NNPredictor):

    def get_estimation_layer(self, x_in):
        x_shape = np.shape(x_in)
        x = x_in

        n_steps = x_shape[1]
        n_features = x_shape[2]

        # layers.Reshape the input from (batch_size, n_steps, n_features) -> (batch_size, n_steps * n_features)
        x = layers.Flatten(input_shape=(n_steps, n_features))(x) # Flatten 3D input to 2D

        for _ in range (3):
            x = layers.Dense(128, activation='tanh')(x) # type: ignore
            x = layers.Dropout(0.1)(x)

        # x = layers.BatchNormalization()(x)

        for _ in range (3):
            x = layers.Dense(64, activation='tanh')(x) # type: ignore
            x = layers.Dropout(0.1)(x)

        # x = layers.BatchNormalization()(x)

        for _ in range (3):
            x = layers.Dense(32, activation='tanh')(x) # type: ignore
            x = layers.Dropout(0.1)(x)

        # x = layers.BatchNormalization()(x)

        for _ in range (3):
            x = layers.Dense(16, activation='tanh')(x) # type: ignore
            x = layers.Dropout(0.1)(x)

        # x = layers.BatchNormalization()(x)


        # layers.Reshape the output back to (batch_size, n_steps * n_features) -> (batch_size, steps, n_features)
        x = layers.Dense(n_steps * n_features)(x)  # 1D output
        x = layers.Reshape((n_steps, n_features))(x)  # layers.Reshape output back to 3D

        # x = layers.Dropout(0.2)(x)
        return x

# ------------------------------


class predictor_multihead(NNPredictor):

    def get_estimation_layer(self, x_in):
        x_shape = np.shape(x_in)
        x = x_in

        seq_len = x_shape[1]
        head_size = max(self.nearest_power_of_2(int(x_shape[-1]/2)-1), 16)
        num_heads = 4
        ff_dim = 4
        dropout = 0.2
        num_blocks = 4

        # "ATTENTION LAYER"
        # x = layers.MultiHeadAttention(key_dim=num_features, num_heads=num_heads, dropout=dropout)(x, x)
        # x = layers.Dropout(0.1)(x)
        # res = x + x

        # # FEED FORWARD Part - you can stick anything here or just delete the whole section - it will still work.
        # x = layers.LayerNormalization(epsilon=1e-6)(res)
        # x = layers.Conv1D(filters=seq_len, kernel_size=1, activation="relu")(x)
        # x = layers.Dropout(dropout)(x)
        # x = layers.Conv1D(filters=num_features, kernel_size=1)(x)
        # x = x + res

        # sort of a simplified Transformer architecture

        for _ in range(num_blocks):
            # MultiHeadAttention
            attention_output = layers.MultiHeadAttention(key_dim=head_size, num_heads=num_heads, dropout=dropout)(x, x)
            x = layers.LayerNormalization(epsilon=1e-6)(attention_output + x)

            # Feed Forward Part
            ff = layers.Dense(ff_dim, activation="relu")(x)
            ff = layers.Dense(x_in.shape[-1])(ff)
            x = layers.LayerNormalization(epsilon=1e-6)(ff + x)

        x = layers.Dropout(0.2)(x)
        return x

# ------------------------------


# Note that this uses the keras-beats package

from NBeatsLayer import NBeatsLayer # type: ignore

class predictor_nbeats(NNPredictor):

    def get_estimation_layer(self, x_in):

        x_shape = np.shape(x_in)
        x = x_in

        seq_len = x_shape[1]

        # NBeats is really slow, so trim down the number of features
        # num_features = max(self.nearest_power_of_2(int(x_shape[-1]/2)-1), 16)
        num_features = max(self.nearest_power_of_2(int(x_shape[-1]/4)-1), 16)

        nbeats = NBeatsLayer(input_dim=num_features, output_dim=num_features, forecast_length=seq_len, backcast_length=seq_len)
        x = nbeats(inputs=x)

        return x

# ------------------------------

from PCALayer import TrainablePCALayer # type: ignore

class predictor_pca(NNPredictor):

    def get_estimation_layer(self, x_in):
        x_shape = np.shape(x_in)
        x = x_in

        num_filters = x_shape[-1]
        pca_size = num_filters

        pca = TrainablePCALayer(n_components=pca_size)
        x = pca(x)

        x = layers.LSTM(pca_size//2, activation='tanh', return_sequences=True)(x)

        # x = layers.Conv1D(filters=128, kernel_size=2, activation="relu", padding="causal")(x)
        # x = layers.GRU(64, return_sequences=True)(x)

        x = layers.Dropout(0.2)(x)
        return x
# ------------------------------


class predictor_tcn(NNPredictor):

    def get_estimation_layer(self, x_in):
        x_shape = np.shape(x_in)
        x = x_in

        num_filters = max(self.nearest_power_of_2(int(x_shape[-1]/2)-1), 16)

        x = TCN(nb_filters=num_filters, return_sequences=True, kernel_initializer='glorot_uniform',  use_layer_norm=True)(x)

        x = layers.Dropout(0.2)(x)
        return x

# ------------------------------


class predictor_transformer(NNPredictor):

    def get_estimation_layer(self, x_in):
        x_shape = np.shape(x_in)
        x = x_in

        head_size = x_shape[-1]
        # num_heads = max(x_shape[1], int(x_shape[2]))
        num_heads = 4
        # ff_dim = x_shape[1]
        ff_dim = 4
        # num_transformer_blocks = x_shape[1]
        num_transformer_blocks = 4
        dropout = 0.2

        # Add positional encoding as a proper Keras layer
        x = PositionalEncoding(x_shape[1], x_shape[2])(x)
        
        for _ in range(num_transformer_blocks):
            x = self.transformer_encoder(x, head_size, num_heads, dropout, ff_dim)


        # x = layers.Dropout(0.2)(x)
        return x

# ------------------------------

class predictor_wavenet(NNPredictor):

    def get_estimation_layer(self, x_in):
        x_shape = np.shape(x_in)
        x = x_in

        num_filters = max(self.nearest_power_of_2(int(x_shape[-1]/2)-1), 16)
        # simplified Wavenet
        for rate in (1, 2, 4, 8) * 2: # type: ignore
            x = layers.Conv1D(filters=num_filters, kernel_size=2, padding="causal", activation="relu", dilation_rate=rate)(x)

        x = layers.Dropout(0.2)(x)
        return x

# ------------------------------


class predictor_wavenet_full(NNPredictor):

    def get_estimation_layer(self, x_in):
        x_shape = np.shape(x_in)
        x = x_in

        num_features = x_shape[-1]

        # Wavenet2 model, which is a series of convolutional layers with increasing dilution rate:
        # This is the full version

        if num_features > 32:
            filter_size = 64
        else:
            filter_size = 32


        x = layers.Convolution1D(filter_size, 2, padding="causal", dilation_rate=1)(x)

        # A, B = self.wavenetBlock(64, 2, 1)(inputs)

        skip_connections = []
        for i in range(1, 3):
            rate = 1
            for j in range (1, 10):
                x, skip = self.wavenetBlock(filter_size, 2, rate)(x)
                skip_connections.append(skip)
                rate = 2 * rate

            x = layers.BatchNormalization()(x)

        x = layers.Add()(skip_connections)
        x = layers.Activation('relu')(x)
        x = layers.Convolution1D(filter_size, 1, padding='same', kernel_regularizer=regularizers.l2(0))(x)
        x = layers.Activation('relu')(x)
        x = layers.Convolution1D(filter_size, 1, padding='same')(x)

        x = layers.Dropout(0.2)(x)
        return x

# ------------------------------

# This is a variant of NNPredictor that applies a prediction to each column individually, then again on those predictions
class ColumnarPredictor(NNPredictor):

    def split_input(self, x):
        return tf.split(x, num_or_size_splits=x.shape[-1], axis=-1)

    def output_shape(self, input_shape):
        output_shape = [input_shape[:-1] + (1,)] * input_shape[-1]
        print(f'input_shape: {input_shape} output_shape:{output_shape}')
        return output_shape

    def create_model(self, seq_len, num_features):

        inputs = Input(shape=(seq_len, num_features))

        # Normalise the input data (to make sure there aren't any pair-specific values like price)
        xn = layers.BatchNormalization()(inputs)

        # # get an LSTM analysis of the combined input
        # in_lstm = layers.LSTM(16, return_sequences=True)(xn)

        # col_outputs = [in_lstm]
        col_outputs = []
        # Iterate through the last dimension (features)
        for i in range(xn.shape[-1]):
            # Extract the i-th feature across all batches and rows
            xfi=xn[:, :, i:i+1]
            # run the estimation layer on this feature
            xfi = self.get_estimation_layer(xfi)
            col_outputs.append(xfi)

        # Concatenate the outputs
        concat_output = layers.Concatenate()(col_outputs)

        # run an LSTM on the concatenated results
        # lstm_size = min(32, self.nearest_power_of_2(num_features-1))
        x = layers.LSTM(64)(concat_output)

        # Get the latest sequences, reduce down to 2D
        # x = self.get_latest(concat_output)
        x = self.get_latest(x)

        # last layer is a linear (float) value
        outputs = layers.Dense(1, activation="linear")(x)

        model = Model(inputs, outputs, name=self.name)

        return model

# ------------------------------

class columnar_lstm(ColumnarPredictor):

    def get_estimation_layer(self, x_in):
        x_shape = tf.keras.backend.int_shape(x_in)
        # print(f'x_in: {x_in.shape} x_shape: {x_shape}')
        lstm_size = max(16, self.nearest_power_of_2(x_shape[-1] - 1))
        x = layers.LSTM(lstm_size, return_sequences=True)(x_in)

        x = layers.Dropout(0.2)(x)
        return x


# ------------------------------

class columnar_mlp(ColumnarPredictor):

    def get_estimation_layer(self, x_in):
        x_shape = np.shape(x_in)
        x = x_in

        n_steps = x_shape[1]
        n_features = x_shape[2]

        # layers.Reshape the input from (batch_size, n_steps, n_features) -> (batch_size, n_steps * n_features)
        x = layers.Flatten(input_shape=(n_steps, n_features))(x) # Flatten 3D input to 2D

        n = 64
        while n >= 16:
            for _ in range (3):
                x = layers.Dense(n, activation='tanh')(x) # type: ignore
                x = layers.Dropout(0.1)(x)
            n = n // 2


        # layers.Reshape the output back to (batch_size, n_steps * n_features) -> (batch_size, steps, n_features)
        x = layers.Dense(n_steps * n_features)(x)  # 1D output
        x = layers.Reshape((n_steps, n_features))(x)  # layers.Reshape output back to 3D

        # x = layers.Dropout(0.2)(x)
        return x


# -----------------------------------

class columnar_transformer(ColumnarPredictor):

    def get_estimation_layer(self, x_in):
        x_shape = np.shape(x_in)
        x = x_in

        n_steps = x_shape[1]
        n_features = x_shape[2]

        head_size = x_shape[-1]
        # num_heads = max(x_shape[1], int(x_shape[2]))
        num_heads = 4
        # ff_dim = x_shape[1]
        ff_dim = 4
        # num_transformer_blocks = x_shape[1]
        num_transformer_blocks = 4
        dropout = 0.2

        # Add positional encoding as a proper Keras layer
        x = PositionalEncoding(n_steps, n_features)(x)
        
        for _ in range(num_transformer_blocks):
            x = self.transformer_encoder(x, head_size, num_heads, dropout, ff_dim)

        return x


# -----------------------------------

# enum of all available predictor types


class PredictorType(Enum):
    ADDITIVE_ATTENTION = predictor_additive_attention
    ATTENTION = predictor_attention
    COMBO = predictor_combo
    CNN = predictor_cnn
    ENCODER = predictor_encoder
    GRU = predictor_gru
    LSTM = predictor_lstm
    LSTM3 = predictor_lstm3
    MLP = predictor_mlp
    MULTIHEAD = predictor_multihead
    NBEATS = predictor_nbeats
    PCA = predictor_pca
    TCN = predictor_tcn
    TRANSFORMER = predictor_transformer
    WAVENET = predictor_wavenet
    WAVENET_FULL = predictor_wavenet_full

    COLUMNAR_LSTM = columnar_lstm
    COLUMNAR_MLP = columnar_mlp
    COLUMNAR_TRANSFORMER = columnar_transformer



# (static) function to create a predictor of the specified type
def make_predictor(predictor_type: PredictorType, pair, seq_len: int, num_features: int) -> NNPredictor:
    return predictor_type.value(pair, seq_len, num_features)


# -----------------------------------
