from typing import Any
# Neural Network Classifiers for Anomaly detection

# usage: classifier, name = NNDetector.create_classifier(classifier_type, pair, nfeatures, seq_len, tag="")

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
import random
import os

warnings.simplefilter(action="ignore", category=FutureWarning)

warnings.filterwarnings("ignore", category=UserWarning, module="keras.src.saving.saving_lib")

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_DETERMINISTIC_OPS'] = '1'
os.environ['TF_RUN_EAGER_OP_AS_FUNCTION'] = '0'

import tensorflow as tf

from utils.ClassifierKerasAnomaly import ClassifierKerasAnomaly

# Base class for anomaly classifiers
class AnomalyClassifierBase(ClassifierKerasAnomaly):
    is_trained = False
    clean_data_required = False  # training data cannot contain anomalies

    def __init__(self, pair, seq_len, num_features, tag=""):
        super().__init__(pair, seq_len, num_features, tag=tag)
        # Override the name set by the parent class
        self.name = self.__class__.__name__

    def create_model(self, seq_len, num_features, pca_W=None, pca_M=None):
        """
        Override the base create_model to implement the multi-task architecture.
        This method creates the input/output layers and calls the subclass-specific architecture.
        """

        self.seq_len = seq_len

        # Input layer
        inputs = tf.keras.layers.Input(
            shape=(seq_len, num_features), name="input_layer"
        )

        latent_size = max(8, num_features // 4)

        # encode the input (returns 3D tensor for non-Dense variants)
        encoded = self.create_encoder(inputs, seq_len, num_features, latent_size)

        # Decoder takes 3D tensor directly from encoder (for non-Dense variants)
        decoded = self.create_decoder(encoded, seq_len, num_features, latent_size)
        
        # Classifier head needs 2D input, so flatten/pool the 3D encoder output
        # For 3D tensors, use GlobalAveragePooling1D; for 2D (Dense variant), use as-is
        if len(encoded.shape) == 3:
            latent_2d = tf.keras.layers.GlobalAveragePooling1D()(encoded)
        else:
            latent_2d = encoded
        classifier_features = self.create_classifier_head(
            latent_2d, seq_len, latent_size
        )

        # transform latent vector back to original and a classification
        reconstruction = decoded
        trading_head = classifier_features

        # set up the outputs (names required)

        reconstruction_output = tf.keras.layers.Dense(
            num_features, activation="linear", name="reconstruction"
        )(reconstruction)

        trading_output = tf.keras.layers.Dense(
            3, activation="softmax", bias_initializer="zeros", name="trading"
        )(trading_head)

        # Create the multi-output model
        model = tf.keras.Model(
            inputs=inputs,
            outputs={
                'reconstruction': reconstruction_output,
                'trading': trading_output,
            },
            name=self.name
        )

        return model

    def create_encoder(self, inputs, seq_len, num_features, latent_size):
        """
        Override the base create_encoder to implement the specific architecture.
        """
        raise NotImplementedError("Subclass must implement create_encoder")

    def create_decoder(self, latent_vector, seq_len, num_features, latent_size):
        """
        Override the base create_decoder to implement the specific architecture.
        """
        raise NotImplementedError("Subclass must implement create_decoder")

    def create_classifier_head(self, latent_vector, seq_len, latent_size):
        """
        takes latent vector and reduces it down to a dense vector ready for classification
        """

        l2_strength = 1e-3
        d1_size = max(8, self.nearest_power_of_2((latent_size//2)+1))
        d2_size = max(4, d1_size//2)

        classifier_head = tf.keras.Sequential(
            [
                # tf.keras.layers.GlobalAveragePooling1D(),
                tf.keras.layers.Dense(
                    d1_size,
                    activation="elu",
                    kernel_regularizer=tf.keras.regularizers.l2(l2_strength),
                ),
                tf.keras.layers.Dropout(0.2),
                tf.keras.layers.Dense(
                    d2_size,
                    activation="elu",
                    kernel_regularizer=tf.keras.regularizers.l2(l2_strength),
                ),
                tf.keras.layers.Dropout(0.2),
            ],
            name="classifier_head",
        )(latent_vector)

        return classifier_head 

    def get_reconstruction_errors(self, data):
        """Calculate reconstruction errors (MSE between input and output)"""
        reconstructions = self.predict(data, verbose=0)

        # For 3D data, average across the sequence dimension
        if len(reconstructions.shape) == 3:
            errors = np.mean(np.square(data - reconstructions), axis=(1, 2))
        else:
            errors = np.mean(np.square(data - reconstructions), axis=1)

        return errors


# --------------------------------------------------------------
# Dense Autoencoder

class AnomalyClassifier_Dense(AnomalyClassifierBase):
    
    def create_encoder(self, inputs, seq_len, num_features, latent_size):
        # Calculate total input features (seq_len * num_features for flattened data)
        total_features = seq_len * num_features
        encoder_size1 = max(32, total_features * 2)
        encoder_size2 = max(16, total_features)

        encoded = tf.keras.layers.Dense(encoder_size1, activation='elu')(inputs)
        encoded = tf.keras.layers.BatchNormalization()(encoded)
        encoded = tf.keras.layers.Dropout(0.2)(encoded)
        
        encoded = tf.keras.layers.Dense(encoder_size2, activation='elu')(encoded)
        encoded = tf.keras.layers.BatchNormalization()(encoded)
        encoded = tf.keras.layers.Dropout(0.2)(encoded)

        return encoded

    def create_decoder(self, latent, seq_len, num_features, latent_size):
        total_features = seq_len * num_features
        decoder_size1 = max(32, total_features * 2)
        decoder_size2 = max(16, total_features)

        decoded = tf.keras.layers.Dense(decoder_size2, activation='elu')(latent)
        decoded = tf.keras.layers.BatchNormalization()(decoded)
        decoded = tf.keras.layers.Dropout(0.2)(decoded)
        
        decoded = tf.keras.layers.Dense(decoder_size1, activation='elu')(decoded)
        decoded = tf.keras.layers.BatchNormalization()(decoded)
        decoded = tf.keras.layers.Dropout(0.2)(decoded)
        return decoded


# --------------------------------------------------------------
# LSTM Autoencoder

class AnomalyClassifier_LSTM(AnomalyClassifierBase):

    def create_encoder(self, inputs, seq_len, num_features, latent_size):
        encoded = tf.keras.layers.LSTM(64, activation='tanh', return_sequences=True)(inputs)
        encoded = tf.keras.layers.Dropout(0.2)(encoded)
        encoded = tf.keras.layers.BatchNormalization()(encoded)

        encoded = tf.keras.layers.LSTM(32, activation='tanh', return_sequences=True)(encoded)
        encoded = tf.keras.layers.Dropout(0.2)(encoded)
        encoded = tf.keras.layers.BatchNormalization()(encoded)
        return encoded

    def create_decoder(self, latent, seq_len, num_features, latent_size):
        # Latent is already 3D tensor from encoder
        decoded = tf.keras.layers.LSTM(32, activation='tanh', return_sequences=True)(latent)
        decoded = tf.keras.layers.Dropout(0.2)(decoded)
        decoded = tf.keras.layers.BatchNormalization()(decoded)

        decoded = tf.keras.layers.LSTM(64, activation='tanh', return_sequences=True)(decoded)
        decoded = tf.keras.layers.Dropout(0.2)(decoded)
        decoded = tf.keras.layers.BatchNormalization()(decoded)

        return decoded

# --------------------------------------------------------------
# Attention-based Autoencoder

class AnomalyClassifier_Attention(AnomalyClassifierBase):

    def create_encoder(self, inputs, seq_len, num_features, latent_size):
        encoded = self.attention_encoder(inputs, head_size=32, num_heads=4, dropout=0.2, ff_dim=128)
        # Return 3D tensor (no GlobalAveragePooling1D)
        return encoded

    def create_decoder(self, latent, seq_len, num_features, latent_size):
        # Latent is already 3D tensor from encoder
        decoded = self.attention_encoder(latent, head_size=32, num_heads=4, dropout=0.2, ff_dim=128)
        return decoded

        
    def attention_encoder(self, inputs, head_size, num_heads, dropout, ff_dim):
        # Normalization and Attention
        x = tf.keras.layers.LayerNormalization(epsilon=1e-6)(inputs)
        
        # Only apply attention if sequence length > 1
        if inputs.shape[1] > 1:
            attention_output = tf.keras.layers.MultiHeadAttention(
                num_heads=num_heads, key_dim=head_size, dropout=dropout
            )(x, x)
            x = tf.keras.layers.Add()([attention_output, x])
        
        # Feed Forward
        x = tf.keras.layers.LayerNormalization(epsilon=1e-6)(x)
        ffn = tf.keras.Sequential([
            tf.keras.layers.Dense(ff_dim, activation="relu"),
            tf.keras.layers.Dropout(dropout),
            tf.keras.layers.Dense(inputs.shape[-1]),
        ])
        x = tf.keras.layers.Add()([ffn(x), x])
        
        return x


# --------------------------------------------------------------
# Full Transformer Autoencoder

class TransformerBlock(tf.keras.layers.Layer):
    """Complete Transformer block with self-attention and feed-forward"""
    
    def __init__(self, t_size, num_heads, dff, rate=0.1, **kwargs):
        super(TransformerBlock, self).__init__(**kwargs)
        self.t_size = t_size
        self.num_heads = num_heads
        self.dff = dff
        self.rate = rate
        
        # Initialize layers in __init__ but don't build them yet
        self.att = None
        self.ffn = None
        self.layernorm1 = None
        self.layernorm2 = None
        self.dropout1 = None
        self.dropout2 = None
    
    def build(self, input_shape):
        """Build the layer with proper input shape"""
        # Create layers with proper input shape
        self.att = tf.keras.layers.MultiHeadAttention(
            num_heads=self.num_heads, 
            key_dim=self.t_size,
            value_dim=self.t_size
        )
        
        self.ffn = tf.keras.Sequential([
            tf.keras.layers.Dense(self.dff, activation='relu'),
            tf.keras.layers.Dense(self.t_size)
        ])
        
        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        
        self.dropout1 = tf.keras.layers.Dropout(self.rate)
        self.dropout2 = tf.keras.layers.Dropout(self.rate)
        
        super(TransformerBlock, self).build(input_shape)
    
    def call(self, x, training=True):
        # Self-attention
        attn_output = self.att(x, x)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(x + attn_output)
        
        # Feed-forward
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        out2 = self.layernorm2(out1 + ffn_output)
        
        return out2
    
    def get_config(self):
        config = super(TransformerBlock, self).get_config()
        config.update({
            't_size': self.t_size,
            'num_heads': self.num_heads,
            'dff': self.dff,
            'rate': self.rate
        })
        return config

class AnomalyClassifier_Transformer(AnomalyClassifierBase):
    
    t_size = 32
    num_heads = 8
    dff = 256  # Feed-forward dimension
    num_layers = 4  # Number of transformer layers
    dropout_rate = 0.1

    def create_encoder(self, inputs, seq_len, num_features, latent_size):

        self.t_size = max(32, self.nearest_power_of_2(num_features//2)+1)  # Ensure t_size >= num_features
        self.num_heads = min(8, self.t_size // 8)  # Ensure num_heads divides t_size evenly

        # Project to t_size dimension
        x = tf.keras.layers.Dense(self.t_size)(inputs)
        
        # Transformer encoder layers
        for i in range(self.num_layers):
            x = TransformerBlock(self.t_size, self.num_heads, self.dff, self.dropout_rate, name=f'transformer_block_{i}')(x)
        
        # Return 3D tensor (no GlobalAveragePooling1D)
        encoded = x

        return encoded

    def create_decoder(self, latent, seq_len, num_features, latent_size):
        # Latent is already 3D tensor from encoder, ensure it's projected to t_size
        if latent.shape[-1] != self.t_size:
            x = tf.keras.layers.Dense(self.t_size)(latent)
        else:
            x = latent
        
        # Transformer decoder layers
        for i in range(self.num_layers):
            x = TransformerBlock(self.t_size, self.num_heads, self.dff, self.dropout_rate, name=f'transformer_block_decoder_{i}')(x)
        
        # Project back to original feature dimension
        decoded = tf.keras.layers.Dense(num_features, activation='linear')(x)

        return decoded


# --------------------------------------------------------------
# Types of Detector

class ClassifierType(Enum):
    Dense = AnomalyClassifier_Dense  # Dense Autoencoder
    LSTM = AnomalyClassifier_LSTM  # LSTM Autoencoder
    Attention = AnomalyClassifier_Attention  # Attention-based Autoencoder
    Transformer = AnomalyClassifier_Transformer  # Full Transformer Autoencoder

# --------------------------------------------------------------

# factory to create classifier based on ID. Returns classifier and name
def create_classifier(classifier_type: ClassifierType, pair, nfeatures, seq_len, tag=""):
    classifier_name = str(classifier_type).split(".")[-1]
    classifier = classifier_type.value(pair, seq_len, nfeatures, tag=tag)

    return classifier, classifier_name

# --------------------------------------------------------------
