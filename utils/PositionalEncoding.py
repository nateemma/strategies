# pragma pylint: disable=C0103, C0114, C0115, C0116, C0301, C0302, C0303, C0325, C0411, C0413
# pragma pylint: disable=W0105, W1203, W1309, W1514, W0613, W0621,
# type: ignore
# pylint: disable=import-error

"""
PositionalEncoding Layer for Transformer models

This module provides a standalone PositionalEncoding layer that can be properly
serialized and deserialized by Keras when saving/loading models.
"""

import numpy as np
import tensorflow as tf
import keras


@keras.saving.register_keras_serializable()
class PositionalEncoding(tf.keras.layers.Layer):
    """
    Positional encoding layer for Transformer models.
    
    This layer adds sinusoidal positional encoding to the input sequence.
    It can be properly serialized and deserialized by Keras.
    """
    
    def __init__(self, seq_len, num_features, **kwargs):
        super().__init__(**kwargs)
        self.seq_len = seq_len
        self.num_features = num_features
        
    def build(self, input_shape):
        # Create the positional encoding matrix
        pos_encoding = np.zeros((self.seq_len, self.num_features))
        
        for pos in range(self.seq_len):
            for i in range(0, self.num_features, 2):
                # Even indices: sin
                pos_encoding[pos, i] = np.sin(pos / (10000 ** (i / self.num_features)))
                # Odd indices: cos
                if i + 1 < self.num_features:
                    pos_encoding[pos, i + 1] = np.cos(pos / (10000 ** (i / self.num_features)))
        
        # Convert to tensor (float32 by default)
        self.pos_encoding = tf.constant(pos_encoding, dtype=tf.float32)
        
    def call(self, inputs):
        # Match dtype (e.g., float16 under mixed precision) before addition
        pos = tf.cast(self.pos_encoding, inputs.dtype)
        return inputs + pos

    # Help Keras infer output spec/shape
    def compute_output_shape(self, input_shape):
        return input_shape

        
    def get_config(self):
        config = super().get_config()
        config.update({
            'seq_len': self.seq_len,
            'num_features': self.num_features,
        })
        return config
    
    @classmethod
    def from_config(cls, config):
        return cls(**config) 