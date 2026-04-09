"""
Attention layers for neural network classifiers.
Provides reusable attention mechanisms for temporal data processing.
"""

import tensorflow as tf


class AttentionPooling1D(tf.keras.layers.Layer):
    """
    Learns a weighted average over the time dimension.
    Input:  (batch, time, features)
    Output: (batch, features)
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.attention_dense = None
        self.softmax = tf.keras.layers.Softmax(axis=1)

    def build(self, input_shape):
        # Create the attention dense layer with proper input shape
        self.attention_dense = tf.keras.layers.Dense(1, activation="tanh")
        # Call build on the dense layer to initialize weights
        self.attention_dense.build(input_shape)
        super().build(input_shape)

    def call(self, inputs):
        # Compute attention scores for each timestep
        scores = self.attention_dense(inputs)  # (batch, time, 1)
        weights = self.softmax(scores)  # (batch, time, 1)
        # Weighted sum over time
        weighted_sum = tf.reduce_sum(inputs * weights, axis=1)  # (batch, features)
        return weighted_sum

    def get_config(self):
        config = super().get_config()
        # No extra params to save, but method is required for serialization
        return config


class MultiHeadAttentionPooling1D(tf.keras.layers.Layer):
    """
    Multi-head attention pooling for more sophisticated temporal focus.
    Input:  (batch, time, features)
    Output: (batch, features)
    """

    def __init__(self, num_heads=4, key_dim=32, **kwargs):
        super().__init__(**kwargs)
        self.num_heads = num_heads
        self.key_dim = key_dim
        self.multi_head_attention = None
        self.output_dense = None

    def build(self, input_shape):
        # Create multi-head attention layer
        self.multi_head_attention = tf.keras.layers.MultiHeadAttention(
            num_heads=self.num_heads, 
            key_dim=self.key_dim
        )
        
        # Output projection
        self.output_dense = tf.keras.layers.Dense(input_shape[-1])
        
        super().build(input_shape)

    def call(self, inputs):
        # Apply multi-head attention (self-attention)
        attended = self.multi_head_attention(inputs, inputs)
        
        # Global average pooling over time dimension
        pooled = tf.reduce_mean(attended, axis=1)
        
        # Final projection
        output = self.output_dense(pooled)
        
        return output

    def get_config(self):
        config = super().get_config()
        config.update({
            'num_heads': self.num_heads,
            'key_dim': self.key_dim
        })
        return config
