# implements PCA (Primary Component Analysis) Layers in keras

import tensorflow as tf
import keras
import numpy as np
from sklearn.decomposition import PCA


# non-trainable version (set weights before calling model to predict)
@keras.saving.register_keras_serializable()
class PCALayer(tf.keras.layers.Layer): # type: ignore
    def __init__(self, n_components, **kwargs):
        super(PCALayer, self).__init__(**kwargs)
        self.n_components = n_components

    def build(self, input_shape):
        self.pca_matrix = self.add_weight(
            name='pca_matrix',
            shape=(input_shape[-1], self.n_components),
            initializer='zeros',
            trainable=False
        )

    def call(self, inputs):
        return tf.matmul(inputs, self.pca_matrix)

    def set_weights_from_pca(self, X):
        pca = PCA(n_components=self.n_components)
        pca.fit(X)
        self.set_weights([pca.components_.T])
'''
# Usage:
input_dim = 256
n_components = 64

inputs = tf.keras.Input(shape=(input_dim,))
pca_layer = PCALayer(n_components)(inputs)
model = tf.keras.Model(inputs=inputs, outputs=pca_layer)

# Generate some random data and fit PCA
X = np.random.randn(1000, input_dim)
pca_layer.set_weights_from_pca(X)

'''


@keras.saving.register_keras_serializable()
class TrainablePCALayer(tf.keras.layers.Layer):
    def __init__(self, n_components, initial_w=None, initial_m=None, **kwargs):
        super(TrainablePCALayer, self).__init__(**kwargs)
        self.n_components = n_components
        self.initial_w = initial_w
        self.initial_m = initial_m

    def build(self, input_shape):
        input_dim = input_shape[-1]

        # Determine initializer based on whether pre-calculated weights were provided
        w_initializer = (
            tf.keras.initializers.Constant(self.initial_w)
            if self.initial_w is not None
            else "glorot_uniform"
        )
        m_initializer = (
            tf.keras.initializers.Constant(self.initial_m)
            if self.initial_m is not None
            else "zeros"
        )

        # 1. Add the Trainable Projection Matrix (W)
        self.W = self.add_weight(
            name="pca_matrix",
            shape=(input_dim, self.n_components),
            initializer=w_initializer,  # Use pre-calculated W
            trainable=True,  # Keep trainable for fine-tuning
        )

        # 2. Add the Trainable Mean Vector (M)
        self.M = self.add_weight(
            name="pca_mean",
            shape=(1, input_dim),
            initializer=m_initializer,  # Use pre-calculated M
            trainable=True,
        )

        super(TrainablePCALayer, self).build(input_shape)

    def call(self, inputs):
        # Center the data using the LEARNED/INITIALIZED mean
        centered = inputs - self.M

        # Project data onto principal components
        projected = tf.matmul(centered, self.W)

        return projected

    def get_config(self):
        config = super(TrainablePCALayer, self).get_config()
        config.update({
            "n_components": self.n_components,
            # Note: You generally DO NOT save large weight matrices (W and M) in the config, 
            # as Keras saves them separately. Only save parameters needed for re-initialization.
        })
        return config
