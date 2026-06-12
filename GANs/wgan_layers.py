"""Shared WGAN-GP Keras layer classes.

Byte-identical between df_wgan_gp.py (single-task) and df_mt_wgan_gp.py
(multi-task); kept in one place. WGAN persists weights (save_weights /
load_weights), rebuilding the architecture from code, so these layers can live
in a shared module without affecting model load.
"""

from __future__ import annotations

import tensorflow as tf
from keras import Layer


class _SplitLayer(Layer):
    """Custom layer to split tensor into two parts (for FiLM gamma/beta)"""

    def call(self, inputs):
        return tf.split(inputs, 2, axis=-1)


class _ResizeToLenLayer(Layer):
    """Custom layer to resize sequence to target length"""

    def __init__(self, target_len, **kwargs):
        super().__init__(**kwargs)
        self.target_len = target_len

    def call(self, inputs):
        cur_len = tf.shape(inputs)[1]

        def resized():
            tt = tf.expand_dims(inputs, 2)
            tt = tf.image.resize(tt, (self.target_len, 1), method="bilinear")
            tt = tf.squeeze(tt, 2)
            return tf.cast(tt, inputs.dtype)

        def identity():
            return inputs

        return tf.cond(tf.equal(cur_len, self.target_len), identity, resized)

    def get_config(self):
        config = super().get_config()
        config.update({"target_len": self.target_len})
        return config


class _MinibatchStdLayer(Layer):
    """Custom layer for minibatch standard deviation"""

    def call(self, inputs):
        m = tf.reduce_mean(inputs, axis=0, keepdims=True)
        v = tf.reduce_mean(tf.square(inputs - m), axis=0, keepdims=True)
        s = tf.sqrt(v + 1e-8)
        s = tf.reduce_mean(s, axis=1, keepdims=True)
        return tf.tile(s, [tf.shape(inputs)[0], 1])
