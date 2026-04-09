"""
WGAN-GP for conditional sequence data (batch, seq_len, num_features).

API: balance_with_wgan_gp(train_data, train_labels, epochs=100, batch_size=256, max_target=None, verbose=True)
Returns augmented (data, labels) where minority classes are upsampled to max_target per class.

Notes:
- train_labels must be one-hot (batch, num_classes)
- Data should be pre-scaled; generator output is tanh-bounded and rescaled to real mean/std then clipped to real min/max
"""

from __future__ import annotations

import numpy as np
import tensorflow as tf
from keras import layers, Model, Layer
from keras.callbacks import Callback
import os
import pickle
from enum import Enum


class WGANArchitecture(str, Enum):
    """WGAN architecture types"""

    BASELINE = "baseline"  # Simple baseline architecture
    CNN_RESIDUAL = "cnn_residual"  # Current CNN with residual connections (default)
    DCGAN = "dcgan"  # DCGAN-compliant architecture (Radford et al. 2015)
    MLP = "mlp"  # MLP architecture using Dense layers instead of CNNs


class GANCollapseCallback(Callback):
    """Callback to detect and stop GAN training when it collapses"""

    def __init__(
        self, patience=3, critic_loss_threshold=9.5, gen_loss_threshold=0.01, verbose=1
    ):
        super().__init__()
        self.patience = patience
        self.critic_loss_threshold = critic_loss_threshold
        self.gen_loss_threshold = gen_loss_threshold
        self.verbose = verbose
        self.collapse_count = 0
        self.best_epoch = -1  # -1 means no valid epoch found yet
        self.best_g_loss = float("inf")
        self.best_d_loss = float("inf")
        # Track a combined score: lower is better (we want low g_loss and low |d_loss|)
        self.best_score = float("inf")

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        g_loss = logs.get("g_loss", 0.0)
        d_loss = logs.get("d_loss", 0.0)

        # Check for collapse conditions FIRST (before tracking best epoch)
        # For WGAN, negative critic losses are normal (critic wants to maximize Wasserstein distance)
        # Only detect collapse when critic loss is positive and high (gradient penalty explosion)
        critic_collapse = d_loss >= self.critic_loss_threshold
        # For WGAN-GP, generator loss is negative of critic score on fakes
        # Negative g_loss is GOOD (generator fooling critic)
        # Collapse is when g_loss is close to zero (abs value), indicating generator not learning
        gen_collapse = abs(g_loss) <= self.gen_loss_threshold
        is_collapsed = critic_collapse or gen_collapse

        # Track best losses using a combined score
        # Score = g_loss + abs(d_loss) - we want both to be low
        # IMPORTANT: Only track non-collapsed epochs as "best"
        # Collapsed epochs (g_loss=0.0) have artificially low scores but are not good
        if not is_collapsed:
            current_score = g_loss + abs(d_loss)
            if current_score < self.best_score:
                self.best_score = current_score
                self.best_g_loss = g_loss
                self.best_d_loss = d_loss
                self.best_epoch = epoch

        if is_collapsed:
            self.collapse_count += 1
            if self.verbose > 0:
                reason = []
                if critic_collapse:
                    reason.append(
                        f"critic_loss={d_loss:.4f} >= {self.critic_loss_threshold}"
                    )
                if gen_collapse:
                    reason.append(
                        f"gen_loss={g_loss:.4f} (abs <= {self.gen_loss_threshold})"
                    )
                msg = (
                    f"    WARNING: GAN collapse detected ({', '.join(reason)}), "
                    f"count: {self.collapse_count}/{self.patience}"
                )
                print(msg)

            if self.collapse_count >= self.patience:
                if self.verbose > 0:
                    print(
                        f"    Stopping GAN training early due to collapse (epoch {epoch + 1})"
                    )
                    if self.best_epoch >= 0:
                        print(
                            f"    Best epoch was {self.best_epoch + 1} with "
                            f"g_loss={self.best_g_loss:.4f}, d_loss={self.best_d_loss:.4f}"
                        )
                        print(
                            f"    Best checkpoint (epoch {self.best_epoch + 1}) will be loaded automatically."
                        )
                    else:
                        print(
                            "    WARNING: No valid checkpoint found. All epochs collapsed."
                        )
                self.model.stop_training = True
        else:
            # Reset counter if training is stable
            if self.collapse_count > 0:
                self.collapse_count = max(0, self.collapse_count - 1)


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


class _SpectralNormalization(Layer):
    """Spectral Normalization wrapper for Dense/Conv layers"""

    def __init__(self, layer, **kwargs):
        super().__init__(**kwargs)
        self.layer = layer
        self.u = None

    def build(self, input_shape):
        self.layer.build(input_shape)
        super().build(input_shape)
        # Initialize u vector for power iteration
        w = self.layer.kernel
        u_shape = (1, w.shape[-1])
        self.u = self.add_weight(
            name="u",
            shape=u_shape,
            initializer="random_normal",
            trainable=False,
            dtype=w.dtype,
        )

    def call(self, inputs, training=False):
        w = self.layer.kernel
        w_shape = w.shape
        w_reshaped = tf.reshape(w, (-1, w_shape[-1]))

        # Power iteration for spectral norm
        u = self.u
        for _ in range(1):  # Single iteration per call (accumulates over training)
            v = tf.nn.l2_normalize(tf.matmul(u, w_reshaped, transpose_b=True), axis=1)
            u = tf.nn.l2_normalize(tf.matmul(v, w_reshaped), axis=1)

        self.u.assign(u)
        sigma = tf.matmul(tf.matmul(u, w_reshaped), v, transpose_b=True)
        w_normalized = w / (sigma + 1e-8)

        # Apply normalized weights
        if isinstance(self.layer, layers.Dense):
            return tf.matmul(inputs, w_normalized) + self.layer.bias
        elif isinstance(self.layer, layers.Conv1D):
            # For Conv1D, we need to reshape and apply convolution
            w_normalized_reshaped = tf.reshape(w_normalized, w_shape)
            return (
                tf.nn.conv1d(
                    inputs,
                    w_normalized_reshaped,
                    stride=self.layer.strides[0],
                    padding=self.layer.padding.upper(),
                )
                + self.layer.bias
            )
        return self.layer(inputs)


class _SelfAttention(Layer):
    """Self-attention layer for sequence data"""

    def __init__(self, dim=None, **kwargs):
        super().__init__(**kwargs)
        self.dim = dim
        self.query = None
        self.key = None
        self.value = None

    def build(self, input_shape):
        # Use input feature dimension if dim not specified
        feature_dim = input_shape[-1]
        if self.dim is None:
            self.dim = feature_dim
        self.query = layers.Dense(self.dim, use_bias=False)
        self.key = layers.Dense(self.dim, use_bias=False)
        self.value = layers.Dense(
            feature_dim, use_bias=False
        )  # Output same dim as input
        self.query.build(input_shape)
        self.key.build(input_shape)
        self.value.build(input_shape)
        super().build(input_shape)

    def call(self, inputs):
        q = self.query(inputs)
        k = self.key(inputs)
        v = self.value(inputs)

        # Scaled dot-product attention
        scores = tf.matmul(q, k, transpose_b=True) / tf.sqrt(float(self.dim))
        attn_weights = tf.nn.softmax(scores, axis=-1)
        attn_output = tf.matmul(attn_weights, v)

        # Residual connection
        return attn_output + inputs


class _ConditionalLayerNorm(Layer):
    """Conditional Layer Normalization using FiLM"""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.ln = None
        self.film_dense = None
        self.feature_dim = None

    def build(self, input_shapes):
        # input_shapes is a list: [x_shape, cond_shape]
        x_shape, cond_shape = input_shapes
        self.feature_dim = x_shape[-1]
        self.ln = layers.LayerNormalization()
        self.ln.build(x_shape)
        self.film_dense = layers.Dense(2 * self.feature_dim)
        self.film_dense.build(cond_shape)
        super().build(input_shapes)

    def call(self, inputs):
        x, cond = inputs
        normalized = self.ln(x)

        # FiLM modulation - use tf.split directly instead of custom layer
        gb = self.film_dense(cond)
        gamma, beta = tf.split(gb, 2, axis=-1)
        gamma = tf.reshape(gamma, (-1, 1, self.feature_dim))
        beta = tf.reshape(beta, (-1, 1, self.feature_dim))
        return normalized * (1.0 + gamma) + beta


class _PixelwiseNormalization(Layer):
    """
    Pixelwise normalization (from ProGAN/StyleGAN).
    Normalizes each spatial location by dividing by the square root of the mean
    of squared values across the feature dimension. Helps stabilize generator training.
    """

    def call(self, inputs):
        # inputs shape: (batch, seq_len, features)
        # Normalize across feature dimension (axis=-1)
        # Formula: x / sqrt(mean(x^2) + eps)
        return inputs / tf.sqrt(
            tf.reduce_mean(tf.square(inputs), axis=-1, keepdims=True) + 1e-8
        )


class _MinibatchStddev(Layer):
    """
    Mini-batch standard deviation layer (from ProGAN/StyleGAN).
    Helps stabilize training and prevent mode collapse by encouraging diversity.

    Computes std across mini-batch groups, averages across spatial dimensions,
    and concatenates as an additional feature map.
    """

    def __init__(self, group_size: int = 4, **kwargs):
        """
        Args:
            group_size: Number of samples to group together when computing std.
                      If batch size < group_size, uses batch size.
        """
        super().__init__(**kwargs)
        self.group_size = group_size

    def call(self, inputs):
        # inputs shape: (batch, seq_len, features)
        batch_size = tf.shape(inputs)[0]
        seq_len = tf.shape(inputs)[1]
        num_features = tf.shape(inputs)[2]

        # ProGAN approach: group samples into mini-batches
        # If batch_size < group_size, use the entire batch as one group
        group_size = tf.minimum(self.group_size, batch_size)

        # Reshape to group samples: (num_groups, group_size, seq_len, features)
        # Handle case where batch_size is not divisible by group_size
        num_groups = batch_size // group_size
        remainder = batch_size % group_size

        if remainder == 0:
            # Perfect division - reshape directly
            grouped = tf.reshape(inputs, (num_groups, group_size, seq_len, num_features))
        else:
            # Not perfectly divisible - pad with zeros or use truncated batch
            # For stability, we'll pad with the mean of the batch
            batch_mean = tf.reduce_mean(inputs, axis=0, keepdims=True)
            padding_size = group_size - remainder
            padding = tf.tile(batch_mean, [padding_size, 1, 1])
            padded = tf.concat([inputs, padding], axis=0)
            num_groups = (batch_size + padding_size) // group_size
            grouped = tf.reshape(padded, (num_groups, group_size, seq_len, num_features))

        # Compute std across group_size dimension (axis=1)
        # Result: (num_groups, seq_len, features)
        group_mean = tf.reduce_mean(grouped, axis=1, keepdims=True)
        group_variance = tf.reduce_mean(
            tf.square(grouped - group_mean), axis=1, keepdims=False
        )
        group_std = tf.sqrt(group_variance + 1e-8)

        # Average across spatial dimensions (seq_len and features) to get per-group std
        # Result: (num_groups,)
        std_per_group = tf.reduce_mean(group_std, axis=[1, 2])

        # Reshape back to match original batch structure
        # For each sample, use the std of its group
        if remainder == 0:
            # Each sample gets its group's std
            std_expanded = tf.reshape(std_per_group, (num_groups, 1, 1))
            std_expanded = tf.tile(std_expanded, [1, group_size, seq_len, 1])
            std_broadcast = tf.reshape(std_expanded, (batch_size, seq_len, 1))
        else:
            # Handle padded case - only use std for original samples
            std_expanded = tf.reshape(std_per_group, (num_groups, 1, 1))
            std_expanded = tf.tile(std_expanded, [1, group_size, seq_len, 1])
            std_padded = tf.reshape(std_expanded, (batch_size + padding_size, seq_len, 1))
            std_broadcast = std_padded[:batch_size]

        # Concatenate as additional feature map
        return tf.concat([inputs, std_broadcast], axis=-1)

    def get_config(self):
        config = super().get_config()
        config.update({"group_size": self.group_size})
        return config


class _GenBaseline(Model):
    """Simple baseline Generator architecture"""

    def __init__(
        self,
        seq_len: int,
        num_features: int,
        num_classes: int,
        latent_dim: int = 64,
        base_filters: int = 128,
        kernel_size: int = 3,
    ):
        super().__init__(name="wgangp_gen_baseline")
        self.seq_len = seq_len
        self.num_features = num_features
        self.num_classes = num_classes
        self.latent_dim = latent_dim
        self.base_filters = base_filters
        self.kernel_size = kernel_size

        z_in = layers.Input(shape=(latent_dim,))
        c_in = layers.Input(shape=(num_classes,))

        # Simple concatenation of noise and class
        x = layers.Concatenate()([z_in, c_in])
        x = layers.Dense(base_filters * seq_len)(x)
        x = layers.Reshape((seq_len, base_filters))(x)

        # Concatenate labels directly (not RepeatVector)
        x_flat = layers.Flatten()(x)
        x = layers.Concatenate()([x_flat, c_in])
        x = layers.Dense(base_filters * seq_len)(x)
        x = layers.Reshape((seq_len, base_filters))(x)

        # Simple Conv1D blocks (no pixelwise normalization)
        x = layers.Conv1D(base_filters, kernel_size, padding="same")(x)
        x = layers.LeakyReLU(0.2)(x)
        x = layers.Conv1D(base_filters // 2, kernel_size, padding="same")(x)
        x = layers.LeakyReLU(0.2)(x)

        # Final projection
        h = layers.Conv1D(
            num_features, kernel_size=kernel_size, padding="same", activation="tanh"
        )(x)

        out = _ResizeToLenLayer(target_len=seq_len)(h)
        self.model = Model([z_in, c_in], out, name="wgangp_gen_baseline")

    def call(self, inputs, training=False):
        z, c = inputs
        return self.model([z, c], training=training)


class _GenLSTM_CNN(Model):
    """CNN-based Generator with residual connections and improved conditioning"""

    def __init__(
        self,
        seq_len: int,
        num_features: int,
        num_classes: int,
        latent_dim: int = 64,
        base_filters: int = 128,
        kernel_size: int = 3,
    ):
        super().__init__(name="wgangp_gen_cnn")
        self.seq_len = seq_len
        self.num_features = num_features
        self.num_classes = num_classes
        self.latent_dim = latent_dim
        self.base_filters = base_filters
        self.kernel_size = kernel_size

        z_in = layers.Input(shape=(latent_dim,))
        c_in = layers.Input(shape=(num_classes,))

        # Project class embedding for stronger conditioning
        c_embed = layers.Dense(latent_dim // 2, use_bias=False)(c_in)
        c_embed = layers.LayerNormalization()(c_embed)

        # Initial projection with conditional input (concatenate z and c_embed)
        x = layers.Concatenate()([z_in, c_embed])
        x = layers.Dense(base_filters * 2)(x)
        x = layers.LayerNormalization()(x)
        x = layers.LeakyReLU(0.2)(x)

        # Reshape to sequence
        x = layers.Reshape((1, base_filters * 2))(x)

        # Expand to sequence length if needed (use Upsampling instead of Conv1DTranspose)
        if seq_len > 1:
            x = layers.UpSampling1D(size=seq_len)(x)

        # Concatenate labels directly (not RepeatVector)
        x_flat = layers.Flatten()(x)
        x = layers.Concatenate()([x_flat, c_in])
        x = layers.Dense(base_filters * seq_len)(x)
        x = layers.Reshape((seq_len, base_filters))(x)

        # CNN blocks with residual connections for stability
        # Block 1
        conv1 = layers.Conv1D(base_filters, kernel_size, padding="same")
        conv1_out = conv1(x)
        conv1_norm = layers.LayerNormalization()(conv1_out)
        conv1_act = layers.LeakyReLU(0.2)(conv1_norm)

        # Block 2 with residual
        conv2 = layers.Conv1D(base_filters, kernel_size, padding="same")
        conv2_out = conv2(conv1_act)
        conv2_norm = layers.LayerNormalization()(conv2_out)
        conv2_res = layers.Add()([conv1_act, conv2_norm])  # Residual connection
        conv2_act = layers.LeakyReLU(0.2)(conv2_res)

        # Block 3
        conv3 = layers.Conv1D(base_filters // 2, kernel_size, padding="same")
        conv3_out = conv3(conv2_act)
        conv3_norm = layers.LayerNormalization()(conv3_out)
        conv3_act = layers.LeakyReLU(0.2)(conv3_norm)

        # Final projection to features
        h = layers.Conv1D(
            num_features, kernel_size=kernel_size, padding="same", activation="tanh"
        )(conv3_act)

        out = _ResizeToLenLayer(target_len=seq_len)(h)
        self.model = Model([z_in, c_in], out, name="wgangp_gen_cnn")

    def call(self, inputs, training=False):
        z, c = inputs
        return self.model([z, c], training=training)


class _CriticBaseline(Model):
    """Simple baseline Critic architecture"""

    def __init__(
        self,
        seq_len: int,
        num_features: int,
        num_classes: int,
        base_filters: int = 64,
        kernel_size: int = 3,
    ):
        super().__init__(name="wgangp_critic_baseline")
        self.seq_len = seq_len
        self.num_features = num_features
        self.num_classes = num_classes
        self.base_filters = base_filters
        self.kernel_size = kernel_size

        xin = layers.Input(shape=(seq_len, num_features))
        cin = layers.Input(shape=(num_classes,))

        # Flatten input and concatenate with labels (not RepeatVector)
        x_flat = layers.Flatten()(xin)
        x = layers.Concatenate()([x_flat, cin])

        # Dense layers (critic must be 1-Lipschitz, no normalization)
        h = layers.Dense(base_filters * seq_len)(x)
        h = layers.LeakyReLU(0.2)(h)
        h = layers.Dense(base_filters * 2)(h)
        h = layers.LeakyReLU(0.2)(h)

        out = layers.Dense(1)(h)
        self.model = Model([xin, cin], out, name="wgangp_critic_baseline")

    def call(self, inputs, training=False):
        x, c = inputs
        return self.model([x, c], training=training)


class _Critic(Model):
    """Stable Critic with improved normalization"""

    def __init__(
        self,
        seq_len: int,
        num_features: int,
        num_classes: int,
        base_filters: int = 64,
        kernel_size: int = 3,
    ):
        super().__init__(name="wgangp_critic")
        self.seq_len = seq_len
        self.num_features = num_features
        self.num_classes = num_classes
        self.base_filters = base_filters
        self.kernel_size = kernel_size

        xin = layers.Input(shape=(seq_len, num_features))
        cin = layers.Input(shape=(num_classes,))

        # Flatten input and concatenate with labels (not RepeatVector)
        x_flat = layers.Flatten()(xin)
        x = layers.Concatenate()([x_flat, cin])

        # Dense layers (critic must be 1-Lipschitz, no normalization, no dropout)
        h = layers.Dense(base_filters * 4)(x)
        h = layers.LeakyReLU(0.2)(h)

        h = layers.Dense(base_filters * 2)(h)
        h = layers.LeakyReLU(0.2)(h)

        h = layers.Dense(base_filters)(h)
        h = layers.LeakyReLU(0.2)(h)

        out = layers.Dense(1)(h)
        self.model = Model([xin, cin], out, name="wgangp_critic")

    def call(self, inputs, training=False):
        x, c = inputs
        return self.model([x, c], training=training)


class _GenDCGAN(Model):
    """
    DCGAN-compliant Generator following Radford et al. 2015 guidelines:
    - Use strided convolutions (Conv1DTranspose) instead of pooling/upsampling
    - Use BatchNormalization (except output layer)
    - Use ReLU activation (except output uses tanh)
    - Remove fully connected hidden layers
    """

    def __init__(
        self,
        seq_len: int,
        num_features: int,
        num_classes: int,
        latent_dim: int = 64,
        base_filters: int = 128,
        kernel_size: int = 3,
    ):
        super().__init__(name="wgangp_gen_dcgan")
        self.seq_len = seq_len
        self.num_features = num_features
        self.num_classes = num_classes
        self.latent_dim = latent_dim
        self.base_filters = base_filters
        self.kernel_size = kernel_size

        z_in = layers.Input(shape=(latent_dim,))
        c_in = layers.Input(shape=(num_classes,))

        # Concatenate noise and class (DCGAN: no FC hidden layers, reshape directly)
        x = layers.Concatenate()([z_in, c_in])

        # Project to initial feature maps (DCGAN: reshape directly to conv feature maps)
        # Calculate initial sequence length (will be upsampled with strided convolutions)
        # For seq_len=1, we start with 1 and don't need upsampling
        initial_seq_len = 1
        x = layers.Dense(base_filters * 4 * initial_seq_len, use_bias=False)(x)
        x = layers.BatchNormalization()(x)
        x = layers.ReLU()(x)

        # Reshape to (batch, seq_len, filters) for Conv1D
        x = layers.Reshape((initial_seq_len, base_filters * 4))(x)

        # Concatenate labels directly (not RepeatVector)
        x_flat = layers.Flatten()(x)
        x = layers.Concatenate()([x_flat, c_in])
        x = layers.Dense(base_filters * 4 * seq_len)(x)
        x = layers.Reshape((seq_len, base_filters * 4))(x)

        # DCGAN: Use strided convolutions (Conv1DTranspose) for upsampling
        # Block 1: Upsample if needed (for seq_len > 1)
        if seq_len > 1:
            # Use Conv1DTranspose with stride to upsample
            # Calculate stride needed to reach seq_len
            stride = seq_len
            x = layers.Conv1DTranspose(
                base_filters * 2,
                kernel_size=kernel_size,
                strides=stride,
                padding="same",
                use_bias=False,
            )(x)
            x = layers.BatchNormalization()(x)
            x = layers.ReLU()(x)
        else:
            # No upsampling needed, just process
            x = layers.Conv1D(
                base_filters * 2, kernel_size=kernel_size, padding="same", use_bias=False
            )(x)
            x = layers.BatchNormalization()(x)
            x = layers.ReLU()(x)

        # Block 2: Process features
        x = layers.Conv1D(
            base_filters, kernel_size=kernel_size, padding="same", use_bias=False
        )(x)
        x = layers.BatchNormalization()(x)
        x = layers.ReLU()(x)

        # Block 3: Final processing
        x = layers.Conv1D(
            base_filters // 2, kernel_size=kernel_size, padding="same", use_bias=False
        )(x)
        x = layers.BatchNormalization()(x)
        x = layers.ReLU()(x)

        # Output layer: DCGAN uses tanh for bounded output, NO BatchNormalization
        h = layers.Conv1D(
            num_features, kernel_size=kernel_size, padding="same", activation="tanh"
        )(x)

        # Ensure output matches exact sequence length
        out = _ResizeToLenLayer(target_len=seq_len)(h)
        self.model = Model([z_in, c_in], out, name="wgangp_gen_dcgan")

    def call(self, inputs, training=False):
        z, c = inputs
        return self.model([z, c], training=training)


class _CriticDCGAN(Model):
    """
    DCGAN-compliant Critic following Radford et al. 2015 guidelines:
    - Use strided convolutions instead of pooling
    - Use BatchNormalization (except input layer)
    - Use LeakyReLU activation (slope ~0.2)
    - Remove fully connected hidden layers (except final output)
    """

    def __init__(
        self,
        seq_len: int,
        num_features: int,
        num_classes: int,
        base_filters: int = 64,
        kernel_size: int = 3,
    ):
        super().__init__(name="wgangp_critic_dcgan")
        self.seq_len = seq_len
        self.num_features = num_features
        self.num_classes = num_classes
        self.base_filters = base_filters
        self.kernel_size = kernel_size

        xin = layers.Input(shape=(seq_len, num_features))
        cin = layers.Input(shape=(num_classes,))

        # Flatten input and concatenate with labels (not RepeatVector)
        x_flat = layers.Flatten()(xin)
        x = layers.Concatenate()([x_flat, cin])

        # Dense layers (critic must be 1-Lipschitz, no normalization, no dropout)
        h = layers.Dense(base_filters * 4)(x)
        h = layers.LeakyReLU(0.2)(h)

        h = layers.Dense(base_filters * 2)(h)
        h = layers.LeakyReLU(0.2)(h)

        h = layers.Dense(base_filters)(h)
        h = layers.LeakyReLU(0.2)(h)

        # DCGAN: Final dense layer for output (no hidden FC layers)
        # But we need at least one dense layer for class conditioning
        # So we use a minimal dense layer
        out = layers.Dense(1)(h)
        self.model = Model([xin, cin], out, name="wgangp_critic_dcgan")

    def call(self, inputs, training=False):
        x, c = inputs
        return self.model([x, c], training=training)


class _GenMLP(Model):
    """MLP Generator using Dense layers instead of CNNs"""

    def __init__(
        self,
        seq_len: int,
        num_features: int,
        num_classes: int,
        latent_dim: int = 64,
        base_filters: int = 128,
        kernel_size: int = 3,  # Not used in MLP, kept for compatibility
    ):
        super().__init__(name="wgangp_gen_mlp")
        self.seq_len = seq_len
        self.num_features = num_features
        self.num_classes = num_classes
        self.latent_dim = latent_dim
        self.base_filters = base_filters

        z_in = layers.Input(shape=(latent_dim,))
        c_in = layers.Input(shape=(num_classes,))

        # Concatenate noise and class labels directly
        x = layers.Concatenate()([z_in, c_in])

        # MLP layers
        x = layers.Dense(base_filters * 4)(x)
        x = layers.BatchNormalization()(x)
        x = layers.ReLU()(x)

        x = layers.Dense(base_filters * 2)(x)
        x = layers.BatchNormalization()(x)
        x = layers.ReLU()(x)

        x = layers.Dense(base_filters)(x)
        x = layers.BatchNormalization()(x)
        x = layers.ReLU()(x)

        # Final projection to output size (num_features for seq_len=1)
        # For MLP, we work directly with 2D: (batch, num_features)
        x = layers.Dense(num_features)(x)
        x = layers.BatchNormalization()(x)
        x = layers.ReLU()(x)

        # Apply tanh activation
        h = layers.Activation("tanh")(x)

        # Reshape to (batch, seq_len, num_features) for compatibility
        # Since seq_len=1, this is just adding a dimension: (batch, 1, num_features)
        out = layers.Reshape((seq_len, num_features))(h)
        self.model = Model([z_in, c_in], out, name="wgangp_gen_mlp")

    def call(self, inputs, training=False):
        z, c = inputs
        return self.model([z, c], training=training)


class _CriticMLP(Model):
    """MLP Critic using Dense layers instead of CNNs (1-Lipschitz, no normalization, no dropout)"""

    def __init__(
        self,
        seq_len: int,
        num_features: int,
        num_classes: int,
        base_filters: int = 64,
        kernel_size: int = 3,  # Not used in MLP, kept for compatibility
    ):
        super().__init__(name="wgangp_critic_mlp")
        self.seq_len = seq_len
        self.num_features = num_features
        self.num_classes = num_classes
        self.base_filters = base_filters

        xin = layers.Input(shape=(seq_len, num_features))
        cin = layers.Input(shape=(num_classes,))

        # For MLP with seq_len=1, flatten immediately and work with 2D
        # Input is (batch, 1, features), flatten to (batch, features)
        x_flat = layers.Flatten()(xin)
        x = layers.Concatenate()([x_flat, cin])

        # MLP layers (critic must be 1-Lipschitz, no normalization, no dropout)
        h = layers.Dense(base_filters * 4)(x)
        h = layers.LeakyReLU(0.2)(h)

        h = layers.Dense(base_filters * 2)(h)
        h = layers.LeakyReLU(0.2)(h)

        h = layers.Dense(base_filters)(h)
        h = layers.LeakyReLU(0.2)(h)

        # Final output layer
        out = layers.Dense(1)(h)
        self.model = Model([xin, cin], out, name="wgangp_critic_mlp")

    def call(self, inputs, training=False):
        x, c = inputs
        return self.model([x, c], training=training)


def create_wgan_models(
    architecture: WGANArchitecture | str,
    seq_len: int,
    num_features: int,
    num_classes: int,
    latent_dim: int = 64,
    base_filters: int = 128,
    gen_kernel_size: int = 3,
    critic_base_filters: int = 64,
    critic_kernel_size: int = 3,
):
    """
    Factory function to create generator and critic models based on architecture type.

    Args:
        architecture: Architecture type (WGANArchitecture enum or string)
        seq_len: Sequence length
        num_features: Number of features
        num_classes: Number of classes
        latent_dim: Latent dimension for generator
        base_filters: Base filters for generator
        gen_kernel_size: Kernel size for generator
        critic_base_filters: Base filters for critic
        critic_kernel_size: Kernel size for critic

    Returns:
        Tuple of (generator, critic) models
    """
    # Convert string to enum if needed
    if isinstance(architecture, str):
        architecture = WGANArchitecture(architecture.lower())

    if architecture == WGANArchitecture.BASELINE:
        generator = _GenBaseline(
            seq_len=seq_len,
            num_features=num_features,
            num_classes=num_classes,
            latent_dim=latent_dim,
            base_filters=base_filters,
            kernel_size=gen_kernel_size,
        )
        critic = _CriticBaseline(
            seq_len=seq_len,
            num_features=num_features,
            num_classes=num_classes,
            base_filters=critic_base_filters,
            kernel_size=critic_kernel_size,
        )
    elif architecture == WGANArchitecture.CNN_RESIDUAL:
        generator = _GenLSTM_CNN(
            seq_len=seq_len,
            num_features=num_features,
            num_classes=num_classes,
            latent_dim=latent_dim,
            base_filters=base_filters,
            kernel_size=gen_kernel_size,
        )
        critic = _Critic(
            seq_len=seq_len,
            num_features=num_features,
            num_classes=num_classes,
            base_filters=critic_base_filters,
            kernel_size=critic_kernel_size,
        )
    elif architecture == WGANArchitecture.DCGAN:
        generator = _GenDCGAN(
            seq_len=seq_len,
            num_features=num_features,
            num_classes=num_classes,
            latent_dim=latent_dim,
            base_filters=base_filters,
            kernel_size=gen_kernel_size,
        )
        critic = _CriticDCGAN(
            seq_len=seq_len,
            num_features=num_features,
            num_classes=num_classes,
            base_filters=critic_base_filters,
            kernel_size=critic_kernel_size,
        )
    elif architecture == WGANArchitecture.MLP:
        generator = _GenMLP(
            seq_len=seq_len,
            num_features=num_features,
            num_classes=num_classes,
            latent_dim=latent_dim,
            base_filters=base_filters,
            kernel_size=gen_kernel_size,
        )
        critic = _CriticMLP(
            seq_len=seq_len,
            num_features=num_features,
            num_classes=num_classes,
            base_filters=critic_base_filters,
            kernel_size=critic_kernel_size,
        )
    else:
        raise ValueError(f"Unknown architecture: {architecture}")

    return generator, critic


class WGAN_GP(Model):
    def __init__(
        self,
        seq_len: int,
        num_features: int,
        num_classes: int,
        latent_dim: int = 64,
        gp_weight: float = 2.0,  # Reduced from 10.0 to prevent gradient penalty from dominating
        n_critic: int = 5,  # Reduced back to 5 for stability
        base_filters: int = 128,
        gen_kernel_size: int = 3,
        architecture: WGANArchitecture | str = WGANArchitecture.CNN_RESIDUAL,
        critic_base_filters: int = 64,
        critic_kernel_size: int = 3,
        class_weights: dict[int, float] | None = None,
        use_feature_matching: bool = False,
        feature_matching_weight: float = 0.1,
    ):
        super().__init__(name="wgan_gp")
        self.seq_len = seq_len
        self.num_features = num_features
        self.num_classes = num_classes
        self.latent_dim = latent_dim
        self.gp_weight = gp_weight
        self.n_critic = n_critic
        self.base_filters = base_filters
        self.architecture = architecture
        self.class_weights = class_weights or {}
        self.use_feature_matching = use_feature_matching
        self.feature_matching_weight = feature_matching_weight

        # Create generator and critic using factory
        self.gen, self.critic = create_wgan_models(
            architecture=architecture,
            seq_len=seq_len,
            num_features=num_features,
            num_classes=num_classes,
            latent_dim=latent_dim,
            base_filters=base_filters,
            gen_kernel_size=gen_kernel_size,
            critic_base_filters=critic_base_filters,
            critic_kernel_size=critic_kernel_size,
        )

        self.gen_loss_tracker = tf.keras.metrics.Mean(name="g_loss")
        self.crit_loss_tracker = tf.keras.metrics.Mean(name="d_loss")
        self.mean_real_tracker = tf.keras.metrics.Mean(name="d_real")
        self.mean_fake_tracker = tf.keras.metrics.Mean(name="d_fake")
        self.gp_tracker = tf.keras.metrics.Mean(name="gp")
        # Combined score for checkpointing: lower is better (g_loss + abs(d_loss))
        self.combined_score_tracker = tf.keras.metrics.Mean(name="combined_score")

        # Post-process params filled later
        self.feature_mean = None
        self.feature_std = None
        self.feature_min = None
        self.feature_max = None

    @property
    def metrics(self):
        return [
            self.gen_loss_tracker,
            self.crit_loss_tracker,
            self.mean_real_tracker,
            self.mean_fake_tracker,
            self.gp_tracker,
            self.combined_score_tracker,
        ]

    def compile(self, g_optimizer, d_optimizer):
        super().compile()
        self.g_optimizer = g_optimizer
        self.d_optimizer = d_optimizer

    def _postprocess(self, x):
        # Generator already outputs tanh, so we just need to rescale (no need to apply tanh again)
        # x is already in [-1, 1] range from generator's tanh activation
        if self.feature_mean is not None and self.feature_std is not None:
            mean = tf.reshape(tf.cast(self.feature_mean, x.dtype), (1, 1, -1))
            std = tf.reshape(tf.cast(self.feature_std, x.dtype), (1, 1, -1))
            x = x * std + mean
        if self.feature_min is not None and self.feature_max is not None:
            fmin = tf.reshape(tf.cast(self.feature_min, x.dtype), (1, 1, -1))
            fmax = tf.reshape(tf.cast(self.feature_max, x.dtype), (1, 1, -1))
            x = tf.clip_by_value(x, fmin, fmax)
        x = tf.where(tf.math.is_finite(x), x, tf.zeros_like(x))
        return x

    def gradient_penalty(self, real, fake, c):
        batch_size = tf.shape(real)[0]
        # Ensure consistent dtype
        real_dtype = real.dtype
        fake = tf.cast(fake, real_dtype)
        alpha = tf.random.uniform((batch_size, 1, 1), 0.0, 1.0, dtype=real_dtype)
        interpolated = real + alpha * (fake - real)
        with tf.GradientTape() as gp_tape:
            gp_tape.watch(interpolated)
            pred = self.critic([interpolated, c], training=True)
        grads = gp_tape.gradient(pred, [interpolated])[0]
        # Guard against NaN gradients
        grads = tf.where(tf.math.is_finite(grads), grads, tf.zeros_like(grads))
        grads = tf.reshape(grads, (batch_size, -1))
        norm = tf.sqrt(tf.reduce_sum(tf.square(grads), axis=1) + 1e-8)
        gp = tf.reduce_mean((norm - 1.0) ** 2)
        # Guard against NaN in gradient penalty
        gp = tf.where(tf.math.is_finite(gp), gp, tf.zeros_like(gp))
        return gp

    def _get_class_weights_tensor(self, c: tf.Tensor) -> tf.Tensor:
        """Get class weights for each sample in the batch"""
        if not self.class_weights:
            return tf.ones(tf.shape(c)[0], dtype=c.dtype)

        # Get class indices (argmax of one-hot)
        class_indices = tf.argmax(c, axis=1)
        # Map to weights
        weights = tf.zeros(tf.shape(c)[0], dtype=c.dtype)
        for class_idx, weight in self.class_weights.items():
            mask = tf.equal(class_indices, class_idx)
            weights = tf.where(mask, tf.cast(weight, c.dtype), weights)
        return weights

    def train_step(self, data):
        real_x, c = data
        batch_size = tf.shape(real_x)[0]

        tf.debugging.assert_equal(
            tf.shape(real_x)[1],
            self.seq_len,
            message="real_x sequence length mismatch",
        )

        # Get class weights if provided
        class_weights = self._get_class_weights_tensor(c)

        # Train critic n_critic steps
        crit_loss = 0.0
        d_real_acc = 0.0
        d_fake_acc = 0.0
        gp_acc = 0.0
        for _ in range(self.n_critic):
            z = tf.random.normal((batch_size, self.latent_dim), dtype=real_x.dtype)
            with tf.GradientTape() as tape:
                fake_x = self.gen([z, c], training=True)
                fake_x = self._postprocess(fake_x)
                real_scores = self.critic([real_x, c], training=True)
                fake_scores = self.critic([fake_x, c], training=True)

                # Apply class weights to scores
                if self.class_weights:
                    # Cast class_weights to match scores dtype (may be float16 with mixed precision)
                    class_weights_cast = tf.cast(class_weights, real_scores.dtype)
                    # Guard against NaN/Inf in weights
                    class_weights_cast = tf.where(
                        tf.math.is_finite(class_weights_cast),
                        class_weights_cast,
                        tf.ones_like(class_weights_cast),
                    )
                    real_scores_weighted = real_scores * tf.expand_dims(
                        class_weights_cast, -1
                    )
                    fake_scores_weighted = fake_scores * tf.expand_dims(
                        class_weights_cast, -1
                    )
                    # Guard against NaN in weighted scores
                    real_scores_weighted = tf.where(
                        tf.math.is_finite(real_scores_weighted),
                        real_scores_weighted,
                        real_scores,
                    )
                    fake_scores_weighted = tf.where(
                        tf.math.is_finite(fake_scores_weighted),
                        fake_scores_weighted,
                        fake_scores,
                    )
                    mean_fake = tf.reduce_mean(fake_scores_weighted)
                    mean_real = tf.reduce_mean(real_scores_weighted)
                else:
                    mean_fake = tf.reduce_mean(fake_scores)
                    mean_real = tf.reduce_mean(real_scores)

                wasserstein = mean_fake - mean_real
                # Guard against NaN in Wasserstein distance
                wasserstein = tf.where(
                    tf.math.is_finite(wasserstein),
                    wasserstein,
                    tf.zeros_like(wasserstein),
                )
                gp = self.gradient_penalty(real_x, fake_x, c)
                # Ensure consistent dtype for loss computation
                wasserstein_dtype = wasserstein.dtype
                gp = tf.cast(gp, wasserstein_dtype)
                gp_weight = tf.cast(self.gp_weight, wasserstein_dtype)
                d_loss = wasserstein + gp_weight * gp
                # Guard against NaN in critic loss
                d_loss = tf.where(
                    tf.math.is_finite(d_loss), d_loss, tf.zeros_like(d_loss)
                )
            grads = tape.gradient(d_loss, self.critic.trainable_variables)
            # Clip gradients to prevent explosion
            # tf.clip_by_norm handles NaN/Inf gracefully, so we can just clip
            grads = [tf.clip_by_norm(g, 1.0) if g is not None else None for g in grads]
            # Filter out None gradients and apply
            valid_grads = [
                (g, v)
                for g, v in zip(grads, self.critic.trainable_variables)
                if g is not None
            ]
            if valid_grads:
                self.d_optimizer.apply_gradients(valid_grads)
            crit_loss += d_loss
            d_real_acc += mean_real
            d_fake_acc += mean_fake
            gp_acc += gp
        n_c = float(self.n_critic)
        crit_loss /= n_c
        d_real_mean = d_real_acc / n_c
        d_fake_mean = d_fake_acc / n_c
        gp_mean = gp_acc / n_c

        # Train generator
        z = tf.random.normal((batch_size, self.latent_dim), dtype=real_x.dtype)
        with tf.GradientTape() as tape:
            fake_x = self.gen([z, c], training=True)
            fake_x = self._postprocess(fake_x)
            fake_scores = self.critic([fake_x, c], training=True)

            # Clip scores to reasonable range to prevent NaN/Inf while preserving gradients
            fake_scores = tf.clip_by_value(fake_scores, -1e6, 1e6)

            # Apply class weights if provided
            if self.class_weights:
                # Cast class_weights to match scores dtype (may be float16 with mixed precision)
                class_weights_cast = tf.cast(class_weights, fake_scores.dtype)
                # Guard against NaN/Inf in weights
                class_weights_cast = tf.where(
                    tf.math.is_finite(class_weights_cast),
                    class_weights_cast,
                    tf.ones_like(class_weights_cast),
                )
                fake_scores_weighted = fake_scores * tf.expand_dims(
                    class_weights_cast, -1
                )
                # Guard against NaN in weighted scores
                fake_scores_weighted = tf.where(
                    tf.math.is_finite(fake_scores_weighted),
                    fake_scores_weighted,
                    fake_scores,
                )
                g_loss_adv = -tf.reduce_mean(fake_scores_weighted)
                # Guard against NaN in loss
                g_loss_adv = tf.where(
                    tf.math.is_finite(g_loss_adv),
                    g_loss_adv,
                    -tf.reduce_mean(fake_scores),
                )
            else:
                g_loss_adv = -tf.reduce_mean(fake_scores)

            g_loss = g_loss_adv

            # Add feature matching loss if enabled
            if self.use_feature_matching:
                # Get real and fake features from critic
                # For feature matching, we compare intermediate representations
                # Simplified: compare mean/std of features
                real_features = tf.reduce_mean(real_x, axis=1)  # (batch, features)
                fake_features = tf.reduce_mean(fake_x, axis=1)  # (batch, features)
                # Ensure dtype consistency
                fake_features = tf.cast(fake_features, real_features.dtype)
                fm_loss = tf.reduce_mean(tf.square(real_features - fake_features))
                # Guard against NaN in feature matching loss
                fm_loss = tf.where(
                    tf.math.is_finite(fm_loss), fm_loss, tf.zeros_like(fm_loss)
                )
                fm_weight = tf.cast(self.feature_matching_weight, g_loss_adv.dtype)
                g_loss = g_loss_adv + fm_weight * fm_loss
                # Guard against NaN in combined loss
                g_loss = tf.where(tf.math.is_finite(g_loss), g_loss, g_loss_adv)

            # Final guard: if loss is still NaN, use unweighted loss
            g_loss = tf.where(
                tf.math.is_finite(g_loss), g_loss, -tf.reduce_mean(fake_scores)
            )

        grads = tape.gradient(g_loss, self.gen.trainable_variables)
        # Ensure we have valid gradients
        if grads is None:
            # Model might not be built - skip this update
            g_loss = tf.constant(0.0, dtype=g_loss.dtype)
        elif all(g is None for g in grads):
            # All gradients are None - skip this update
            g_loss = tf.constant(0.0, dtype=g_loss.dtype)
        else:
            # Clip gradients to prevent explosion
            # Use tighter clipping when class weights are used to prevent instability
            clip_norm = 0.5 if self.class_weights else 1.0
            # tf.clip_by_norm handles NaN/Inf gracefully, so we can just clip
            grads = [
                tf.clip_by_norm(g, clip_norm) if g is not None else None for g in grads
            ]
            # Filter out None gradients and apply
            valid_grads = [
                (g, v)
                for g, v in zip(grads, self.gen.trainable_variables)
                if g is not None
            ]
            if valid_grads:
                self.g_optimizer.apply_gradients(valid_grads)

        # Update trackers (Mean metrics handle NaN gracefully by ignoring them)
        self.gen_loss_tracker.update_state(g_loss)
        self.crit_loss_tracker.update_state(crit_loss)
        self.mean_real_tracker.update_state(d_real_mean)
        self.mean_fake_tracker.update_state(d_fake_mean)
        self.gp_tracker.update_state(gp_mean)

        # Update combined score: g_loss + abs(d_loss) - lower is better
        combined_score = g_loss + tf.abs(crit_loss)
        self.combined_score_tracker.update_state(combined_score)

        # Return safe values (use 0.0 if NaN)
        g_loss_safe = tf.where(
            tf.math.is_finite(g_loss), g_loss, tf.constant(0.0, dtype=g_loss.dtype)
        )
        crit_loss_safe = tf.where(
            tf.math.is_finite(crit_loss),
            crit_loss,
            tf.constant(0.0, dtype=crit_loss.dtype),
        )
        # Return combined_score so ModelCheckpoint can monitor it
        combined_score_safe = tf.where(
            tf.math.is_finite(combined_score),
            combined_score,
            tf.constant(float("inf"), dtype=combined_score.dtype),
        )
        return {
            "g_loss": g_loss_safe,
            "d_loss": crit_loss_safe,
            "combined_score": combined_score_safe,
        }

    def generate(self, n: int, one_hot: np.ndarray) -> np.ndarray:
        z = tf.random.normal((n, self.latent_dim))
        c = tf.convert_to_tensor(one_hot, dtype=tf.float32)
        x = self.gen([z, c], training=False)
        x = self._postprocess(x)
        return x.numpy()


def _save_wgan_model(
    gan: WGAN_GP, metadata: dict, save_path: str, verbose: bool = False
):
    """Save WGAN-GP model and metadata"""
    os.makedirs(save_path, exist_ok=True)
    model_path = os.path.join(save_path, "wgangp_model.weights.h5")
    meta_path = os.path.join(save_path, "wgangp_metadata.pkl")
    # Build sub-models and mark parent as built
    dummy_z = tf.zeros((1, gan.latent_dim))
    dummy_c = tf.zeros((1, gan.num_classes))
    dummy_x = tf.zeros((1, gan.seq_len, gan.num_features))
    _ = gan.gen([dummy_z, dummy_c], training=False)
    _ = gan.critic([dummy_x, dummy_c], training=False)
    gan.built = True  # Explicitly mark parent model as built
    gan.save_weights(model_path)
    with open(meta_path, "wb") as f:
        pickle.dump(metadata, f)
    if verbose:
        print(f"    WGAN-GP model saved to {save_path}")


def _load_wgan_model(
    save_path: str, verbose: bool = False
) -> tuple[WGAN_GP | None, dict | None]:
    """Load WGAN-GP model and metadata if they exist"""
    model_path = os.path.join(save_path, "wgangp_model.weights.h5")
    meta_path = os.path.join(save_path, "wgangp_metadata.pkl")
    if not (os.path.exists(model_path) and os.path.exists(meta_path)):
        return None, None
    try:
        with open(meta_path, "rb") as f:
            metadata = pickle.load(f)
        # Get architecture from metadata, default to CNN_RESIDUAL for backward compatibility
        saved_arch = metadata.get("architecture", WGANArchitecture.CNN_RESIDUAL.value)
        gan = WGAN_GP(
            seq_len=metadata["seq_len"],
            num_features=metadata["num_features"],
            num_classes=metadata["num_classes"],
            latent_dim=metadata.get("latent_dim", 64),
            gp_weight=metadata.get("gp_weight", 5.0),  # Default reduced from 10.0
            n_critic=metadata.get("n_critic", 5),
            base_filters=metadata.get("base_filters", 128),
            gen_kernel_size=metadata.get("gen_kernel_size", 3),
            architecture=saved_arch,
        )
        gan.feature_mean = tf.convert_to_tensor(
            metadata.get("feature_mean"), dtype=tf.float32
        )
        gan.feature_std = tf.convert_to_tensor(
            metadata.get("feature_std"), dtype=tf.float32
        )
        gan.feature_min = tf.convert_to_tensor(
            metadata.get("feature_min"), dtype=tf.float32
        )
        gan.feature_max = tf.convert_to_tensor(
            metadata.get("feature_max"), dtype=tf.float32
        )
        gan.compile(
            # Same learning rate for stability, with gradient clipping
            g_optimizer=tf.keras.optimizers.Adam(
                learning_rate=1e-4, beta_1=0.5, beta_2=0.9, clipnorm=1.0
            ),
            d_optimizer=tf.keras.optimizers.Adam(
                learning_rate=1e-4, beta_1=0.5, beta_2=0.9, clipnorm=1.0
            ),
        )
        # Ensure model is built before loading weights by calling sub-models
        dummy_z = tf.zeros((1, gan.latent_dim))
        dummy_c = tf.zeros((1, gan.num_classes))
        dummy_x = tf.zeros((1, gan.seq_len, gan.num_features))
        # Build generator and critic
        _ = gan.gen([dummy_z, dummy_c], training=False)
        _ = gan.critic([dummy_x, dummy_c], training=False)
        gan.built = True  # Explicitly mark parent model as built
        gan.load_weights(model_path)
        if verbose:
            print(f"    WGAN-GP model loaded from {save_path}")
        return gan, metadata
    except Exception as e:
        if verbose:
            print(f"    Failed to load WGAN-GP model: {e}")
        return None, None


def _one_hot_encode(labels: np.ndarray) -> np.ndarray:
    labels = np.asarray(labels)
    if labels.ndim == 1:
        num_classes = int(labels.max()) + 1 if labels.size else 1
        eye = np.eye(num_classes, dtype=np.float32)
        return eye[labels.astype(int)]
    return labels.astype(np.float32)


def balance_with_wgan_gp(
    train_data: np.ndarray,
    train_labels: np.ndarray,
    epochs: int = 100,
    batch_size: int = 256,
    max_target: int | None = None,
    augmentation_ratio: float = 1.5,  # Limit augmentation to ratio * each class size (prevents overfitting)
    augmentation_target_ratio: float | None = None,  # If set, augment minority classes to this ratio of majority
    # (e.g., 0.5 = 50% of majority)
    verbose: bool = True,
    save_path: str | None = None,
    assess_quality: bool = True,
    n_critic: int = 5,
    seq_len: int | None = None,
    architecture: WGANArchitecture | str = WGANArchitecture.CNN_RESIDUAL,
    use_class_weights: bool = True,
    class_weight_power: float = 0.5,  # Power for inverse frequency weighting (0.5 = sqrt)
    class_weight_max: float = 2.0,  # Maximum class weight to prevent instability (reduced from 2.5)
    use_feature_matching: bool = False,
    feature_matching_weight: float = 0.1,
    noise_std: float = 0.02,  # Standard deviation of Gaussian noise to add to generated samples (reduces overfitting)
):
    _ = seq_len  # retained for backward-compatible signature

    # Ensure data is 2D (batch, features)
    if train_data.ndim != 2:
        raise ValueError("balance_with_wgan_gp expects 2D train_data (batch, features)")

    # Convert labels to one-hot if necessary
    labels_f32 = _one_hot_encode(train_labels)
    if labels_f32.ndim != 2:
        raise ValueError("train_labels must be 1D (class indices) or 2D one-hot array")

    num_classes = labels_f32.shape[1]
    num_features = train_data.shape[1]

    # Convert architecture to enum for checking
    if isinstance(architecture, str):
        arch_enum = WGANArchitecture(architecture.lower())
    else:
        arch_enum = architecture

    # For MLP, we can work with 2D data directly (no sequence dimension needed)
    # For other architectures, convert to 3D (batch, seq_len=1, features)
    if arch_enum == WGANArchitecture.MLP:
        # MLP works with 2D: (batch, features) - no reshape needed
        seq_len_value = 1  # Still track for compatibility, but models handle 2D
        # Keep data as 2D, models will handle the interface
        train_data_3d = train_data[:, np.newaxis, :]  # For compatibility with WGAN_GP interface
    else:
        # Other architectures need 3D tensors
        train_data_3d = train_data[:, np.newaxis, :]
        seq_len_value = 1
    train_data = train_data_3d

    gan = None
    loaded_meta = None
    model_was_loaded = False
    if save_path:
        gan, loaded_meta = _load_wgan_model(save_path, verbose)
        if gan is not None:
            saved_seq_len = loaded_meta.get("seq_len", 1)
            saved_features = loaded_meta.get("num_features")
            saved_classes = loaded_meta.get("num_classes")
            saved_arch = loaded_meta.get(
                "architecture", WGANArchitecture.CNN_RESIDUAL.value
            )
            # Convert architecture to string for comparison
            arch_str = (
                architecture.value
                if isinstance(architecture, WGANArchitecture)
                else str(architecture).lower()
            )
            # Check architecture compatibility first (before trying to load weights)
            arch_compatible = saved_arch.lower() == arch_str.lower()
            if not arch_compatible:
                raise ValueError(
                    f"Architecture mismatch: saved model uses '{saved_arch}' but requested '{arch_str}'. "
                    f"Please delete the saved model at {save_path} or use the same architecture."
                )
            compatible = (
                saved_seq_len == seq_len_value
                and saved_features == num_features
                and saved_classes == num_classes
            )
            if compatible:
                if saved_seq_len != seq_len_value:
                    if saved_seq_len > 1:
                        train_data = np.repeat(train_data, saved_seq_len, axis=1)
                    else:
                        train_data = train_data
                    seq_len_value = saved_seq_len
                model_was_loaded = True
            else:
                if verbose:
                    print(
                        "    Saved model incompatible with current data; retraining from scratch"
                    )
                gan = None
                train_data = train_data
                seq_len_value = 1

    n = train_data.shape[0]
    data_f32 = train_data.astype("float32")

    if verbose:
        print(
            f"    GAN dataset shapes → data: {data_f32.shape}, labels: {labels_f32.shape}"
        )

    # Create class-balanced dataset for better minority class learning
    # Split data by class
    class_indices = {}
    for c in range(num_classes):
        class_indices[c] = np.where(np.argmax(labels_f32, axis=1) == c)[0]

    # Find minimum class size for balanced sampling
    min_class_size = (
        min(len(indices) for indices in class_indices.values()) if class_indices else n
    )
    samples_per_class_per_batch = max(1, batch_size // num_classes)

    # Calculate batches per epoch: ensure we cover all data at least once
    # Use the minimum needed to cover smallest class, but also ensure reasonable coverage
    batches_per_epoch = max(
        (
            max(1, min_class_size // samples_per_class_per_batch)
            if min_class_size > 0
            else 1
        ),
        n // batch_size,  # At least cover the dataset once
    )

    def balanced_batch_generator():
        """Generate balanced batches with equal representation from each class"""
        # Create shuffled indices for each class
        shuffled_indices = {
            c: np.random.permutation(indices) for c, indices in class_indices.items()
        }
        class_positions = {c: 0 for c in range(num_classes)}

        # Generate infinite batches (steps_per_epoch will limit each epoch)
        while True:
            batch_x = []
            batch_y = []

            # Sample equally from each class
            for c in range(num_classes):
                pos = class_positions[c]
                indices = shuffled_indices[c]
                # Wrap around if we've exhausted this class
                if pos >= len(indices):
                    shuffled_indices[c] = np.random.permutation(class_indices[c])
                    pos = 0
                    class_positions[c] = 0

                # Take samples_per_class_per_batch samples from this class
                end_pos = min(pos + samples_per_class_per_batch, len(indices))
                batch_indices = indices[pos:end_pos]

                # If we need more samples, wrap around
                if len(batch_indices) < samples_per_class_per_batch:
                    remaining = samples_per_class_per_batch - len(batch_indices)
                    batch_indices = np.concatenate(
                        [batch_indices, shuffled_indices[c][:remaining]]
                    )
                    shuffled_indices[c] = np.random.permutation(class_indices[c])
                    class_positions[c] = remaining
                else:
                    class_positions[c] = end_pos

                batch_x.append(data_f32[batch_indices])
                batch_y.append(labels_f32[batch_indices])

            # Combine and shuffle within batch
            batch_x = np.concatenate(batch_x, axis=0)
            batch_y = np.concatenate(batch_y, axis=0)
            perm = np.random.permutation(len(batch_x))
            batch_x = batch_x[perm]
            batch_y = batch_y[perm]

            yield (batch_x, batch_y)

    # Use balanced generator if classes are imbalanced, otherwise use standard dataset
    class_counts = [len(class_indices[c]) for c in range(num_classes)]
    is_imbalanced = (
        max(class_counts) / min(class_counts) > 1.5 if min(class_counts) > 0 else False
    )

    # Calculate steps_per_epoch for progress tracking
    steps_per_epoch = None

    if is_imbalanced and not model_was_loaded:
        # Use balanced batches for training
        ds = tf.data.Dataset.from_generator(
            balanced_batch_generator,
            output_signature=(
                tf.TensorSpec(
                    shape=(None, seq_len_value, num_features), dtype=tf.float32
                ),
                tf.TensorSpec(shape=(None, num_classes), dtype=tf.float32),
            ),
        ).prefetch(tf.data.AUTOTUNE)
        steps_per_epoch = batches_per_epoch
        if verbose:
            print(
                f"    Using class-balanced batches "
                f"(samples per class per batch: {samples_per_class_per_batch}, "
                f"batches per epoch: {batches_per_epoch})"
            )
    else:
        # Use standard shuffled dataset
        ds = (
            tf.data.Dataset.from_tensor_slices((data_f32, labels_f32))
            .shuffle(min(8192, n))
            .batch(batch_size)
            .prefetch(tf.data.AUTOTUNE)
        )
        steps_per_epoch = (n + batch_size - 1) // batch_size  # Ceiling division

    # Stats for postprocess
    feat_mean = data_f32.mean(axis=(0, 1)).astype("float32")
    feat_std = data_f32.std(axis=(0, 1)).astype("float32") + 1e-8
    feat_min = data_f32.min(axis=(0, 1)).astype("float32")
    feat_max = data_f32.max(axis=(0, 1)).astype("float32")

    if verbose:
        # Debug: show range of mean, std to understand rescaling
        print(
            f"    Data stats for rescaling: mean range [{feat_mean.min():.3f}, {feat_mean.max():.3f}], "
            f"std range [{feat_std.min():.3f}, {feat_std.max():.3f}], "
            f"value range [{feat_min.min():.3f}, {feat_max.max():.3f}]"
        )

    # Calculate class weights if requested
    class_weights_dict = None
    if use_class_weights:
        # Calculate class frequencies
        class_idx = np.argmax(labels_f32, axis=1)
        unique, counts = np.unique(class_idx, return_counts=True)
        max_count = float(np.max(counts))
        # Calculate weights: weight = (max_count / count) ** power
        # This gives higher weights to minority classes
        # Cap weights to prevent instability
        class_weights_dict = {
            int(c): min(
                float((max_count / count) ** class_weight_power), class_weight_max
            )
            for c, count in zip(unique, counts)
        }
        if verbose:
            print(
                f"    Class weights (capped at {class_weight_max}): {class_weights_dict}"
            )

    # Try to load existing model if save_path provided
    if gan is None:
        gan = WGAN_GP(
            seq_len_value,
            num_features,
            num_classes,
            latent_dim=64,
            gp_weight=2.0,  # Reduced from 10.0 to prevent gradient penalty from dominating
            n_critic=n_critic,
            architecture=architecture,
            class_weights=class_weights_dict,
            use_feature_matching=use_feature_matching,
            feature_matching_weight=feature_matching_weight,
        )
        gan.feature_mean = tf.convert_to_tensor(feat_mean)
        gan.feature_std = tf.convert_to_tensor(feat_std)
        gan.feature_min = tf.convert_to_tensor(feat_min)
        gan.feature_max = tf.convert_to_tensor(feat_max)

        gan.compile(
            # Same learning rate for stability, with gradient clipping
            g_optimizer=tf.keras.optimizers.Adam(
                learning_rate=1e-4, beta_1=0.5, beta_2=0.9, clipnorm=1.0
            ),
            d_optimizer=tf.keras.optimizers.Adam(
                learning_rate=1e-4, beta_1=0.5, beta_2=0.9, clipnorm=1.0
            ),
        )
        # Build models explicitly by calling with dummy inputs
        dummy_z = tf.zeros((1, gan.latent_dim))
        dummy_c = tf.zeros((1, gan.num_classes))
        dummy_x = tf.zeros((1, gan.seq_len, gan.num_features))
        _ = gan.gen([dummy_z, dummy_c], training=False)
        _ = gan.critic([dummy_x, dummy_c], training=False)
        # Ensure models are marked as built
        gan.gen.built = True
        gan.critic.built = True
        gan.built = True  # Mark parent model as built
        if verbose:
            print("    GAN generator summary:")
            # Show summary of internal Functional model, not the wrapper
            if hasattr(gan.gen, "model"):
                gan.gen.model.summary(print_fn=lambda line: print("        " + line))
            else:
                gan.gen.summary(print_fn=lambda line: print("        " + line))
            print("    GAN critic summary:")
            if hasattr(gan.critic, "model"):
                gan.critic.model.summary(print_fn=lambda line: print("        " + line))
            else:
                gan.critic.summary(print_fn=lambda line: print("        " + line))

    # Train only if model wasn't loaded
    if not model_was_loaded:
        # Ensure models are fully built before training by running one training step
        try:
            sample_batch = next(iter(ds.take(1)))
            sample_x, sample_c = sample_batch
            # Run a training step to fully build the model and initialize metrics
            gan.train_step((sample_x, sample_c))
            # Mark parent model as built
            gan.built = True
            # Ensure sub-models are also marked as built
            if hasattr(gan, 'gen'):
                gan.gen.built = True
            if hasattr(gan, 'critic'):
                gan.critic.built = True
        except Exception:
            # If training step fails, just ensure models are marked as built
            gan.built = True
            if hasattr(gan, 'gen'):
                gan.gen.built = True
            if hasattr(gan, 'critic'):
                gan.critic.built = True
        fit_kwargs = {"epochs": max(1, epochs), "verbose": 1 if verbose else 0}
        if steps_per_epoch is not None:
            fit_kwargs["steps_per_epoch"] = steps_per_epoch

        # Create checkpoint directory for best model
        checkpoint_dir = None
        checkpoint_path = None
        if save_path:
            checkpoint_dir = os.path.join(save_path, "checkpoints")
            os.makedirs(checkpoint_dir, exist_ok=True)
            checkpoint_path = os.path.join(checkpoint_dir, "best_model.weights.h5")

        # Add callback to detect GAN collapse
        collapse_callback = GANCollapseCallback(
            patience=3,  # Stop after 3 consecutive epochs of collapse
            critic_loss_threshold=3.0,  # Positive critic loss threshold (indicates gradient penalty explosion)
            gen_loss_threshold=0.01,  # Generator loss too close to zero
            verbose=1 if verbose else 0,
        )

        callbacks = [collapse_callback]

        # Add ModelCheckpoint to save best model based on combined_score
        # Use a custom callback wrapper to ensure model is built and metric exists
        if checkpoint_path:
            # Create a wrapper callback that only saves after first epoch
            class SafeModelCheckpoint(tf.keras.callbacks.ModelCheckpoint):
                def __init__(self, *args, **kwargs):
                    super().__init__(*args, **kwargs)
                    self._first_epoch = True
                    self._has_combined_score = False

                def on_epoch_end(self, epoch, logs=None):
                    logs = logs or {}
                    # Only save if combined_score is available (after first epoch)
                    if "combined_score" in logs:
                        self._has_combined_score = True
                        self._first_epoch = False
                        # Ensure model is built before saving
                        if hasattr(self.model, 'built') and not self.model.built:
                            self.model.built = True
                        super().on_epoch_end(epoch, logs)
                    elif self._first_epoch:
                        # First epoch - skip saving but mark as no longer first
                        self._first_epoch = False
                    # If combined_score not available, don't call parent (prevents saving)

            checkpoint_callback = SafeModelCheckpoint(
                filepath=checkpoint_path,
                save_weights_only=True,
                monitor="combined_score",
                mode="min",  # Lower combined_score is better
                save_best_only=True,
                verbose=0,
            )
            callbacks.append(checkpoint_callback)

        fit_kwargs["callbacks"] = callbacks

        gan.fit(ds, **fit_kwargs)

        # Load best checkpoint if it exists
        if checkpoint_path and os.path.exists(checkpoint_path):
            if verbose:
                print(f"    Loading best checkpoint from {checkpoint_path}")
            try:
                # Ensure model is built before loading weights
                dummy_z = tf.zeros((1, gan.latent_dim))
                dummy_c = tf.zeros((1, gan.num_classes))
                dummy_x = tf.zeros((1, gan.seq_len, gan.num_features))
                _ = gan.gen([dummy_z, dummy_c], training=False)
                _ = gan.critic([dummy_x, dummy_c], training=False)
                gan.built = True
                # Load the best weights
                gan.load_weights(checkpoint_path)
                if verbose:
                    print("    Best checkpoint loaded successfully")
            except Exception as e:
                if verbose:
                    print(f"    Warning: Could not load checkpoint: {e}")
                    print("    Using final model weights instead")

        # Save model if save_path provided
        if save_path:
            # Convert architecture enum to string for serialization
            arch_str = (
                architecture.value
                if isinstance(architecture, WGANArchitecture)
                else str(architecture)
            )
            metadata = {
                "seq_len": seq_len_value,
                "num_features": num_features,
                "num_classes": num_classes,
                "latent_dim": gan.latent_dim,
                "gp_weight": gan.gp_weight,
                "n_critic": n_critic,  # Save the n_critic value used
                "base_filters": getattr(gan.gen, "base_filters", 128),
                "gen_kernel_size": getattr(gan.gen, "kernel_size", 3),
                "architecture": arch_str,
                "feature_mean": feat_mean,
                "feature_std": feat_std,
                "feature_min": feat_min,
                "feature_max": feat_max,
            }
            _save_wgan_model(gan, metadata, save_path, verbose)

    # Compute targets with augmentation ratio limit to prevent overfitting
    idx = np.argmax(train_labels, axis=1)
    unique, counts = np.unique(idx, return_counts=True)
    class_counts = {int(k): int(v) for k, v in zip(unique, counts)}
    current_max = int(np.max(counts)) if counts.size > 0 else 0

    # Apply augmentation based on strategy
    gen_x_list = [train_data]
    gen_y_list = [train_labels]
    targets_map = {}

    # Determine augmentation strategy
    if augmentation_target_ratio is not None:
        # Strategy: Augment minority classes to a percentage of majority class
        # e.g., augmentation_target_ratio=0.5 means 50% of majority class size
        target_base = int(current_max * augmentation_target_ratio)
        if verbose:
            print(
                f"    WGAN augmentation strategy: minority classes to "
                f"{augmentation_target_ratio*100:.0f}% of majority ({current_max}) = {target_base}"
            )
    else:
        target_base = None
        if verbose:
            print(
                f"    WGAN augmentation strategy: each class to "
                f"{augmentation_ratio}x its own size"
            )

    for c in range(num_classes):
        have = class_counts.get(c, 0)

        # Calculate target for this class
        if augmentation_target_ratio is not None:
            # Use percentage of majority class
            target = target_base
            # Don't augment if already at or above target
            if have >= target:
                target = have
        else:
            # Use ratio of each class's own size
            if max_target is not None:
                ratio_target = int(have * augmentation_ratio)
                target = min(max_target, ratio_target)
            else:
                target = int(have * augmentation_ratio)
                # Don't exceed the maximum class size
                target = min(target, current_max)

        targets_map[c] = target
        need = target - have
        if need <= 0:
            continue
        one_hot = np.zeros((need, num_classes), dtype=np.float32)
        one_hot[:, c] = 1.0
        synth = gan.generate(need, one_hot)

        # Add noise to generated samples to reduce overfitting
        # This makes synthetic samples less "memorizable" and more diverse
        if noise_std > 0:
            noise = np.random.normal(0, noise_std, synth.shape).astype(np.float32)
            synth = synth + noise
            # Clip to maintain valid feature ranges (assuming normalized [-1, 1])
            synth = np.clip(synth, -1.0, 1.0)

        gen_x_list.append(synth)
        gen_y_list.append(one_hot)

    if verbose:
        targets_str = ", ".join([f"{c}: {t}" for c, t in targets_map.items()])
        if augmentation_target_ratio is not None:
            print(
                f"    WGAN targets per class: {targets_str} "
                f"(target: {augmentation_target_ratio*100:.0f}% of majority={current_max})"
            )
        else:
            print(
                f"    WGAN targets per class: {targets_str} "
                f"(augmentation_ratio={augmentation_ratio}x each class size)"
            )

    aug_x = np.concatenate(gen_x_list, axis=0)
    aug_y = np.concatenate(gen_y_list, axis=0)

    perm = np.random.permutation(len(aug_x))
    aug_x = aug_x[perm]
    aug_y = aug_y[perm]

    aug_x_for_quality = aug_x
    aug_x = aug_x[:, 0, :]

    # Assess quality if requested
    if assess_quality and len(aug_x_for_quality) > len(train_data):
        if verbose:
            print("\n    Assessing generated sample quality...")
        _ = assess_generation_quality(
            train_data,
            labels_f32,
            aug_x_for_quality,
            aug_y,
            verbose=verbose,
        )

    return aug_x, aug_y


def assess_generation_quality(
    real_data: np.ndarray,
    real_labels: np.ndarray,
    generated_data: np.ndarray,
    generated_labels: np.ndarray,
    verbose: bool = True,
) -> dict:
    """
    Assess the quality of generated samples by comparing them to real data.

    Returns a dictionary with quality metrics including:
    - mean_rmse: RMSE of feature means
    - std_rmse: RMSE of feature standard deviations
    - feature_correlations: Correlation coefficients for each feature
    - class_specific_metrics: Per-class quality metrics
    """
    metrics = {}

    # Separate real and generated data
    # Use the original real_data parameter (more reliable than extracting from generated_data)
    n_real = len(real_data)
    real_only = real_data  # Use original real data
    gen_only = (
        generated_data[n_real:]
        if len(generated_data) > n_real
        else np.array([]).reshape(0, *real_data.shape[1:])
    )

    if len(gen_only) == 0:
        if verbose:
            print("    [Quality] No generated samples to assess")
        return metrics

    # Overall statistics comparison
    real_mean = real_only.mean(axis=(0, 1))  # (num_features,)
    gen_mean = gen_only.mean(axis=(0, 1))
    real_std = real_only.std(axis=(0, 1)) + 1e-8
    gen_std = gen_only.std(axis=(0, 1)) + 1e-8

    mean_rmse = float(np.sqrt(np.mean((real_mean - gen_mean) ** 2)))
    std_rmse = float(np.sqrt(np.mean((real_std - gen_std) ** 2)))

    metrics["mean_rmse"] = mean_rmse
    metrics["std_rmse"] = std_rmse

    # Feature-wise correlation (flatten sequences for correlation)
    real_flat = real_only.reshape(len(real_only), -1)
    gen_flat = gen_only.reshape(len(gen_only), -1)

    # Sample same size for correlation
    n_samples = min(len(real_flat), len(gen_flat))
    real_sample = real_flat[:n_samples]
    gen_sample = gen_flat[:n_samples]

    feature_corrs = []
    for f in range(real_flat.shape[1]):
        if real_sample[:, f].std() > 1e-8 and gen_sample[:, f].std() > 1e-8:
            corr = np.corrcoef(real_sample[:, f], gen_sample[:, f])[0, 1]
            feature_corrs.append(float(corr))

    metrics["mean_correlation"] = (
        float(np.mean(feature_corrs)) if feature_corrs else 0.0
    )
    metrics["min_correlation"] = float(np.min(feature_corrs)) if feature_corrs else 0.0

    # Class-specific metrics
    num_classes = real_labels.shape[1]
    class_metrics = {}

    for c in range(num_classes):
        real_mask = np.argmax(real_labels, axis=1) == c
        real_class = real_only[real_mask]

        # Extract generated samples of this class (they start after n_real)
        all_gen_mask = np.argmax(generated_labels, axis=1) == c
        gen_indices = np.where(all_gen_mask)[0]
        gen_indices = gen_indices[gen_indices >= n_real]  # Only generated samples
        gen_class = (
            generated_data[gen_indices]
            if len(gen_indices) > 0
            else np.array([]).reshape(0, *real_data.shape[1:])
        )

        if len(gen_class) == 0:
            continue

        if len(real_class) > 0 and len(gen_class) > 0:
            r_mean = real_class.mean(axis=(0, 1))
            g_mean = gen_class.mean(axis=(0, 1))
            r_std = real_class.std(axis=(0, 1)) + 1e-8
            g_std = gen_class.std(axis=(0, 1)) + 1e-8

            class_metrics[c] = {
                "mean_rmse": float(np.sqrt(np.mean((r_mean - g_mean) ** 2))),
                "std_rmse": float(np.sqrt(np.mean((r_std - g_std) ** 2))),
                "real_count": len(real_class),
                "gen_count": len(gen_class),
            }

    metrics["class_specific"] = class_metrics

    # Value range comparison
    real_min = real_only.min()
    real_max = real_only.max()
    gen_min = gen_only.min()
    gen_max = gen_only.max()

    metrics["range_coverage"] = {
        "real_range": (float(real_min), float(real_max)),
        "gen_range": (float(gen_min), float(gen_max)),
        "overlap": float(
            max(0, min(real_max, gen_max) - max(real_min, gen_min))
            / (real_max - real_min + 1e-8)
        ),
    }

    if verbose:
        print(f"    [Quality] Mean RMSE: {mean_rmse:.4f}, Std RMSE: {std_rmse:.4f}")
        print(
            f"    [Quality] Feature correlation: "
            f"mean={metrics['mean_correlation']:.4f}, "
            f"min={metrics['min_correlation']:.4f}"
        )
        print(
            f"    [Quality] Value range overlap: {metrics['range_coverage']['overlap']:.2%}"
        )
        if class_metrics:
            print("    [Quality] Per-class metrics:")
            for c, cm in class_metrics.items():
                print(
                    f"      Class {c}: mean_RMSE={cm['mean_rmse']:.4f}, "
                    f"std_RMSE={cm['std_rmse']:.4f} "
                    f"(real={cm['real_count']}, gen={cm['gen_count']})"
                )

    return metrics
