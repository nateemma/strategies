"""
WGAN-GP for conditional sequence data (batch, seq_len, num_features).

API: balance_with_wgan_gp(train_data, train_labels, epochs=100, batch_size=256, max_target=None, verbose=True)
Returns augmented (data, labels) where minority classes are upsampled to max_target per class.

Notes:
- train_labels must be one-hot (batch, num_classes)
- Data should be pre-scaled; generator output is tanh-bounded and rescaled to real mean/std then clipped to real min/max
"""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd
import tensorflow as tf
from keras import layers, Model, Layer
import os
import pickle

from user_data.strategies.GAN.GANBase import GANBase


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


class _GenCNN(Model):
    def __init__(
        self,
        seq_len: int,
        num_features: int,
        num_classes: int,
        latent_dim: int = 64,
        base_filters: int = 128,
        kernel_size: int = 3,
        up_blocks: int = 2,
    ):
        super().__init__(name="wgangp_gen_cnn")
        self.seq_len = seq_len
        self.num_features = num_features
        self.num_classes = num_classes
        self.latent_dim = latent_dim
        self.base_filters = base_filters
        self.kernel_size = kernel_size
        self.up_blocks = up_blocks

        z_in = layers.Input(shape=(latent_dim,))
        c_in = layers.Input(shape=(num_classes,))
        x = layers.Concatenate()([z_in, c_in])

        effective_up_blocks = up_blocks if seq_len > 1 else 0
        start_len = max(1, seq_len // (2**effective_up_blocks))
        channels = base_filters
        x = layers.Dense(start_len * channels)(x)
        x = layers.LeakyReLU(0.2)(x)
        x = layers.Reshape((start_len, channels))(x)

        def film(y, cond, ch):
            gb = layers.Dense(2 * ch)(cond)
            split_layer = _SplitLayer()
            split_outputs = split_layer(gb)
            gamma = layers.Reshape((1, ch))(split_outputs[0])
            beta = layers.Reshape((1, ch))(split_outputs[1])
            return y * (1.0 + gamma) + beta

        h = x
        for _ in range(effective_up_blocks):
            h = layers.Conv1DTranspose(
                filters=channels, kernel_size=kernel_size, strides=2, padding="same"
            )(h)
            h = layers.LeakyReLU(0.2)(h)
            h = film(h, c_in, channels)
            channels = max(channels // 2, 32)

        h = layers.Conv1D(
            num_features, kernel_size=kernel_size, padding="same", activation="tanh"
        )(h)

        out = _ResizeToLenLayer(target_len=seq_len)(h)
        self.model = Model([z_in, c_in], out, name="wgangp_gen_cnn")
        self.model.summary()

    def call(self, inputs, training=False):
        z, c = inputs
        return self.model([z, c], training=training)


class _Critic(Model):
    def __init__(
        self,
        seq_len: int,
        num_features: int,
        num_classes: int,
        base_filters: int = 128,
        kernel_size: int = 3,
    ):
        super().__init__(name="wgangp_critic")

        xin = layers.Input(shape=(seq_len, num_features))
        cin = layers.Input(shape=(num_classes,))
        c_rep = layers.RepeatVector(seq_len)(cin)
        x = layers.Concatenate()([xin, c_rep])
        h = layers.Conv1D(base_filters, kernel_size, padding="same")(x)
        h = layers.LeakyReLU(0.2)(h)
        h = layers.Conv1D(base_filters * 2, kernel_size, strides=2, padding="same")(h)
        h = layers.LeakyReLU(0.2)(h)
        h = layers.GlobalAveragePooling1D()(h)
        # Minibatch stddev removed - can cause instability and WGAN-GP gradient penalty already prevents mode collapse
        # sfeat = _MinibatchStdLayer()(h)
        # h = layers.Concatenate()([h, sfeat])
        h = layers.Dense(128)(h)
        h = layers.LeakyReLU(0.2)(h)
        out = layers.Dense(1)(h)
        self.model = Model([xin, cin], out, name="wgangp_critic")
        self.model.summary()

    def call(self, inputs, training=False):
        x, c = inputs
        # tf.print("    [DEBUG] critic x:", tf.shape(x), "c:", tf.shape(c), summarize=-1)
        tf.debugging.assert_equal(
            tf.shape(x)[1],
            1,
            message="critic expected seq_len 1",
        )
        return self.model([x, c], training=training)


class WGAN_GP(Model):
    def __init__(
        self,
        seq_len: int,
        num_features: int,
        num_classes: int,
        latent_dim: int = 64,
        gp_weight: float = 10.0,
        n_critic: int = 5,
        gen_base_filters: int = 128,
        gen_kernel_size: int = 3,
        gen_up_blocks: int = 2,
    ):
        super().__init__(name="wgan_gp")
        self.seq_len = seq_len
        self.num_features = num_features
        self.num_classes = num_classes
        self.latent_dim = latent_dim
        self.gp_weight = gp_weight
        self.n_critic = n_critic
        self.gen = _GenCNN(
            seq_len,
            num_features,
            num_classes,
            latent_dim,
            gen_base_filters,
            gen_kernel_size,
            gen_up_blocks,
        )
        self.critic = _Critic(seq_len, num_features, num_classes)

        self.gen_loss_tracker = tf.keras.metrics.Mean(name="g_loss")
        self.crit_loss_tracker = tf.keras.metrics.Mean(name="d_loss")
        self.mean_real_tracker = tf.keras.metrics.Mean(name="d_real")
        self.mean_fake_tracker = tf.keras.metrics.Mean(name="d_fake")
        self.gp_tracker = tf.keras.metrics.Mean(name="gp")

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

    def train_step(self, data):
        real_x, c = data
        batch_size = tf.shape(real_x)[0]

        tf.debugging.assert_equal(
            tf.shape(real_x)[1],
            self.seq_len,
            message="real_x sequence length mismatch",
        )

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
            grads = tape.gradient(d_loss, self.critic.trainable_variables)
            self.d_optimizer.apply_gradients(
                zip(grads, self.critic.trainable_variables)
            )
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
            g_loss = -tf.reduce_mean(fake_scores)
        grads = tape.gradient(g_loss, self.gen.trainable_variables)
        self.g_optimizer.apply_gradients(zip(grads, self.gen.trainable_variables))

        self.gen_loss_tracker.update_state(g_loss)
        self.crit_loss_tracker.update_state(crit_loss)
        self.mean_real_tracker.update_state(d_real_mean)
        self.mean_fake_tracker.update_state(d_fake_mean)
        self.gp_tracker.update_state(gp_mean)
        return {
            "g_loss": self.gen_loss_tracker.result(),
            "d_loss": self.crit_loss_tracker.result(),
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
        gan = WGAN_GP(
            seq_len=metadata["seq_len"],
            num_features=metadata["num_features"],
            num_classes=metadata["num_classes"],
            latent_dim=metadata.get("latent_dim", 64),
            gp_weight=metadata.get("gp_weight", 10.0),
            n_critic=metadata.get("n_critic", 5),
            gen_base_filters=metadata.get("gen_base_filters", 128),
            gen_kernel_size=metadata.get("gen_kernel_size", 3),
            gen_up_blocks=metadata.get("gen_up_blocks", 2),
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
            # Use same learning rate for both, but generator has tighter gradient clipping
            g_optimizer=tf.keras.optimizers.Adam(
                learning_rate=1e-4, beta_1=0.5, beta_2=0.9, clipnorm=0.5
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
    verbose: bool = True,
    save_path: str | None = None,
    assess_quality: bool = True,
    n_critic: int = 5,
):
    # Ensure data is 2D (batch, features)
    if train_data.ndim != 2:
        raise ValueError("balance_with_wgan_gp expects 2D train_data (batch, features)")

    # Convert labels to one-hot if necessary
    labels_f32 = _one_hot_encode(train_labels)
    if labels_f32.ndim != 2:
        raise ValueError("train_labels must be 1D (class indices) or 2D one-hot array")

    num_classes = labels_f32.shape[1]
    num_features = train_data.shape[1]

    # Expand to (batch, 1, features) for the Conv1D GAN
    train_data = train_data[:, np.newaxis, :]
    seq_len_value = 1

    gan = None
    loaded_meta = None
    if save_path:
        gan, loaded_meta = _load_wgan_model(save_path, verbose)
        if gan is not None:
            if (
                loaded_meta["seq_len"] != 1
                or loaded_meta["num_features"] != num_features
                or loaded_meta["num_classes"] != num_classes
            ):
                if verbose:
                    print(
                        "    Saved model incompatible with current data; retraining from scratch"
                    )
                gan = None

    n = train_data.shape[0]
    data_f32 = train_data.astype("float32")

    if verbose:
        print(
            f"    GAN dataset shapes → data: {data_f32.shape}, labels: {labels_f32.shape}"
        )

    ds = (
        tf.data.Dataset.from_tensor_slices((data_f32, labels_f32))
        .shuffle(min(8192, n))
        .batch(batch_size)
        .prefetch(tf.data.AUTOTUNE)
    )

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

    # Try to load existing model if save_path provided
    model_was_loaded = gan is not None
    if gan is None:
        gan = WGAN_GP(
            seq_len_value,
            num_features,
            num_classes,
            latent_dim=64,
            gp_weight=10.0,
            n_critic=n_critic,
        )
        gan.feature_mean = tf.convert_to_tensor(feat_mean)
        gan.feature_std = tf.convert_to_tensor(feat_std)
        gan.feature_min = tf.convert_to_tensor(feat_min)
        gan.feature_max = tf.convert_to_tensor(feat_max)

        gan.compile(
            # Use same learning rate for both, but generator has tighter gradient clipping
            g_optimizer=tf.keras.optimizers.Adam(
                learning_rate=1e-4, beta_1=0.5, beta_2=0.9, clipnorm=0.5
            ),
            d_optimizer=tf.keras.optimizers.Adam(
                learning_rate=1e-4, beta_1=0.5, beta_2=0.9, clipnorm=1.0
            ),
        )
        if verbose:
            print("    GAN generator summary:")
            gan.gen.summary(print_fn=lambda line: print("        " + line))
            print("    GAN critic summary:")
            gan.critic.summary(print_fn=lambda line: print("        " + line))

    # Train only if model wasn't loaded
    if not model_was_loaded:
        gan.fit(ds, epochs=max(1, epochs), verbose=1 if verbose else 0)

        # Save model if save_path provided
        if save_path:
            metadata = {
                "seq_len": seq_len_value,
                "num_features": num_features,
                "num_classes": num_classes,
                "latent_dim": gan.latent_dim,
                "gp_weight": gan.gp_weight,
                "n_critic": n_critic,  # Save the n_critic value used
                "gen_base_filters": getattr(gan.gen, "base_filters", 128),
                "gen_kernel_size": getattr(gan.gen, "kernel_size", 3),
                "gen_up_blocks": getattr(gan.gen, "up_blocks", 2),
                "feature_mean": feat_mean,
                "feature_std": feat_std,
                "feature_min": feat_min,
                "feature_max": feat_max,
            }
            _save_wgan_model(gan, metadata, save_path, verbose)

    # Compute targets
    idx = np.argmax(train_labels, axis=1)
    unique, counts = np.unique(idx, return_counts=True)
    class_counts = {int(k): int(v) for k, v in zip(unique, counts)}
    current_max = int(np.max(counts)) if counts.size > 0 else 0
    target = max_target if (max_target is not None) else current_max

    gen_x_list = [train_data]
    gen_y_list = [train_labels]
    for c in range(num_classes):
        have = class_counts.get(c, 0)
        need = target - have
        if need <= 0:
            continue
        one_hot = np.zeros((need, num_classes), dtype=np.float32)
        one_hot[:, c] = 1.0
        synth = gan.generate(need, one_hot)
        gen_x_list.append(synth)
        gen_y_list.append(one_hot)

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
        quality_metrics = assess_generation_quality(
            train_data,
            labels_f32,
            aug_x_for_quality,
            aug_y,
            verbose=verbose,
        )
        # Store metrics in returned data (can be accessed if needed)

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
            f"    [Quality] Feature correlation: mean={metrics['mean_correlation']:.4f}, min={metrics['min_correlation']:.4f}"
        )
        print(
            f"    [Quality] Value range overlap: {metrics['range_coverage']['overlap']:.2%}"
        )
        if class_metrics:
            print("    [Quality] Per-class metrics:")
            for c, cm in class_metrics.items():
                print(
                    f"      Class {c}: mean_RMSE={cm['mean_rmse']:.4f}, std_RMSE={cm['std_rmse']:.4f} (real={cm['real_count']}, gen={cm['gen_count']})"
                )

    return metrics


class SingleTaskWGANGP(GANBase):
    """Object-oriented wrapper around the single-task WGAN-GP implementation."""

    MODEL_FILENAME = "wgangp_model.weights.h5"
    METADATA_FILENAME = "wgangp_metadata.pkl"
    DEFAULT_CONFIG: Dict[str, Any] = {
        "train_kwargs": {
            "epochs": 100,
            "batch_size": 256,
            "n_critic": 5,
            "verbose": True,
            "assess_quality": False,
        },
        "augment_kwargs": {
            "epochs": 100,
            "batch_size": 256,
            "n_critic": 5,
            "verbose": True,
            "assess_quality": True,
        },
    }

    def __init__(
        self,
        identifier: str,
        root_dir: str,
        train_kwargs: Optional[Dict[str, Any]] = None,
        augment_kwargs: Optional[Dict[str, Any]] = None,
        save_path: Optional[str] = None,
        **config: Any,
    ) -> None:
        merged = {
            "train_kwargs": train_kwargs or {},
            "augment_kwargs": augment_kwargs or {},
            **config,
        }
        super().__init__(
            identifier=identifier,
            root_dir=root_dir,
            train_kwargs=merged.get("train_kwargs"),
            augment_kwargs=merged.get("augment_kwargs"),
            save_path=save_path,
        )

    # ------------------------------------------------------------------
    # Public overrides
    # ------------------------------------------------------------------
    def augment(
        self,
        dataframe: pd.DataFrame,
        labels: np.ndarray | list,
        *,
        target_ratio: Optional[float] = None,
        max_target: Optional[int] = None,
        **kwargs: Any,
    ) -> Tuple[pd.DataFrame, np.ndarray]:
        extra = dict(kwargs)
        if target_ratio is not None and max_target is None:
            arr = np.asarray(labels)
            if arr.size:
                counts = np.bincount(arr.astype(int, copy=False))
                current_max = int(counts.max())
                extra["max_target"] = int(current_max * target_ratio)
        elif max_target is not None:
            extra["max_target"] = max_target
        return super().augment(dataframe, labels, **extra)

    # ------------------------------------------------------------------
    # Hooks
    # ------------------------------------------------------------------
    def _train_impl(
        self,
        dataframe: pd.DataFrame,
        labels: np.ndarray | list,
        kwargs: Dict[str, Any],
    ) -> int:
        values, _ = self._extract_dataframe(dataframe)
        encoded, _ = _encode_single_labels(labels)
        balance_with_wgan_gp(values, encoded, **kwargs)
        return int(values.shape[0])

    def _augment_impl(
        self,
        dataframe: pd.DataFrame,
        labels: np.ndarray | list,
        kwargs: Dict[str, Any],
    ) -> Tuple[pd.DataFrame, np.ndarray]:
        values, columns = self._extract_dataframe(dataframe)
        encoded, dtype = _encode_single_labels(labels)
        aug_values, aug_labels = balance_with_wgan_gp(values, encoded, **kwargs)
        aug_df = pd.DataFrame(aug_values, columns=columns)
        restored = _decode_single_labels(aug_labels, dtype)
        return aug_df, restored


def _encode_single_labels(labels: np.ndarray | list) -> Tuple[np.ndarray, np.dtype]:
    arr = np.asarray(labels)
    if arr.ndim != 1:
        raise ValueError("Single-task labels must be 1D class indices")
    arr_int = arr.astype(int, copy=False)
    num_classes = int(arr_int.max()) + 1 if arr_int.size else 1
    encoded = np.eye(num_classes, dtype=np.float32)[arr_int]
    return encoded, arr.dtype


def _decode_single_labels(labels: np.ndarray, dtype: np.dtype) -> np.ndarray:
    decoded = np.argmax(labels, axis=1)
    return decoded.astype(dtype, copy=False)


__all__ = ["SingleTaskWGANGP"]
