"""
Conditional GAN for generic tabular/time-series tensors shaped as
(batch, seq_len, num_features).

Primary entrypoint: balance_with_cgan()
- Trains a lightweight conditional GAN on (train_data, train_labels)
- Generates additional samples for minority classes to reach a target count
- Returns augmented (data, labels)

Notes:
- Labels are expected to be one-hot encoded with shape (batch, num_classes)
- Data is expected to be float32, scaled/normalized prior to calling this
"""

# pragma: disable=import-error
from __future__ import annotations

import numpy as np
import tensorflow as tf
from keras import layers, Model
from keras.callbacks import Callback
import os
import pickle


class _DFGenerator(Model):
    def __init__(self, seq_len: int, num_features: int, num_classes: int, latent_dim: int = 64, hidden_units: int = 256):
        super().__init__(name="df_generator_mlp")
        self.seq_len = seq_len
        self.num_features = num_features
        self.num_classes = num_classes
        self.latent_dim = latent_dim

        input_z = layers.Input(shape=(latent_dim,))
        input_c = layers.Input(shape=(num_classes,))
        x = layers.Concatenate(axis=-1)([input_z, input_c])
        x = layers.Dense(hidden_units, activation="relu")(x)
        x = layers.Dense(hidden_units, activation="relu")(x)
        x = layers.Dense(seq_len * num_features, activation="linear")(x)
        out = layers.Reshape((seq_len, num_features))(x)

        self.model = Model([input_z, input_c], out, name="df_generator_mlp")

    def call(self, inputs, training=False):
        z, c = inputs
        return self.model([z, c], training=training)


class _DFGeneratorCNN(Model):
    def __init__(
        self,
        seq_len: int,
        num_features: int,
        num_classes: int,
        latent_dim: int = 64,
        base_filters: int = 128,
        kernel_size: int = 3,
        num_upsample_blocks: int = 2,
    ):
        super().__init__(name="df_generator_cnn")
        self.seq_len = seq_len
        self.num_features = num_features
        self.num_classes = num_classes
        self.latent_dim = latent_dim
        self.base_filters = base_filters
        self.kernel_size = kernel_size
        self.num_upsample_blocks = num_upsample_blocks

        import math

        # Cap upsample blocks so we never overshoot seq_len.
        # Each stride-2 Conv1DTranspose doubles the time dimension, so we can
        # safely apply at most floor(log2(seq_len)) blocks before seq_len is
        # reached.  For seq_len=1 this is 0; for seq_len=4 this is 2; etc.
        effective_up_blocks = min(
            num_upsample_blocks,
            max(0, int(math.floor(math.log2(max(1, seq_len))))),
        )
        start_len = max(1, seq_len // (2 ** effective_up_blocks))

        input_z = layers.Input(shape=(latent_dim,))
        input_c = layers.Input(shape=(num_classes,))
        x = layers.Concatenate(axis=-1)([input_z, input_c])

        channels = base_filters
        x = layers.Dense(start_len * channels, activation=None)(x)
        x = layers.LeakyReLU(negative_slope=0.2)(x)
        x = layers.Reshape((start_len, channels))(x)

        def film_block(y, cond, ch):
            gamma_beta = layers.Dense(2 * ch, activation=None)(cond)
            gamma, beta = layers.Lambda(lambda t: tf.split(t, num_or_size_splits=2, axis=-1))(gamma_beta)
            gamma = layers.Reshape((1, ch))(gamma)
            beta = layers.Reshape((1, ch))(beta)
            return y * (1.0 + gamma) + beta

        # Upsample blocks (at most effective_up_blocks, never overshooting seq_len)
        h = x
        for _ in range(effective_up_blocks):
            h = layers.Conv1DTranspose(filters=channels, kernel_size=kernel_size, strides=2, padding="same", activation=None)(h)
            h = layers.LeakyReLU(negative_slope=0.2)(h)
            h = film_block(h, input_c, channels)
            channels = max(channels // 2, 32)

        # Final projection to features
        h = layers.Conv1D(filters=num_features, kernel_size=kernel_size, padding="same", activation="linear")(h)

        # Resize to exact seq_len only when the upsample output doesn't already
        # match (e.g. non-power-of-2 seq_len).  Decided statically at construction
        # time — no tf.cond needed, so the shape is always fully known to Keras.
        upsample_output_len = start_len * (2 ** effective_up_blocks)
        if upsample_output_len != seq_len:
            _target_seq_len = seq_len  # capture for closure
            def _resize_fn(t, _sl=_target_seq_len):
                tt = tf.expand_dims(t, axis=2)          # (B, L, 1, C)
                tt = tf.image.resize(tt, size=(_sl, 1), method="bilinear")
                tt = tf.squeeze(tt, axis=2)              # (B, seq_len, C)
                return tf.cast(tt, t.dtype)

            out = layers.Lambda(
                _resize_fn,
                output_shape=(seq_len, num_features),
                name="resize_to_seq_len",
            )(h)
        else:
            out = h

        self.model = Model([input_z, input_c], out, name="df_generator_cnn")

    def call(self, inputs, training=False):
        z, c = inputs
        return self.model([z, c], training=training)


class _DFDiscriminator(Model):
    def __init__(self, seq_len: int, num_features: int, num_classes: int, hidden_units: int = 256):
        super().__init__(name="df_discriminator")
        self.seq_len = seq_len
        self.num_features = num_features
        self.num_classes = num_classes

        input_x = layers.Input(shape=(seq_len, num_features))
        input_c = layers.Input(shape=(num_classes,))

        # Broadcast class condition across time and concatenate to features
        c_rep = layers.RepeatVector(seq_len)(input_c)
        xc = layers.Concatenate(axis=-1)([input_x, c_rep])

        # Lightweight temporal feature extractor
        y = layers.Conv1D(128, kernel_size=3, padding="same", activation=None)(xc)
        y = layers.LeakyReLU(negative_slope=0.2)(y)
        y = layers.Conv1D(128, kernel_size=3, padding="same", activation=None)(y)
        y = layers.LeakyReLU(negative_slope=0.2)(y)
        feat = layers.GlobalAveragePooling1D(name="disc_features")(y)

        # Minibatch standard deviation trick (adds batch variation signal)
        def minibatch_stddev(t):
            # t: (B, F)
            mean = tf.reduce_mean(t, axis=0, keepdims=True)
            var = tf.reduce_mean(tf.square(t - mean), axis=0, keepdims=True)
            std = tf.sqrt(var + 1e-8)
            # average across features to a single scalar, then tile to batch
            std_scalar = tf.reduce_mean(std, axis=1, keepdims=True)  # (1,1)
            std_feat = tf.tile(std_scalar, [tf.shape(t)[0], 1])      # (B,1)
            return std_feat

        std_feat = layers.Lambda(minibatch_stddev, name="minibatch_std")(feat)
        y2 = layers.Concatenate()([feat, std_feat])
        y2 = layers.Dense(hidden_units, activation=None)(y2)
        y2 = layers.LeakyReLU(negative_slope=0.2)(y2)
        out = layers.Dense(1)(y2)  # logits

        # Expose both logits and intermediate features for feature matching
        self.model = Model([input_x, input_c], [out, feat], name="df_discriminator")

    def call(self, inputs, training=False):
        x, c = inputs
        return self.model([x, c], training=training)


class DFCGAN(Model):
    def __init__(
        self,
        seq_len: int,
        num_features: int,
        num_classes: int,
        latent_dim: int = 64,
        d_steps: int = 2,
        instance_noise_std: float = 0.01,
        label_smoothing: bool = True,
        fm_weight: float = 1.0,
        fm_var_weight: float = 0.5,
        mmd_weight: float = 0.5,
        feature_mean: np.ndarray | None = None,
        feature_std: np.ndarray | None = None,
        feature_min: np.ndarray | None = None,
        feature_max: np.ndarray | None = None,
        generator_arch: str = "cnn",
        gen_base_filters: int = 128,
        gen_kernel_size: int = 3,
        gen_upsample_blocks: int = 2,
    ):
        super().__init__(name="df_cgan")
        self.seq_len = seq_len
        self.num_features = num_features
        self.num_classes = num_classes
        self.latent_dim = latent_dim
        self.d_steps = max(1, int(d_steps))
        self.instance_noise_std = max(0.0, float(instance_noise_std))
        self.label_smoothing = bool(label_smoothing)
        self.fm_weight = float(fm_weight)
        self.fm_var_weight = float(fm_var_weight)
        self.mmd_weight = float(mmd_weight)
        # Store feature stats for output rescaling/clipping
        self.feature_mean = None if feature_mean is None else tf.convert_to_tensor(feature_mean, dtype=tf.float32)
        self.feature_std = None if feature_std is None else tf.convert_to_tensor(feature_std, dtype=tf.float32)
        self.feature_min = None if feature_min is None else tf.convert_to_tensor(feature_min, dtype=tf.float32)
        self.feature_max = None if feature_max is None else tf.convert_to_tensor(feature_max, dtype=tf.float32)

        if generator_arch == "cnn":
            self.generator = _DFGeneratorCNN(
                seq_len, num_features, num_classes, latent_dim,
                base_filters=gen_base_filters, kernel_size=gen_kernel_size, num_upsample_blocks=gen_upsample_blocks
            )
        else:
            self.generator = _DFGenerator(seq_len, num_features, num_classes, latent_dim)
        self.discriminator = _DFDiscriminator(seq_len, num_features, num_classes)

        self.gen_loss_tracker = tf.keras.metrics.Mean(name="g_loss")
        self.disc_loss_tracker = tf.keras.metrics.Mean(name="d_loss")

    @property
    def metrics(self):
        return [self.gen_loss_tracker, self.disc_loss_tracker]

    def compile(self, d_optimizer, g_optimizer, loss_fn):
        super().compile()
        self.d_optimizer = d_optimizer
        self.g_optimizer = g_optimizer
        self.loss_fn = loss_fn

    def train_step(self, data):
        real_x, one_hot_c = data  # x: (B, T, F), c: (B, C)
        batch_size = tf.shape(real_x)[0]

        # Helper: add instance noise
        def add_noise(x):
            if self.instance_noise_std > 0.0:
                x_dtype = x.dtype
                noise = tf.random.normal(shape=tf.shape(x), stddev=self.instance_noise_std, dtype=x_dtype)
                return x + noise
            return x

        # Helper: label smoothing
        def real_labels_like(logits):
            labels = tf.ones_like(logits, dtype=logits.dtype)
            if self.label_smoothing:
                high = 1.0
                low = 0.9
                smooth = tf.random.uniform(shape=tf.shape(labels), minval=low, maxval=high, dtype=logits.dtype)
                return smooth
            return labels

        def fake_labels_like(logits):
            labels = tf.zeros_like(logits, dtype=logits.dtype)
            if self.label_smoothing:
                high = 0.1
                low = 0.0
                smooth = tf.random.uniform(shape=tf.shape(labels), minval=low, maxval=high, dtype=logits.dtype)
                return smooth
            return labels

        # 1) Train discriminator (possibly multiple steps)
        d_loss_accum = 0.0
        for _ in range(self.d_steps):
            random_z = tf.random.normal(shape=(batch_size, self.latent_dim))
            fake_x = self.generator([random_z, one_hot_c], training=True)
            with tf.GradientTape() as tape:
                real_logits, _ = self.discriminator([add_noise(real_x), one_hot_c], training=True)
                fake_logits, _ = self.discriminator([add_noise(fake_x), one_hot_c], training=True)
                d_loss_real = self.loss_fn(real_labels_like(real_logits), real_logits)
                d_loss_fake = self.loss_fn(fake_labels_like(fake_logits), fake_logits)
                d_loss = (d_loss_real + d_loss_fake) / 2.0
            d_grads = tape.gradient(d_loss, self.discriminator.trainable_variables)
            self.d_optimizer.apply_gradients(zip(d_grads, self.discriminator.trainable_variables))
            d_loss_accum += d_loss

        d_loss_mean = d_loss_accum / float(self.d_steps)

        # 2) Train generator (single step)
        random_z = tf.random.normal(shape=(batch_size, self.latent_dim))
        with tf.GradientTape() as tape:
            # v2: linear output, z-scored space. real_x is z-scored upstream
            # so both real and gen go through the discriminator in matching
            # space — no _postprocess_gen inversion at training time.
            gen_x = self.generator([random_z, one_hot_c], training=True)
            gen_logits, gen_feat = self.discriminator([add_noise(gen_x), one_hot_c], training=True)
            # want discriminator to predict real (1)
            adv_loss = self.loss_fn(tf.ones_like(gen_logits, dtype=gen_logits.dtype), gen_logits)
            # feature matching: match discriminator intermediate feature mean and variance
            _, real_feat_ref = self.discriminator([add_noise(real_x), one_hot_c], training=True)
            feat_dtype = gen_feat.dtype
            real_feat_ref = tf.cast(real_feat_ref, feat_dtype)
            real_feat_mean = tf.reduce_mean(real_feat_ref, axis=0)
            gen_feat_mean = tf.reduce_mean(gen_feat, axis=0)
            fm_mean = tf.reduce_mean(tf.abs(real_feat_mean - gen_feat_mean))

            real_feat_var = tf.math.reduce_variance(real_feat_ref, axis=0)
            gen_feat_var = tf.math.reduce_variance(gen_feat, axis=0)
            fm_var = tf.reduce_mean(tf.abs(real_feat_var - gen_feat_var))

            # MMD with RBF kernel in feature space
            def _rbf(x, y, sigma=1.0):
                x2 = tf.reduce_sum(tf.square(x), axis=1, keepdims=True)
                y2 = tf.reduce_sum(tf.square(y), axis=1, keepdims=True)
                xy = tf.matmul(x, y, transpose_b=True)
                d2 = x2 - 2.0 * xy + tf.transpose(y2)
                k = tf.exp(-d2 / (2.0 * sigma * sigma))
                return k
            # Normalize features for MMD stability (zero mean, unit var)
            def _norm_feats(f):
                mu = tf.reduce_mean(f, axis=0, keepdims=True)
                sd = tf.math.reduce_std(f, axis=0, keepdims=True) + 1e-6
                return (f - mu) / sd
            gen_feat_n = _norm_feats(gen_feat)
            real_feat_n = _norm_feats(real_feat_ref)
            sigma = 1.0
            K_xx = _rbf(gen_feat_n, gen_feat_n, sigma)
            K_yy = _rbf(real_feat_n, real_feat_n, sigma)
            K_xy = _rbf(gen_feat_n, real_feat_n, sigma)
            mmd = tf.reduce_mean(K_xx) + tf.reduce_mean(K_yy) - 2.0 * tf.reduce_mean(K_xy)

            g_loss = (
                tf.cast(adv_loss, feat_dtype)
                + tf.cast(self.fm_weight, feat_dtype) * fm_mean
                + tf.cast(self.fm_var_weight, feat_dtype) * fm_var
                + tf.cast(self.mmd_weight, feat_dtype) * mmd
            )
        g_grads = tape.gradient(g_loss, self.generator.trainable_variables)
        self.g_optimizer.apply_gradients(zip(g_grads, self.generator.trainable_variables))

        # Guard NaNs/Infs in losses (tensor-safe: avoid Python conditionals on tensors)
        g_loss = tf.where(tf.math.is_finite(g_loss), g_loss, tf.zeros_like(g_loss))
        d_loss_mean = tf.where(tf.math.is_finite(d_loss_mean), d_loss_mean, tf.zeros_like(d_loss_mean))

        self.gen_loss_tracker.update_state(g_loss)
        self.disc_loss_tracker.update_state(d_loss_mean)
        return {"g_loss": self.gen_loss_tracker.result(), "d_loss": self.disc_loss_tracker.result()}

    def generate(self, n_samples: int, class_one_hot: np.ndarray) -> np.ndarray:
        z = tf.random.normal(shape=(n_samples, self.latent_dim))
        c = tf.convert_to_tensor(class_one_hot, dtype=tf.float32)
        x = self.generator([z, c], training=False)
        x = self._postprocess_gen(x)
        return x.numpy()

    def _postprocess_gen(self, x):
        """Inverse z-score: maps z-scored generator output → real feature space.

        Under v2 (since the tanh-bounded path was removed) this is called
        only at generate time. Training-time discriminator comparison
        happens entirely in z-scored space so train_step bypasses it.
        """
        # Broadcast mean/std to (1,1,F)
        if self.feature_mean is not None and self.feature_std is not None:
            mean = tf.reshape(tf.cast(self.feature_mean, x.dtype), (1, 1, -1))
            std = tf.reshape(tf.cast(self.feature_std, x.dtype), (1, 1, -1))
            x = x * std + mean
        # Clip to observed min/max if provided
        if self.feature_min is not None and self.feature_max is not None:
            fmin = tf.reshape(tf.cast(self.feature_min, x.dtype), (1, 1, -1))
            fmax = tf.reshape(tf.cast(self.feature_max, x.dtype), (1, 1, -1))
            x = tf.clip_by_value(x, fmin, fmax)
        # Guard NaN/Inf
        x = tf.where(tf.math.is_finite(x), x, tf.zeros_like(x))
        return x


def _one_hot_to_indices(labels: np.ndarray) -> np.ndarray:
    return np.argmax(labels, axis=1)


def _save_cgan_model(gan: DFCGAN, metadata: dict, save_path: str, verbose: bool = False):
    """Save CGAN model and metadata"""
    os.makedirs(save_path, exist_ok=True)
    model_path = os.path.join(save_path, "cgan_model.weights.h5")
    meta_path = os.path.join(save_path, "cgan_metadata.pkl")
    # Build sub-models and mark parent as built
    dummy_z = tf.zeros((1, gan.latent_dim))
    dummy_c = tf.zeros((1, gan.num_classes))
    dummy_x = tf.zeros((1, gan.seq_len, gan.num_features))
    _ = gan.generator([dummy_z, dummy_c], training=False)
    _ = gan.discriminator([dummy_x, dummy_c], training=False)
    gan.built = True  # Explicitly mark parent model as built
    gan.save_weights(model_path)
    with open(meta_path, "wb") as f:
        pickle.dump(metadata, f)
    if verbose:
        print(f"    CGAN model saved to {save_path}")


def _load_cgan_model(save_path: str, verbose: bool = False) -> tuple[DFCGAN | None, dict | None]:
    """Load CGAN model and metadata if they exist"""
    model_path = os.path.join(save_path, "cgan_model.weights.h5")
    meta_path = os.path.join(save_path, "cgan_metadata.pkl")
    if not (os.path.exists(model_path) and os.path.exists(meta_path)):
        return None, None
    try:
        with open(meta_path, "rb") as f:
            metadata = pickle.load(f)
        gan = DFCGAN(
            seq_len=metadata["seq_len"],
            num_features=metadata["num_features"],
            num_classes=metadata["num_classes"],
            latent_dim=metadata.get("latent_dim", 64),
            d_steps=metadata.get("d_steps", 2),
            instance_noise_std=metadata.get("instance_noise_std", 0.01),
            label_smoothing=metadata.get("label_smoothing", True),
            fm_weight=metadata.get("fm_weight", 1.0),
            fm_var_weight=metadata.get("fm_var_weight", 0.5),
            mmd_weight=metadata.get("mmd_weight", 0.5),
            feature_mean=metadata.get("feature_mean"),
            feature_std=metadata.get("feature_std"),
            feature_min=metadata.get("feature_min"),
            feature_max=metadata.get("feature_max"),
            generator_arch=metadata.get("generator_arch", "cnn"),
            gen_base_filters=metadata.get("gen_base_filters", 128),
            gen_kernel_size=metadata.get("gen_kernel_size", 3),
            gen_upsample_blocks=metadata.get("gen_upsample_blocks", 2),
        )
        gan.compile(
            d_optimizer=tf.keras.optimizers.Adam(learning_rate=metadata.get("learning_rate", 3e-4), clipnorm=1.0),
            g_optimizer=tf.keras.optimizers.Adam(learning_rate=metadata.get("learning_rate", 3e-4), clipnorm=1.0),
            loss_fn=tf.keras.losses.BinaryCrossentropy(from_logits=True),
        )
        # Ensure model is built before loading weights by calling sub-models
        dummy_z = tf.zeros((1, gan.latent_dim))
        dummy_c = tf.zeros((1, gan.num_classes))
        dummy_x = tf.zeros((1, gan.seq_len, gan.num_features))
        # Build generator and discriminator
        _ = gan.generator([dummy_z, dummy_c], training=False)
        _ = gan.discriminator([dummy_x, dummy_c], training=False)
        gan.built = True  # Explicitly mark parent model as built
        gan.load_weights(model_path)
        if verbose:
            print(f"    CGAN model loaded from {save_path}")
        return gan, metadata
    except Exception as e:
        if verbose:
            print(f"    Failed to load CGAN model: {e}")
        return None, None


def balance_with_cgan(
    train_data: np.ndarray,
    train_labels: np.ndarray,
    epochs: int = 10,
    batch_size: int = 256,
    max_target: int | None = None,
    learning_rate: float = 3e-4,
    verbose: bool = False,
    steps_per_epoch: int | None = 30,
    d_steps: int = 2,
    instance_noise_std: float = 0.01,
    label_smoothing: bool = True,
    fm_weight: float = 1.0,
    fm_var_weight: float = 0.5,
    mmd_weight: float = 0.5,
    # Learning-rate plateau controls
    lr_plateau_enable: bool = True,
    lr_factor: float = 0.5,
    lr_patience: int = 5,
    lr_min: float = 1e-6,
    lr_cooldown: int = 0,
    lr_monitor: str = "d_loss",
    lr_mode: str = "min",
    save_path: str | None = None,
):
    """
    Train a small conditional GAN on (train_data, train_labels) and augment the
    dataset so each class reaches max_target (or the max existing class count).

    Returns (aug_data, aug_labels) including originals and generated samples.
    """
    assert train_data.ndim == 3, "train_data must be (batch, seq_len, num_features)"
    assert train_labels.ndim == 2, "train_labels must be one-hot (batch, num_classes)"

    num_samples, seq_len, num_features = train_data.shape
    num_classes = train_labels.shape[1]

    # v2 pipeline: z-score the training data BEFORE building the dataset
    # so the generator trains in standardised space. Stats are persisted
    # in metadata so _postprocess_gen inverts the transform at generate.
    real_feat_mean = train_data.mean(axis=(0, 1)).astype("float32")
    real_feat_std = train_data.std(axis=(0, 1)).astype("float32") + 1e-8
    real_feat_min = train_data.min(axis=(0, 1)).astype("float32")
    real_feat_max = train_data.max(axis=(0, 1)).astype("float32")
    _ZSCORE_CLIP = 4.0
    train_data_zscored = (train_data.astype("float32") - real_feat_mean) / real_feat_std
    train_data_zscored = np.clip(
        train_data_zscored, -_ZSCORE_CLIP, _ZSCORE_CLIP
    ).astype("float32")

    # Build tf.data pipeline from z-scored data.
    ds = tf.data.Dataset.from_tensor_slices((train_data_zscored, train_labels.astype("float32")))
    ds = ds.shuffle(buffer_size=min(8192, num_samples)).batch(batch_size).prefetch(tf.data.AUTOTUNE)

    # Try to load existing model if save_path provided
    gan = None
    if save_path:
        gan, loaded_meta = _load_cgan_model(save_path, verbose)
        if gan is not None and loaded_meta is not None:
            # Verify dimensions match
            if (
                loaded_meta["seq_len"] == seq_len
                and loaded_meta["num_features"] == num_features
                and loaded_meta["num_classes"] == num_classes
            ):
                if verbose:
                    print(f"    Using existing CGAN model from {save_path}")
            else:
                if verbose:
                    print("    Saved model dimensions don't match; training new model")
                gan = None

    # Create new model if not loaded
    model_was_loaded = gan is not None
    if gan is None:
        gan = DFCGAN(
            seq_len=seq_len,
            num_features=num_features,
            num_classes=num_classes,
            latent_dim=64,
            d_steps=d_steps,
            instance_noise_std=instance_noise_std,
            label_smoothing=label_smoothing,
            fm_weight=fm_weight,
            fm_var_weight=fm_var_weight,
            mmd_weight=mmd_weight,
            feature_mean=real_feat_mean,
            feature_std=real_feat_std + 1e-8,
            feature_min=real_feat_min,
            feature_max=real_feat_max,
        )
        gan.compile(
            d_optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate, clipnorm=1.0),
            g_optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate, clipnorm=1.0),
            loss_fn=tf.keras.losses.BinaryCrossentropy(from_logits=True),
        )

    class _LossLogger(Callback):
        def on_epoch_end(self, epoch, logs=None):
            if logs is None:
                logs = {}
            g = logs.get("g_loss")
            d = logs.get("d_loss")
            if g is not None and d is not None:
                print(f"      [CGAN] epoch {epoch+1}: g_loss={float(g):.4f} d_loss={float(d):.4f}")

    class _EarlyStopOnCollapse(Callback):
        def __init__(self, min_d_loss=0.1, patience=5):
            super().__init__()
            self.min_d_loss = min_d_loss
            self.patience = patience
            self.counter = 0

        def on_epoch_end(self, _epoch, logs=None):
            if logs is None:
                logs = {}
            d = logs.get("d_loss")
            if d is None:
                return
            if float(d) < self.min_d_loss:
                self.counter += 1
            else:
                self.counter = 0
            if self.counter >= self.patience:
                print(f"      [CGAN] Early stopping: d_loss<{self.min_d_loss} for {self.patience} epochs")
                self.model.stop_training = True

    callbacks = []
    if verbose:
        callbacks.append(_LossLogger())
        callbacks.append(_EarlyStopOnCollapse(min_d_loss=0.1, patience=5))

    # Dual-optimizer ReduceLROnPlateau
    if lr_plateau_enable:
        class _DualReduceLROnPlateau(Callback):
            def __init__(self, d_opt, g_opt, monitor, factor, patience, min_lr, cooldown, mode):
                super().__init__()
                self.d_opt = d_opt
                self.g_opt = g_opt
                self.monitor = monitor
                self.factor = factor
                self.patience = patience
                self.min_lr = min_lr
                self.cooldown = cooldown
                self.mode = mode
                self.best = None
                self.wait = 0
                self.cooldown_counter = 0

            def _get_lr(self, opt):
                lr = getattr(opt, "learning_rate", None)
                try:
                    return float(lr.numpy())
                except Exception:
                    try:
                        return float(lr)
                    except Exception:
                        return None

            def _set_lr(self, opt, value):
                lr_attr = getattr(opt, "learning_rate", None)
                try:
                    if hasattr(lr_attr, "assign"):
                        lr_attr.assign(value)
                    else:
                        setattr(opt, "learning_rate", value)
                except Exception:
                    try:
                        setattr(opt, "lr", value)
                    except Exception:
                        pass

            def _is_improvement(self, current, best):
                if best is None:
                    return True
                return (current < best) if self.mode == "min" else (current > best)

            def on_epoch_end(self, _epoch, logs=None):
                if logs is None:
                    return
                current = logs.get(self.monitor)
                if current is None or not float(current) == float(current):  # NaN guard
                    return
                if self.cooldown_counter > 0:
                    self.cooldown_counter -= 1
                if self._is_improvement(float(current), self.best):
                    self.best = float(current)
                    self.wait = 0
                else:
                    if self.cooldown_counter == 0:
                        self.wait += 1
                        if self.wait >= self.patience:
                            # reduce both lrs
                            for opt in (self.d_opt, self.g_opt):
                                lr = self._get_lr(opt)
                                if lr is None:
                                    continue
                                new_lr = max(self.min_lr, lr * self.factor)
                                self._set_lr(opt, new_lr)
                            self.cooldown_counter = self.cooldown
                            self.wait = 0
                            d_lr = self._get_lr(self.d_opt)
                            g_lr = self._get_lr(self.g_opt)
                            if d_lr is not None and g_lr is not None:
                                print(f"      [CGAN] LR plateau: reducing lrs to d={d_lr:.6g} g={g_lr:.6g}")

        callbacks.append(
            _DualReduceLROnPlateau(
                gan.d_optimizer,
                gan.g_optimizer,
                monitor=lr_monitor,
                factor=lr_factor,
                patience=lr_patience,
                min_lr=lr_min,
                cooldown=lr_cooldown,
                mode=lr_mode,
            )
        )

    # Fidelity reporter: compare mean/std between real and generated mini-batches
    if verbose:
        class _FidelityReport(Callback):
            def __init__(self, real_x_all, real_y_all, gan_ref, latent_dim_ref, num_classes_ref):
                super().__init__()
                self.real_x = real_x_all
                self.real_y = real_y_all
                self.gan = gan_ref
                self.latent_dim = latent_dim_ref
                self.num_classes = num_classes_ref

            def on_epoch_end(self, _epoch, _logs=None):
                # sample small real batch
                n = min(256, len(self.real_x))
                if n == 0:
                    return
                idx = np.random.choice(len(self.real_x), size=n, replace=False)
                real_batch = self.real_x[idx]
                # sample class labels from empirical distribution
                priors = self.real_y.mean(axis=0)
                gen_classes = np.random.choice(self.num_classes, size=n, p=priors)
                one_hot = np.zeros((n, self.num_classes), dtype=np.float32)
                one_hot[np.arange(n), gen_classes] = 1.0
                fake_batch = self.gan.generate(n, one_hot)
                # compute mean/std RMSE across features
                r_mean = real_batch.mean(axis=(0, 1))
                f_mean = fake_batch.mean(axis=(0, 1))
                r_std = real_batch.std(axis=(0, 1)) + 1e-8
                f_std = fake_batch.std(axis=(0, 1)) + 1e-8
                mean_rmse = float(np.sqrt(np.mean((r_mean - f_mean) ** 2)))
                std_rmse = float(np.sqrt(np.mean((r_std - f_std) ** 2)))
                print(f"      [CGAN] fidelity: mean_RMSE={mean_rmse:.4f} std_RMSE={std_rmse:.4f}")

        # Attach with access to real data for fidelity comparison
        callbacks.append(_FidelityReport(train_data.astype("float32"), train_labels.astype("float32"), gan, gan.latent_dim, num_classes))

    # Train only if model wasn't loaded
    if not model_was_loaded:
        if steps_per_epoch is not None and steps_per_epoch > 0:
            gan.fit(ds.take(steps_per_epoch), epochs=max(1, epochs), verbose=0, callbacks=callbacks)
        else:
            gan.fit(ds, epochs=max(1, epochs), verbose=0, callbacks=callbacks)
        
        # Save model if save_path provided
        if save_path:
            metadata = {
                "seq_len": seq_len,
                "num_features": num_features,
                "num_classes": num_classes,
                "latent_dim": 64,
                "d_steps": d_steps,
                "instance_noise_std": instance_noise_std,
                "label_smoothing": label_smoothing,
                "fm_weight": fm_weight,
                "fm_var_weight": fm_var_weight,
                "mmd_weight": mmd_weight,
                "learning_rate": learning_rate,
                "feature_mean": real_feat_mean,
                "feature_std": real_feat_std + 1e-8,
                "feature_min": real_feat_min,
                "feature_max": real_feat_max,
                "generator_arch": "cnn" if isinstance(gan.generator, _DFGeneratorCNN) else "mlp",
                "gen_base_filters": getattr(gan.generator, "base_filters", 128),
                "gen_kernel_size": getattr(gan.generator, "kernel_size", 3),
                "gen_upsample_blocks": getattr(gan.generator, "num_upsample_blocks", 2),
            }
            _save_cgan_model(gan, metadata, save_path, verbose)

    # Compute class counts
    idx = _one_hot_to_indices(train_labels)
    unique, counts = np.unique(idx, return_counts=True)
    class_counts = {int(k): int(v) for k, v in zip(unique, counts)}
    current_max = int(np.max(counts)) if counts.size > 0 else 0
    # Respect caller-provided max_target even if below current majority
    target = max_target if (max_target is not None) else current_max

    gen_data_list = [train_data]
    gen_label_list = [train_labels]

    # Generate additional samples per minority class
    for c in range(num_classes):
        have = class_counts.get(c, 0)
        need = target - have
        if need <= 0:
            continue

        # Create one-hot labels for the generated batch
        one_hot = np.zeros((need, num_classes), dtype=np.float32)
        one_hot[:, c] = 1.0
        synth = gan.generate(need, one_hot)
        gen_data_list.append(synth)
        gen_label_list.append(one_hot)

    aug_data = np.concatenate(gen_data_list, axis=0)
    aug_labels = np.concatenate(gen_label_list, axis=0)

    # Shuffle to mix real and synthetic
    perm = np.random.permutation(len(aug_data))
    aug_data = aug_data[perm]
    aug_labels = aug_labels[perm]

    return aug_data, aug_labels


