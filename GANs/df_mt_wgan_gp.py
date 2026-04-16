"""
WGAN-GP for conditional sequence data with multi-task labels (batch, seq_len, num_features).

API: balance_with_mt_wgan_gp(train_data, train_labels, epochs=100, batch_size=256, max_target=None, verbose=True)
Returns augmented (data, labels) where minority classes are upsampled to max_target per class.

Notes:
- train_labels must be a dictionary of one-hot encoded arrays: {"trading": (batch, 3), "regime": (batch, 3), ...}
- All tasks must have the same batch size
- Data should be pre-scaled; generator output is tanh-bounded and rescaled to real mean/std then clipped to real min/max
- Balancing is done based on the primary_task (default: "trading")
"""

from __future__ import annotations

import numpy as np
import tensorflow as tf
from keras import layers, Model, Layer
import os
import pickle


# Global cache for loaded GAN models/metadata to avoid redundant reloads during multi-pair strategy runs
_GAN_CACHE = {}
_META_CACHE = {}


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


class _MTGenCNN(Model):
    def __init__(self, seq_len: int, num_features: int, task_label_dims: dict[str, int], latent_dim: int = 64, base_filters: int = 128, kernel_size: int = 3, up_blocks: int = 2):
        super().__init__(name="mt_wgangp_gen_cnn")
        self.seq_len = seq_len
        self.num_features = num_features
        self.task_label_dims = task_label_dims  # {"trading": 3, "regime": 3, ...}
        self.latent_dim = latent_dim
        self.base_filters = base_filters
        self.kernel_size = kernel_size
        self.up_blocks = up_blocks
        
        # Total condition dimension is sum of all task label dimensions
        self.total_cond_dim = sum(task_label_dims.values())

        z_in = layers.Input(shape=(latent_dim,))
        # Condition input: concatenated one-hot encodings from all tasks
        c_in = layers.Input(shape=(self.total_cond_dim,))

        effective_up_blocks = up_blocks if seq_len > 1 else 0
        start_len = max(1, seq_len // (2 ** effective_up_blocks))
        channels = base_filters

        if effective_up_blocks == 0:
            # seq_len=1 path: use a deeper MLP with LayerNorm so the conditioning
            # signal can compete with the high-dimensional noise z.
            # Mirrors the architecture of _GenMLP in df_wgan_gp.py.
            #
            # Why: with seq_len=1 the FiLM blocks (which provide strong conditioning
            # for seq_len>1) are skipped entirely.  A simple Dense(z+c)→Conv1D
            # with 5 conditioning dims out of 69 total inputs causes c to be
            # drowned by z.  A dedicated c embedding + deeper net + LayerNorm
            # (not BatchNorm) gives the conditioning signal room to grow.
            c_embed = layers.Dense(latent_dim // 2, use_bias=False)(c_in)
            c_embed = layers.LayerNormalization()(c_embed)

            x = layers.Concatenate()([z_in, c_embed])
            x = layers.Dense(channels)(x)
            x = layers.LayerNormalization()(x)
            x = layers.LeakyReLU(0.2)(x)

            x = layers.Dense(channels // 2)(x)
            x = layers.LayerNormalization()(x)
            x = layers.LeakyReLU(0.2)(x)

            # Final projection: Dense → tanh directly (no BN/ReLU before tanh)
            h = layers.Dense(num_features, activation="tanh")(x)
            out = layers.Reshape((seq_len, num_features))(h)
        else:
            # seq_len>1 path: original CNN with FiLM conditioning
            x = layers.Concatenate()([z_in, c_in])
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
                h = layers.Conv1DTranspose(filters=channels, kernel_size=kernel_size, strides=2, padding="same")(h)
                h = layers.LeakyReLU(0.2)(h)
                h = film(h, c_in, channels)
                channels = max(channels // 2, 32)

            h = layers.Conv1D(num_features, kernel_size=kernel_size, padding="same", activation="tanh")(h)
            out = _ResizeToLenLayer(target_len=seq_len)(h)

        self.model = Model([z_in, c_in], out, name="mt_wgangp_gen_cnn")

    def call(self, inputs, training=False):
        z, c = inputs
        return self.model([z, c], training=training)


class _MTCritic(Model):
    def __init__(self, seq_len: int, num_features: int, task_label_dims: dict[str, int], base_filters: int = 128, kernel_size: int = 3):
        super().__init__(name="mt_wgangp_critic")
        self.task_label_dims = task_label_dims
        self.total_cond_dim = sum(task_label_dims.values())
        self.sorted_tasks = sorted(list(task_label_dims.keys()))
        
        xin = layers.Input(shape=(seq_len, num_features))
        cin = layers.Input(shape=(self.total_cond_dim,))
        c_rep = layers.RepeatVector(seq_len)(cin)
        x = layers.Concatenate()([xin, c_rep])
        h = layers.Conv1D(base_filters, kernel_size, padding="same")(x)
        h = layers.LeakyReLU(0.2)(h)
        h = layers.Conv1D(base_filters * 2, kernel_size, strides=2, padding="same")(h)
        h = layers.LeakyReLU(0.2)(h)
        h = layers.GlobalAveragePooling1D()(h)
        # Minibatch stddev removed - can cause instability and WGAN-GP gradient penalty already prevents mode collapse
        # sfeat = _MinibatchStdLayer()(h)
        # trunk = layers.Concatenate()([h, sfeat])
        trunk = h
        trunk = layers.Dense(128)(trunk)
        trunk = layers.LeakyReLU(0.2)(trunk)

        # Adversarial head
        adv_out = layers.Dense(1, name="adv_head")(trunk)

        # Task heads
        task_outputs = {}
        for task in self.sorted_tasks:
            n_out = task_label_dims[task]
            th = layers.Dense(64)(trunk)
            th = layers.LeakyReLU(0.2)(th)
            task_outputs[task] = layers.Dense(n_out, activation="softmax", name=f"{task}_head")(th)

        outputs = {"adv": adv_out, **task_outputs}
        self.model = Model([xin, cin], outputs, name="mt_wgangp_critic")

    def call(self, inputs, training=False):
        x, c = inputs
        return self.model([x, c], training=training)


class MT_WGAN_GP(Model):
    def __init__(
        self,
        seq_len: int,
        num_features: int,
        task_label_dims: dict[str, int],
        latent_dim: int = 64,
        gp_weight: float = 10.0,
        n_critic: int = 5,
        gen_base_filters: int = 128,
        gen_kernel_size: int = 3,
        gen_up_blocks: int = 2,
        primary_task: str = "trading",
        class_means: np.ndarray | None = None,
        cond_loss_weight: float = 2.0,
    ):
        super().__init__(name="mt_wgan_gp")
        self.seq_len = seq_len
        self.num_features = num_features
        self.task_label_dims = task_label_dims
        self.latent_dim = latent_dim
        self.gp_weight = gp_weight
        self.n_critic = n_critic
        self.total_cond_dim = sum(task_label_dims.values())
        self.primary_task = primary_task
        # class_means: (num_classes, num_features) per-class feature means for the primary task,
        # in real-data space.  Used for the class mean matching loss (see train_step).
        self.class_means = (
            tf.constant(class_means, dtype=tf.float32) if class_means is not None else None
        )
        self.cond_loss_weight = cond_loss_weight
        self.gen = _MTGenCNN(seq_len, num_features, task_label_dims, latent_dim, gen_base_filters, gen_kernel_size, gen_up_blocks)
        self.critic = _MTCritic(seq_len, num_features, task_label_dims)

        self.gen_loss_tracker = tf.keras.metrics.Mean(name="g_loss")
        self.crit_loss_tracker = tf.keras.metrics.Mean(name="d_loss")
        self.mean_real_tracker = tf.keras.metrics.Mean(name="d_real")
        self.mean_fake_tracker = tf.keras.metrics.Mean(name="d_fake")
        self.gp_tracker = tf.keras.metrics.Mean(name="gp")

        # Aux classification
        self.task_loss_weights: dict[str, float] = {}
        self.aux_loss_fn = tf.keras.losses.CategoricalCrossentropy()

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

    def compile(self, g_optimizer, d_optimizer, task_loss_weights: dict[str, float] | None = None):
        super().compile()
        self.g_optimizer = g_optimizer
        self.d_optimizer = d_optimizer
        # default weights: trading=0.1, others=0.02 (reduced to prevent auxiliary losses from dominating)
        # Auxiliary losses should be secondary to the adversarial loss for stable training
        if task_loss_weights is None:
            weights = {}
            for k in self.task_label_dims.keys():
                weights[k] = 0.1 if k == "trading" else 0.02
            self.task_loss_weights = weights
        else:
            self.task_loss_weights = dict(task_loss_weights)

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
            pred_adv = self.critic([interpolated, c], training=True)["adv"]
        grads = gp_tape.gradient(pred_adv, [interpolated])[0]
        # Guard against NaN gradients
        grads = tf.where(tf.math.is_finite(grads), grads, tf.zeros_like(grads))
        grads = tf.reshape(grads, (batch_size, -1))
        norm = tf.sqrt(tf.reduce_sum(tf.square(grads), axis=1) + 1e-8)
        gp = tf.reduce_mean((norm - 1.0) ** 2)
        # Guard against NaN in gradient penalty
        gp = tf.where(tf.math.is_finite(gp), gp, tf.zeros_like(gp))
        return gp

    def train_step(self, data):
        # Expect (real_x, cond_vector, labels_dict)
        real_x, c, labels = data
        batch_size = tf.shape(real_x)[0]

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
                real_pred = self.critic([real_x, c], training=True)
                fake_pred = self.critic([fake_x, c], training=True)
                mean_fake = tf.reduce_mean(real_pred["adv"] * 0.0 + fake_pred["adv"])  # ensure tensor type
                mean_real = tf.reduce_mean(real_pred["adv"])
                wasserstein = mean_fake - mean_real
                # Guard against NaN in Wasserstein distance
                wasserstein = tf.where(tf.math.is_finite(wasserstein), wasserstein, tf.zeros_like(wasserstein))
                gp = self.gradient_penalty(real_x, fake_x, c)
                # Ensure consistent dtype for loss computation
                wasserstein_dtype = wasserstein.dtype
                gp = tf.cast(gp, wasserstein_dtype)
                gp_weight = tf.cast(self.gp_weight, wasserstein_dtype)
                # Auxiliary classification losses on real and fake
                aux_loss = 0.0
                for task, w in self.task_loss_weights.items():
                    y_real = tf.cast(labels[task], dtype=tf.float32)
                    y_fake = y_real  # by default use same conditioning; labels fed via c
                    p_real = tf.cast(real_pred[task], dtype=tf.float32)
                    p_fake = tf.cast(fake_pred[task], dtype=tf.float32)
                    # Guard against NaN in predictions before computing loss
                    p_real = tf.where(tf.math.is_finite(p_real), p_real, tf.zeros_like(p_real))
                    p_fake = tf.where(tf.math.is_finite(p_fake), p_fake, tf.zeros_like(p_fake))
                    aux_real = self.aux_loss_fn(y_real, p_real)
                    aux_fake = self.aux_loss_fn(y_fake, p_fake)
                    # Guard against NaN in auxiliary loss
                    aux_real = tf.where(tf.math.is_finite(aux_real), aux_real, tf.zeros_like(aux_real))
                    aux_fake = tf.where(tf.math.is_finite(aux_fake), aux_fake, tf.zeros_like(aux_fake))
                    aux_loss += w * (aux_real + aux_fake)
                aux_loss = tf.where(tf.math.is_finite(aux_loss), aux_loss, tf.zeros_like(aux_loss))
                # Clip auxiliary loss to prevent it from dominating (limit to reasonable range)
                aux_loss = tf.clip_by_value(aux_loss, 0.0, 10.0)
                d_loss = wasserstein + gp_weight * gp + tf.cast(aux_loss, wasserstein_dtype)
                # Final guard against NaN in total loss and clip extreme values
                d_loss = tf.where(tf.math.is_finite(d_loss), d_loss, tf.zeros_like(d_loss))
                # Clip total loss to prevent explosion (though this shouldn't happen with proper training)
                d_loss = tf.clip_by_value(d_loss, -100.0, 100.0)
            grads = tape.gradient(d_loss, self.critic.trainable_variables)
            self.d_optimizer.apply_gradients(zip(grads, self.critic.trainable_variables))
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
            fake_x_raw = self.gen([z, c], training=True)   # tanh space: [-1, 1]
            fake_x = self._postprocess(fake_x_raw)         # real data space
            fake_pred = self.critic([fake_x, c], training=True)
            adv_loss = -tf.reduce_mean(fake_pred["adv"])  # generator tries to maximize critic score
            adv_loss = tf.cast(adv_loss, dtype=tf.float32)  # Cast to float32 for loss computation
            # Guard against NaN in adversarial loss
            adv_loss = tf.where(tf.math.is_finite(adv_loss), adv_loss, tf.zeros_like(adv_loss))
            aux_g_loss = 0.0
            for task, w in self.task_loss_weights.items():
                y_fake = tf.cast(labels[task], dtype=tf.float32)
                p_fake = tf.cast(fake_pred[task], dtype=tf.float32)
                # Guard against NaN in predictions before computing loss
                p_fake = tf.where(tf.math.is_finite(p_fake), p_fake, tf.zeros_like(p_fake))
                aux_task = self.aux_loss_fn(y_fake, p_fake)
                # Guard against NaN in auxiliary loss
                aux_task = tf.where(tf.math.is_finite(aux_task), aux_task, tf.zeros_like(aux_task))
                aux_g_loss += w * aux_task
            aux_g_loss = tf.where(tf.math.is_finite(aux_g_loss), aux_g_loss, tf.zeros_like(aux_g_loss))
            # Clip auxiliary loss to prevent it from dominating
            aux_g_loss = tf.clip_by_value(aux_g_loss, 0.0, 10.0)
            g_loss = adv_loss + aux_g_loss

            # Class mean matching loss for the primary task (same logic as df_wgan_gp.py).
            # Operates in tanh space so targets are always reachable; see df_wgan_gp.py for details.
            if (
                self.class_means is not None
                and self.feature_mean is not None
                and self.feature_std is not None
                and self.primary_task in labels
            ):
                dtype = fake_x_raw.dtype
                feat_mean = tf.reshape(tf.cast(self.feature_mean, dtype), (1, 1, -1))
                feat_std  = tf.reshape(tf.cast(self.feature_std,  dtype), (1, 1, -1))
                class_means_norm = (tf.cast(self.class_means, dtype) - feat_mean[0]) / feat_std[0]
                class_means_norm = tf.clip_by_value(class_means_norm, -1.0, 1.0)
                c_task = tf.cast(labels[self.primary_task], dtype)      # (batch, num_classes)
                target = tf.matmul(c_task, class_means_norm)            # (batch, num_features)
                target = tf.expand_dims(target, 1)                      # (batch, 1, num_features)
                fake_means = tf.reduce_mean(fake_x_raw, axis=1, keepdims=True)
                cond_loss = tf.reduce_mean(tf.square(fake_means - target))
                cond_loss = tf.where(tf.math.is_finite(cond_loss), cond_loss, tf.zeros_like(cond_loss))
                g_loss = g_loss + tf.cast(self.cond_loss_weight, g_loss.dtype) * cond_loss

            # Final guard against NaN in total loss and clip extreme values
            g_loss = tf.where(tf.math.is_finite(g_loss), g_loss, tf.zeros_like(g_loss))
            # Clip total loss to prevent explosion
            g_loss = tf.clip_by_value(g_loss, -100.0, 100.0)
        grads = tape.gradient(g_loss, self.gen.trainable_variables)
        self.g_optimizer.apply_gradients(zip(grads, self.gen.trainable_variables))

        self.gen_loss_tracker.update_state(g_loss)
        self.crit_loss_tracker.update_state(crit_loss)
        self.mean_real_tracker.update_state(d_real_mean)
        self.mean_fake_tracker.update_state(d_fake_mean)
        self.gp_tracker.update_state(gp_mean)
        return {"g_loss": self.gen_loss_tracker.result(), "d_loss": self.crit_loss_tracker.result()}

    def generate(self, n: int, task_labels: dict[str, np.ndarray]) -> tuple[np.ndarray, dict[str, np.ndarray]]:
        """
        Generate synthetic samples with specified task labels.
        
        Args:
            n: Number of samples to generate
            task_labels: Dictionary of one-hot encoded labels for each task
                        e.g., {"trading": (n, 3), "regime": (n, 3), ...}
        
        Returns:
            (synthetic_data, task_labels_dict)
        """
        z = tf.random.normal((n, self.latent_dim))
        # Concatenate all task labels into condition vector
        cond_list = [tf.convert_to_tensor(task_labels[task], dtype=tf.float32) for task in sorted(task_labels.keys())]
        c = tf.concat(cond_list, axis=1)
        x = self.gen([z, c], training=False)
        x = self._postprocess(x)
        return x.numpy(), task_labels


def _concatenate_task_labels(task_labels_dict: dict[str, np.ndarray]) -> np.ndarray:
    """Concatenate all task labels into a single condition vector"""
    # Sort keys for consistent ordering
    sorted_keys = sorted(task_labels_dict.keys())
    return np.concatenate([task_labels_dict[k] for k in sorted_keys], axis=1)


def _save_mt_wgan_model(gan: MT_WGAN_GP, metadata: dict, save_path: str, verbose: bool = False):
    """Save MT WGAN-GP model and metadata"""
    os.makedirs(save_path, exist_ok=True)
    model_path = os.path.join(save_path, "mt_wgangp_model.weights.h5")
    meta_path = os.path.join(save_path, "mt_wgangp_metadata.pkl")
    # Build sub-models and mark parent as built
    dummy_z = tf.zeros((1, gan.latent_dim))
    dummy_c = tf.zeros((1, gan.total_cond_dim))
    dummy_x = tf.zeros((1, gan.seq_len, gan.num_features))
    _ = gan.gen([dummy_z, dummy_c], training=False)
    _ = gan.critic([dummy_x, dummy_c], training=False)
    gan.built = True  # Explicitly mark parent model as built
    gan.save_weights(model_path)
    with open(meta_path, "wb") as f:
        pickle.dump(metadata, f)
    if verbose:
        print(f"    MT WGAN-GP model saved to {save_path}")


def _load_mt_wgan_model(save_path: str, verbose: bool = False) -> tuple[MT_WGAN_GP | None, dict | None]:
    """Load MT WGAN-GP model and metadata if they exist"""
    model_path = os.path.join(save_path, "mt_wgangp_model.weights.h5")
    meta_path = os.path.join(save_path, "mt_wgangp_metadata.pkl")
    if not (os.path.exists(model_path) and os.path.exists(meta_path)):
        return None, None
    if save_path in _GAN_CACHE:
        return _GAN_CACHE[save_path]

    try:
        with open(meta_path, "rb") as f:
            metadata = pickle.load(f)
        gan = MT_WGAN_GP(
            seq_len=metadata["seq_len"],
            num_features=metadata["num_features"],
            task_label_dims=metadata["task_label_dims"],
            latent_dim=metadata.get("latent_dim", 64),
            gp_weight=metadata.get("gp_weight", 10.0),
            n_critic=metadata.get("n_critic", 5),
            gen_base_filters=metadata.get("gen_base_filters", 128),
            gen_kernel_size=metadata.get("gen_kernel_size", 3),
            gen_up_blocks=metadata.get("gen_up_blocks", 2),
        )
        gan.feature_mean = tf.convert_to_tensor(metadata.get("feature_mean"), dtype=tf.float32)
        gan.feature_std = tf.convert_to_tensor(metadata.get("feature_std"), dtype=tf.float32)
        gan.feature_min = tf.convert_to_tensor(metadata.get("feature_min"), dtype=tf.float32)
        gan.feature_max = tf.convert_to_tensor(metadata.get("feature_max"), dtype=tf.float32)
        gan.compile(
            # Use same learning rate for both, but generator has tighter gradient clipping
            g_optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4, beta_1=0.5, beta_2=0.9, clipnorm=0.5),
            d_optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4, beta_1=0.5, beta_2=0.9, clipnorm=1.0),
            task_loss_weights=metadata.get("task_loss_weights", None),
        )
        # Ensure model is built before loading weights by calling sub-models
        dummy_z = tf.zeros((1, gan.latent_dim))
        dummy_c = tf.zeros((1, gan.total_cond_dim))
        dummy_x = tf.zeros((1, gan.seq_len, gan.num_features))
        # Build generator and critic
        _ = gan.gen([dummy_z, dummy_c], training=False)
        _ = gan.critic([dummy_x, dummy_c], training=False)
        gan.built = True  # Explicitly mark parent model as built
        gan.load_weights(model_path)
        if verbose:
            print(f"    MT WGAN-GP model loaded from {save_path}")
        
        _GAN_CACHE[save_path] = (gan, metadata)
        return gan, metadata
    except Exception as e:
        if verbose:
            print(f"    Failed to load MT WGAN-GP model: {e}")
        return None, None


def load_mt_wgan_thresholds(save_path: str) -> dict[str, float | int | None]:
    """
    Load threshold values and training_type from MT WGAN-GP metadata.
    
    Returns:
        Dictionary with 'min_buy_gain_threshold', 'min_sell_loss_threshold', and 'training_type' if present in metadata
    """
    if save_path in _META_CACHE:
        return _META_CACHE[save_path]

    meta_path = os.path.join(save_path, "mt_wgangp_metadata.pkl")
    if not os.path.exists(meta_path):
        return {"min_buy_gain_threshold": None, "min_sell_loss_threshold": None, "training_type": None}
    try:
        with open(meta_path, "rb") as f:
            metadata = pickle.load(f)
        
        result = {
            "min_buy_gain_threshold": metadata.get("min_buy_gain_threshold"),
            "min_sell_loss_threshold": metadata.get("min_sell_loss_threshold"),
            "training_type": metadata.get("training_type"),
        }
        _META_CACHE[save_path] = result
        return result
    except Exception:
        return {"min_buy_gain_threshold": None, "min_sell_loss_threshold": None, "training_type": None}


def balance_with_mt_wgan_gp(
    train_data: np.ndarray,
    train_labels: dict[str, np.ndarray],
    epochs: int = 100,
    batch_size: int = 256,
    max_target: int | None = None,
    verbose: bool = True,
    save_path: str | None = None,
    assess_quality: bool = True,
    n_critic: int = 5,
    primary_task: str = "trading",
    task_target_ratios: dict[str, float | dict[int, float] | None] | None = None,
    task_loss_weights: dict[str, float] | None = None,  # Default: trading=0.1, others=0.02
    seq_len: int | None = None,
    min_buy_gain_threshold: float | None = None,
    min_sell_loss_threshold: float | None = None,
    training_type: int | None = None,
    _return_model: bool = False,  # When True, return (aug_x, aug_y, trained_gan) instead of (aug_x, aug_y)
):
    """
    Balance training data using WGAN-GP with multi-task labels.
    
    Args:
        train_data: (batch, seq_len, num_features) numpy array
        train_labels: Dictionary of one-hot encoded labels: {"trading": (batch, 3), "regime": (batch, 3), ...}
        epochs: Training epochs
        batch_size: Batch size
        max_target: Target count per class (for primary_task, used if task_target_ratios not provided). 
                   If None, uses majority class count of primary_task
        verbose: Print progress
        save_path: Path to save/load model
        assess_quality: Whether to assess generation quality
        n_critic: Number of critic steps per generator step
        primary_task: Task to use for determining class balance if task_target_ratios not provided (default: "trading")
        task_target_ratios: Dictionary mapping task names to target ratios (0.0-1.0) relative to majority class.
                          Can be a single float (applies to all classes) or dict[int, float] (per-class ratios).
                          Target = majority_class_count * ratio.
                          If None for a task, that task is not balanced.
                          If None for entire dict, uses primary_task and max_target/wgan_target_ratio.
                          Example: {"trading": 0.5, "profit": {0: 0.6, 1: 0.4, 2: 0.5}, "regime": None}
        task_loss_weights: Dictionary of loss weights for each task in auxiliary classification
    
    Returns:
        (augmented_data, augmented_labels_dict)
    """
    original_2d = train_data.ndim == 2
    base_train_data = None
    if original_2d:
        batch_size_data, num_features = train_data.shape
        base_train_data = train_data[:, np.newaxis, :]
        train_data = base_train_data
        seq_len_value = 1
    else:
        assert train_data.ndim == 3
        batch_size_data, seq_len_value, num_features = train_data.shape
        if seq_len is not None and seq_len != seq_len_value:
            if verbose:
                print("    Provided seq_len differs from data; using data sequence length")

    # Validate all labels have same batch size
    batch_sizes = {task: labels.shape[0] for task, labels in train_labels.items()}
    if len(set(batch_sizes.values())) > 1:
        raise ValueError(f"All task labels must have the same batch size. Found: {batch_sizes}")
    if batch_sizes[list(train_labels.keys())[0]] != batch_size_data:
        raise ValueError(
            f"train_data batch size ({batch_size_data}) doesn't match labels batch size ({batch_sizes[list(train_labels.keys())[0]]})"
        )
    
    # Get task label dimensions
    task_label_dims = {task: labels.shape[1] for task, labels in train_labels.items()}
    
    # Validate primary_task if task_target_ratios not provided
    if task_target_ratios is None:
        if primary_task not in train_labels:
            raise ValueError(f"primary_task '{primary_task}' not found in train_labels. Available: {list(train_labels.keys())}")
    
    # Validate task_target_ratios if provided
    if task_target_ratios is not None:
        for task in task_target_ratios.keys():
            if task is not None and task not in train_labels:
                raise ValueError(f"task '{task}' in task_target_ratios not found in train_labels. Available: {list(train_labels.keys())}")
    
    gan = None
    loaded_meta = None
    if save_path:
        gan, loaded_meta = _load_mt_wgan_model(save_path, verbose)
        if gan is not None and loaded_meta is not None:
            saved_seq_len = loaded_meta.get("seq_len", 1)
            saved_features = loaded_meta.get("num_features")
            saved_dims = loaded_meta.get("task_label_dims")
            compatible = (
                saved_seq_len == seq_len_value
                and saved_features == num_features
                and saved_dims == task_label_dims
            )
            if not compatible:
                if verbose:
                    print("    Saved model dimensions don't match; training new model")
                gan = None
                if original_2d and base_train_data is not None:
                    train_data = base_train_data
                elif original_2d:
                    train_data = train_data[:, :1, :]
                seq_len_value = 1

    if seq_len_value is None:
        seq_len_value = loaded_meta["seq_len"] if loaded_meta else 1

    n, current_seq_len, _ = train_data.shape
    if current_seq_len != seq_len_value:
        if original_2d and base_train_data is not None:
            if seq_len_value > 1:
                train_data = np.repeat(base_train_data, seq_len_value, axis=1)
            else:
                train_data = base_train_data
        else:
            if current_seq_len > seq_len_value:
                train_data = train_data[:, :seq_len_value, :]
            else:
                repeats = seq_len_value // current_seq_len
                remainder = seq_len_value % current_seq_len
                train_data = np.repeat(train_data, repeats, axis=1)
                if remainder:
                    train_data = np.concatenate(
                        [train_data, train_data[:, :remainder, :]], axis=1
                    )
        n, current_seq_len, _ = train_data.shape

    data_f32 = train_data.astype("float32")
    labels_f32 = {task: labels.astype("float32") for task, labels in train_labels.items()}
    
    # Create concatenated condition vectors
    cond_vectors = _concatenate_task_labels(labels_f32)
    
    # Package labels dict into tensors for dataset
    labels_tensors = {task: tf.convert_to_tensor(arr, dtype=tf.float32) for task, arr in labels_f32.items()}
    # Create dataset yielding (x, cond, labels_dict)
    ds = (
        tf.data.Dataset.from_tensor_slices((data_f32, cond_vectors, labels_tensors))
        .shuffle(min(8192, n))
        .batch(batch_size)
        .prefetch(tf.data.AUTOTUNE)
    )
    
    # Stats for postprocess
    feat_mean = data_f32.mean(axis=(0, 1)).astype("float32")
    feat_std = data_f32.std(axis=(0, 1)).astype("float32") + 1e-8
    feat_min = data_f32.min(axis=(0, 1)).astype("float32")
    feat_max = data_f32.max(axis=(0, 1)).astype("float32")

    # Compute per-class feature means for the primary task (real-data space).
    # Used by the class mean matching loss to enforce conditioning (same approach as df_wgan_gp.py).
    primary_labels = labels_f32.get(primary_task)
    if primary_labels is not None:
        primary_idx = np.argmax(primary_labels, axis=1)
        num_primary_classes = task_label_dims[primary_task]
        class_means_arr = np.zeros((num_primary_classes, num_features), dtype=np.float32)
        for ci in range(num_primary_classes):
            mask = primary_idx == ci
            if mask.any():
                class_means_arr[ci] = data_f32[mask].mean(axis=(0, 1))
    else:
        class_means_arr = None

    # Create new model if not loaded
    model_was_loaded = gan is not None
    if gan is None:
        gan = MT_WGAN_GP(
            seq_len_value,
            num_features,
            task_label_dims,
            latent_dim=64,
            gp_weight=10.0,
            n_critic=n_critic,
            primary_task=primary_task,
            class_means=class_means_arr,
            cond_loss_weight=10.0,
        )
        gan.feature_mean = tf.convert_to_tensor(feat_mean)
        gan.feature_std = tf.convert_to_tensor(feat_std)
        gan.feature_min = tf.convert_to_tensor(feat_min)
        gan.feature_max = tf.convert_to_tensor(feat_max)
        
        gan.compile(
            # Use same learning rate for both, but generator has tighter gradient clipping
            g_optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4, beta_1=0.5, beta_2=0.9, clipnorm=0.5),
            d_optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4, beta_1=0.5, beta_2=0.9, clipnorm=1.0),
            task_loss_weights=task_loss_weights,
        )
    
    # Train only if model wasn't loaded
    if not model_was_loaded:
        gan.fit(ds, epochs=max(1, epochs), verbose=1 if verbose else 0)

        # Build metadata (always, so _return_model callers can access it via gan._interface_metadata)
        metadata = {
            "seq_len": seq_len_value,
            "num_features": num_features,
            "task_label_dims": task_label_dims,
            "latent_dim": gan.latent_dim,
            "gp_weight": gan.gp_weight,
            "n_critic": n_critic,
            "gen_base_filters": getattr(gan.gen, "base_filters", 128),
            "gen_kernel_size": getattr(gan.gen, "kernel_size", 3),
            "gen_up_blocks": getattr(gan.gen, "up_blocks", 2),
            "feature_mean": feat_mean,
            "feature_std": feat_std,
            "feature_min": feat_min,
            "feature_max": feat_max,
            "task_loss_weights": dict(gan.task_loss_weights),
        }
        if min_buy_gain_threshold is not None:
            metadata["min_buy_gain_threshold"] = float(min_buy_gain_threshold)
        if min_sell_loss_threshold is not None:
            metadata["min_sell_loss_threshold"] = float(min_sell_loss_threshold)
        if training_type is not None:
            metadata["training_type"] = int(training_type)
        gan._interface_metadata = metadata
        if save_path:
            _save_mt_wgan_model(gan, metadata, save_path, verbose)
    
    # Determine balancing targets for each task using ratios
    if task_target_ratios is None:
        # Fall back to primary_task behavior (using wgan_target_ratio or max_target)
        primary_labels = labels_f32[primary_task]
        idx = np.argmax(primary_labels, axis=1)
        unique, counts = np.unique(idx, return_counts=True)
        class_counts = {int(k): int(v) for k, v in zip(unique, counts)}
        current_max = int(np.max(counts)) if counts.size > 0 else 0
        
        # Use max_target if provided, otherwise use current_max (100% ratio)
        if max_target is not None:
            target = max_target
        else:
            target = current_max
        
        task_target_ratios = {primary_task: target / current_max if current_max > 0 else 1.0}
        if verbose:
            print(f"    Using primary_task '{primary_task}' with target ratio: {task_target_ratios[primary_task]:.3f}")
    
    # For each task, determine what needs to be generated using ratios
    # Structure: {(task, class): need_count}
    generation_needs = {}
    
    for task, ratio_spec in task_target_ratios.items():
        if ratio_spec is None:
            continue  # Skip this task
        if task not in train_labels:
            if verbose:
                print(f"    Warning: task '{task}' not found in labels, skipping")
            continue
        
        task_labels = labels_f32[task]
        task_idx = np.argmax(task_labels, axis=1)
        unique, counts = np.unique(task_idx, return_counts=True)
        class_counts = {int(k): int(v) for k, v in zip(unique, counts)}
        current_max = int(np.max(counts)) if counts.size > 0 else 0
        
        # Determine ratios per class
        if isinstance(ratio_spec, dict):
            # Per-class ratios provided
            class_ratios = ratio_spec
        else:
            # Single ratio applies to all classes
            class_ratios = {c: ratio_spec for c in range(task_label_dims[task])}
        
        if verbose:
            print(f"    Task '{task}': ratios={class_ratios}, current_counts={class_counts}")
        
        for c in range(task_label_dims[task]):
            ratio = class_ratios.get(c, 0.0)
            if ratio <= 0.0:
                continue  # Skip this class
            
            # Calculate target: ratio of majority class
            target = int(current_max * ratio) if current_max > 0 else 0
            have = class_counts.get(c, 0)
            need = target - have
            
            if need > 0:
                if verbose:
                    print(f"      Class {c}: ratio={ratio:.3f}, target={target}, have={have}, need={need}")
                generation_needs[(task, c)] = need
    
    if not generation_needs:
        if verbose:
            print("    No classes need balancing")
        if _return_model:
            return train_data, train_labels, gan
        return train_data, train_labels
    
    gen_x_list = [train_data]
    gen_y_dict = {task: [labels] for task, labels in train_labels.items()}
    
    # Generate samples for each task/class combination
    for (task, class_c), need in generation_needs.items():
        if verbose:
            print(f"    Generating {need} samples for {task} class {class_c}")
        
        # Find real samples that match this task/class
        task_labels = labels_f32[task]
        task_idx = np.argmax(task_labels, axis=1)
        matching_mask = task_idx == class_c
        matching_samples = np.where(matching_mask)[0]
        
        if len(matching_samples) == 0:
            # No real samples of this class, generate with default labels
            task_labels_gen = {}
            for other_task in sorted(train_labels.keys()):
                if other_task == task:
                    one_hot = np.zeros((need, task_label_dims[other_task]), dtype=np.float32)
                    one_hot[:, class_c] = 1.0
                else:
                    # Use most common class from real data
                    other_idx = np.argmax(train_labels[other_task], axis=1)
                    most_common = int(np.bincount(other_idx).argmax())
                    one_hot = np.zeros((need, task_label_dims[other_task]), dtype=np.float32)
                    one_hot[:, most_common] = 1.0
                task_labels_gen[other_task] = one_hot
        else:
            # Sample from real data labels that match this task/class
            sampled_indices = np.random.choice(matching_samples, size=min(need, len(matching_samples)), replace=True)
            # If we need more than available, duplicate
            if need > len(sampled_indices):
                sampled_indices = np.concatenate([
                    sampled_indices,
                    np.random.choice(matching_samples, size=need - len(sampled_indices), replace=True)
                ])
            
            # Copy labels from sampled real data
            task_labels_gen = {t: train_labels[t][sampled_indices].copy() for t in train_labels.keys()}
            # Ensure the target task/class is correct
            task_labels_gen[task] = np.zeros((need, task_label_dims[task]), dtype=np.float32)
            task_labels_gen[task][:, class_c] = 1.0
        
        # Generate synthetic data
        synth, _ = gan.generate(need, task_labels_gen)
        gen_x_list.append(synth)
        
        # Append generated labels
        for t in train_labels.keys():
            gen_y_dict[t].append(task_labels_gen[t])
    
    aug_x = np.concatenate(gen_x_list, axis=0)
    aug_y = {task: np.concatenate(gen_y_dict[task], axis=0) for task in train_labels.keys()}

    # NOTE: do NOT shuffle here. Real samples occupy [0:n_real] and generated samples
    # occupy [n_real:] — callers and tests rely on this layout to extract gen-only slices.
    # Training loops that consume the augmented data handle their own shuffling.

    aug_x_for_quality = aug_x

    if original_2d:
        aug_x = aug_x[:, 0, :]

    # Assess quality if requested
    if assess_quality and len(aug_x_for_quality) > len(train_data):
        if verbose:
            print("\n    Assessing generated sample quality...")
        quality_metrics = assess_mt_generation_quality(
            train_data, train_labels, aug_x_for_quality, aug_y, primary_task=primary_task, verbose=verbose
        )

    # For the loaded-model path, store metadata so _return_model callers can access it
    if model_was_loaded and not hasattr(gan, "_interface_metadata"):
        gan._interface_metadata = loaded_meta or {}

    if _return_model:
        return aug_x, aug_y, gan

    return aug_x, aug_y


def assess_mt_generation_quality(
    real_data: np.ndarray,
    real_labels: dict[str, np.ndarray],
    generated_data: np.ndarray,
    generated_labels: dict[str, np.ndarray],
    primary_task: str = "trading",
    verbose: bool = True,
) -> dict:
    """
    Assess the quality of generated samples by comparing them to real data.
    
    Returns a dictionary with quality metrics.
    """
    metrics = {}
    
    n_real = len(real_data)
    real_only = real_data
    gen_only = generated_data[n_real:] if len(generated_data) > n_real else np.array([]).reshape(0, *real_data.shape[1:])
    
    if len(gen_only) == 0:
        if verbose:
            print("    [Quality] No generated samples to assess")
        return metrics
    
    # Overall statistics comparison
    real_mean = real_only.mean(axis=(0, 1))
    gen_mean = gen_only.mean(axis=(0, 1))
    real_std = real_only.std(axis=(0, 1)) + 1e-8
    gen_std = gen_only.std(axis=(0, 1)) + 1e-8
    
    mean_rmse = float(np.sqrt(np.mean((real_mean - gen_mean) ** 2)))
    std_rmse = float(np.sqrt(np.mean((real_std - gen_std) ** 2)))
    
    metrics["mean_rmse"] = mean_rmse
    metrics["std_rmse"] = std_rmse
    
    # Feature-wise correlation
    real_flat = real_only.reshape(len(real_only), -1)
    gen_flat = gen_only.reshape(len(gen_only), -1)
    
    n_samples = min(len(real_flat), len(gen_flat))
    real_sample = real_flat[:n_samples]
    gen_sample = gen_flat[:n_samples]
    
    feature_corrs = []
    for f in range(real_flat.shape[1]):
        if real_sample[:, f].std() > 1e-8 and gen_sample[:, f].std() > 1e-8:
            corr = np.corrcoef(real_sample[:, f], gen_sample[:, f])[0, 1]
            feature_corrs.append(float(corr))
    
    metrics["mean_correlation"] = float(np.mean(feature_corrs)) if feature_corrs else 0.0
    metrics["min_correlation"] = float(np.min(feature_corrs)) if feature_corrs else 0.0
    
    # Class-specific metrics for primary_task
    if primary_task in real_labels and primary_task in generated_labels:
        primary_real_labels = real_labels[primary_task]
        primary_gen_labels = generated_labels[primary_task]
        num_classes = primary_real_labels.shape[1]
        class_metrics = {}
        
        for c in range(num_classes):
            real_mask = np.argmax(primary_real_labels, axis=1) == c
            real_class = real_only[real_mask]
            
            all_gen_mask = np.argmax(primary_gen_labels, axis=1) == c
            gen_indices = np.where(all_gen_mask)[0]
            gen_indices = gen_indices[gen_indices >= n_real]
            gen_class = generated_data[gen_indices] if len(gen_indices) > 0 else np.array([]).reshape(0, *real_data.shape[1:])
            
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
        "overlap": float(max(0, min(real_max, gen_max) - max(real_min, gen_min)) / (real_max - real_min + 1e-8)),
    }
    
    if verbose:
        print(f"    [Quality] Mean RMSE: {mean_rmse:.4f}, Std RMSE: {std_rmse:.4f}")
        print(f"    [Quality] Feature correlation: mean={metrics['mean_correlation']:.4f}, min={metrics['min_correlation']:.4f}")
        print(f"    [Quality] Value range overlap: {metrics['range_coverage']['overlap']:.2%}")
        if "class_specific" in metrics and metrics["class_specific"]:
            print(f"    [Quality] Per-class metrics ({primary_task}):")
            for c, cm in metrics["class_specific"].items():
                print(f"      Class {c}: mean_RMSE={cm['mean_rmse']:.4f}, std_RMSE={cm['std_rmse']:.4f} (real={cm['real_count']}, gen={cm['gen_count']})")
    
    return metrics

