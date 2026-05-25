import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np
import pandas as pd
import pickle
from typing import Tuple, Optional, List, Dict, Any
import time
import os
import shutil
import tempfile

class Generator(nn.Module):
    """Generator for single-task WGAN-GP (tabular).

    Linear output (no Tanh). WGAN-GP enforces the Lipschitz constraint
    via gradient penalty on the critic, so the generator does not need
    bounded output. Tanh was a v1-pipeline leftover that squashed the
    range to ±1σ in z-scored space; under v2 the input is z-scored and
    clipped to ±4σ, so a bounded output gave away the tails.
    """
    def __init__(self, latent_dim: int, num_classes: int, num_features: int, hidden_dim: int = 256):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(latent_dim + num_classes, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.LayerNorm(hidden_dim * 2),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim * 2, num_features),
        )

    def __call__(self, z, c):
        x = mx.concatenate([z, c], axis=-1)
        return self.model(x)

class Critic(nn.Module):
    def __init__(self, num_features: int, num_classes: int, hidden_dim: int = 256):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(num_features + num_classes, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim * 2, 1)
        )

    def __call__(self, x, c):
        xc = mx.concatenate([x, c], axis=-1)
        return self.model(xc)

class WGANMLX:
    # Z-score outlier-clip threshold — mirrors TabDDPMMLX._ZSCORE_CLIP and
    # MTWGANMLX._ZSCORE_CLIP. Training data is clipped to ±_ZSCORE_CLIP σ
    # before being fed to the generator/critic.
    _ZSCORE_CLIP: float = 4.0

    def __init__(self, num_features: int, num_classes: int, latent_dim: int = 64, gp_weight: float = 10.0, learning_rate: float = 1e-4):
        self.latent_dim = latent_dim
        self.num_classes = num_classes
        self.num_features = num_features
        self.gp_weight = gp_weight
        self.gen = Generator(latent_dim, num_classes, num_features)
        self.critic = Critic(num_features, num_classes)
        self.gen_opt = optim.Adam(learning_rate=learning_rate, betas=(0.5, 0.9))
        self.critic_opt = optim.Adam(learning_rate=learning_rate, betas=(0.5, 0.9))
        # Stats for inverse z-score in generate(). Set by balance_with_wgan_mlx
        # at fit time, persisted in metadata at save() time.
        self.feature_mean: Optional[np.ndarray] = None
        self.feature_std: Optional[np.ndarray] = None

    def loss_critic(self, model, real_x, real_c, z, fake_c):
        fake_x = self.gen(z, fake_c)
        real_score = self.critic(real_x, real_c)
        fake_score = self.critic(fake_x, fake_c)
        w_loss = mx.mean(fake_score) - mx.mean(real_score)
        
        # Gradient Penalty
        alpha = mx.random.uniform(shape=(real_x.shape[0], 1))
        interpolated = alpha * real_x + (1 - alpha) * fake_x
        
        def critic_score_fn(x):
            return mx.mean(self.critic(x, real_c))
        
        gp_grad = mx.grad(critic_score_fn)(interpolated)
        gp_norm = mx.sqrt(mx.sum(mx.square(gp_grad), axis=1) + 1e-8)
        gp = mx.mean((gp_norm - 1.0) ** 2)
        return w_loss + self.gp_weight * gp

    def loss_gen(self, model, z, c):
        fake_x = self.gen(z, c)
        fake_score = self.critic(fake_x, c)
        return -mx.mean(fake_score)

    def save(self, path: str):
        """Persist generator weights + the z-score stats needed by generate().

        The critic is intentionally NOT saved — only needed for training.
        """
        os.makedirs(path, exist_ok=True)
        self.gen.save_weights(os.path.join(path, "wgan_gen_mlx.safetensors"))
        meta = {
            "num_features": int(self.num_features),
            "num_classes":  int(self.num_classes),
            "latent_dim":   int(self.latent_dim),
            "feature_mean": (
                np.asarray(self.feature_mean, dtype=np.float32)
                if self.feature_mean is not None else None
            ),
            "feature_std": (
                np.asarray(self.feature_std, dtype=np.float32)
                if self.feature_std is not None else None
            ),
        }
        with open(os.path.join(path, "wgan_meta_mlx.pkl"), "wb") as f:
            pickle.dump(meta, f)

    def load(self, path: str):
        """Restore generator weights AND z-score stats. Both required for
        the v2 path; missing stats raise to surface stale (pre-v2) models.
        """
        weight_path = os.path.join(path, "wgan_gen_mlx.safetensors")
        meta_path = os.path.join(path, "wgan_meta_mlx.pkl")
        if not os.path.exists(weight_path):
            return False
        self.gen.load_weights(weight_path)
        if os.path.exists(meta_path):
            with open(meta_path, "rb") as f:
                meta = pickle.load(f)
            fm = meta.get("feature_mean")
            fs = meta.get("feature_std")
            if fm is not None:
                self.feature_mean = np.asarray(fm, dtype=np.float32)
            if fs is not None:
                self.feature_std = np.asarray(fs, dtype=np.float32)
        # Pre-v2 saved models have weights but no metadata — leave stats
        # as None and let generate() fall back to no-inverse-z-score so
        # the old format keeps working until retrained.
        return True

    def generate(self, n: int, one_hot: np.ndarray) -> np.ndarray:
        """Generate n samples conditioned on one_hot labels.

        Applies inverse z-score (multiply by std, add mean) when those
        stats are available — they're set by ``balance_with_wgan_mlx`` at
        fit time and persisted through save/load. Returns features in the
        ORIGINAL (un-normalised) feature space, matching what
        ``balance_with_mt_wgan_mlx._postprocess`` produces.

        Returns:
            np.ndarray of shape (n, 1, num_features) — 3-D to match the
            WGANGP.generate() convention expected by GANInterface.
        """
        c = mx.array(one_hot, dtype=mx.float32)
        z = mx.random.normal((n, self.latent_dim))
        synth = self.gen(z, c)
        mx.eval(synth)
        out = np.array(synth)
        if self.feature_mean is not None and self.feature_std is not None:
            out = out * self.feature_std[None, :] + self.feature_mean[None, :]
        return out[:, np.newaxis, :]  # (n, 1, F)

def balance_with_wgan_mlx(
    train_data: np.ndarray,
    train_labels: np.ndarray,
    epochs: int = 100,
    batch_size: int = 1024,
    n_critic: int = 5,
    augmentation_target_ratio: float = 0.5,
    noise_std: float = 0.0,
    verbose: bool = True,
    **kwargs
) -> Tuple[np.ndarray, np.ndarray]:
    """MLX-accelerated WGAN-GP for tabular data balancing."""
    
    # Check for seq_len=1 (this is a tabular GAN)
    if train_data.ndim == 3:
        train_data = train_data[:, 0, :]

    num_samples, num_features = train_data.shape
    num_classes = train_labels.shape[1]

    # 1. Preparation
    gan = WGANMLX(num_features, num_classes)

    # Compute z-score stats from raw input. The generator trains in
    # z-scored space; the augmentation step below inverts the z-score on
    # synth output so the returned aug_x is in the ORIGINAL (raw) feature
    # space — matching the strategy's expected post-GAN input convention.
    # Mirrors MTWGANMLX (see df_mt_wgan_mlx.py).
    original_train_data = train_data  # keep raw for the augmentation output
    gan.feature_mean = train_data.mean(axis=0).astype("float32")
    std = train_data.std(axis=0).astype("float32")
    gan.feature_std = np.maximum(std, 1e-6)  # std floor

    # Apply forward z-score + ±_ZSCORE_CLIP σ clip — training-time only.
    train_data_zscored = (train_data - gan.feature_mean) / gan.feature_std
    train_data_zscored = np.clip(
        train_data_zscored, -WGANMLX._ZSCORE_CLIP, WGANMLX._ZSCORE_CLIP
    ).astype(np.float32)

    X = mx.array(train_data_zscored, dtype=mx.float32)
    y = mx.array(train_labels, dtype=mx.float32)
    
    save_path = kwargs.get("save_path")
    model_loaded = False
    
    if save_path:
        model_loaded = gan.load(save_path)
        if model_loaded:
            if verbose:
                print(f"    Loaded existing MLX WGAN model from {save_path}; skipping training.")
    
    if not model_loaded:
        critic_grad_fn = nn.value_and_grad(gan.critic, gan.loss_critic)
        gen_grad_fn = nn.value_and_grad(gan.gen, gan.loss_gen)

        # Divergence / collapse detection — mirrors TF GANCollapseCallback.
        # Stop if combined |D Loss| + |G Loss| has grown far above the best
        # seen for `diverge_patience` consecutive epochs, or if either loss
        # individually exceeds a hard ceiling.
        diverge_patience = 5          # consecutive bad epochs before stopping
        abs_d_ceil = 300.0            # hard ceiling for |D Loss|
        abs_g_ceil = 500.0            # hard ceiling for |G Loss|
        degrade_factor = 10.0        # score must be > factor × best to count as bad
        diverge_count = 0
        best_score = float("inf")
        best_epoch = -1
        checkpoint_dir = tempfile.mkdtemp(prefix="wgan_mlx_ckpt_")
        best_gen_path = os.path.join(checkpoint_dir, "best_gen.safetensors")
        best_critic_path = os.path.join(checkpoint_dir, "best_critic.safetensors")

        if verbose:
            print(f"    Starting MLX WGAN-GP training ({epochs} epochs, bs={batch_size})...")
            start_time = time.time()

        stopped_early = False
        last_epoch = 0
        for epoch in range(epochs):
            last_epoch = epoch
            if verbose and epoch == 0:
                print(f"    Epoch 1/{epochs} starting...")

            perm = mx.array(np.random.permutation(num_samples))
            for i in range(0, num_samples, batch_size):
                indices = perm[i : i + batch_size]
                real_x, real_c = X[indices], y[indices]
                bs = real_x.shape[0]
                z = mx.random.normal((bs, gan.latent_dim))

                # Update critic n_critic times per generator step.
                for _ in range(n_critic):
                    z_c = mx.random.normal((bs, gan.latent_dim))
                    loss_c, grads_c = critic_grad_fn(gan.critic, real_x, real_c, z_c, real_c)
                    gan.critic_opt.update(gan.critic, grads_c)
                    mx.eval(gan.critic.parameters())
                loss_g, grads_g = gen_grad_fn(gan.gen, z, real_c)
                gan.gen_opt.update(gan.gen, grads_g)

                if verbose and epoch == 0 and (i // batch_size) % 100 == 0:
                    print(f"        Batch {i // batch_size} / {num_samples // batch_size}...")

            mx.eval(gan.gen.parameters(), gan.critic.parameters())

            d_val = loss_c.item()
            g_val = loss_g.item()
            epoch_score = abs(d_val) + abs(g_val)

            if verbose:
                if (epoch < 10) or ((epoch + 1) % 5 == 0):
                    print(f"      Epoch {epoch+1}/{epochs} | D Loss: {d_val:.4f} | G Loss: {g_val:.4f}")

            # --- best checkpoint ---
            if epoch_score < best_score:
                best_score = epoch_score
                best_epoch = epoch
                diverge_count = 0          # reset patience on improvement
                try:
                    gan.gen.save_weights(best_gen_path)
                    gan.critic.save_weights(best_critic_path)
                except Exception:
                    pass  # checkpoint is best-effort
            else:
                # Count as divergent if score is far worse than best OR above ceiling
                is_divergent = (
                    epoch_score > degrade_factor * best_score
                    or abs(d_val) > abs_d_ceil
                    or abs(g_val) > abs_g_ceil
                )
                if is_divergent:
                    diverge_count += 1
                    if verbose:
                        print(
                            f"      WARNING: loss divergence detected "
                            f"(D={d_val:.2f}, G={g_val:.2f}), "
                            f"count {diverge_count}/{diverge_patience}"
                        )
                    if diverge_count >= diverge_patience:
                        if verbose:
                            print(
                                f"    Stopping early at epoch {epoch+1} due to divergence. "
                                f"Best was epoch {best_epoch+1} "
                                f"(score={best_score:.4f})."
                            )
                        stopped_early = True
                        break
                else:
                    diverge_count = max(0, diverge_count - 1)

        if verbose:
            print(f"    MLX WGAN training complete in {time.time() - start_time:.2f}s")

        # Restore best checkpoint if training diverged or ended past the best epoch
        if best_epoch >= 0 and (stopped_early or best_epoch < last_epoch - 1):
            try:
                gan.gen.load_weights(best_gen_path)
                gan.critic.load_weights(best_critic_path)
                mx.eval(gan.gen.parameters(), gan.critic.parameters())
                if verbose:
                    print(
                        f"    Restored best model from epoch {best_epoch+1} "
                        f"(score={best_score:.4f})"
                    )
            except Exception as e:
                if verbose:
                    print(f"    WARNING: Could not restore best checkpoint: {e}")

        shutil.rmtree(checkpoint_dir, ignore_errors=True)

        if save_path:
            gan.save(save_path)
            if verbose:
                print(f"    Saved MLX WGAN model to {save_path}")

    # 2. Augmentation Strategy
    y_idx = np.argmax(train_labels, axis=1)
    unique, counts = np.unique(y_idx, return_counts=True)
    max_count = np.max(counts)
    target_base = int(max_count * augmentation_target_ratio)

    # Concatenate raw (un-z-scored) real data with inverse-z-scored synth.
    # The output is in the ORIGINAL feature distribution — the strategy's
    # post-GAN tensor scaler runs on the combined real+synth tensor next.
    gen_x_list = [original_train_data]
    gen_y_list = [train_labels]

    for c in range(num_classes):
        have = np.sum(y_idx == c)
        need = max(0, target_base - have)
        if need <= 0: continue

        c_target = np.zeros((need, num_classes), dtype=np.float32)
        c_target[:, c] = 1.0
        c_target_mx = mx.array(c_target)
        z = mx.random.normal((need, gan.latent_dim))

        synth = gan.gen(z, c_target_mx)
        mx.eval(synth)
        synth_np = np.array(synth)

        # Inverse z-score → raw feature space.
        synth_np = synth_np * gan.feature_std[None, :] + gan.feature_mean[None, :]

        if noise_std > 0:
            synth_np += np.random.normal(0, noise_std, synth_np.shape)

        gen_x_list.append(synth_np)
        gen_y_list.append(c_target)

    aug_x = np.concatenate(gen_x_list, axis=0)
    aug_y = np.concatenate(gen_y_list, axis=0)
    p = np.random.permutation(len(aug_x))
    if kwargs.get("_return_model"):
        return aug_x[p], aug_y[p], gan
    return aug_x[p], aug_y[p]
