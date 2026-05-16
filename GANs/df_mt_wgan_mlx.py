import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np
import pandas as pd
from typing import Tuple, Optional, List, Dict, Any
import time
import os
import pickle

# Global cache for loaded GAN models/metadata to avoid redundant reloads during multi-pair strategy runs
_GAN_CACHE = {}
_META_CACHE = {}

class MTGenerator(nn.Module):
    def __init__(self, seq_len: int, num_features: int, total_cond_dim: int, latent_dim: int = 64, hidden_dim: int = 256):
        super().__init__()
        self.seq_len = seq_len
        self.num_features = num_features
        
        if seq_len > 1:
            # Simple CNN-based generator for sequences
            self.initial_dense = nn.Linear(latent_dim + total_cond_dim, hidden_dim * 2)
            self.model = nn.Sequential(
                nn.Linear(hidden_dim * 2, seq_len * hidden_dim),
                nn.LeakyReLU(0.2),
                # Note: For simplicity in tabular/time-series data, we can use Dense layers 
                # or Conv1D. Here we use a flexible reshape + Dense approach for now.
                nn.Linear(hidden_dim, num_features),
                nn.Tanh()
            )
        else:
            # Tabular generator
            self.model = nn.Sequential(
                nn.Linear(latent_dim + total_cond_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.LeakyReLU(0.2),
                nn.Linear(hidden_dim, hidden_dim * 2),
                nn.LayerNorm(hidden_dim * 2),
                nn.LeakyReLU(0.2),
                nn.Linear(hidden_dim * 2, num_features),
                nn.Tanh()
            )

    def __call__(self, z, c):
        # Concatenate z and condition
        x = mx.concatenate([z, c], axis=-1)
        
        if self.seq_len > 1:
            h = self.initial_dense(x)
            h = mx.reshape(self.model[0](h), (-1, self.seq_len, 256)) # Manually handling part of Sequential for reshape
            # Actually, let's just use a more structured approach for sequences
            pass
        
        return self.model(x)

# Redefining for sequence support more robustly
class MTGeneratorMLX(nn.Module):
    def __init__(self, seq_len: int, num_features: int, total_cond_dim: int, latent_dim: int = 64, hidden_dim: int = 256):
        super().__init__()
        self.seq_len = seq_len
        self.num_features = num_features
        
        if seq_len > 1:
            # Sequence generation: Dense -> Reshape -> MLP per step 
            # (Works well for shorter financial sequences like 10-30 steps)
            self.trunk = nn.Sequential(
                nn.Linear(latent_dim + total_cond_dim, hidden_dim * 2),
                nn.LeakyReLU(0.2),
                nn.Linear(hidden_dim * 2, seq_len * hidden_dim),
                nn.LeakyReLU(0.2)
            )
            self.head = nn.Linear(hidden_dim, num_features)
        else:
            # Tabular generation
            self.model = nn.Sequential(
                nn.Linear(latent_dim + total_cond_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.LeakyReLU(0.2),
                nn.Linear(hidden_dim, hidden_dim * 2),
                nn.LayerNorm(hidden_dim * 2),
                nn.LeakyReLU(0.2),
                nn.Linear(hidden_dim * 2, num_features),
                nn.Tanh()
            )

    def __call__(self, z, c):
        x = mx.concatenate([z, c], axis=-1)
        if self.seq_len > 1:
            h = self.trunk(x)
            h = mx.reshape(h, (-1, self.seq_len, 256))
            out = self.head(h)
            return mx.tanh(out)
        else:
            out = self.model(x)
            # If seq_len=1, return (B, 1, F) to match training data shape
            return mx.expand_dims(out, 1) if out.ndim == 2 else out

class MTCriticMLX(nn.Module):
    def __init__(self, seq_len: int, num_features: int, task_label_dims: Dict[str, int], total_cond_dim: int, hidden_dim: int = 256):
        super().__init__()
        self.seq_len = seq_len
        self.task_label_dims = task_label_dims
        self.sorted_tasks = sorted(list(task_label_dims.keys()))
        
        # Shared trunk
        if seq_len > 1:
            self.trunk = nn.Sequential(
                nn.Linear(num_features + total_cond_dim, hidden_dim),
                nn.LeakyReLU(0.2),
                # If sequence, we flatten for the trunk output
                nn.Linear(hidden_dim, 64),
                nn.LeakyReLU(0.2)
            )
            # Actually, for Critic with sequence, we should process the sequence then global pool or flatten
        else:
            self.trunk = nn.Sequential(
                nn.Linear(num_features + total_cond_dim, hidden_dim),
                nn.LeakyReLU(0.2),
                nn.Linear(hidden_dim, hidden_dim * 2),
                nn.LeakyReLU(0.2)
            )
        
        # Head for Adversarial Score
        if seq_len > 1:
            # Sequence trunk: Process xin + repeat(cin)
            pass
        
        # We'll use a simplified architecture that handles both
        self.fc1 = nn.Linear((num_features if seq_len==1 else num_features*seq_len) + total_cond_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        
        self.adv_head = nn.Linear(hidden_dim, 1)
        
        # We manually register these heads as attributes so MLX finds their parameters
        self._task_names = []
        for i, (task, dim) in enumerate(task_label_dims.items()):
            head_name = f"task_head_{i}"
            setattr(self, head_name, nn.Sequential(
                nn.Linear(hidden_dim, 64),
                nn.LeakyReLU(0.2),
                nn.Linear(64, dim)
            ))
            self._task_names.append((task, head_name))

    def __call__(self, x, c):
        # Flatten x if sequence
        if self.seq_len > 1:
            x_flat = mx.reshape(x, (x.shape[0], -1))
        else:
            # If seq_len=1, the input has shape (B, 1, F), squeeze to (B, F)
            x_flat = mx.squeeze(x, axis=1) if x.ndim == 3 else x
            
        h = mx.concatenate([x_flat, c], axis=-1)
        h = nn.leaky_relu(self.fc1(h), 0.2)
        h = nn.leaky_relu(self.fc2(h), 0.2)
        
        adv_score = self.adv_head(h)
        
        task_outputs = {}
        for task, head_name in self._task_names:
            task_outputs[task] = getattr(self, head_name)(h)
            
        return adv_score, task_outputs

class MTWGANMLX:
    # Z-score outlier-clip threshold — mirrors TabDDPMMLX._ZSCORE_CLIP and
    # MTDDPMMLX._ZSCORE_CLIP.  Training data is clipped to ±_ZSCORE_CLIP σ
    # before being fed to the generator/critic.
    _ZSCORE_CLIP: float = 4.0

    def __init__(self, seq_len: int, num_features: int, task_label_dims: dict[str, int], latent_dim: int = 64, gp_weight: float = 10.0, learning_rate: float = 1e-4, task_loss_weights: dict[str, float] | None = None):
        self.latent_dim = latent_dim
        self.task_label_dims = task_label_dims
        self.total_cond_dim = sum(task_label_dims.values())
        self.gp_weight = gp_weight
        self.seq_len = seq_len
        self.num_features = num_features
        
        self.gen = MTGeneratorMLX(seq_len, num_features, self.total_cond_dim, latent_dim)
        self.critic = MTCriticMLX(seq_len, num_features, task_label_dims, self.total_cond_dim)
        
        self.n_critic = 5
        self.task_loss_weights = task_loss_weights or {}
        
        # Initialize attributes for static analysis
        self.gen_opt = None
        self.critic_opt = None
        self.grad_clip = 10.0
        self.feature_mean = None
        self.feature_std = None
        self.feature_min = None
        self.feature_max = None

        self._create_optimizers(learning_rate, task_label_dims)
        
    def _get_paths(self, path: str):
        suffix = f"_s{self.seq_len}" if self.seq_len > 1 else ""
        gen_path = os.path.join(path, f"mt_wgan_gen_mlx{suffix}.safetensors")
        crit_path = os.path.join(path, f"mt_wgan_crit_mlx{suffix}.safetensors")
        meta_path = os.path.join(path, f"mt_wgan_meta_mlx{suffix}.pkl")
        return gen_path, crit_path, meta_path

    # Optimizers
    def _create_optimizers(self, learning_rate: float, task_label_dims: dict[str, int]):
        self.gen_opt = optim.Adam(learning_rate=learning_rate, betas=(0.5, 0.9))
        self.critic_opt = optim.Adam(learning_rate=learning_rate, betas=(0.5, 0.9))
        
        # Default task loss weights if not provided
        if not self.task_loss_weights:
            self.task_loss_weights = {k: (0.1 if k == "trading" else 0.02) for k in task_label_dims}

    def _postprocess(self, x):
        if self.feature_mean is not None and self.feature_std is not None:
            mean = mx.array(self.feature_mean)
            std = mx.array(self.feature_std)
            # Handle sequence vs tabular broadcasting
            if x.ndim == 3:
                mean = mx.reshape(mean, (1, 1, -1))
                std = mx.reshape(std, (1, 1, -1))
            else:
                mean = mx.reshape(mean, (1, -1))
                std = mx.reshape(std, (1, -1))
            x = x * std + mean
            
        if self.feature_min is not None and self.feature_max is not None:
            fmin = mx.array(self.feature_min)
            fmax = mx.array(self.feature_max)
            if x.ndim == 3:
                fmin = mx.reshape(fmin, (1, 1, -1))
                fmax = mx.reshape(fmax, (1, 1, -1))
            else:
                fmin = mx.reshape(fmin, (1, -1))
                fmax = mx.reshape(fmax, (1, -1))
            x = mx.clip(x, fmin, fmax)
        return x

    def loss_critic(self, model, real_x, real_c, z, batch_labels_mx):
        # We use the 'model' argument for the critic to ensure MLX traces gradients correctly
        fake_x = self.gen(z, real_c)
        fake_x = self._postprocess(fake_x)
        
        real_score, real_task_preds = model(real_x, real_c)
        fake_score, fake_task_preds = model(fake_x, real_c)
        
        w_loss = mx.mean(fake_score) - mx.mean(real_score)
        
        # Gradient Penalty
        alpha_shape = [real_x.shape[0]] + [1] * (real_x.ndim - 1)
        alpha = mx.random.uniform(shape=alpha_shape)
        interpolated = alpha * real_x + (1 - alpha) * fake_x
        
        # Using the current model weights for GP
        def critic_score_fn(x):
            score, _ = model(x, real_c)
            return mx.sum(score) # Use Sum to get individual sample gradients via grad()
        
        gp_grad = mx.grad(critic_score_fn)(interpolated)
        gp_norm = mx.sqrt(mx.sum(mx.square(gp_grad), axis=tuple(range(1, interpolated.ndim))) + 1e-8)
        gp = mx.mean((gp_norm - 1.0) ** 2)
        
        # Auxiliary Task Losses
        aux_loss = 0.0
        for task, head_name in model._task_names:
            weight = self.task_loss_weights.get(task, 0.02)
            y_true = batch_labels_mx[task]
            
            # Cross Entropy Loss
            pred_real = real_task_preds[task]
            l_real = mx.mean(nn.losses.cross_entropy(pred_real, y_true))
            
            pred_fake = fake_task_preds[task]
            l_fake = mx.mean(nn.losses.cross_entropy(pred_fake, y_true))
            
            aux_loss += weight * (l_real + l_fake)
        
        # Clip aux loss to prevent it from overwhelming Wasserstein loss
        aux_loss = mx.clip(aux_loss, 0.0, 5.0)
            
        return w_loss + self.gp_weight * gp + aux_loss

    def loss_gen(self, model, z, c, batch_labels_mx):
        # We use the 'model' argument for the generator to ensure MLX traces gradients correctly
        fake_x = model(z, c)
        fake_x = self._postprocess(fake_x)
        
        fake_score, fake_task_preds = self.critic(fake_x, c)
        
        adv_loss = -mx.mean(fake_score)
        
        aux_loss = 0.0
        for task, head_name in self.critic._task_names:
            weight = self.task_loss_weights.get(task, 0.02)
            y_true = batch_labels_mx[task]
            pred_fake = fake_task_preds[task]
            aux_loss += weight * mx.mean(nn.losses.cross_entropy(pred_fake, y_true))
            
        aux_loss = mx.clip(aux_loss, 0.0, 5.0)
            
        return adv_loss + aux_loss

    def save(self, path: str, metadata: dict):
        os.makedirs(path, exist_ok=True)
        gen_path, crit_path, meta_path = self._get_paths(path)
        self.gen.save_weights(gen_path)
        self.critic.save_weights(crit_path)
        with open(meta_path, "wb") as f:
            pickle.dump(metadata, f)

    def generate(
        self,
        n: int,
        task_labels: dict[str, np.ndarray],
    ) -> tuple[np.ndarray, dict[str, np.ndarray]]:
        """Generate n samples conditioned on task_labels.

        Args:
            n:            Number of samples to generate.
            task_labels:  Dict mapping task name → one-hot array (n, C).

        Returns:
            (gen_x, gen_labels) where gen_x is np.ndarray (n, seq_len, F)
            and gen_labels mirrors task_labels.
        """
        sorted_tasks = sorted(task_labels.keys())
        c_parts = [mx.array(task_labels[t], dtype=mx.float32) for t in sorted_tasks]
        c = mx.concatenate(c_parts, axis=-1)
        z = mx.random.normal((n, self.latent_dim))
        synth = self.gen(z, c)
        synth = self._postprocess(synth)
        mx.eval(synth)
        gen_x = np.array(synth)
        return gen_x, {t: task_labels[t] for t in sorted_tasks}

    def load(self, path: str):
        gen_path, crit_path, meta_path = self._get_paths(path)
        if not (os.path.exists(gen_path) and os.path.exists(meta_path)):
            return False, None
            
        try:
            with open(meta_path, "rb") as f:
                metadata = pickle.load(f)
            
            # Check for fundamental architectural incompatibility before loading weights
            saved_seq_len = metadata.get("seq_len", 1)
            saved_features = metadata.get("num_features")
            saved_dims = metadata.get("task_label_dims")
            
            if (saved_seq_len != self.seq_len or 
                saved_features != self.num_features or 
                saved_dims != self.task_label_dims):
                # We return False but still provide the metadata so the caller can check it
                return False, metadata
                
            self.gen.load_weights(gen_path)
            # We don't necessarily need the critic weights for generation, but nice to have
            if os.path.exists(crit_path):
                try:
                    self.critic.load_weights(crit_path)
                except:
                    pass
            return True, metadata
        except Exception:
            return False, None

def balance_with_mt_wgan_mlx(
    train_data: np.ndarray,
    train_labels: dict[str, np.ndarray],
    epochs: int = 100,
    batch_size: int = 1024,
    verbose: bool = True,
    primary_task: str = "trading",
    task_target_ratios: dict[str, Any] | None = None,
    **kwargs
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """MLX-accelerated Multi-Task WGAN-GP."""
    
    # sequence identification
    original_2d = train_data.ndim == 2
    if original_2d:
        train_data = train_data[:, np.newaxis, :]
        seq_len = 1
    else:
        seq_len = train_data.shape[1]
    
    num_samples, _, num_features = train_data.shape
    task_label_dims = {task: labels.shape[1] for task, labels in train_labels.items()}
    
    latent_dim = kwargs.get("latent_dim", 128)
    learning_rate = kwargs.get("learning_rate", 1e-4)
    gp_weight = kwargs.get("gp_weight", 10.0)
    task_loss_weights = kwargs.get("task_loss_weights")
    
    # 1. Preparation
    gan = MTWGANMLX(
        seq_len, num_features, task_label_dims,
        latent_dim=latent_dim,
        learning_rate=learning_rate,
        gp_weight=gp_weight,
        task_loss_weights=task_loss_weights
    )
    
    # Stats for postprocess — computed from raw data BEFORE normalisation.
    # _postprocess inverts the z-score applied below, so these must reflect
    # the original (un-normalised) distribution.
    gan.feature_mean = train_data.mean(axis=(0, 1)).astype("float32")
    std = train_data.std(axis=(0, 1)).astype("float32")
    gan.feature_std = np.maximum(std, 1e-6)  # std floor — mirrors MTDDPMMLX
    gan.feature_min = train_data.min(axis=(0, 1)).astype("float32")
    gan.feature_max = train_data.max(axis=(0, 1)).astype("float32")

    # Apply forward z-score so the generator learns in standardised space.
    # _postprocess already applies the algebraic inverse (x * std + mean),
    # so the two transforms cancel correctly on the way out.
    # Mirrors MTDDPMMLX._zscore_fit_3d and TabDDPMMLX._zscore_fit.
    #
    # Note: the generator's final Tanh activation bounds its output to
    # (-1, 1) in z-scored space, which corresponds to roughly ±1σ in
    # real-data space.  MT_WGAN can reproduce the central mass of each
    # feature's distribution but cannot reach the tails beyond ±1σ.
    # To produce wider distributions the Tanh should be replaced with a
    # scaled variant (e.g. `4 * tanh(x)`) or removed.  Not addressed in
    # this patch — out of scope; see findings doc
    # 2026-05-15-mt-wgan-audit-findings.md.
    train_data = (train_data - gan.feature_mean) / gan.feature_std
    train_data = np.clip(train_data, -MTWGANMLX._ZSCORE_CLIP, MTWGANMLX._ZSCORE_CLIP).astype(np.float32)

    save_path = kwargs.get("save_path")
    model_loaded = False

    if save_path:
        cache_key = (save_path, seq_len, num_features, tuple(sorted(task_label_dims.items())))
        if cache_key in _GAN_CACHE:
            gan = _GAN_CACHE[cache_key]
            model_loaded = True
            if verbose: print(f"    Using cached MT-MLX WGAN model from {save_path}")
        else:
            model_loaded, loaded_meta = gan.load(save_path)
            if model_loaded:
                # Check compatibility
                comp = (loaded_meta.get("seq_len") == seq_len and 
                        loaded_meta.get("num_features") == num_features and
                        loaded_meta.get("task_label_dims") == task_label_dims)
                if not comp:
                    if verbose: print("    Loaded MT-WGAN model incompatible. Re-training...")
                    model_loaded = False
                else:
                    if verbose: print(f"    Loaded existing MT-MLX WGAN model from {save_path}")
                    _GAN_CACHE[cache_key] = gan

    if not model_loaded:
        critic_grad_fn = nn.value_and_grad(gan.critic, gan.loss_critic)
        gen_grad_fn = nn.value_and_grad(gan.gen, gan.loss_gen)
        
        # Pre-convert all data and labels to MLX arrays to avoid CPU/Numpy syncs in hot loop
        sorted_tasks = sorted(list(train_labels.keys()))
        C_full = np.concatenate([train_labels[t] for t in sorted_tasks], axis=1)
        C = mx.array(C_full, dtype=mx.float32)
        X = mx.array(train_data, dtype=mx.float32)
        Y_device = {t: mx.array(train_labels[t], dtype=mx.float32) for t in sorted_tasks}

        if verbose:
            print(f"    Starting MLX MT-WGAN-GP training ({epochs} epochs, bs={batch_size})...")
            start_time = time.time()

        for epoch in range(epochs):
            perm = mx.array(np.random.permutation(num_samples))
            for i in range(0, num_samples, batch_size):
                indices = perm[i : i + batch_size]
                real_x, real_c = X[indices], C[indices]
                
                # Purely device-side sliced labels dict
                batch_labels_mx = {t: Y_device[t][indices] for t in sorted_tasks}
                
                bs = real_x.shape[0]
                z = mx.random.normal((bs, gan.latent_dim))
                
                # Train critic n_critic steps
                for _ in range(gan.n_critic):
                    loss_c, grads_c = critic_grad_fn(gan.critic, real_x, real_c, z, batch_labels_mx)
                    # Apply gradient clipping
                    grads_c, _ = optim.clip_grad_norm(grads_c, gan.grad_clip)
                    gan.critic_opt.update(gan.critic, grads_c)
                
                # Train generator 1 step
                loss_g, grads_g = gen_grad_fn(gan.gen, z, real_c, batch_labels_mx)
                # Apply gradient clipping
                grads_g, _ = optim.clip_grad_norm(grads_g, gan.grad_clip)
                gan.gen_opt.update(gan.gen, grads_g)
                
                # Optional: evaluate occasionally inside the batch loop if epochs are very long
                if i % (10 * batch_size) == 0:
                     mx.eval(loss_c, loss_g)
            
            mx.eval(gan.gen.parameters(), gan.critic.parameters())
            if verbose and ((epoch < 20) or ((epoch + 1) % 10 == 0)):
                print(f"      Epoch {epoch+1}/{epochs} | D Loss: {loss_c.item():.4f} | G Loss: {loss_g.item():.4f}")

        if verbose:
            print(f"    MLX MT-WGAN training complete in {time.time() - start_time:.2f}s")
        
        if save_path:
            meta = {
                "seq_len": seq_len,
                "num_features": num_features,
                "task_label_dims": task_label_dims,
                "latent_dim": gan.latent_dim
            }
            # Add thresholds if provided
            for k in ["min_buy_gain_threshold", "min_sell_loss_threshold", "training_type"]:
                if k in kwargs: meta[k] = kwargs[k]
            gan.save(save_path, meta)

    # In-fit augmentation is intentionally a no-op. The backend always
    # passes task_target_ratios={k: 0.0 for k in labels} via
    # _build_mt_balance_config, so any in-function balancing here would
    # be dead code at production callsites. Augmentation happens later
    # in BaseNNMTStrategy.preprocess_training_data via the shared
    # GANs.balance.balance_multi_task, which iterates with leak-aware
    # deficit recomputation across all MT GAN backends.
    #
    # Previously this function carried its own single-pass augmentation
    # loop; removed 2026-05-16 to match balance_with_mt_ddpm_mlx and
    # consolidate iteration semantics in balance_multi_task.
    if original_2d:
        train_data = train_data.reshape(num_samples, num_features)
    if kwargs.get("_return_model"):
        return train_data, train_labels, gan
    return train_data, train_labels

def load_mt_wgan_thresholds_mlx(save_path: str) -> Dict[str, Any]:
    if save_path in _META_CACHE:
        return _META_CACHE[save_path]
    
    meta_path = os.path.join(save_path, "mt_wgan_meta_mlx.pkl")
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
    except:
        return {"min_buy_gain_threshold": None, "min_sell_loss_threshold": None, "training_type": None}
