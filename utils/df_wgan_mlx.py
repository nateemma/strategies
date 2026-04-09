import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np
import pandas as pd
from typing import Tuple, Optional, List, Dict, Any
import time
import os

class Generator(nn.Module):
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
            nn.Tanh()
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
    def __init__(self, num_features: int, num_classes: int, latent_dim: int = 64, gp_weight: float = 10.0, learning_rate: float = 1e-4):
        self.latent_dim = latent_dim
        self.num_classes = num_classes
        self.num_features = num_features
        self.gp_weight = gp_weight
        self.gen = Generator(latent_dim, num_classes, num_features)
        self.critic = Critic(num_features, num_classes)
        self.gen_opt = optim.Adam(learning_rate=learning_rate, betas=(0.5, 0.9))
        self.critic_opt = optim.Adam(learning_rate=learning_rate, betas=(0.5, 0.9))

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
        os.makedirs(path, exist_ok=True)
        self.gen.save_weights(os.path.join(path, "wgan_gen_mlx.safetensors"))

    def load(self, path: str):
        weight_path = os.path.join(path, "wgan_gen_mlx.safetensors")
        if os.path.exists(weight_path):
            self.gen.load_weights(weight_path)
            return True
        return False

def balance_with_wgan_mlx(
    train_data: np.ndarray,
    train_labels: np.ndarray,
    epochs: int = 100,
    batch_size: int = 1024,
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
    X = mx.array(train_data, dtype=mx.float32)
    y = mx.array(train_labels, dtype=mx.float32)
    gan = WGANMLX(num_features, num_classes)
    
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
        
        if verbose:
            print(f"    Starting MLX WGAN-GP training ({epochs} epochs, bs={batch_size})...")
            start_time = time.time()

        # Training
        for epoch in range(epochs):
            if verbose and epoch == 0:
                print(f"    Epoch 1/{epochs} starting...")
                
            perm = mx.array(np.random.permutation(num_samples))
            for i in range(0, num_samples, batch_size):
                indices = perm[i : i + batch_size]
                real_x, real_c = X[indices], y[indices]
                bs = real_x.shape[0]
                z = mx.random.normal((bs, gan.latent_dim))
                
                # WGAN-GP Training: Typically update critic more than generator (n_critic=5)
                # For speed, we do 1:1 but with strong GP
                loss_c, grads_c = critic_grad_fn(gan.critic, real_x, real_c, z, real_c)
                gan.critic_opt.update(gan.critic, grads_c)
                loss_g, grads_g = gen_grad_fn(gan.gen, z, real_c)
                gan.gen_opt.update(gan.gen, grads_g)
                
                if verbose and epoch == 0 and (i // batch_size) % 100 == 0:
                    print(f"        Batch {i // batch_size} / {num_samples // batch_size}...")
            
            mx.eval(gan.gen.parameters(), gan.critic.parameters())
            if verbose:
                if (epoch < 10) or ((epoch + 1) % 5 == 0):
                    print(f"      Epoch {epoch+1}/{epochs} | D Loss: {loss_c.item():.4f} | G Loss: {loss_g.item():.4f}")

        if verbose:
            print(f"    MLX WGAN training complete in {time.time() - start_time:.2f}s")
        
        if save_path:
            gan.save(save_path)
            if verbose:
                print(f"    Saved MLX WGAN model to {save_path}")

    # 2. Augmentation Strategy
    y_idx = np.argmax(train_labels, axis=1)
    unique, counts = np.unique(y_idx, return_counts=True)
    max_count = np.max(counts)
    target_base = int(max_count * augmentation_target_ratio)
    
    gen_x_list = [train_data]
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
        
        if noise_std > 0:
            synth_np += np.random.normal(0, noise_std, synth_np.shape)
        
        gen_x_list.append(synth_np)
        gen_y_list.append(c_target)

    aug_x = np.concatenate(gen_x_list, axis=0)
    aug_y = np.concatenate(gen_y_list, axis=0)
    p = np.random.permutation(len(aug_x))
    return aug_x[p], aug_y[p]
