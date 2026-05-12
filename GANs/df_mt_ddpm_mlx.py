"""
Multi-task tabular DDPM (MLX-only).

Tensor-aware sibling of TabDDPMMLX: operates on (B, seq_len, F) sequences
with a Dict[str, np.ndarray] label set, mirroring the MT_WGAN convention.
Single-task usage = labels = {"trading": one_hot(B, C)}.

Backbone: flattened MLP over (B, seq_len * F). The whole sequence is one
"sample" from the model's POV — the time axis is preserved in the input
and output, and the LSTM classifier downstream sees real and synthetic
windows of identical structure.

Phase 1 design — see docs/superpowers/specs/2026-05-12-mt-ddpm-design.md
for motivation and full design rationale.
"""

from __future__ import annotations

import os
import pickle
import time
from typing import Any, Dict, List, Optional, Tuple

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np

from GANs.diffusion_mlx import (
    cosine_beta_schedule,
    make_schedule,
    q_sample,
    ddim_sample,
)


# ---------------------------------------------------------------------------
# Parameter-tree helpers (mirrors df_tabddpm_mlx._tree_lerp)
# ---------------------------------------------------------------------------

def _tree_lerp(a: Any, b: Any, t: float) -> Any:
    """Element-wise (1-t)*a + t*b on nested mlx parameter trees."""
    if isinstance(a, dict):
        return {k: _tree_lerp(a[k], b[k], t) for k in a}
    if isinstance(a, list):
        return [_tree_lerp(ai, bi, t) for ai, bi in zip(a, b)]
    if isinstance(a, mx.array):
        return (1.0 - t) * a + t * b
    return a  # non-array leaves passed through


# ---------- submodules ----------

class _SinusoidalTimeEmbed(nn.Module):
    """Standard sinusoidal time embedding. Mirrors df_tabddpm_mlx.py."""

    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def __call__(self, t: mx.array) -> mx.array:
        half = self.dim // 2
        freqs = mx.exp(
            -mx.log(mx.array(10000.0)) * mx.arange(half, dtype=mx.float32) / half
        )
        args = t.astype(mx.float32)[:, None] * freqs[None, :]
        return mx.concatenate([mx.sin(args), mx.cos(args)], axis=-1)


class _TaskLabelEmbed(nn.Module):
    """One Embedding per task; outputs are summed.

    For single-task ``{"trading": 3}`` this collapses to a single
    ``nn.Embedding(3, d_model)`` lookup.
    """

    def __init__(self, task_label_dims: Dict[str, int], d_model: int):
        super().__init__()
        self.task_label_dims = dict(task_label_dims)
        self._embeds: Dict[str, nn.Embedding] = {
            name: nn.Embedding(dim, d_model)
            for name, dim in task_label_dims.items()
        }
        for name, emb in self._embeds.items():
            setattr(self, f"embed_{name}", emb)

    def __call__(self, task_labels: Dict[str, mx.array]) -> mx.array:
        out = None
        for name, idx in task_labels.items():
            emb = self._embeds[name](idx.astype(mx.int32))
            out = emb if out is None else out + emb
        return out


class _MLPBlock(nn.Module):
    """LayerNorm -> Linear -> GELU -> Dropout. Same as TabDDPM."""

    def __init__(self, dim: int, dropout: float):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fc = nn.Linear(dim, dim)
        self.dropout_p = dropout
        self._dropout = nn.Dropout(dropout) if dropout > 0.0 else None

    def __call__(self, x: mx.array, training: bool) -> mx.array:
        h = self.norm(x)
        h = self.fc(h)
        h = nn.gelu(h)
        if training and self._dropout is not None:
            h = self._dropout(h)
        return x + h


class _FlatMLPBackbone(nn.Module):
    """Flatten (B, T, F) -> (B, T*F), run MLP, reshape back.

    Inputs concatenated to the flattened features:
      * time embedding (d_model)
      * task-label embedding (d_model)
    """

    def __init__(
        self,
        seq_len: int,
        num_features: int,
        d_model: int,
        d_layers: int,
        dropout: float,
        task_label_dims: Dict[str, int],
    ):
        super().__init__()
        self.seq_len = seq_len
        self.num_features = num_features
        flat_dim = seq_len * num_features

        self.in_proj = nn.Linear(flat_dim + 2 * d_model, d_model)
        self.blocks = [_MLPBlock(d_model, dropout) for _ in range(d_layers)]
        self.out_proj = nn.Linear(d_model, flat_dim)

        self.time_embed = _SinusoidalTimeEmbed(d_model)
        self.label_embed = _TaskLabelEmbed(task_label_dims, d_model)

    def __call__(
        self,
        x: mx.array,
        t: mx.array,
        task_labels: Dict[str, mx.array],
        training: bool,
    ) -> mx.array:
        b = x.shape[0]
        flat = x.reshape(b, -1)
        t_emb = self.time_embed(t)
        l_emb = self.label_embed(task_labels)
        h = mx.concatenate([flat, t_emb, l_emb], axis=-1)
        h = self.in_proj(h)
        for blk in self.blocks:
            h = blk(h, training=training)
        out = self.out_proj(h)
        return out.reshape(b, self.seq_len, self.num_features)


# ---------- outer model ----------

class MTDDPMMLX:
    """Multi-task tabular DDPM (MLX, flattened-MLP backbone).

    Phase 1 implementation — see spec for design rationale and the
    Phase 2/3/4 follow-ons that are deliberately out of scope here.
    """

    def __init__(
        self,
        seq_len: int,
        num_features: int,
        task_label_dims: Dict[str, int],
        d_model: int = 256,
        d_layers: int = 4,
        dropout: float = 0.1,
        num_timesteps: int = 1000,
        num_sample_steps: int = 50,
        epochs: int = 300,
        batch_size: int = 256,
        learning_rate: float = 2e-4,
        weight_decay: float = 0.0,
        ema_decay: float = 0.999,
        eval_frequency: int = 10,
        lr_min_ratio: float = 0.1,
        min_snr_gamma: float = 0.0,
        class_balanced_sampling: bool = False,
        p_uncond: float = 0.0,
        guidance_scale: float = 1.0,
        use_edm_schedule: bool = False,
        edm_p_mean: float = -1.2,
        edm_p_std: float = 1.2,
        edm_sigma_min: float = 0.002,
        edm_sigma_max: float = 10.0,
        edm_rho: float = 7.0,
        edm_sigma_data: float = 1.0,
        verbose: bool = True,
    ):
        self.seq_len = seq_len
        self.num_features = num_features
        self.task_label_dims = dict(task_label_dims)
        self.d_model = d_model
        self.d_layers = d_layers
        self.dropout = dropout
        self.num_timesteps = num_timesteps
        self.num_sample_steps = num_sample_steps
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.ema_decay = ema_decay
        self.eval_frequency = eval_frequency
        self.lr_min_ratio = lr_min_ratio
        self.min_snr_gamma = min_snr_gamma
        self.class_balanced_sampling = class_balanced_sampling
        self.p_uncond = p_uncond
        self.guidance_scale = guidance_scale
        self.use_edm_schedule = use_edm_schedule
        self.edm_p_mean = edm_p_mean
        self.edm_p_std = edm_p_std
        self.edm_sigma_min = edm_sigma_min
        self.edm_sigma_max = edm_sigma_max
        self.edm_rho = edm_rho
        self.edm_sigma_data = edm_sigma_data
        self.verbose = verbose

        self._mlp = _FlatMLPBackbone(
            seq_len=seq_len,
            num_features=num_features,
            d_model=d_model,
            d_layers=d_layers,
            dropout=dropout,
            task_label_dims=task_label_dims,
        )
        self._ema_mlp: Optional[_FlatMLPBackbone] = None
        self._schedule = None

        self.feature_mean: Optional[np.ndarray] = None
        self.feature_std: Optional[np.ndarray] = None

    def fit(
        self,
        data: np.ndarray,
        labels: Dict[str, np.ndarray],
        categorical_columns: Optional[List[str]] = None,
        pair_labels: Optional[np.ndarray] = None,
        pair_names: Optional[List[str]] = None,
    ) -> None:
        """Train the diffusion model. Persists best-by-train-loss snapshot."""
        import copy

        if data.ndim != 3:
            raise ValueError(
                f"MTDDPMMLX.fit expects data of shape (N, seq_len, F); "
                f"got {data.shape}"
            )
        if not isinstance(labels, dict):
            raise TypeError(
                f"MTDDPMMLX.fit expects labels as Dict[str, np.ndarray]; "
                f"got {type(labels).__name__}"
            )

        N, T, F = data.shape
        assert T == self.seq_len and F == self.num_features

        self.feature_mean = data.reshape(-1, F).mean(axis=0).astype(np.float32)
        self.feature_std = data.reshape(-1, F).std(axis=0).astype(np.float32) + 1e-6

        x_data = mx.array(data, dtype=mx.float32)
        task_arrays = {
            name: mx.array(np.argmax(onehot, axis=1), dtype=mx.int32)
            for name, onehot in labels.items()
        }

        self._schedule = make_schedule(self.num_timesteps)

        optimizer = optim.AdamW(
            learning_rate=self.learning_rate, weight_decay=self.weight_decay
        )

        self._ema_mlp = copy.deepcopy(self._mlp)

        # Labels dict is captured by closure — passing a Python dict as a
        # positional arg to nn.value_and_grad causes MLX to attempt gradient
        # tracing through dict keys, which is unsupported. Only MLX arrays
        # that need gradients (x0, noise) go as positional args.
        _batch_labels: Dict[str, mx.array] = {}

        def loss_fn(mlp, x0, t, noise):
            # q_sample uses [:, None] broadcast — designed for (B, F).
            # Flatten (B, T, F) → (B, T*F) before the call, then reshape.
            b = x0.shape[0]
            x0_flat = x0.reshape(b, -1)
            noise_flat = noise.reshape(b, -1)
            x_noisy_flat = q_sample(x0_flat, t, noise_flat, self._schedule)
            x_noisy = x_noisy_flat.reshape(b, T, F)
            eps_hat = mlp(x_noisy, t.astype(mx.float32), _batch_labels, training=True)
            noise_shaped = noise_flat.reshape(b, T, F)
            return mx.mean((eps_hat - noise_shaped) ** 2)

        loss_and_grad = nn.value_and_grad(self._mlp, loss_fn)

        self._loss_history: List[float] = []
        self._best_train_loss: Optional[float] = None

        rng = np.random.default_rng(42)
        steps_per_epoch = max(N // self.batch_size, 1)

        for epoch in range(self.epochs):
            epoch_losses = []
            perm = rng.permutation(N)
            for step in range(steps_per_epoch):
                idx = perm[step * self.batch_size : (step + 1) * self.batch_size]
                if len(idx) == 0:
                    continue
                x0 = x_data[mx.array(idx, dtype=mx.int32)]
                t = mx.random.randint(0, self.num_timesteps, (len(idx),))
                noise = mx.random.normal(x0.shape)
                # Update the closure-captured dict in place before each step.
                for name, arr in task_arrays.items():
                    _batch_labels[name] = arr[mx.array(idx, dtype=mx.int32)]

                loss, grads = loss_and_grad(self._mlp, x0, t, noise)
                optimizer.update(self._mlp, grads)

                self._ema_update()

                # CRITICAL: eval ALL updated state so Metal doesn't accumulate
                # a lazy graph that overflows the resource limit.
                mx.eval(
                    self._mlp.parameters(),
                    self._ema_mlp.parameters(),
                    optimizer.state,
                    loss,
                )

                epoch_losses.append(float(loss.item()))

            epoch_loss = float(np.mean(epoch_losses)) if epoch_losses else float("nan")
            self._loss_history.append(epoch_loss)

            if self._best_train_loss is None or epoch_loss < self._best_train_loss:
                self._best_train_loss = epoch_loss
                self._best_state = copy.deepcopy(self._ema_mlp)

            if self.verbose and (epoch % self.eval_frequency == 0 or epoch == self.epochs - 1):
                print(f"  MT_DDPM epoch {epoch + 1}/{self.epochs} loss={epoch_loss:.5f}")

    def _ema_update(self) -> None:
        """In-place EMA: ema = ema * decay + live * (1 - decay).

        Uses the same _tree_lerp pattern as TabDDPMMLX._ema_update to walk
        the parameter tree without manual recursion.
        """
        d = self.ema_decay
        live = self._mlp.parameters()
        ema = self._ema_mlp.parameters()
        new_ema = _tree_lerp(ema, live, 1.0 - d)
        self._ema_mlp.update(new_ema)

    def generate(
        self,
        n: int,
        task_labels: Dict[str, np.ndarray],
        pair_label: Optional[int] = None,
    ) -> np.ndarray:
        """Reverse-process samples. Returns shape (n, seq_len, F), float32."""
        if not isinstance(task_labels, dict):
            raise TypeError(
                f"generate() expects task_labels as Dict[str, np.ndarray]; "
                f"got {type(task_labels).__name__}"
            )

        # Convert one-hot labels to argmax indices for embedding lookup.
        label_arrays = {
            name: mx.array(np.argmax(onehot, axis=1), dtype=mx.int32)
            for name, onehot in task_labels.items()
        }

        # Use best-loss EMA snapshot if available, else current EMA.
        model = getattr(self, "_best_state", None) or self._ema_mlp or self._mlp

        # ddim_sample is 2D-hardcoded. Flatten-before-call: have ddim see
        # (n, T*F), reshape inside model_fn and on output.
        T = self.seq_len
        F = self.num_features

        def model_fn(x_t_flat: mx.array, t: mx.array, cond) -> mx.array:
            # cond is passed through by ddim_sample but unused here — labels
            # come from the closure over `label_arrays`.
            x_t_3d = x_t_flat.reshape(x_t_flat.shape[0], T, F)
            eps_3d = model(x_t_3d, t.astype(mx.float32), label_arrays, training=False)
            return eps_3d.reshape(eps_3d.shape[0], T * F)

        flat_shape = (n, T * F)
        # The `cond` arg is unused by our model_fn; pass a dummy placeholder.
        dummy_cond = mx.zeros((n,), dtype=mx.int32)

        samples_flat = ddim_sample(
            model_fn,
            flat_shape,
            dummy_cond,
            self._schedule,
            num_steps=self.num_sample_steps,
        )
        samples_3d = samples_flat.reshape(n, T, F)
        return np.asarray(samples_3d, dtype=np.float32)

    def save(self, *args, **kwargs):
        raise NotImplementedError("filled in by Task 5")

    @classmethod
    def load_from(cls, *args, **kwargs):
        raise NotImplementedError("filled in by Task 5")
