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

    def __call__(self, x: mx.array, training: bool) -> mx.array:
        h = self.norm(x)
        h = self.fc(h)
        h = nn.gelu(h)
        if training and self.dropout_p > 0.0:
            h = nn.dropout(h, p=self.dropout_p)
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

    def fit(self, *args, **kwargs):
        raise NotImplementedError("filled in by Task 3")

    def generate(self, *args, **kwargs):
        raise NotImplementedError("filled in by Task 4")

    def save(self, *args, **kwargs):
        raise NotImplementedError("filled in by Task 5")

    @classmethod
    def load_from(cls, *args, **kwargs):
        raise NotImplementedError("filled in by Task 5")
