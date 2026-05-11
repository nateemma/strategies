"""
TabDDPMMLX — MLX-native, continuous-only, single-task TabDDPM trainer
and sampler.

Lifecycle: fit() / generate() / save() / load() — same shape as WGANMLX
so the GANInterface backend adapter is a thin wrapper.

The diffusion math lives in `diffusion_mlx`; this module owns the model
class (MLP backbone + time embedding + class embedding), the training
loop, the EMA copy used for sampling, and the safetensors + pickle
save/load lifecycle.

See `docs/superpowers/specs/2026-05-11-tabddpm-design.md` for the
design and `docs/superpowers/plans/2026-05-11-tabddpm-implementation.md`
for the implementation plan.
"""

from __future__ import annotations

import math
import os
import pickle
import time
from typing import Any, Dict, Optional, Sequence, Tuple

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np

from GANs.diffusion_mlx import Schedule, ddim_sample, make_schedule, q_sample


_META_FILENAME = "tabddpm_metadata.pkl"
_WEIGHTS_FILENAME = "tabddpm_gen_mlx.safetensors"


# ---------------------------------------------------------------------------
# Backbone
# ---------------------------------------------------------------------------


class _SinusoidalTimeEmbed(nn.Module):
    """Sinusoidal positional embedding for diffusion timesteps,
    followed by two SiLU-activated Linear layers projecting to d_model.
    Same shape the TabDDPM paper uses."""

    def __init__(self, d_model: int, sinusoid_dim: int = 128):
        super().__init__()
        self.sinusoid_dim = sinusoid_dim
        self.proj1 = nn.Linear(sinusoid_dim, d_model)
        self.proj2 = nn.Linear(d_model, d_model)

    def __call__(self, t: mx.array) -> mx.array:
        # t: (B,) int32 → (B, sinusoid_dim) sin/cos features → (B, d_model).
        half = self.sinusoid_dim // 2
        freqs = mx.exp(
            -math.log(10000.0) * mx.arange(half, dtype=mx.float32) / half
        )
        args = t.astype(mx.float32)[:, None] * freqs[None, :]
        emb = mx.concatenate([mx.sin(args), mx.cos(args)], axis=-1)
        emb = nn.silu(self.proj1(emb))
        emb = nn.silu(self.proj2(emb))
        return emb


class _MLPBlock(nn.Module):
    """Linear → ReLU → Dropout, the TabDDPM paper's block primitive."""

    def __init__(self, d_in: int, d_out: int, dropout: float = 0.0):
        super().__init__()
        self.linear = nn.Linear(d_in, d_out)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else None

    def __call__(self, x: mx.array) -> mx.array:
        x = nn.relu(self.linear(x))
        if self.dropout is not None:
            x = self.dropout(x)
        return x


class _TabDDPMMLP(nn.Module):
    """MLP backbone: x_proj + t_embed + class_embed → stacked blocks → head."""

    def __init__(
        self,
        num_features: int,
        num_classes: int,
        d_model: int = 256,
        d_layers: Sequence[int] = (256, 256),
        dropout: float = 0.0,
    ):
        super().__init__()
        self.x_proj = nn.Linear(num_features, d_model)
        self.t_embed = _SinusoidalTimeEmbed(d_model)
        self.class_embed = nn.Embedding(num_classes, d_model)

        dims = [d_model, *d_layers]
        self.blocks = [
            _MLPBlock(dims[i], dims[i + 1], dropout=dropout)
            for i in range(len(dims) - 1)
        ]
        self.head = nn.Linear(dims[-1], num_features)

    def __call__(self, x_t: mx.array, t: mx.array, class_idx: mx.array) -> mx.array:
        h = self.x_proj(x_t) + self.t_embed(t) + self.class_embed(class_idx)
        for blk in self.blocks:
            h = blk(h)
        return self.head(h)


# ---------------------------------------------------------------------------
# TabDDPMMLX — outer class
# ---------------------------------------------------------------------------


class TabDDPMMLX:
    """Continuous-only, single-task, MLX-native TabDDPM.

    Construct with feature/class dimensions and (optional) hyperparams;
    call fit(data, labels) once; then generate() / save() / load().
    """

    def __init__(
        self,
        num_features: int = 0,
        num_classes: int = 0,
        *,
        d_model: int = 256,
        d_layers: Sequence[int] = (256, 256),
        dropout: float = 0.0,
        num_timesteps: int = 1000,
        num_sample_steps: int = 50,
        epochs: int = 300,
        batch_size: int = 4096,
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-5,
        ema_decay: float = 0.999,
        eval_frequency: int = 20,
        verbose: bool = True,
    ):
        self.num_features = num_features
        self.num_classes = num_classes
        self.d_model = d_model
        self.d_layers = tuple(d_layers)
        self.dropout = dropout
        self.num_timesteps = num_timesteps
        self.num_sample_steps = num_sample_steps
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.ema_decay = ema_decay
        self.eval_frequency = eval_frequency
        self.verbose = verbose

        # Feature stats populated by fit(); used by _postprocess.
        self.feature_min: Optional[np.ndarray] = None
        self.feature_max: Optional[np.ndarray] = None

        # Models created lazily in fit() once we know num_features/num_classes.
        # Skeleton instantiation (e.g. before load_from) still needs the
        # MLPs so the test can inspect their shapes — only build them when
        # dimensions are known.
        if num_features > 0 and num_classes > 0:
            self._build_models()
        else:
            self._mlp = None
            self._ema_mlp = None

        self._sched: Schedule = make_schedule(self.num_timesteps)

    def _build_models(self) -> None:
        self._mlp = _TabDDPMMLP(
            self.num_features, self.num_classes,
            d_model=self.d_model, d_layers=self.d_layers,
            dropout=self.dropout,
        )
        self._ema_mlp = _TabDDPMMLP(
            self.num_features, self.num_classes,
            d_model=self.d_model, d_layers=self.d_layers,
            dropout=self.dropout,
        )
