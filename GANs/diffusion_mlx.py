"""
Pure-MLX diffusion math used by TabDDPM (and reusable by any future
diffusion-based GAN type).

This module deliberately has no model dependency — every function takes
arrays in and arrays out.  The cosine β-schedule, forward `q_sample`,
and DDIM η=0 reverse sampler all live here so they can be unit-tested
in isolation; the model class in `df_tabddpm_mlx.py` calls into them.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, Optional, Tuple

import mlx.core as mx


# ---------------------------------------------------------------------------
# Schedule
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Schedule:
    """Precomputed diffusion-process constants.

    All arrays are shape (T,) and aligned: index t corresponds to the
    t-th diffusion step (0 = least noisy, T-1 = pure noise).
    """

    betas: mx.array                       # β_t
    alphas: mx.array                       # α_t = 1 - β_t
    alphas_cumprod: mx.array               # ᾱ_t = Π_{s≤t} α_s
    sqrt_alphas_cumprod: mx.array          # √ᾱ_t
    sqrt_one_minus_alphas_cumprod: mx.array  # √(1 - ᾱ_t)
    posterior_variance: mx.array           # σ²_t for stochastic DDPM reverse


def cosine_beta_schedule(T: int, s: float = 0.008) -> mx.array:
    """Nichol & Dhariwal cosine schedule — also TabDDPM's default.

    Defines  f(t) = cos((t/T + s) / (1+s) · π/2)²  for t ∈ [0, T],
    then ᾱ_t = f(t)/f(0), β_t = 1 − ᾱ_t/ᾱ_{t−1}, clipped to (0, 0.999).
    """
    steps = mx.arange(T + 1, dtype=mx.float32)
    f = mx.cos(((steps / T) + s) / (1.0 + s) * math.pi / 2.0) ** 2
    alpha_bar = f / f[0]
    betas = 1.0 - (alpha_bar[1:] / alpha_bar[:-1])
    # Clip to keep numerical sanity at the tails.
    betas = mx.clip(betas, 1e-8, 0.999)
    return betas


def make_schedule(T: int, s: float = 0.008) -> Schedule:
    """Build a full Schedule from a cosine β-schedule of length T."""
    betas = cosine_beta_schedule(T, s=s)
    alphas = 1.0 - betas
    alphas_cumprod = mx.cumprod(alphas, axis=0)
    # ᾱ_{t-1} with ᾱ_{-1} = 1
    alphas_cumprod_prev = mx.concatenate(
        [mx.ones((1,), dtype=alphas_cumprod.dtype), alphas_cumprod[:-1]], axis=0
    )
    posterior_variance = betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)
    return Schedule(
        betas=betas,
        alphas=alphas,
        alphas_cumprod=alphas_cumprod,
        sqrt_alphas_cumprod=mx.sqrt(alphas_cumprod),
        sqrt_one_minus_alphas_cumprod=mx.sqrt(1.0 - alphas_cumprod),
        posterior_variance=posterior_variance,
    )
