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


# ---------------------------------------------------------------------------
# Forward (training-time noising)
# ---------------------------------------------------------------------------


def q_sample(
    x0: mx.array, t: mx.array, noise: mx.array, sched: Schedule
) -> mx.array:
    """Forward diffusion: x_t = √ᾱ_t · x_0 + √(1-ᾱ_t) · ε.

    Args:
        x0:    (B, F) clean samples in [-1, 1].
        t:     (B,) int32 timestep indices in [0, T-1].
        noise: (B, F) standard-normal noise, same shape as x0.
        sched: Schedule from make_schedule.

    Returns:
        (B, F) noised samples x_t.
    """
    sqrt_ac = sched.sqrt_alphas_cumprod[t]               # (B,)
    sqrt_omac = sched.sqrt_one_minus_alphas_cumprod[t]   # (B,)
    return sqrt_ac[:, None] * x0 + sqrt_omac[:, None] * noise


# ---------------------------------------------------------------------------
# Reverse (DDIM η=0 sampler)
# ---------------------------------------------------------------------------


def ddim_sample(
    model_fn: Callable[[mx.array, mx.array, mx.array], mx.array],
    shape: Tuple[int, ...],
    cond: mx.array,
    sched: Schedule,
    num_steps: int = 50,
    x_init: Optional[mx.array] = None,
    key: Optional[mx.array] = None,
) -> mx.array:
    """Deterministic DDIM reverse process (η=0).

    Args:
        model_fn: Callable(x_t, t, cond) -> ε̂ of shape `shape`.
                  No model dependency in this module — caller passes
                  a closure over its trained network.
        shape:    Output shape (e.g. (n, F)).
        cond:     Conditioning tensor (e.g. (n,) class indices) passed
                  through to model_fn untouched.
        sched:    Schedule from make_schedule.
        num_steps: Number of DDIM steps (50 is a good default).
        x_init:   Optional starting x_T. If None, sampled from N(0, I).
        key:      Optional mx.random key for reproducible x_init.

    Returns:
        Raw x_0 of shape `shape`. No clipping in this module — the
        caller's _postprocess handles clipping + inverse minmax.
    """
    T = sched.betas.shape[0]
    if x_init is None:
        x = mx.random.normal(shape, key=key)
    else:
        x = x_init

    # Build a num_steps-length sub-sequence of timesteps T-1, ..., 0.
    # Use evenly spaced integer indices into [0, T-1].
    step_idx = mx.linspace(0, T - 1, num_steps, dtype=mx.float32)
    step_idx = mx.round(step_idx).astype(mx.int32)
    # Convert to a python list of ints (sub-sequences are short — fine
    # for a 50-step loop, and indexed scalar gather is simpler this way).
    timesteps = [int(step_idx[i].item()) for i in range(num_steps)][::-1]

    batch = shape[0]
    for i, t_int in enumerate(timesteps):
        t_arr = mx.full((batch,), t_int, dtype=mx.int32)
        eps_hat = model_fn(x, t_arr, cond)

        sqrt_ac = sched.sqrt_alphas_cumprod[t_arr][:, None]
        sqrt_omac = sched.sqrt_one_minus_alphas_cumprod[t_arr][:, None]
        x0_hat = (x - sqrt_omac * eps_hat) / sqrt_ac

        if i == len(timesteps) - 1:
            # Final step: return x̂_0 directly (t_prev would be -1).
            x = x0_hat
        else:
            t_prev = timesteps[i + 1]
            ac_prev = sched.alphas_cumprod[t_prev]
            sqrt_ac_prev = mx.sqrt(ac_prev)
            sqrt_omac_prev = mx.sqrt(1.0 - ac_prev)
            x = sqrt_ac_prev * x0_hat + sqrt_omac_prev * eps_hat

    return x
