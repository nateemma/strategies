"""
Pure-MLX EDM-style sigma-schedule diffusion helpers
(Karras et al. 2022, "Elucidating the Design Space of Diffusion-Based
Generative Models").

Drop-in alternative to the cosine-β schedule in ``diffusion_mlx.py``,
toggled via ``TabDDPMMLX(use_edm_schedule=True)``.  Uses ε-prediction
(no preconditioning) for minimal divergence from the existing model
contract — what changes is the noise schedule (log-uniform σ instead
of integer-t cosine-β) and the reverse-process sampler (Heun's 2nd-order
ODE solver with a ρ-parameterised σ ramp instead of DDIM-η=0).

Why sigma schedules suit tabular data better than cosine-β:
  * Cosine-β was tuned for unit-variance image pixels — feature mix
    homogeneous across channels.  Tabular features have heterogeneous
    variances even after z-score, and cosine spends most of its β
    budget at the high-noise end where tabular structure is already
    drowned in noise.
  * Log-uniform σ sampling naturally covers many noise-amplitude scales,
    so the model spends gradient signal where it matters per-feature
    rather than per-timestep.
"""

from __future__ import annotations

from typing import Callable, Optional, Tuple

import mlx.core as mx


# ---------------------------------------------------------------------------
# Default hyperparameters (EDM paper Section 5.4 + Table 1)
# ---------------------------------------------------------------------------

# σ sampling: log-normal during training, log(σ) ~ N(P_mean, P_std).
DEFAULT_P_MEAN: float = -1.2
DEFAULT_P_STD: float = 1.2

# σ schedule for sampling: σ_i = (σ_max^(1/ρ) + i/(N-1) · (σ_min^(1/ρ) - σ_max^(1/ρ)))^ρ
DEFAULT_SIGMA_MIN: float = 0.002
DEFAULT_SIGMA_MAX: float = 80.0
DEFAULT_RHO: float = 7.0


# ---------------------------------------------------------------------------
# Training-time σ sampling
# ---------------------------------------------------------------------------


def sample_log_normal_sigma(
    batch_size: int,
    p_mean: float = DEFAULT_P_MEAN,
    p_std: float = DEFAULT_P_STD,
    key: Optional[mx.array] = None,
) -> mx.array:
    """Sample σ values from a log-normal distribution.

    ln(σ) ~ N(p_mean, p_std²)  →  σ = exp(ln_σ).  Standard EDM training
    samples σ this way at every step instead of picking a discrete
    timestep uniformly.

    Returns:
        (batch_size,) mx.array of σ values, all > 0.
    """
    ln_sigma = p_mean + p_std * mx.random.normal((batch_size,), key=key)
    return mx.exp(ln_sigma)


# ---------------------------------------------------------------------------
# Sampling-time σ schedule (ρ-ramp from σ_max down to σ_min)
# ---------------------------------------------------------------------------


def build_sigma_schedule(
    num_steps: int,
    sigma_min: float = DEFAULT_SIGMA_MIN,
    sigma_max: float = DEFAULT_SIGMA_MAX,
    rho: float = DEFAULT_RHO,
) -> mx.array:
    """Build the σ ramp used by Heun sampling.

    Returns:
        (num_steps + 1,) mx.array.  Element i is σ_i;  σ_0 = sigma_max,
        σ_{num_steps-1} = sigma_min, and an appended σ_num_steps = 0
        so the loop's final step lands on the data manifold exactly.
    """
    ramp = mx.arange(num_steps, dtype=mx.float32) / max(num_steps - 1, 1)
    inv_rho = 1.0 / rho
    smax_pow = sigma_max ** inv_rho
    smin_pow = sigma_min ** inv_rho
    sigmas = (smax_pow + ramp * (smin_pow - smax_pow)) ** rho
    # Append σ=0 so the final Heun step terminates on the data manifold.
    return mx.concatenate([sigmas, mx.zeros((1,), dtype=mx.float32)], axis=0)


# ---------------------------------------------------------------------------
# Reverse process — Heun's 2nd-order ODE solver
# ---------------------------------------------------------------------------


def heun_sample(
    model_fn: Callable[[mx.array, mx.array, mx.array], mx.array],
    shape: Tuple[int, ...],
    cond: mx.array,
    sigmas: mx.array,
    x_init: Optional[mx.array] = None,
    key: Optional[mx.array] = None,
) -> mx.array:
    """Heun's 2nd-order deterministic ODE sampler.

    Solves the probability-flow ODE dx/dσ = ε(x, σ) where ε is the
    network's noise prediction.  Each step combines an Euler half and
    a correction half evaluated at the next σ — quadratic accuracy in
    the step size for half the FLOPs of a fixed-step 4th-order method.

    Args:
        model_fn:  callable(x_σ, σ_array, cond) -> ε_hat. σ_array has
                   shape (batch_size,) and is broadcast across feature dim.
        shape:     output shape (e.g. (n, F)).
        cond:      conditioning tensor (e.g. class indices), passed through.
        sigmas:    (num_steps + 1,) σ schedule; sigmas[-1] should be 0.
        x_init:    optional starting x at σ_max.  Defaults to
                   σ_max · N(0, I).
        key:       optional mlx.random key for reproducible x_init.

    Returns:
        x_0 of shape `shape` — the denoised samples.
    """
    sigma_max = float(sigmas[0].item())
    if x_init is None:
        x = sigma_max * mx.random.normal(shape, key=key)
    else:
        x = x_init

    batch = shape[0]
    num_steps = sigmas.shape[0] - 1

    for i in range(num_steps):
        sigma_cur = float(sigmas[i].item())
        sigma_next = float(sigmas[i + 1].item())
        sigma_cur_b = mx.full((batch,), sigma_cur, dtype=mx.float32)

        # ε at current σ — this IS the ODE derivative when working in
        # the ε-prediction parameterisation (since D = x − σ · ε
        # ⇒ dx/dσ = (x − D)/σ = ε).
        eps_cur = model_fn(x, sigma_cur_b, cond)
        dt = sigma_next - sigma_cur                # negative — σ decreases

        # Euler half-step.
        x_euler = x + dt * eps_cur

        if sigma_next > 0:
            # Correction half-step at σ_next using the Euler estimate.
            sigma_next_b = mx.full((batch,), sigma_next, dtype=mx.float32)
            eps_next = model_fn(x_euler, sigma_next_b, cond)
            x = x + dt * 0.5 * (eps_cur + eps_next)
        else:
            # Last step lands on σ=0: Euler is already exact since
            # there's no σ_next to evaluate ε at.
            x = x_euler

    return x
