"""Unit tests for diffusion_mlx — pure-math diffusion utilities."""

from __future__ import annotations

import sys
from pathlib import Path

STRATEGIES_ROOT = str(Path(__file__).parent.parent.parent)
if STRATEGIES_ROOT not in sys.path:
    sys.path.insert(0, STRATEGIES_ROOT)

import mlx.core as mx
import numpy as np
import pytest

from GANs.diffusion_mlx import (
    Schedule,
    cosine_beta_schedule,
    make_schedule,
)


class TestCosineSchedule:
    def test_returns_T_betas(self):
        betas = cosine_beta_schedule(T=1000)
        assert betas.shape == (1000,)

    def test_betas_in_open_unit_interval(self):
        betas = np.asarray(cosine_beta_schedule(T=1000))
        assert (betas > 0).all()
        assert (betas < 1).all()

    def test_alpha_cumprod_decays_to_near_zero(self):
        sched = make_schedule(T=1000)
        alphas_cumprod = np.asarray(sched.alphas_cumprod)
        assert alphas_cumprod[0] > 0.99
        assert alphas_cumprod[-1] < 1e-2

    def test_schedule_fields_have_right_shape(self):
        sched = make_schedule(T=200)
        for field_name in (
            "betas", "alphas", "alphas_cumprod",
            "sqrt_alphas_cumprod", "sqrt_one_minus_alphas_cumprod",
            "posterior_variance",
        ):
            arr = getattr(sched, field_name)
            assert arr.shape == (200,), f"{field_name}: got {arr.shape}"


class TestQSample:
    def test_identity_at_t_zero(self):
        from GANs.diffusion_mlx import q_sample

        sched = make_schedule(T=1000)
        x0 = mx.random.normal((16, 8))
        noise = mx.random.normal((16, 8))
        t = mx.zeros((16,), dtype=mx.int32)

        x_t = q_sample(x0, t, noise, sched)
        # At t=0, ᾱ_0 ≈ 1, so x_t ≈ x_0 (small residual noise from
        # sqrt(1-ᾱ_0) ≈ 0 but not exactly zero — assert close-ish).
        diff = mx.mean(mx.abs(x_t - x0)).item()
        assert diff < 0.05, f"q_sample at t=0 drifted by {diff:.4f}"

    def test_unit_variance_at_t_T_minus_one(self):
        from GANs.diffusion_mlx import q_sample

        T = 1000
        sched = make_schedule(T=T)
        x0 = mx.random.normal((4096, 8))
        noise = mx.random.normal((4096, 8))
        t = mx.full((4096,), T - 1, dtype=mx.int32)

        x_t = np.asarray(q_sample(x0, t, noise, sched))
        # At t=T-1, ᾱ_t ≈ 0, so x_t ≈ noise → per-column variance ≈ 1.
        var = x_t.var(axis=0)
        assert (var > 0.85).all() and (var < 1.15).all(), f"per-col var={var}"
