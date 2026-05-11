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


class TestDDIMSample:
    def test_oracle_inverts_to_x0(self):
        """If model_fn returns the exact ε used in q_sample, DDIM
        should recover x_0 to high accuracy."""
        from GANs.diffusion_mlx import ddim_sample, q_sample

        T = 1000
        sched = make_schedule(T=T)
        rng_key = mx.random.key(0)

        # Build a known (x_0, ε) pair and a fixed starting timestep.
        x0_true = mx.random.normal((32, 8), key=rng_key)

        # Oracle: given (x_t, t, cond), return the ε that produced x_t
        # from x0_true. We bake the relationship via q_sample math.
        def oracle_eps(x_t, t, cond):
            sqrt_ac = sched.sqrt_alphas_cumprod[t][:, None]
            sqrt_omac = sched.sqrt_one_minus_alphas_cumprod[t][:, None]
            # x_t = sqrt_ac * x0 + sqrt_omac * eps  =>  eps = (x_t - sqrt_ac*x0)/sqrt_omac
            return (x_t - sqrt_ac * x0_true) / sqrt_omac

        # Start from x_T computed by q_sample on a known noise sample.
        noise = mx.random.normal((32, 8), key=mx.random.key(1))
        t_T = mx.full((32,), T - 1, dtype=mx.int32)
        x_T = q_sample(x0_true, t_T, noise, sched)

        x_recovered = ddim_sample(
            model_fn=oracle_eps,
            shape=(32, 8),
            cond=mx.zeros((32,), dtype=mx.int32),
            sched=sched,
            num_steps=50,
            x_init=x_T,
        )

        err = mx.mean(mx.abs(x_recovered - x0_true)).item()
        assert err < 1e-2, f"oracle inversion error {err:.4f}"

    def test_determinism(self):
        """Same x_init + same model_fn → identical output."""
        from GANs.diffusion_mlx import ddim_sample

        sched = make_schedule(T=1000)
        x_T = mx.random.normal((8, 4), key=mx.random.key(42))

        def model_fn(x_t, t, cond):
            return mx.zeros_like(x_t)  # trivial model

        out1 = ddim_sample(
            model_fn=model_fn, shape=(8, 4),
            cond=mx.zeros((8,), dtype=mx.int32),
            sched=sched, num_steps=20, x_init=x_T,
        )
        out2 = ddim_sample(
            model_fn=model_fn, shape=(8, 4),
            cond=mx.zeros((8,), dtype=mx.int32),
            sched=sched, num_steps=20, x_init=x_T,
        )
        assert mx.array_equal(out1, out2).item()
