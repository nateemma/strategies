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
