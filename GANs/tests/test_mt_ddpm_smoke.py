"""Smoke tests for MTDDPMMLX construction and basic forward pass."""

import os
import sys

import mlx.core as mx
import numpy as np
import pytest

from GANs.df_mt_ddpm_mlx import MTDDPMMLX


@pytest.fixture(scope="module")
def small_model():
    return MTDDPMMLX(
        seq_len=8,
        num_features=6,
        task_label_dims={"trading": 3},
        d_model=64,
        d_layers=2,
        num_timesteps=100,
        num_sample_steps=10,
        epochs=2,
        batch_size=16,
        verbose=False,
    )


def test_construction(small_model):
    assert small_model.seq_len == 8
    assert small_model.num_features == 6
    assert small_model.task_label_dims == {"trading": 3}


def test_total_params_finite(small_model):
    # Tree of all params — should be non-empty and all finite.
    flat = []
    for _, v in small_model._mlp.parameters().items():
        def walk(x):
            if isinstance(x, dict):
                for vv in x.values():
                    walk(vv)
            elif isinstance(x, list):
                for vv in x:
                    walk(vv)
            else:
                flat.append(x)
        walk(v)
    assert len(flat) > 0
    for p in flat:
        assert mx.all(mx.isfinite(p)).item()
