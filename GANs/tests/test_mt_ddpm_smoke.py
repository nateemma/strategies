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


def test_fit_runs_and_loss_decreases():
    rng = np.random.default_rng(0)
    seq_len, F = 8, 6
    N = 256
    t_grid = np.linspace(0, 2 * np.pi, seq_len, dtype=np.float32)
    freqs = rng.uniform(0.5, 2.0, size=(N, 1)).astype(np.float32)
    data = np.sin(freqs * t_grid)[..., None] * rng.normal(0, 1, size=(N, 1, F)).astype(np.float32)
    labels = {"trading": np.eye(3, dtype=np.float32)[rng.integers(0, 3, size=N)]}

    model = MTDDPMMLX(
        seq_len=seq_len,
        num_features=F,
        task_label_dims={"trading": 3},
        d_model=64,
        d_layers=2,
        num_timesteps=100,
        num_sample_steps=10,
        epochs=10,
        batch_size=64,
        verbose=False,
    )
    model.fit(data, labels)

    assert model._best_train_loss is not None
    assert model._loss_history[0] > model._best_train_loss
