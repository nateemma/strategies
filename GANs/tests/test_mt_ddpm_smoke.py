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


def test_generate_shape_and_finiteness():
    rng = np.random.default_rng(1)
    seq_len, F, C = 8, 6, 3
    N = 128
    data = rng.normal(0, 1, size=(N, seq_len, F)).astype(np.float32)
    labels = {"trading": np.eye(C, dtype=np.float32)[rng.integers(0, C, size=N)]}

    model = MTDDPMMLX(
        seq_len=seq_len,
        num_features=F,
        task_label_dims={"trading": C},
        d_model=32,
        d_layers=2,
        num_timesteps=50,
        num_sample_steps=10,
        epochs=3,
        batch_size=32,
        verbose=False,
    )
    model.fit(data, labels)

    n_gen = 64
    gen_labels = {"trading": np.eye(C, dtype=np.float32)[rng.integers(0, C, size=n_gen)]}
    samples = model.generate(n_gen, gen_labels)
    assert samples.shape == (n_gen, seq_len, F)
    assert np.all(np.isfinite(samples))
    # Per-feature std should be > 0 (no collapse on tiny training).
    assert (samples.reshape(-1, F).std(axis=0) > 0).all()


def test_save_load_roundtrip(tmp_path):
    rng = np.random.default_rng(2)
    seq_len, F, C = 8, 6, 3
    N = 64
    data = rng.normal(0, 1, size=(N, seq_len, F)).astype(np.float32)
    labels = {"trading": np.eye(C, dtype=np.float32)[rng.integers(0, C, size=N)]}

    model = MTDDPMMLX(
        seq_len=seq_len,
        num_features=F,
        task_label_dims={"trading": C},
        d_model=32,
        d_layers=2,
        num_timesteps=50,
        num_sample_steps=10,
        epochs=2,
        batch_size=32,
        verbose=False,
    )
    model.fit(data, labels)

    save_dir = str(tmp_path / "ckpt")
    os.makedirs(save_dir, exist_ok=True)
    model.save(save_dir)

    loaded, meta = MTDDPMMLX.load_from(save_dir)
    assert loaded.seq_len == seq_len
    assert loaded.num_features == F
    assert loaded.task_label_dims == {"trading": C}

    # Generate must work on the loaded model.
    gen_labels = {"trading": np.eye(C, dtype=np.float32)[rng.integers(0, C, size=8)]}
    samples = loaded.generate(8, gen_labels)
    assert samples.shape == (8, seq_len, F)
    assert np.all(np.isfinite(samples))
