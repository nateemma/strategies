"""Test the balance_with_mt_ddpm_mlx function wrapper."""

import numpy as np
import pytest

from GANs.df_mt_ddpm_mlx import balance_with_mt_ddpm_mlx


def test_balance_returns_expected_tuple():
    rng = np.random.default_rng(3)
    seq_len, F, C = 8, 6, 3
    N = 128
    data = rng.normal(0, 1, size=(N, seq_len, F)).astype(np.float32)
    labels = {"trading": np.eye(C, dtype=np.float32)[rng.integers(0, C, size=N)]}

    out_data, out_labels, model = balance_with_mt_ddpm_mlx(
        data, labels,
        epochs=2,
        batch_size=32,
        save_path=None,
        assess_quality=False,
        task_target_ratios={k: 0.0 for k in labels},
        _return_model=True,
        d_model=32,
        d_layers=2,
        num_timesteps=50,
        num_sample_steps=10,
        verbose=False,
    )

    # When task_target_ratios = 0, no augmentation — data/labels pass through.
    assert out_data.shape == data.shape
    for k in labels:
        assert out_labels[k].shape == labels[k].shape
    assert model is not None
    assert hasattr(model, "generate")
