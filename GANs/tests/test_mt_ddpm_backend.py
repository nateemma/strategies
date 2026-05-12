"""Test the MTDDPMMLXBackend GANBackend adapter."""

import numpy as np
import pytest

from GANs.GANType import GANType
from GANs.backends.mt_ddpm import MTDDPMMLXBackend


def test_backend_registered_with_correct_type():
    backend = MTDDPMMLXBackend()
    assert backend.GAN_TYPE == GANType.MT_DDPM


def test_backend_fit_generate_roundtrip():
    rng = np.random.default_rng(4)
    seq_len, F, C = 8, 6, 3
    N = 64
    data = rng.normal(0, 1, size=(N, seq_len, F)).astype(np.float32)
    labels = {"trading": np.eye(C, dtype=np.float32)[rng.integers(0, C, size=N)]}

    backend = MTDDPMMLXBackend()
    backend.fit(
        data, labels,
        epochs=2, batch_size=32,
        d_model=32, d_layers=2,
        num_timesteps=50, num_sample_steps=10,
        verbose=False,
    )

    gen_labels = {"trading": np.eye(C, dtype=np.float32)[rng.integers(0, C, size=16)]}
    samples = backend.generate(16, task_labels=gen_labels)
    assert samples.shape == (16, seq_len, F)
    assert np.all(np.isfinite(samples))


def test_backend_generate_before_fit_raises():
    backend = MTDDPMMLXBackend()
    with pytest.raises(RuntimeError, match="before fit"):
        backend.generate(8)
