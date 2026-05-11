"""Backend adapter tests for TabDDPM."""

from __future__ import annotations

import os
import sys
import shutil
import tempfile
from pathlib import Path

STRATEGIES_ROOT = str(Path(__file__).parent.parent.parent)
if STRATEGIES_ROOT not in sys.path:
    sys.path.insert(0, STRATEGIES_ROOT)

import numpy as np
import pytest

import GANs.backends  # noqa: F401 — side-effect registration
from GANs.GANBackend import resolve_backend
from GANs.GANType import GANType


def _toy_dataset(n=200, f=8, c=3, seed=0):
    rng = np.random.default_rng(seed)
    data = rng.normal(size=(n, f)).astype(np.float32)
    labels_int = rng.integers(0, c, size=(n,))
    return data, np.eye(c, dtype=np.float32)[labels_int]


class TestBackendRegistry:
    def test_backend_registered(self):
        backend_cls = resolve_backend(GANType.TAB_DDPM, prefer_mlx=True)
        assert backend_cls.__name__ == "TabDDPMMLXBackend"
        assert backend_cls.PREFERS_MLX is True


class TestBackendLifecycle:
    def test_fit_generate_save_load_roundtrip(self):
        backend_cls = resolve_backend(GANType.TAB_DDPM, prefer_mlx=True)
        backend = backend_cls()

        data, labels = _toy_dataset()
        backend.fit(
            data, labels,
            d_model=16, d_layers=(16, 16),
            num_timesteps=50, num_sample_steps=10,
            epochs=2, batch_size=64, verbose=False,
        )

        one_hot = np.zeros((5, 3), dtype=np.float32)
        one_hot[:, 0] = 1.0
        gen = backend.generate(5, one_hot=one_hot)
        assert gen.shape == (5, 1, 8)

        tmp = tempfile.mkdtemp()
        try:
            backend.save(tmp, training_type=2)
            backend2, meta = backend_cls.load(tmp)
            assert meta["training_type"] == 2
            gen2 = backend2.generate(5, one_hot=one_hot)
            assert gen2.shape == (5, 1, 8)
        finally:
            shutil.rmtree(tmp, ignore_errors=True)

    def test_generate_requires_one_hot(self):
        backend_cls = resolve_backend(GANType.TAB_DDPM, prefer_mlx=True)
        backend = backend_cls()
        data, labels = _toy_dataset()
        backend.fit(data, labels,
                    d_model=16, d_layers=(16, 16),
                    num_timesteps=20, num_sample_steps=5,
                    epochs=1, batch_size=64, verbose=False)
        with pytest.raises(ValueError, match="one_hot"):
            backend.generate(5)

    def test_load_missing_files_raises_filenotfound(self):
        backend_cls = resolve_backend(GANType.TAB_DDPM, prefer_mlx=True)
        tmp = tempfile.mkdtemp()
        try:
            with pytest.raises(FileNotFoundError):
                backend_cls.load(tmp)
        finally:
            shutil.rmtree(tmp, ignore_errors=True)
