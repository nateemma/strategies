"""GANInterface plumbing tests for TabDDPM."""

from __future__ import annotations

import sys
import tempfile
import shutil
from pathlib import Path

STRATEGIES_ROOT = str(Path(__file__).parent.parent.parent)
if STRATEGIES_ROOT not in sys.path:
    sys.path.insert(0, STRATEGIES_ROOT)

import numpy as np
import pytest

from GANs.GANInterface import GANInterface, _BACKEND_MIGRATED
from GANs.GANType import GANType


def _toy_dataset(n=200, f=8, c=3, seed=0):
    rng = np.random.default_rng(seed)
    data = rng.normal(size=(n, f)).astype(np.float32)
    labels_int = rng.integers(0, c, size=(n,))
    return data, np.eye(c, dtype=np.float32)[labels_int]


class TestInterfaceWiring:
    def test_tab_ddpm_in_backend_migrated(self):
        assert GANType.TAB_DDPM in _BACKEND_MIGRATED

    def test_defaults_present(self):
        assert GANType.TAB_DDPM in GANInterface._DEFAULTS
        d = GANInterface._DEFAULTS[GANType.TAB_DDPM]
        assert d["num_timesteps"] == 1000
        assert d["num_sample_steps"] == 50

    def test_fit_generate_via_interface(self):
        tmp = tempfile.mkdtemp()
        try:
            iface = GANInterface(GANType.TAB_DDPM, save_path=tmp)
            data, labels = _toy_dataset()
            iface.fit(data, labels,
                      d_model=16, d_layers=(16, 16),
                      num_timesteps=50, num_sample_steps=10,
                      epochs=2, batch_size=64, verbose=False)

            one_hot = np.zeros((10, 3), dtype=np.float32)
            one_hot[:, 0] = 1.0
            gen = iface.generate(10, one_hot=one_hot)
            assert gen.shape == (10, 1, 8)

            iface.save(training_type=2)
            iface2 = GANInterface(GANType.TAB_DDPM, save_path=tmp)
            meta = iface2.load(expected={"training_type": 2})
            assert meta["training_type"] == 2
        finally:
            shutil.rmtree(tmp, ignore_errors=True)
