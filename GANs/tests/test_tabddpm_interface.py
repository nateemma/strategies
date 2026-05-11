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


class TestBalanceIntegration:
    def test_balance_single_task_with_tab_ddpm(self):
        from GANs.balance import balance_single_task

        tmp = tempfile.mkdtemp()
        try:
            iface = GANInterface(GANType.TAB_DDPM, save_path=tmp)
            # Use imbalanced dataset: 100 class 0, 20 class 1, 10 class 2.
            # With target_ratio=1.0, we want all classes to have 100 samples.
            rng = np.random.default_rng(seed=42)
            data_c0 = rng.normal(size=(100, 8)).astype(np.float32)
            data_c1 = rng.normal(size=(20, 8)).astype(np.float32)
            data_c2 = rng.normal(size=(10, 8)).astype(np.float32)
            data = np.vstack([data_c0, data_c1, data_c2])
            labels_c0 = np.eye(3, dtype=np.float32)[np.zeros(100, dtype=int)]
            labels_c1 = np.eye(3, dtype=np.float32)[np.ones(20, dtype=int)]
            labels_c2 = np.eye(3, dtype=np.float32)[np.full(10, 2, dtype=int)]
            labels = np.vstack([labels_c0, labels_c1, labels_c2])

            iface.fit(data, labels,
                      d_model=16, d_layers=(16, 16),
                      num_timesteps=50, num_sample_steps=10,
                      epochs=2, batch_size=64, verbose=False)

            aug_data, aug_labels = balance_single_task(
                interface=iface, data=data, labels=labels, target_ratio=1.0,
                log=lambda *a, **kw: None, debug_log=lambda *a, **kw: None,
            )
            # Augmented set is larger than the input (we added samples).
            assert aug_data.shape[0] > data.shape[0]
            # 2-D output (squeeze path was taken) — same shape as input.
            assert aug_data.ndim == 2
            assert aug_data.shape[1] == 8
        finally:
            shutil.rmtree(tmp, ignore_errors=True)
