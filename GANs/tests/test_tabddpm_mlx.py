"""Smoke and integration tests for TabDDPMMLX (the model class)."""

from __future__ import annotations

import os
import sys
import shutil
import tempfile
from pathlib import Path

STRATEGIES_ROOT = str(Path(__file__).parent.parent.parent)
if STRATEGIES_ROOT not in sys.path:
    sys.path.insert(0, STRATEGIES_ROOT)

import mlx.core as mx
import numpy as np
import pytest

from GANs.df_tabddpm_mlx import TabDDPMMLX


def _toy_dataset(n=400, f=8, c=3, seed=0):
    rng = np.random.default_rng(seed)
    data = rng.normal(size=(n, f)).astype(np.float32)
    labels_int = rng.integers(0, c, size=(n,))
    labels = np.eye(c, dtype=np.float32)[labels_int]
    return data, labels


class TestSkeleton:
    def test_instantiates_with_expected_attrs(self):
        m = TabDDPMMLX(num_features=8, num_classes=3,
                       d_model=16, d_layers=(16, 16),
                       num_timesteps=100, num_sample_steps=10,
                       epochs=1, batch_size=64, verbose=False)
        assert m.num_features == 8
        assert m.num_classes == 3
        assert m.num_timesteps == 100
        assert m.num_sample_steps == 10
        # Backbone exists and produces the right output shape on a
        # synthetic forward pass.
        x = mx.zeros((4, 8))
        t = mx.zeros((4,), dtype=mx.int32)
        c = mx.zeros((4,), dtype=mx.int32)
        out = m._mlp(x, t, c)
        assert out.shape == (4, 8)
