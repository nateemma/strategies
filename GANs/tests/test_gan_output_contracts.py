"""
Generated-output contract tests for the GAN backends.

These tests document and enforce what every GAN backend's ``generate()``
output must look like — shape, dtype, count, finiteness — when called
through GANInterface.  They use mock models with controllable output
so the tests run without TF or MLX.

What they catch:
  * Shape regressions during refactor (e.g. forgetting the seq_len axis
    for sequential GANs, dropping the labels dict from MT_*).
  * Sample-count drift (callers ask for n, expect n back).
  * Dtype changes (float32 vs float64).

What they intentionally don't catch (covered by Phase 1C, the
real-backend smoke tests, since they need actual training):
  * The actual generator producing NaN or out-of-range samples.

Tests marked @expectedFailure document the contract we WANT — not
what's enforced today.  When Phase 3 lands the unified GANBackend
contract with output validation, the xfail decorators come off.

Run from the strategies root:
    python -m pytest GANs/tests/test_gan_output_contracts.py -v
"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path
from typing import Tuple
from unittest.mock import MagicMock

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------

STRATEGIES_ROOT = str(Path(__file__).parent.parent.parent)
if STRATEGIES_ROOT not in sys.path:
    sys.path.insert(0, STRATEGIES_ROOT)

_ft_stub = MagicMock()
sys.modules.setdefault("freqtrade", _ft_stub)
sys.modules.setdefault("freqtrade.persistence", _ft_stub)
sys.modules.setdefault("freqtrade.strategy", _ft_stub)

from GANs.GANType import GANType  # noqa: E402
from GANs.GANInterface import GANInterface  # noqa: E402


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _interface_with_model(gan_type: GANType, model: MagicMock) -> GANInterface:
    iface = GANInterface(gan_type, save_path="/tmp", prefer_mlx=False)
    iface._model = model
    return iface


def _one_hot(n: int, num_classes: int = 3, cls: int = 1) -> np.ndarray:
    out = np.zeros((n, num_classes), dtype=np.float32)
    out[:, cls] = 1.0
    return out


# ---------------------------------------------------------------------------
# WGAN — single-task, conditioned on one_hot, returns (n, 1, F)
# ---------------------------------------------------------------------------


class TestWGANOutputContract(unittest.TestCase):

    def test_returns_three_dim_tensor(self):
        n, F = 50, 25
        model = MagicMock()
        model.generate.return_value = np.zeros((n, 1, F), dtype=np.float32)

        iface = _interface_with_model(GANType.WGAN, model)
        out = iface.generate(n, one_hot=_one_hot(n))

        self.assertEqual(out.ndim, 3, "WGAN output must be (n, seq_len, F)")
        self.assertEqual(out.shape[0], n, "WGAN must return exactly n samples")
        self.assertEqual(out.shape[2], F, "WGAN feature dim must be preserved")

    def test_dtype_is_float32(self):
        n = 10
        model = MagicMock()
        model.generate.return_value = np.zeros((n, 1, 25), dtype=np.float32)

        iface = _interface_with_model(GANType.WGAN, model)
        out = iface.generate(n, one_hot=_one_hot(n))

        self.assertEqual(np.asarray(out).dtype, np.float32)

    def test_rejects_non_finite_output(self):
        """When the underlying generator emits NaN/Inf, GANInterface
        raises ValueError — non-finite samples must not reach the
        consumer."""
        n = 50
        bad = np.zeros((n, 1, 25), dtype=np.float32)
        bad[3, 0, 7] = np.nan
        bad[10, 0, 0] = np.inf

        model = MagicMock()
        model.generate.return_value = bad

        iface = _interface_with_model(GANType.WGAN, model)
        with self.assertRaises(ValueError):
            iface.generate(n, one_hot=_one_hot(n))


# ---------------------------------------------------------------------------
# MT_WGAN — multi-task, conditioned on dict, returns (samples, labels_dict)
# ---------------------------------------------------------------------------


class TestMTWGANOutputContract(unittest.TestCase):

    @staticmethod
    def _task_labels(n: int):
        return {
            "trading":  _one_hot(n, 3, 1),
            "regime":   _one_hot(n, 3, 1),
            "risk":     _one_hot(n, 3, 1),
            "momentum": _one_hot(n, 3, 1),
            "flow":     _one_hot(n, 3, 1),
            "profit":   _one_hot(n, 3, 1),
        }

    def test_returns_samples_and_labels_tuple(self):
        n, S, F = 50, 16, 25
        labels = self._task_labels(n)
        model = MagicMock()
        model.generate.return_value = (
            np.zeros((n, S, F), dtype=np.float32),
            labels,
        )

        iface = _interface_with_model(GANType.MT_WGAN, model)
        out = iface.generate(n, task_labels=labels)

        self.assertIsInstance(out, tuple)
        self.assertEqual(len(out), 2)

    def test_sample_tensor_shape(self):
        n, S, F = 50, 16, 25
        labels = self._task_labels(n)
        model = MagicMock()
        model.generate.return_value = (
            np.zeros((n, S, F), dtype=np.float32),
            labels,
        )

        iface = _interface_with_model(GANType.MT_WGAN, model)
        gen_x, _ = iface.generate(n, task_labels=labels)

        self.assertEqual(gen_x.shape, (n, S, F))

    def test_labels_dict_round_trip_keys(self):
        n = 50
        labels = self._task_labels(n)
        model = MagicMock()
        model.generate.return_value = (
            np.zeros((n, 16, 25), dtype=np.float32),
            labels,
        )

        iface = _interface_with_model(GANType.MT_WGAN, model)
        _, gen_labels = iface.generate(n, task_labels=labels)

        self.assertEqual(set(gen_labels.keys()), set(labels.keys()))
        for task in labels:
            self.assertEqual(gen_labels[task].shape, labels[task].shape,
                             f"shape mismatch for task {task!r}")

    def test_rejects_non_finite_samples(self):
        n, S, F = 50, 16, 25
        labels = self._task_labels(n)
        bad = np.zeros((n, S, F), dtype=np.float32)
        bad[7, 4, 12] = np.nan

        model = MagicMock()
        model.generate.return_value = (bad, labels)

        iface = _interface_with_model(GANType.MT_WGAN, model)
        with self.assertRaises(ValueError):
            iface.generate(n, task_labels=labels)


# ---------------------------------------------------------------------------
# CGAN — single-task, conditioned on one_hot, returns (n, seq_len, F)
# ---------------------------------------------------------------------------


class TestCGANOutputContract(unittest.TestCase):

    def test_returns_three_dim_tensor(self):
        n, S, F = 50, 16, 25
        model = MagicMock()
        model.generate.return_value = np.zeros((n, S, F), dtype=np.float32)

        iface = _interface_with_model(GANType.CGAN, model)
        out = iface.generate(n, one_hot=_one_hot(n))

        self.assertEqual(out.shape, (n, S, F))

    def test_rejects_non_finite_output(self):
        n = 50
        bad = np.zeros((n, 16, 25), dtype=np.float32)
        bad[0, 0, 0] = np.inf

        model = MagicMock()
        model.generate.return_value = bad

        iface = _interface_with_model(GANType.CGAN, model)
        with self.assertRaises(ValueError):
            iface.generate(n, one_hot=_one_hot(n))


# ---------------------------------------------------------------------------
# CTAB-GAN family — returns DataFrames (single-task) or (DataFrame, dict)
# ---------------------------------------------------------------------------


class TestCTABGANOutputContract(unittest.TestCase):

    def test_returns_dataframe_with_n_rows(self):
        n, F = 100, 25
        cols = [f"f{i}" for i in range(F)]
        model = MagicMock()
        model.generate.return_value = pd.DataFrame(
            np.zeros((n, F), dtype=np.float32), columns=cols
        )

        iface = _interface_with_model(GANType.CTAB_GAN, model)
        out = iface.generate(n, class_label=1)

        self.assertIsInstance(out, pd.DataFrame, "CTAB-GAN must return DataFrame")
        self.assertEqual(len(out), n)
        self.assertEqual(list(out.columns), cols)

    def test_n_routes_to_num_samples_kwarg(self):
        """GANInterface remaps `n` to the model's `num_samples=` kwarg
        for the CTAB family — that contract must be preserved during refactor."""
        n = 100
        model = MagicMock()
        model.generate.return_value = pd.DataFrame(
            np.zeros((n, 25), dtype=np.float32)
        )

        iface = _interface_with_model(GANType.CTAB_GAN, model)
        iface.generate(n, class_label=1)

        # The interface should call model.generate(num_samples=n, class_label=1).
        call_kwargs = model.generate.call_args.kwargs
        self.assertEqual(call_kwargs.get("num_samples"), n)
        self.assertEqual(call_kwargs.get("class_label"), 1)


class TestMTCTABGANOutputContract(unittest.TestCase):

    def test_returns_dataframe_and_labels_tuple(self):
        n, F = 100, 25
        cols = [f"f{i}" for i in range(F)]
        labels = {
            "trading": _one_hot(n, 3, 2),
            "regime":  _one_hot(n, 3, 1),
        }
        model = MagicMock()
        model.generate.return_value = (
            pd.DataFrame(np.zeros((n, F), dtype=np.float32), columns=cols),
            labels,
        )

        iface = _interface_with_model(GANType.MT_CTAB_GAN, model)
        out = iface.generate(n, task_labels=labels)

        self.assertIsInstance(out, tuple)
        df, gen_labels = out
        self.assertIsInstance(df, pd.DataFrame)
        self.assertEqual(len(df), n)
        self.assertEqual(set(gen_labels.keys()), set(labels.keys()))


# ---------------------------------------------------------------------------
# Cross-backend invariants
# ---------------------------------------------------------------------------


class TestSampleCountInvariants(unittest.TestCase):
    """Every backend must return exactly ``n`` samples — no silent truncation
    or duplication during the n→batch breakdown."""

    def _check(self, gan_type: GANType, model_return, get_count, **gen_kwargs):
        model = MagicMock()
        model.generate.return_value = model_return
        iface = _interface_with_model(gan_type, model)
        out = iface.generate(50, **gen_kwargs)
        self.assertEqual(get_count(out), 50)

    def test_wgan(self):
        self._check(
            GANType.WGAN,
            np.zeros((50, 1, 25), dtype=np.float32),
            lambda x: x.shape[0],
            one_hot=_one_hot(50),
        )

    def test_cgan(self):
        self._check(
            GANType.CGAN,
            np.zeros((50, 16, 25), dtype=np.float32),
            lambda x: x.shape[0],
            one_hot=_one_hot(50),
        )

    def test_mt_wgan(self):
        labels = {
            "trading":  _one_hot(50),
            "regime":   _one_hot(50),
            "risk":     _one_hot(50),
            "momentum": _one_hot(50),
            "flow":     _one_hot(50),
            "profit":   _one_hot(50),
        }
        self._check(
            GANType.MT_WGAN,
            (np.zeros((50, 16, 25), dtype=np.float32), labels),
            lambda x: x[0].shape[0],
            task_labels=labels,
        )


# ---------------------------------------------------------------------------

if __name__ == "__main__":
    unittest.main(verbosity=2)
