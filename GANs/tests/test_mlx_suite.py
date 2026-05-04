"""
MLX functional tests — fit / generate / save / load for GAN types that have
an MLX backend (WGAN, MT_WGAN, CTAB_GAN, MT_CTAB_GAN).

All tests are automatically skipped when MLX is not installed or Metal is not
available, so this file is safe to run on any machine.

Run (from freqtrade root):
    python -m pytest user_data/strategies/GANs/tests/test_mlx_suite.py -v

Or force-skip check with:
    python -m pytest user_data/strategies/GANs/tests/test_mlx_suite.py -v -k "not skip"
"""

from __future__ import annotations

import os
import shutil
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------------------
# Path / environment setup
# ---------------------------------------------------------------------------
STRATEGIES_ROOT = str(Path(__file__).parent.parent.parent)
if STRATEGIES_ROOT not in sys.path:
    sys.path.insert(0, STRATEGIES_ROOT)

# Suppress GPU noise
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
os.environ.setdefault("TF_DISABLE_MPS", "1")
os.environ.setdefault("TF_METAL_DEVICE_ENABLE", "0")

# ---------------------------------------------------------------------------
# MLX availability check — evaluated once at import time
# ---------------------------------------------------------------------------

def _mlx_available() -> bool:
    try:
        import mlx.core as mx  # type: ignore
        return hasattr(mx, "metal") and mx.metal.is_available()
    except (ImportError, ModuleNotFoundError):
        return False


_HAS_MLX: bool = _mlx_available()

_SKIP_REASON = "MLX not available (requires Apple Silicon + mlx package)"
_skip_if_no_mlx = unittest.skipUnless(_HAS_MLX, _SKIP_REASON)

# ---------------------------------------------------------------------------
# Dataset constants (small — fast training)
# ---------------------------------------------------------------------------
N_SAMPLES  = 65
N_FEATURES = 8
N_CLASSES  = 3
SEQ_LEN    = 1

N_MT_TRADING = 3
N_MT_REGIME  = 2

FAST_EPOCHS = 3
FAST_BATCH  = 16
FAST_N_CRITIC = 1

# ---------------------------------------------------------------------------
# Dataset factories
# ---------------------------------------------------------------------------

def _make_wgan_dataset(seed: int = 42):
    rng = np.random.default_rng(seed)
    data = rng.uniform(-1, 1, (N_SAMPLES, N_FEATURES)).astype("float32")
    idx  = np.concatenate([np.zeros(40, int), np.ones(15, int), np.full(10, 2, int)])
    return data, np.eye(N_CLASSES, dtype="float32")[idx]


def _make_mt_dataset(seed: int = 42):
    rng = np.random.default_rng(seed)
    data = rng.uniform(-1, 1, (N_SAMPLES, SEQ_LEN, N_FEATURES)).astype("float32")
    trading = np.eye(N_MT_TRADING, dtype="float32")[
        np.concatenate([np.zeros(40, int), np.ones(15, int), np.full(10, 2, int)])
    ]
    regime = np.eye(N_MT_REGIME, dtype="float32")[
        np.concatenate([np.zeros(45, int), np.ones(20, int)])
    ]
    return data, {"trading": trading, "regime": regime}


def _make_ctab_dataset(seed: int = 42):
    import pandas as pd
    data, labels = _make_wgan_dataset(seed)
    return pd.DataFrame(data, columns=[f"f{i}" for i in range(N_FEATURES)]), labels


def _make_mt_ctab_dataset(seed: int = 42):
    """Multi-task CTAB-GAN dataset — DataFrame + dict of one-hot task labels."""
    import pandas as pd
    rng = np.random.default_rng(seed)
    data = rng.uniform(-1, 1, (N_SAMPLES, N_FEATURES)).astype("float32")
    df = pd.DataFrame(data, columns=[f"f{i}" for i in range(N_FEATURES)])
    trading = np.eye(N_MT_TRADING, dtype="float32")[
        np.concatenate([np.zeros(40, int), np.ones(15, int), np.full(10, 2, int)])
    ]
    regime = np.eye(N_MT_REGIME, dtype="float32")[
        np.concatenate([np.zeros(45, int), np.ones(20, int)])
    ]
    return df, {"trading": trading, "regime": regime}


# ---------------------------------------------------------------------------
# 1. WGAN MLX — fit / generate / save / load
# ---------------------------------------------------------------------------

@_skip_if_no_mlx
class TestWGANMLXFitGen(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        from GANs.GANInterface import GANInterface
        cls.data, cls.labels = _make_wgan_dataset()
        cls.tmp = tempfile.mkdtemp(prefix="wgan_mlx_")
        cls.iface = GANInterface(GANType.WGAN, save_path=None, prefer_mlx=True)
        cls.iface.fit(
            cls.data.copy(), cls.labels.copy(),
            epochs=FAST_EPOCHS, batch_size=FAST_BATCH, n_critic=FAST_N_CRITIC, verbose=False,
        )

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp, ignore_errors=True)

    def _one_hot(self, n, cls_idx=0):
        oh = np.zeros((n, N_CLASSES), dtype="float32")
        oh[:, cls_idx] = 1.0
        return oh

    def test_model_is_set_after_fit(self):
        self.assertIsNotNone(self.iface._model)

    def test_generate_returns_correct_shape(self):
        n = 10
        result = self.iface.generate(n, one_hot=self._one_hot(n))
        self.assertIsInstance(result, np.ndarray)
        self.assertEqual(result.shape, (n, 1, N_FEATURES))

    def test_generate_varies_with_class(self):
        n = 20
        gen0 = self.iface.generate(n, one_hot=self._one_hot(n, 0))
        gen1 = self.iface.generate(n, one_hot=self._one_hot(n, 1))
        # Different class condition should (statistically) give different output means
        self.assertFalse(np.allclose(gen0.mean(), gen1.mean(), atol=1e-6))


@_skip_if_no_mlx
class TestWGANMLXSaveLoad(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.data, cls.labels = _make_wgan_dataset()
        cls.tmp = tempfile.mkdtemp(prefix="wgan_mlx_sl_")

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp, ignore_errors=True)

    def _fit_and_save(self, subdir):
        from GANs.GANInterface import GANInterface
        save_path = os.path.join(self.tmp, subdir)
        iface = GANInterface(GANType.WGAN, save_path=save_path, prefer_mlx=True)
        iface.fit(
            self.data.copy(), self.labels.copy(),
            epochs=FAST_EPOCHS, batch_size=FAST_BATCH, n_critic=FAST_N_CRITIC, verbose=False,
        )
        iface.save()
        return save_path, iface

    def test_model_file_written_after_save(self):
        save_path, _ = self._fit_and_save("file_test")
        self.assertTrue(
            os.path.exists(os.path.join(save_path, "wgan_gen_mlx.safetensors")),
        )

    def test_metadata_file_written_after_save(self):
        save_path, _ = self._fit_and_save("meta_test")
        self.assertTrue(
            os.path.exists(os.path.join(save_path, "wgangp_metadata.pkl")),
        )

    def test_loaded_model_generates_correct_shape(self):
        save_path, _ = self._fit_and_save("load_test")
        from GANs.GANInterface import GANInterface
        iface2 = GANInterface(GANType.WGAN, save_path=save_path, prefer_mlx=True)
        iface2.load()
        n = 5
        oh = np.zeros((n, N_CLASSES), dtype="float32")
        oh[:, 0] = 1.0
        result = iface2.generate(n, one_hot=oh)
        self.assertEqual(result.shape, (n, 1, N_FEATURES))

    def test_round_trip_generates_finite_values(self):
        save_path, _ = self._fit_and_save("finite_test")
        from GANs.GANInterface import GANInterface
        iface2 = GANInterface(GANType.WGAN, save_path=save_path, prefer_mlx=True)
        iface2.load()
        oh = np.zeros((10, N_CLASSES), dtype="float32")
        oh[:, 0] = 1.0
        result = iface2.generate(10, one_hot=oh)
        self.assertTrue(np.isfinite(result).all(), "Generated values must all be finite")


# ---------------------------------------------------------------------------
# 2. MT_WGAN MLX — fit / generate / save / load
# ---------------------------------------------------------------------------

@_skip_if_no_mlx
class TestMTWGANMLXFitGen(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        from GANs.GANInterface import GANInterface
        cls.data, cls.labels = _make_mt_dataset()
        cls.tmp = tempfile.mkdtemp(prefix="mt_wgan_mlx_")
        cls.iface = GANInterface(GANType.MT_WGAN, save_path=None, prefer_mlx=True)
        cls.iface.fit(
            cls.data.copy(),
            {k: v.copy() for k, v in cls.labels.items()},
            epochs=FAST_EPOCHS, batch_size=FAST_BATCH, n_critic=FAST_N_CRITIC, verbose=False,
        )

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp, ignore_errors=True)

    def _task_labels(self, n, trading_cls=0, regime_cls=0):
        trading = np.zeros((n, N_MT_TRADING), dtype="float32")
        trading[:, trading_cls] = 1.0
        regime  = np.zeros((n, N_MT_REGIME), dtype="float32")
        regime[:, regime_cls] = 1.0
        return {"trading": trading, "regime": regime}

    def test_model_is_set_after_fit(self):
        self.assertIsNotNone(self.iface._model)

    def test_generate_returns_tuple_with_correct_shapes(self):
        n = 8
        result = self.iface.generate(n, task_labels=self._task_labels(n))
        self.assertIsInstance(result, tuple)
        gen_x, gen_labels = result
        self.assertIsInstance(gen_x, np.ndarray)
        self.assertEqual(gen_x.shape[0], n)
        self.assertEqual(gen_x.ndim, 3)
        self.assertEqual(gen_x.shape[2], N_FEATURES)
        self.assertIn("trading", gen_labels)
        self.assertIn("regime",  gen_labels)

    def test_generate_finite_values(self):
        n = 10
        gen_x, _ = self.iface.generate(n, task_labels=self._task_labels(n))
        self.assertTrue(np.isfinite(gen_x).all())


@_skip_if_no_mlx
class TestMTWGANMLXSaveLoad(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.data, cls.labels = _make_mt_dataset()
        cls.tmp = tempfile.mkdtemp(prefix="mt_wgan_mlx_sl_")

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp, ignore_errors=True)

    def _fit_and_save(self, subdir):
        from GANs.GANInterface import GANInterface
        save_path = os.path.join(self.tmp, subdir)
        iface = GANInterface(GANType.MT_WGAN, save_path=save_path, prefer_mlx=True)
        iface.fit(
            self.data.copy(),
            {k: v.copy() for k, v in self.labels.items()},
            epochs=FAST_EPOCHS, batch_size=FAST_BATCH, n_critic=FAST_N_CRITIC, verbose=False,
        )
        iface.save()
        return save_path

    def _task_oh(self, n):
        trading = np.zeros((n, N_MT_TRADING), dtype="float32"); trading[:, 0] = 1.0
        regime  = np.zeros((n, N_MT_REGIME),  dtype="float32"); regime[:, 0]  = 1.0
        return {"trading": trading, "regime": regime}

    def test_model_files_written_after_save(self):
        save_path = self._fit_and_save("file_test")
        self.assertTrue(
            os.path.exists(os.path.join(save_path, "mt_wgan_gen_mlx.safetensors")),
        )

    def test_metadata_file_written_after_save(self):
        # MLX backend writes mt_wgan_meta_mlx.pkl (not the TF variant)
        save_path = self._fit_and_save("meta_test")
        self.assertTrue(
            os.path.exists(os.path.join(save_path, "mt_wgan_meta_mlx.pkl")),
        )

    def test_loaded_model_generates_correct_shape(self):
        save_path = self._fit_and_save("load_test")
        from GANs.GANInterface import GANInterface
        iface2 = GANInterface(GANType.MT_WGAN, save_path=save_path, prefer_mlx=True)
        iface2.load()
        n = 5
        gen_x, gen_labels = iface2.generate(n, task_labels=self._task_oh(n))
        self.assertEqual(gen_x.shape[0], n)
        self.assertEqual(gen_x.ndim, 3)
        self.assertIn("trading", gen_labels)

    def test_round_trip_generates_finite_values(self):
        save_path = self._fit_and_save("finite_test")
        from GANs.GANInterface import GANInterface
        iface2 = GANInterface(GANType.MT_WGAN, save_path=save_path, prefer_mlx=True)
        iface2.load()
        gen_x, _ = iface2.generate(10, task_labels=self._task_oh(10))
        self.assertTrue(np.isfinite(gen_x).all())


# ---------------------------------------------------------------------------
# 3. CTAB_GAN MLX — fit / generate / save / load
# ---------------------------------------------------------------------------

@_skip_if_no_mlx
class TestCTABGANMLXFitGen(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        from GANs.GANInterface import GANInterface
        cls.df, cls.labels = _make_ctab_dataset()
        cls.tmp = tempfile.mkdtemp(prefix="ctab_mlx_")
        cls.iface = GANInterface(GANType.CTAB_GAN, save_path=None, prefer_mlx=True)
        cls.iface.fit(
            cls.df.copy(), cls.labels.copy(),
            epochs=FAST_EPOCHS, batch_size=FAST_BATCH, latent_dim=32, verbose=False,
        )

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp, ignore_errors=True)

    def test_model_is_set_after_fit(self):
        self.assertIsNotNone(self.iface._model)

    def test_generate_returns_dataframe(self):
        import pandas as pd
        result = self.iface.generate(10, class_label=0)
        self.assertIsInstance(result, pd.DataFrame)
        self.assertEqual(result.shape[0], 10)
        self.assertEqual(result.shape[1], N_FEATURES)

    def test_generate_finite_values(self):
        result = self.iface.generate(10, class_label=0)
        self.assertTrue(np.isfinite(result.values).all())


@_skip_if_no_mlx
class TestCTABGANMLXSaveLoad(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.df, cls.labels = _make_ctab_dataset()
        cls.tmp = tempfile.mkdtemp(prefix="ctab_mlx_sl_")

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp, ignore_errors=True)

    def _fit_and_save(self, subdir):
        from GANs.GANInterface import GANInterface
        save_path = os.path.join(self.tmp, subdir)
        iface = GANInterface(GANType.CTAB_GAN, save_path=save_path, prefer_mlx=True)
        iface.fit(
            self.df.copy(), self.labels.copy(),
            epochs=FAST_EPOCHS, batch_size=FAST_BATCH, latent_dim=32, verbose=False,
        )
        iface.save()
        return save_path

    def test_mlx_metadata_file_written(self):
        save_path = self._fit_and_save("mlx_meta_test")
        self.assertTrue(
            os.path.exists(os.path.join(save_path, "metadata_mlx.pkl")),
            "metadata_mlx.pkl must exist so that load() selects the MLX path",
        )

    def test_mlx_weights_file_written(self):
        save_path = self._fit_and_save("mlx_weights_test")
        self.assertTrue(
            os.path.exists(os.path.join(save_path, "gen_mlx.safetensors")),
        )

    def test_loaded_model_generates_dataframe(self):
        import pandas as pd
        save_path = self._fit_and_save("load_test")
        from GANs.GANInterface import GANInterface
        iface2 = GANInterface(GANType.CTAB_GAN, save_path=save_path, prefer_mlx=True)
        iface2.load()
        result = iface2.generate(5, class_label=0)
        self.assertIsInstance(result, pd.DataFrame)
        self.assertEqual(result.shape[0], 5)

    def test_round_trip_finite_values(self):
        save_path = self._fit_and_save("finite_test")
        from GANs.GANInterface import GANInterface
        iface2 = GANInterface(GANType.CTAB_GAN, save_path=save_path, prefer_mlx=True)
        iface2.load()
        result = iface2.generate(10, class_label=0)
        self.assertTrue(np.isfinite(result.values).all())


# ---------------------------------------------------------------------------
# 4. MT_CTAB_GAN MLX — fit / generate / save / load
# ---------------------------------------------------------------------------


@_skip_if_no_mlx
class TestMTCTABGANMLXFitGen(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        from GANs.GANInterface import GANInterface
        cls.df, cls.labels = _make_mt_ctab_dataset()
        cls.tmp = tempfile.mkdtemp(prefix="mt_ctab_mlx_")
        cls.iface = GANInterface(GANType.MT_CTAB_GAN, save_path=None, prefer_mlx=True)
        # eval_frequency=0 keeps these fast tests on the divergence-only path
        # and avoids the cost of evaluation runs on tiny synthetic data.
        cls.iface.fit(
            cls.df.copy(),
            {k: v.copy() for k, v in cls.labels.items()},
            epochs=FAST_EPOCHS,
            batch_size=FAST_BATCH,
            n_critic=FAST_N_CRITIC,
            latent_dim=32,
            eval_frequency=0,
            verbose=False,
        )

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp, ignore_errors=True)

    def _task_labels(self, n, trading_cls=0, regime_cls=0):
        trading = np.zeros((n, N_MT_TRADING), dtype="float32")
        trading[:, trading_cls] = 1.0
        regime = np.zeros((n, N_MT_REGIME), dtype="float32")
        regime[:, regime_cls] = 1.0
        return {"trading": trading, "regime": regime}

    def test_model_is_set_after_fit(self):
        self.assertIsNotNone(self.iface._model)

    def test_generate_returns_tuple_of_dataframe_and_labels(self):
        import pandas as pd
        result = self.iface.generate(8, task_labels=self._task_labels(8))
        self.assertIsInstance(result, tuple)
        gen_df, gen_labels = result
        self.assertIsInstance(gen_df, pd.DataFrame)
        self.assertEqual(gen_df.shape, (8, N_FEATURES))
        self.assertIn("trading", gen_labels)
        self.assertIn("regime", gen_labels)

    def test_generate_finite_values(self):
        gen_df, _ = self.iface.generate(10, task_labels=self._task_labels(10))
        self.assertTrue(np.isfinite(gen_df.values).all())

    def test_generate_without_task_labels_uses_uniform_sampling(self):
        # Multi-task generate() accepts task_labels=None and samples
        # one-hots per task internally.  Verify shape and finiteness.
        gen_df, gen_labels = self.iface.generate(6, task_labels=None)
        self.assertEqual(gen_df.shape, (6, N_FEATURES))
        self.assertEqual(gen_labels["trading"].shape, (6, N_MT_TRADING))
        self.assertEqual(gen_labels["regime"].shape, (6, N_MT_REGIME))


@_skip_if_no_mlx
class TestMTCTABGANMLXSaveLoad(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.df, cls.labels = _make_mt_ctab_dataset()
        cls.tmp = tempfile.mkdtemp(prefix="mt_ctab_mlx_sl_")

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp, ignore_errors=True)

    def _fit_and_save(self, subdir):
        from GANs.GANInterface import GANInterface
        save_path = os.path.join(self.tmp, subdir)
        iface = GANInterface(GANType.MT_CTAB_GAN, save_path=save_path, prefer_mlx=True)
        iface.fit(
            self.df.copy(),
            {k: v.copy() for k, v in self.labels.items()},
            epochs=FAST_EPOCHS,
            batch_size=FAST_BATCH,
            n_critic=FAST_N_CRITIC,
            latent_dim=32,
            eval_frequency=0,
            verbose=False,
        )
        iface.save()
        return save_path

    def _task_oh(self, n):
        trading = np.zeros((n, N_MT_TRADING), dtype="float32"); trading[:, 0] = 1.0
        regime = np.zeros((n, N_MT_REGIME), dtype="float32"); regime[:, 0] = 1.0
        return {"trading": trading, "regime": regime}

    def test_mlx_metadata_file_written(self):
        save_path = self._fit_and_save("mlx_meta_test")
        self.assertTrue(
            os.path.exists(os.path.join(save_path, "metadata_mlx.pkl")),
            "metadata_mlx.pkl must exist so that load() selects the MLX path",
        )

    def test_mlx_weights_files_written(self):
        save_path = self._fit_and_save("mlx_weights_test")
        self.assertTrue(
            os.path.exists(os.path.join(save_path, "gen_mlx.safetensors")),
        )
        self.assertTrue(
            os.path.exists(os.path.join(save_path, "critic_mlx.safetensors")),
        )

    def test_loaded_model_generates_dataframe(self):
        import pandas as pd
        save_path = self._fit_and_save("load_test")
        from GANs.GANInterface import GANInterface
        iface2 = GANInterface(GANType.MT_CTAB_GAN, save_path=save_path, prefer_mlx=True)
        iface2.load()
        result = iface2.generate(5, task_labels=self._task_oh(5))
        self.assertIsInstance(result, tuple)
        gen_df, _ = result
        self.assertIsInstance(gen_df, pd.DataFrame)
        self.assertEqual(gen_df.shape, (5, N_FEATURES))

    def test_round_trip_finite_values(self):
        save_path = self._fit_and_save("finite_test")
        from GANs.GANInterface import GANInterface
        iface2 = GANInterface(GANType.MT_CTAB_GAN, save_path=save_path, prefer_mlx=True)
        iface2.load()
        gen_df, _ = iface2.generate(10, task_labels=self._task_oh(10))
        self.assertTrue(np.isfinite(gen_df.values).all())


# ---------------------------------------------------------------------------
# Module-level import — GANType must be accessible
# ---------------------------------------------------------------------------
from GANs.GANType import GANType  # noqa: E402


if __name__ == "__main__":
    unittest.main(verbosity=2)
