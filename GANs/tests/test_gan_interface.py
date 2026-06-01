"""
Tests for GANInterface.

These tests run without TensorFlow, MLX, or any actual GAN model files.
All GAN backends are replaced with lightweight mocks, so the test suite
validates the interface contract:
  - fit() / generate() / save() / load() routing for all GAN types
  - Default configs are applied (callers don't need to pass internals)
  - save_path is threaded through to the backend
  - Errors are raised for unsupported method/type combinations

Run from the strategies root:
    python -m pytest GANs/tests/test_gan_interface.py -v
"""

from __future__ import annotations

import sys
import os
import unittest
from pathlib import Path
from typing import Any, Dict
from unittest.mock import MagicMock, patch

import numpy as np

# ---------------------------------------------------------------------------
# Path setup — make sure 'GANs' and 'Framework' are importable without
# requiring freqtrade to be installed.
# ---------------------------------------------------------------------------
STRATEGIES_ROOT = str(Path(__file__).parent.parent.parent)
if STRATEGIES_ROOT not in sys.path:
    sys.path.insert(0, STRATEGIES_ROOT)

# Stub out freqtrade before any strategies-framework import runs.
_ft_stub = MagicMock()
sys.modules.setdefault("freqtrade", _ft_stub)
sys.modules.setdefault("freqtrade.persistence", _ft_stub)
sys.modules.setdefault("freqtrade.strategy", _ft_stub)

from GANs.GANType import GANType  # noqa: E402  (import after path setup)
from GANs.GANInterface import GANInterface, _HAS_MLX  # noqa: E402


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _dummy_data(n: int = 100, features: int = 20) -> np.ndarray:
    rng = np.random.default_rng(0)
    return rng.random((n, features)).astype("float32")


def _dummy_labels_1hot(n: int = 100, classes: int = 3) -> np.ndarray:
    rng = np.random.default_rng(1)
    idx = rng.integers(0, classes, n)
    return np.eye(classes, dtype="float32")[idx]


def _dummy_labels_dict(n: int = 100) -> Dict[str, np.ndarray]:
    return {
        "trading": _dummy_labels_1hot(n, 3),
        "regime":  _dummy_labels_1hot(n, 4),
    }


# ---------------------------------------------------------------------------
# 1. Initialisation
# ---------------------------------------------------------------------------

class TestInit(unittest.TestCase):

    def test_stores_gan_type(self):
        iface = GANInterface(GANType.WGAN, save_path="/tmp/gans")
        self.assertEqual(iface.gan_type, GANType.WGAN)

    def test_stores_save_path(self):
        iface = GANInterface(GANType.MT_WGAN, save_path="/my/path")
        self.assertEqual(iface.save_path, "/my/path")

    def test_prefer_mlx_respected(self):
        iface_no_mlx = GANInterface(GANType.WGAN, save_path="/tmp", prefer_mlx=False)
        self.assertFalse(iface_no_mlx._use_mlx)

    def test_model_initially_none(self):
        iface = GANInterface(GANType.CTAB_GAN, save_path="/tmp")
        self.assertIsNone(iface._model)

    def test_all_types_have_defaults(self):
        for gtype in (GANType.WGAN, GANType.MT_WGAN, GANType.CGAN,
                      GANType.CTAB_GAN, GANType.MT_CTAB_GAN):
            with self.subTest(gtype=gtype):
                self.assertIn(gtype, GANInterface._DEFAULTS)


# ---------------------------------------------------------------------------
# 2. Error cases
# ---------------------------------------------------------------------------

class TestErrors(unittest.TestCase):

    def test_generate_before_load_or_fit_raises(self):
        iface = GANInterface(GANType.CTAB_GAN, save_path="/tmp")
        with self.assertRaises(RuntimeError):
            iface.generate(10)

    def test_save_before_fit_raises(self):
        iface = GANInterface(GANType.CTAB_GAN, save_path="/tmp")
        with self.assertRaises(RuntimeError):
            iface.save()

    def test_wgan_generate_without_one_hot_raises(self):
        iface = GANInterface(GANType.WGAN, save_path="/tmp")
        iface._model = MagicMock()
        with self.assertRaises(ValueError):
            iface.generate(10)  # missing one_hot=

    def test_mt_wgan_generate_without_task_labels_raises(self):
        iface = GANInterface(GANType.MT_WGAN, save_path="/tmp")
        iface._model = MagicMock()
        with self.assertRaises(ValueError):
            iface.generate(10)  # missing task_labels=

    def test_cgan_generate_without_one_hot_raises(self):
        iface = GANInterface(GANType.CGAN, save_path="/tmp")
        iface._model = MagicMock()
        with self.assertRaises(ValueError):
            iface.generate(10)  # missing one_hot=

    def test_fit_raises_for_unsupported_type(self):
        iface = GANInterface(GANType.NONE, save_path="/tmp")
        with self.assertRaises(ValueError):
            iface.fit(_dummy_data(), _dummy_labels_1hot())


# ---------------------------------------------------------------------------
# 3. Model-based API (CTAB_GAN) — fit / generate / save / load
# ---------------------------------------------------------------------------

class TestModelBasedAPI(unittest.TestCase):

    def _mock_ctab_model(self) -> MagicMock:
        model = MagicMock()
        model.load.return_value = {"min_buy_gain_threshold": 0.016}
        model.generate.return_value = MagicMock()  # DataFrame-like
        return model

    def test_fit_calls_underlying_model(self):
        mock_cls  = MagicMock(return_value=self._mock_ctab_model())
        mock_mod  = MagicMock()
        mock_mod.CTABGANPlusEnhanced = mock_cls

        data   = _dummy_data()
        labels = _dummy_labels_1hot()
        import pandas as pd
        df = pd.DataFrame(data)

        with patch.dict("sys.modules", {"GANs.df_ctab_gan": mock_mod}):
            iface = GANInterface(GANType.CTAB_GAN, save_path="/tmp/ctab", prefer_mlx=False)
            iface.fit(df, labels, categorical_columns=[])

        mock_cls.return_value.fit.assert_called_once()

    def test_generate_after_fit(self):
        mock_model = self._mock_ctab_model()
        mock_cls   = MagicMock(return_value=mock_model)
        mock_mod   = MagicMock()
        mock_mod.CTABGANPlusEnhanced = mock_cls

        import pandas as pd
        df = pd.DataFrame(_dummy_data())

        with patch.dict("sys.modules", {"GANs.df_ctab_gan": mock_mod}):
            iface = GANInterface(GANType.CTAB_GAN, save_path="/tmp/ctab", prefer_mlx=False)
            iface.fit(df, _dummy_labels_1hot())
            result = iface.generate(50, class_label=0)

        mock_model.generate.assert_called_once_with(num_samples=50, class_label=0)

    def test_save_uses_save_path(self):
        mock_model = self._mock_ctab_model()
        mock_cls   = MagicMock(return_value=mock_model)
        mock_mod   = MagicMock()
        mock_mod.CTABGANPlusEnhanced = mock_cls

        import pandas as pd
        df = pd.DataFrame(_dummy_data())

        with patch.dict("sys.modules", {"GANs.df_ctab_gan": mock_mod}):
            iface = GANInterface(GANType.CTAB_GAN, save_path="/expected/path", prefer_mlx=False)
            iface.fit(df, _dummy_labels_1hot())
            iface.save(min_buy_gain_threshold=0.02)

        mock_model.save.assert_called_once_with(
            "/expected/path", min_buy_gain_threshold=0.02
        )

    def test_load_returns_metadata(self):
        mock_model = self._mock_ctab_model()
        mock_cls   = MagicMock(return_value=mock_model)
        mock_mod   = MagicMock()
        mock_mod.CTABGANPlusEnhanced = mock_cls

        with patch.dict("sys.modules", {"GANs.df_ctab_gan": mock_mod}):
            iface     = GANInterface(GANType.CTAB_GAN, save_path="/tmp/ctab", prefer_mlx=False)
            metadata  = iface.load()

        self.assertEqual(metadata["min_buy_gain_threshold"], 0.016)
        mock_model.load.assert_called_once_with("/tmp/ctab")

    def test_generate_after_load(self):
        mock_model = self._mock_ctab_model()
        mock_cls   = MagicMock(return_value=mock_model)
        mock_mod   = MagicMock()
        mock_mod.CTABGANPlusEnhanced = mock_cls

        with patch.dict("sys.modules", {"GANs.df_ctab_gan": mock_mod}):
            iface = GANInterface(GANType.CTAB_GAN, save_path="/tmp/ctab", prefer_mlx=False)
            iface.load()
            iface.generate(100, class_label=2)

        mock_model.generate.assert_called_once_with(num_samples=100, class_label=2)


# ---------------------------------------------------------------------------
# 4. CGAN fit / generate / save / load via mocked backend
# ---------------------------------------------------------------------------

class TestCGANLifecycle(unittest.TestCase):

    def _mock_cgan_module(self, gen_return=None):
        """Build a mock df_cgan module with DFCGAN, _save_cgan_model, _load_cgan_model."""
        mock_model = MagicMock()
        mock_model.latent_dim           = 64
        mock_model.d_steps              = 2
        mock_model.instance_noise_std   = 0.01
        mock_model.label_smoothing      = True
        mock_model.fm_weight            = 1.0
        mock_model.fm_var_weight        = 0.5
        mock_model.mmd_weight           = 0.5
        mock_model.generate.return_value = (
            gen_return if gen_return is not None
            else np.zeros((5, 1, 20), dtype="float32")
        )

        mock_mod = MagicMock()
        mock_mod.DFCGAN.return_value = mock_model
        mock_mod._save_cgan_model    = MagicMock()
        meta = {"seq_len": 1, "num_features": 20, "num_classes": 3}
        mock_mod._load_cgan_model.return_value = (mock_model, meta)
        return mock_mod, mock_model

    def test_fit_dispatches_to_cgan(self):
        """fit() for CGAN drives a DFCGAN through training and stores it."""
        # Phase 2: CGAN goes through the backend registry.  Drive a fully
        # mocked DFCGAN through the new path and assert the constructor
        # was invoked + the resulting model lands on the interface.
        mock_mod, mock_model = self._mock_cgan_module()

        # The CGAN backend imports tensorflow inside fit; stub it so the
        # import resolves to a mock, then the rest of the fit body sees
        # placeholder tf.* calls.
        mock_tf = MagicMock()

        data = np.zeros((30, 1, 20), dtype="float32")
        labels = np.eye(3, dtype="float32")[[0] * 10 + [1] * 10 + [2] * 10]

        with patch.dict("sys.modules", {
            "GANs.df_cgan": mock_mod,
            "tensorflow":    mock_tf,
        }):
            iface = GANInterface(GANType.CGAN, save_path="/tmp/cgan", prefer_mlx=False)
            iface.fit(data, labels)

        mock_mod.DFCGAN.assert_called_once()
        # GANInterface mirrors the backend's underlying _model onto self._model.
        self.assertIs(iface._model, mock_model)

    def test_save_calls_save_cgan_model(self):
        """save() for CGAN ultimately invokes _save_cgan_model."""
        mock_mod, mock_model = self._mock_cgan_module()
        mock_model._interface_metadata = {
            "seq_len": 1, "num_features": 20, "num_classes": 3,
        }

        # Drive through the backend rather than setting _model directly,
        # since save() now delegates via self._backend when present.
        from GANs.backends.cgan import CGANTFBackend  # noqa: E402
        backend = CGANTFBackend()
        backend._model = mock_model

        iface = GANInterface(GANType.CGAN, save_path="/tmp/cgan", prefer_mlx=False)
        iface._backend = backend

        with patch.dict("sys.modules", {"GANs.df_cgan": mock_mod}):
            iface.save(extra_key="extra_val")

        mock_mod._save_cgan_model.assert_called_once()
        call_args = mock_mod._save_cgan_model.call_args
        saved_model, saved_meta, saved_path = call_args[0]
        self.assertEqual(saved_path, "/tmp/cgan")
        self.assertEqual(saved_meta["extra_key"], "extra_val")

    def test_load_returns_metadata_and_sets_model(self):
        mock_mod, mock_model = self._mock_cgan_module()

        iface = GANInterface(GANType.CGAN, save_path="/tmp/cgan", prefer_mlx=False)
        with patch.dict("sys.modules", {"GANs.df_cgan": mock_mod}):
            meta = iface.load()

        self.assertEqual(meta["seq_len"], 1)
        self.assertIs(iface._model, mock_model)

    def test_generate_after_load_calls_model_generate(self):
        mock_mod, mock_model = self._mock_cgan_module()

        iface = GANInterface(GANType.CGAN, save_path="/tmp/cgan", prefer_mlx=False)
        with patch.dict("sys.modules", {"GANs.df_cgan": mock_mod}):
            iface.load()

        one_hot = np.eye(3, dtype="float32")[:5]
        iface.generate(5, one_hot=one_hot)
        mock_model.generate.assert_called_once_with(5, one_hot)


# ---------------------------------------------------------------------------
# 5. GANType standalone enum
# ---------------------------------------------------------------------------

class TestGANType(unittest.TestCase):

    def test_all_expected_members_present(self):
        expected = {
            "NONE", "WGAN", "MT_WGAN", "CTAB_GAN", "MT_CTAB_GAN", "CGAN", "BOTH",
            "TAB_DDPM", "MT_DDPM",
        }
        actual   = {m.name for m in GANType}
        self.assertEqual(actual, expected)

    def test_enum_members_are_unique(self):
        values = [m.value for m in GANType]
        self.assertEqual(len(values), len(set(values)))

    def test_gantype_importable_without_freqtrade(self):
        """GANType must not depend on freqtrade (already proven by running this test)."""
        import importlib
        spec = importlib.util.find_spec("GANs.GANType")
        self.assertIsNotNone(spec)


# ---------------------------------------------------------------------------
# 6. MLX routing — verify GANInterface dispatches to MLX backends when
#    _HAS_MLX is True, and falls back to TF when the import fails.
#    No actual MLX hardware or installation required: all backends are mocked.
# ---------------------------------------------------------------------------

class TestMLXRouting(unittest.TestCase):
    """GANInterface must route fit/load calls to the MLX backend when
    _HAS_MLX=True, and fall back to the TF backend on ImportError."""

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _data(n: int = 80, f: int = 8) -> np.ndarray:
        return np.random.default_rng(0).random((n, f)).astype("float32")

    @staticmethod
    def _labels(n: int = 80, c: int = 3) -> np.ndarray:
        idx = np.random.default_rng(1).integers(0, c, n)
        return np.eye(c, dtype="float32")[idx]

    @staticmethod
    def _mt_labels(n: int = 80) -> Dict[str, np.ndarray]:
        rng = np.random.default_rng(2)
        return {
            "trading": np.eye(3, dtype="float32")[rng.integers(0, 3, n)],
            "regime":  np.eye(2, dtype="float32")[rng.integers(0, 2, n)],
        }

    # ------------------------------------------------------------------
    # _use_mlx flag
    # ------------------------------------------------------------------

    def test_use_mlx_true_when_has_mlx_and_prefer_mlx(self):
        with patch("GANs.GANInterface._HAS_MLX", True):
            iface = GANInterface(GANType.WGAN, save_path="/tmp", prefer_mlx=True)
        self.assertTrue(iface._use_mlx)

    def test_use_mlx_false_when_prefer_mlx_false(self):
        """prefer_mlx=False must win even when _HAS_MLX=True."""
        with patch("GANs.GANInterface._HAS_MLX", True):
            iface = GANInterface(GANType.WGAN, save_path="/tmp", prefer_mlx=False)
        self.assertFalse(iface._use_mlx)

    def test_use_mlx_false_when_has_mlx_false(self):
        """prefer_mlx=True does nothing when _HAS_MLX=False."""
        with patch("GANs.GANInterface._HAS_MLX", False):
            iface = GANInterface(GANType.WGAN, save_path="/tmp", prefer_mlx=True)
        self.assertFalse(iface._use_mlx)

    # ------------------------------------------------------------------
    # WGAN — fit dispatches to MLX backend
    # ------------------------------------------------------------------

    def test_wgan_fit_calls_mlx_backend(self):
        data   = self._data()
        labels = self._labels()
        mock_gan = MagicMock()
        mock_mod = MagicMock()
        mock_mod.balance_with_wgan_mlx.return_value = (data, labels, mock_gan)

        with patch("GANs.GANInterface._HAS_MLX", True):
            with patch.dict("sys.modules", {"GANs.df_wgan_mlx": mock_mod}):
                iface = GANInterface(GANType.WGAN, save_path="/tmp", prefer_mlx=True)
                iface.fit(data, labels, epochs=1, batch_size=16)

        mock_mod.balance_with_wgan_mlx.assert_called_once()
        # _return_model=True must have been forwarded
        call_kwargs = mock_mod.balance_with_wgan_mlx.call_args[1]
        self.assertTrue(call_kwargs.get("_return_model"))
        self.assertIs(iface._model, mock_gan)

    def test_wgan_fit_falls_back_to_tf_on_import_error(self):
        data   = self._data()
        labels = self._labels()
        mock_gan   = MagicMock()
        mock_tf    = MagicMock()
        mock_tf.balance_with_wgan_gp.return_value = (data, labels, mock_gan)

        with patch("GANs.GANInterface._HAS_MLX", True):
            with patch.dict("sys.modules", {
                "GANs.df_wgan_mlx": None,   # None → ImportError on import
                "GANs.df_wgan_gp":  mock_tf,
            }):
                iface = GANInterface(GANType.WGAN, save_path="/tmp", prefer_mlx=True)
                iface.fit(data, labels, epochs=1, batch_size=16)

        mock_tf.balance_with_wgan_gp.assert_called_once()
        self.assertIs(iface._model, mock_gan)

    # ------------------------------------------------------------------
    # MT_WGAN — fit dispatches to MLX backend
    # ------------------------------------------------------------------

    def test_mt_wgan_fit_calls_mlx_backend(self):
        data   = self._data().reshape(80, 1, 8)
        labels = self._mt_labels()
        mock_gan = MagicMock()
        mock_mod = MagicMock()
        mock_mod.balance_with_mt_wgan_mlx.return_value = (data, labels, mock_gan)

        with patch("GANs.GANInterface._HAS_MLX", True):
            with patch.dict("sys.modules", {"GANs.df_mt_wgan_mlx": mock_mod}):
                iface = GANInterface(GANType.MT_WGAN, save_path="/tmp", prefer_mlx=True)
                iface.fit(data, labels, epochs=1, batch_size=16)

        mock_mod.balance_with_mt_wgan_mlx.assert_called_once()
        call_kwargs = mock_mod.balance_with_mt_wgan_mlx.call_args[1]
        self.assertTrue(call_kwargs.get("_return_model"))
        self.assertIs(iface._model, mock_gan)

    def test_mt_wgan_fit_falls_back_to_tf_on_import_error(self):
        data   = self._data().reshape(80, 1, 8)
        labels = self._mt_labels()
        mock_gan = MagicMock()
        mock_tf  = MagicMock()
        mock_tf.balance_with_mt_wgan_gp.return_value = (data, labels, mock_gan)

        with patch("GANs.GANInterface._HAS_MLX", True):
            with patch.dict("sys.modules", {
                "GANs.df_mt_wgan_mlx": None,
                "GANs.df_mt_wgan_gp":  mock_tf,
            }):
                iface = GANInterface(GANType.MT_WGAN, save_path="/tmp", prefer_mlx=True)
                iface.fit(data, labels, epochs=1, batch_size=16)

        mock_tf.balance_with_mt_wgan_gp.assert_called_once()
        self.assertIs(iface._model, mock_gan)

    # ------------------------------------------------------------------
    # CTAB_GAN — fit builds CTABGANMLX; load uses CTABGANMLX when
    # metadata_mlx.pkl exists
    # ------------------------------------------------------------------

    def test_ctab_fit_builds_mlx_model(self):
        import pandas as pd
        df     = pd.DataFrame(self._data(), columns=[f"f{i}" for i in range(8)])
        labels = self._labels()

        mock_ctab_mlx = MagicMock()
        mock_mlx_mod  = MagicMock()
        mock_mlx_mod.CTABGANMLX.return_value = mock_ctab_mlx

        with patch("GANs.GANInterface._HAS_MLX", True):
            with patch.dict("sys.modules", {"GANs.df_ctab_gan_mlx": mock_mlx_mod}):
                iface = GANInterface(GANType.CTAB_GAN, save_path="/tmp", prefer_mlx=True)
                iface.fit(df, labels, epochs=1, batch_size=16)

        mock_mlx_mod.CTABGANMLX.assert_called_once()
        mock_ctab_mlx.fit.assert_called_once()
        self.assertIs(iface._model, mock_ctab_mlx)

    def test_ctab_fit_falls_back_to_tf_on_import_error(self):
        import pandas as pd
        df     = pd.DataFrame(self._data(), columns=[f"f{i}" for i in range(8)])
        labels = self._labels()

        mock_ctab_tf = MagicMock()
        mock_tf_mod  = MagicMock()
        mock_tf_mod.CTABGANPlusEnhanced.return_value = mock_ctab_tf

        with patch("GANs.GANInterface._HAS_MLX", True):
            with patch.dict("sys.modules", {
                "GANs.df_ctab_gan_mlx": None,
                "GANs.df_ctab_gan":     mock_tf_mod,
            }):
                iface = GANInterface(GANType.CTAB_GAN, save_path="/tmp", prefer_mlx=True)
                iface.fit(df, labels, epochs=1, batch_size=16)

        mock_tf_mod.CTABGANPlusEnhanced.assert_called_once()
        mock_ctab_tf.fit.assert_called_once()

    def test_ctab_load_uses_mlx_when_metadata_file_present(self):
        """load() for CTAB_GAN must use CTABGANMLX when metadata_mlx.pkl exists."""
        import tempfile, os
        with tempfile.TemporaryDirectory() as tmp:
            # Create a metadata_mlx.pkl sentinel so load() takes the MLX path
            open(os.path.join(tmp, "metadata_mlx.pkl"), "w").close()

            mock_ctab_mlx = MagicMock()
            mock_ctab_mlx.load.return_value = {"mlx_loaded": True}
            mock_mlx_mod  = MagicMock()
            mock_mlx_mod.CTABGANMLX.return_value = mock_ctab_mlx

            with patch("GANs.GANInterface._HAS_MLX", True):
                with patch.dict("sys.modules", {"GANs.df_ctab_gan_mlx": mock_mlx_mod}):
                    iface = GANInterface(GANType.CTAB_GAN, save_path=tmp, prefer_mlx=True)
                    meta  = iface.load()

        mock_mlx_mod.CTABGANMLX.assert_called_once()
        mock_ctab_mlx.load.assert_called_once_with(tmp)
        self.assertTrue(meta.get("mlx_loaded"))

    def test_ctab_load_uses_tf_when_no_mlx_metadata(self):
        """load() must fall back to CTABGANPlus when metadata_mlx.pkl is absent."""
        mock_ctab_tf = MagicMock()
        mock_ctab_tf.load.return_value = {"tf_loaded": True}
        mock_tf_mod  = MagicMock()
        mock_tf_mod.CTABGANPlusEnhanced.return_value = mock_ctab_tf

        with patch("GANs.GANInterface._HAS_MLX", True):
            # save_path with no metadata_mlx.pkl — even with prefer_mlx=True
            with patch.dict("sys.modules", {"GANs.df_ctab_gan": mock_tf_mod}):
                iface = GANInterface(GANType.CTAB_GAN, save_path="/tmp/no_mlx_meta", prefer_mlx=True)
                meta  = iface.load()

        mock_tf_mod.CTABGANPlusEnhanced.assert_called_once()
        self.assertTrue(meta.get("tf_loaded"))


if __name__ == "__main__":
    unittest.main(verbosity=2)
