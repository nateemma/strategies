"""
Parameterized GAN functional suite — one file covers all GAN types.

Each entry in _FITGEN_CONFIGS generates three TestCase subclasses (contract,
save/load, interface) at module load time.  All GAN types use the explicit
fit() / generate() / save() / load() lifecycle.

Run all types:
    python -m pytest GANs/tests/test_functional_suite.py -v

Run a single type (e.g. WGAN):
    python -m pytest GANs/tests/test_functional_suite.py -k "WGAN" -v

Run a single class:
    python -m pytest "GANs/tests/test_functional_suite.py::TestWGANFitGenSaveLoad" -v

Run via unittest:
    python -m unittest GANs.tests.test_functional_suite -v
"""

from __future__ import annotations

import os
import shutil
import sys
import tempfile
import unittest
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import numpy as np

# ---------------------------------------------------------------------------
# Path / environment setup
# ---------------------------------------------------------------------------
STRATEGIES_ROOT = str(Path(__file__).parent.parent.parent)
if STRATEGIES_ROOT not in sys.path:
    sys.path.insert(0, STRATEGIES_ROOT)

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
os.environ.setdefault("TF_DISABLE_MPS", "1")
os.environ.setdefault("TF_METAL_DEVICE_ENABLE", "0")

# ---------------------------------------------------------------------------
# Dataset constants
# ---------------------------------------------------------------------------
N_MAJORITY           = 40
N_MINORITY_A         = 15
N_MINORITY_B         = 10
N_FEATURES           = 8
N_SAMPLES            = N_MAJORITY + N_MINORITY_A + N_MINORITY_B   # 65
N_WGAN_CLASSES       = 3
N_MT_TRADING_CLASSES = 3
N_MT_REGIME_CLASSES  = 2
SEQ_LEN              = 1
N_REGIME_MAJOR       = 45
N_REGIME_MINOR       = 20

# Fast training constants shared by all fit() calls in the suite
FAST_EPOCHS   = 2
FAST_BATCH    = 16
FAST_N_CRITIC = 1


# ---------------------------------------------------------------------------
# Dataset factories
# ---------------------------------------------------------------------------

def _make_wgan_dataset(seed: int = 42):
    """Return (data_2d_f32, labels_1hot) — 2D data, array labels."""
    rng = np.random.default_rng(seed)
    data = rng.uniform(-1.0, 1.0, (N_SAMPLES, N_FEATURES)).astype("float32")
    labels_int = np.concatenate([
        np.zeros(N_MAJORITY,   dtype=int),
        np.ones(N_MINORITY_A,  dtype=int),
        np.full(N_MINORITY_B, 2, dtype=int),
    ])
    labels_1hot = np.eye(N_WGAN_CLASSES, dtype="float32")[labels_int]
    perm = rng.permutation(N_SAMPLES)
    return data[perm], labels_1hot[perm]


def _make_mt_dataset(seed: int = 42):
    """Return (data_3d_f32, labels_dict) — 3D data (N, 1, F), dict labels."""
    rng = np.random.default_rng(seed)
    data = rng.uniform(-1, 1, (N_SAMPLES, SEQ_LEN, N_FEATURES)).astype("float32")
    trading_idx = np.concatenate([
        np.zeros(N_MAJORITY,   int),
        np.ones(N_MINORITY_A,  int),
        np.full(N_MINORITY_B,  2, int),
    ])
    trading_labels = np.eye(N_MT_TRADING_CLASSES, dtype="float32")[trading_idx]
    regime_idx = np.concatenate([
        np.zeros(N_REGIME_MAJOR, int),
        np.ones(N_REGIME_MINOR,  int),
    ])
    regime_labels = np.eye(N_MT_REGIME_CLASSES, dtype="float32")[regime_idx]
    return data, {"trading": trading_labels, "regime": regime_labels}


def _make_mt_2d_dataset(seed: int = 42):
    """Return (data_2d, labels_dict) — 2D data, dict labels."""
    data_3d, labels = _make_mt_dataset(seed)
    return data_3d[:, 0, :], labels


def _make_cgan_dataset(seed: int = 42):
    """Return (data_3d_f32, labels_1hot) — 3D data (N, seq_len=1, F), array labels."""
    rng = np.random.default_rng(seed)
    data = rng.uniform(-1.0, 1.0, (N_SAMPLES, SEQ_LEN, N_FEATURES)).astype("float32")
    labels_int = np.concatenate([
        np.zeros(N_MAJORITY,   dtype=int),
        np.ones(N_MINORITY_A,  dtype=int),
        np.full(N_MINORITY_B, 2, dtype=int),
    ])
    labels_1hot = np.eye(N_WGAN_CLASSES, dtype="float32")[labels_int]
    perm = rng.permutation(N_SAMPLES)
    return data[perm], labels_1hot[perm]


import pandas as pd  # noqa: E402 — needed for CTAB-GAN DataFrame input

_CTAB_COLUMNS = [f"f{i}" for i in range(N_FEATURES)]


def _make_ctab_dataset(seed: int = 42):
    """Single-task DataFrame dataset for CTAB-GAN."""
    data_2d, labels = _make_wgan_dataset(seed)
    return pd.DataFrame(data_2d, columns=_CTAB_COLUMNS), labels


def _make_mt_ctab_dataset(seed: int = 42):
    """Multi-task DataFrame dataset for MT-CTAB-GAN."""
    data_2d, labels = _make_mt_2d_dataset(seed)
    return pd.DataFrame(data_2d, columns=_CTAB_COLUMNS), labels


# ---------------------------------------------------------------------------
# Fast fit kwargs
# ---------------------------------------------------------------------------

def _wgan_fast_fit_kwargs(**overrides) -> dict:
    base = dict(
        epochs=FAST_EPOCHS,
        batch_size=FAST_BATCH,
        n_critic=FAST_N_CRITIC,
        architecture="mlp",
        verbose=False,
    )
    base.update(overrides)
    return base


def _mt_wgan_fast_fit_kwargs(**overrides) -> dict:
    base = dict(
        epochs=FAST_EPOCHS,
        batch_size=FAST_BATCH,
        n_critic=FAST_N_CRITIC,
        verbose=False,
    )
    base.update(overrides)
    return base


def _cgan_fast_fit_kwargs(**overrides) -> dict:
    base = dict(
        epochs=FAST_EPOCHS,
        batch_size=FAST_BATCH,
        d_steps=1,
        steps_per_epoch=5,
        verbose=False,
    )
    base.update(overrides)
    return base


def _ctab_fast_fit_kwargs(**overrides) -> dict:
    base = dict(
        epochs=FAST_EPOCHS,
        batch_size=FAST_BATCH,
        latent_dim=32,
        generator_layers=[64],
        discriminator_layers=[64],
        verbose=False,
    )
    base.update(overrides)
    return base


# ---------------------------------------------------------------------------
# Generate / check callables — WGAN
# ---------------------------------------------------------------------------

def _wgan_do_generate(iface, n):
    one_hot = np.zeros((n, N_WGAN_CLASSES), dtype="float32")
    one_hot[:, 0] = 1.0
    return iface.generate(n, one_hot=one_hot)


def _wgan_check_gen_output(self, result, n):
    self.assertIsInstance(result, np.ndarray)
    self.assertEqual(result.shape[0], n)
    self.assertEqual(result.ndim, 3)          # (n, 1, F)
    self.assertEqual(result.shape[2], N_FEATURES)


def _wgan_check_metadata_dims(self, meta):
    self.assertEqual(meta["num_features"], N_FEATURES)
    self.assertEqual(meta["num_classes"],  N_WGAN_CLASSES)
    self.assertEqual(meta["seq_len"],      1)


# ---------------------------------------------------------------------------
# Generate / check callables — MT_WGAN
# ---------------------------------------------------------------------------

def _mt_wgan_do_generate(iface, n):
    trading_oh = np.zeros((n, N_MT_TRADING_CLASSES), dtype="float32")
    trading_oh[:, 0] = 1.0
    regime_oh  = np.zeros((n, N_MT_REGIME_CLASSES),  dtype="float32")
    regime_oh[:, 0] = 1.0
    return iface.generate(n, task_labels={"trading": trading_oh, "regime": regime_oh})


def _mt_wgan_check_gen_output(self, result, n):
    self.assertIsInstance(result, tuple)
    self.assertEqual(len(result), 2)
    gen_data, gen_labels = result
    self.assertIsInstance(gen_data, np.ndarray)
    self.assertEqual(gen_data.shape[0], n)
    self.assertEqual(gen_data.ndim, 3)        # (n, seq_len, F)
    self.assertEqual(gen_data.shape[2], N_FEATURES)
    self.assertIsInstance(gen_labels, dict)
    for task in ("trading", "regime"):
        self.assertIn(task, gen_labels)


def _mt_wgan_check_metadata_dims(self, meta):
    self.assertEqual(meta["seq_len"],      SEQ_LEN)
    self.assertEqual(meta["num_features"], N_FEATURES)
    expected_dims = {"trading": N_MT_TRADING_CLASSES, "regime": N_MT_REGIME_CLASSES}
    self.assertEqual(meta["task_label_dims"], expected_dims)


# ---------------------------------------------------------------------------
# Generate / check callables — CGAN
# ---------------------------------------------------------------------------

def _cgan_do_generate(iface, n):
    one_hot = np.zeros((n, N_WGAN_CLASSES), dtype="float32")
    one_hot[:, 0] = 1.0
    return iface.generate(n, one_hot=one_hot)


def _cgan_check_gen_output(self, result, n):
    self.assertIsInstance(result, np.ndarray)
    self.assertEqual(result.shape[0], n)
    self.assertEqual(result.ndim, 3)          # (n, seq_len, F)
    self.assertEqual(result.shape[1], SEQ_LEN)
    self.assertEqual(result.shape[2], N_FEATURES)


def _cgan_check_metadata_dims(self, meta):
    self.assertEqual(meta["seq_len"],      SEQ_LEN)
    self.assertEqual(meta["num_features"], N_FEATURES)
    self.assertEqual(meta["num_classes"],  N_WGAN_CLASSES)


# ---------------------------------------------------------------------------
# Generate / check callables — CTAB_GAN
# ---------------------------------------------------------------------------

def _ctab_do_generate(iface, n):
    return iface.generate(n, class_label=0)


def _ctab_check_gen_output(self, result, n):
    self.assertIsInstance(result, pd.DataFrame)
    self.assertEqual(result.shape[0], n)
    self.assertEqual(result.shape[1], N_FEATURES)


def _ctab_check_metadata_dims(self, meta):
    self.assertEqual(meta["num_features"], N_FEATURES)
    self.assertEqual(meta["num_classes"],  N_WGAN_CLASSES)


# ---------------------------------------------------------------------------
# Generate / check callables — MT_CTAB_GAN
# ---------------------------------------------------------------------------

def _mt_ctab_do_generate(iface, n):
    trading_oh = np.zeros((n, N_MT_TRADING_CLASSES), dtype="float32")
    trading_oh[:, 0] = 1.0
    regime_oh  = np.zeros((n, N_MT_REGIME_CLASSES),  dtype="float32")
    regime_oh[:, 0] = 1.0
    return iface.generate(n, task_labels={"trading": trading_oh, "regime": regime_oh})


def _mt_ctab_check_gen_output(self, result, n):
    self.assertIsInstance(result, tuple)
    self.assertEqual(len(result), 2)
    gen_df, labels_dict = result
    self.assertIsInstance(gen_df, pd.DataFrame)
    self.assertEqual(gen_df.shape[0], n)
    self.assertEqual(gen_df.shape[1], N_FEATURES)
    self.assertIsInstance(labels_dict, dict)


def _mt_ctab_check_metadata_dims(self, meta):
    total = meta["num_continuous_features"] + meta["num_categorical_features"]
    self.assertEqual(total, N_FEATURES)
    expected_dims = {"trading": N_MT_TRADING_CLASSES, "regime": N_MT_REGIME_CLASSES}
    self.assertEqual(meta["task_label_dims"], expected_dims)


# ---------------------------------------------------------------------------
# FitGen config dataclass
# ---------------------------------------------------------------------------

from GANs.GANType import GANType  # noqa: E402


@dataclass
class FitGenSuiteConfig:
    """Config for fit() / generate() / save() / load() GAN types.

    All GAN types share this config structure.
    """
    name: str
    gan_type: Any
    n_samples: int
    n_features: int
    make_dataset: Callable
    copy_labels: Callable
    fit_kwargs: dict
    model_files: list              # file names expected after save()
    metadata_filename: str
    required_metadata_keys: set
    check_metadata_dims: Callable  # (test_self, meta) -> None
    do_generate: Callable          # (iface, n) -> result
    check_gen_output: Callable     # (test_self, result, n) -> None
    extra_contract_tests: dict = field(default_factory=dict)
    extra_saveload_tests: dict = field(default_factory=dict)
    extra_interface_tests: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Suite configs — all GAN types
# ---------------------------------------------------------------------------

_FITGEN_CONFIGS: list[FitGenSuiteConfig] = [
    FitGenSuiteConfig(
        name="WGAN",
        gan_type=GANType.WGAN,
        n_samples=N_SAMPLES,
        n_features=N_FEATURES,
        make_dataset=_make_wgan_dataset,
        copy_labels=lambda labels: labels.copy(),
        fit_kwargs=_wgan_fast_fit_kwargs(),
        model_files=["wgangp_model.weights.h5"],
        metadata_filename="wgangp_metadata.pkl",
        required_metadata_keys={
            "seq_len", "num_features", "num_classes",
            "feature_mean", "feature_std", "feature_min", "feature_max",
        },
        check_metadata_dims=_wgan_check_metadata_dims,
        do_generate=_wgan_do_generate,
        check_gen_output=_wgan_check_gen_output,
    ),
    FitGenSuiteConfig(
        name="MTWGAN",
        gan_type=GANType.MT_WGAN,
        n_samples=N_SAMPLES,
        n_features=N_FEATURES,
        make_dataset=_make_mt_dataset,
        copy_labels=lambda labels: {k: v.copy() for k, v in labels.items()},
        fit_kwargs=_mt_wgan_fast_fit_kwargs(),
        model_files=["mt_wgangp_model.weights.h5"],
        metadata_filename="mt_wgangp_metadata.pkl",
        required_metadata_keys={"seq_len", "num_features", "task_label_dims"},
        check_metadata_dims=_mt_wgan_check_metadata_dims,
        do_generate=_mt_wgan_do_generate,
        check_gen_output=_mt_wgan_check_gen_output,
    ),
    FitGenSuiteConfig(
        name="CGAN",
        gan_type=GANType.CGAN,
        n_samples=N_SAMPLES,
        n_features=N_FEATURES,
        make_dataset=_make_cgan_dataset,
        copy_labels=lambda labels: labels.copy(),
        fit_kwargs=_cgan_fast_fit_kwargs(),
        model_files=["cgan_model.weights.h5"],
        metadata_filename="cgan_metadata.pkl",
        required_metadata_keys={"seq_len", "num_features", "num_classes"},
        check_metadata_dims=_cgan_check_metadata_dims,
        do_generate=_cgan_do_generate,
        check_gen_output=_cgan_check_gen_output,
    ),
    FitGenSuiteConfig(
        name="CTABGAN",
        gan_type=GANType.CTAB_GAN,
        n_samples=N_SAMPLES,
        n_features=N_FEATURES,
        make_dataset=_make_ctab_dataset,
        copy_labels=lambda labels: labels.copy(),
        fit_kwargs=_ctab_fast_fit_kwargs(),
        model_files=["generator.keras", "discriminator.keras"],
        metadata_filename="metadata.pkl",
        required_metadata_keys={
            "categorical_columns", "continuous_columns",
            "num_classes", "num_features",
        },
        check_metadata_dims=_ctab_check_metadata_dims,
        do_generate=_ctab_do_generate,
        check_gen_output=_ctab_check_gen_output,
    ),
    FitGenSuiteConfig(
        name="MTCTABGAN",
        gan_type=GANType.MT_CTAB_GAN,
        n_samples=N_SAMPLES,
        n_features=N_FEATURES,
        make_dataset=_make_mt_ctab_dataset,
        copy_labels=lambda labels: {k: v.copy() for k, v in labels.items()},
        fit_kwargs=_ctab_fast_fit_kwargs(),
        model_files=["generator.keras", "discriminator.keras"],
        metadata_filename="metadata.pkl",
        required_metadata_keys={
            "categorical_columns", "continuous_columns",
            "num_continuous_features", "task_label_dims",
        },
        check_metadata_dims=_mt_ctab_check_metadata_dims,
        do_generate=_mt_ctab_do_generate,
        check_gen_output=_mt_ctab_check_gen_output,
    ),
]


# ---------------------------------------------------------------------------
# Factory — Contract (fit + generate)
# ---------------------------------------------------------------------------

def _make_fitgen_contract_class(config: FitGenSuiteConfig) -> type:

    @classmethod
    def setUpClass(cls):
        import gc
        import tensorflow as tf
        tf.keras.backend.clear_session()
        gc.collect()
        from GANs.GANInterface import GANInterface
        cls.data, cls.labels = config.make_dataset()
        cls.tmp = tempfile.mkdtemp(prefix=f"{config.name.lower()}_fg_contract_")
        cls.iface = GANInterface(config.gan_type, save_path=None, prefer_mlx=False)
        cls.iface.fit(
            cls.data.copy(),
            config.copy_labels(cls.labels),
            **config.fit_kwargs,
        )

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp, ignore_errors=True)

    def test_generate_returns_expected_output(
        self,
        _do=config.do_generate,
        _check=config.check_gen_output,
    ):
        result = _do(self.iface, 5)
        _check(self, result, 5)

    def test_generate_minority_class(
        self,
        _do=config.do_generate,
        _check=config.check_gen_output,
    ):
        result = _do(self.iface, 10)
        _check(self, result, 10)

    attrs: dict[str, Any] = {
        "setUpClass":                        setUpClass,
        "tearDownClass":                     tearDownClass,
        "test_generate_returns_expected_output": test_generate_returns_expected_output,
        "test_generate_minority_class":      test_generate_minority_class,
    }
    attrs.update(config.extra_contract_tests)

    return type(
        f"Test{config.name}FitGenContract",
        (unittest.TestCase,),
        attrs,
    )


# ---------------------------------------------------------------------------
# Factory — SaveLoad
# ---------------------------------------------------------------------------

def _make_fitgen_saveload_class(config: FitGenSuiteConfig) -> type:

    @classmethod
    def setUpClass(cls):
        import gc
        import tensorflow as tf
        tf.keras.backend.clear_session()
        gc.collect()
        cls.data, cls.labels = config.make_dataset()
        cls.tmp = tempfile.mkdtemp(prefix=f"{config.name.lower()}_fg_saveload_")

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp, ignore_errors=True)

    def _fit_and_save(self, subdir, _cfg=config):
        """Helper: fit a new model and save it to tmp/subdir."""
        from GANs.GANInterface import GANInterface
        save_path = os.path.join(self.tmp, subdir)
        iface = GANInterface(_cfg.gan_type, save_path=save_path, prefer_mlx=False)
        iface.fit(
            self.data.copy(),
            _cfg.copy_labels(self.labels),
            **_cfg.fit_kwargs,
        )
        iface.save()
        return save_path

    def test_model_files_written_after_training(self, _cfg=config):
        save_path = self._fit_and_save("file_write_test")
        for fname in _cfg.model_files:
            self.assertTrue(
                os.path.exists(os.path.join(save_path, fname)),
                msg=f"{fname!r} should exist after training",
            )
        self.assertTrue(
            os.path.exists(os.path.join(save_path, _cfg.metadata_filename)),
            msg=f"{_cfg.metadata_filename!r} should exist after training",
        )

    def test_metadata_contains_expected_keys(self, _cfg=config):
        import pickle
        save_path = self._fit_and_save("metadata_keys_test")
        with open(os.path.join(save_path, _cfg.metadata_filename), "rb") as fh:
            meta = pickle.load(fh)
        missing = _cfg.required_metadata_keys - meta.keys()
        self.assertFalse(missing, msg=f"Metadata missing keys: {missing}")

    def test_metadata_dimensions_match_training_data(self, _cfg=config):
        import pickle
        save_path = self._fit_and_save("metadata_dims_test")
        with open(os.path.join(save_path, _cfg.metadata_filename), "rb") as fh:
            meta = pickle.load(fh)
        _cfg.check_metadata_dims(self, meta)

    def test_loaded_model_generates_correct_output(self, _cfg=config):
        save_path = self._fit_and_save("reload_test")
        from GANs.GANInterface import GANInterface
        iface2 = GANInterface(_cfg.gan_type, save_path=save_path, prefer_mlx=False)
        iface2.load()
        result = _cfg.do_generate(iface2, 5)
        _cfg.check_gen_output(self, result, 5)

    attrs: dict[str, Any] = {
        "setUpClass":    setUpClass,
        "tearDownClass": tearDownClass,
        "_fit_and_save": _fit_and_save,
        "test_model_files_written_after_training":      test_model_files_written_after_training,
        "test_metadata_contains_expected_keys":         test_metadata_contains_expected_keys,
        "test_metadata_dimensions_match_training_data": test_metadata_dimensions_match_training_data,
        "test_loaded_model_generates_correct_output":   test_loaded_model_generates_correct_output,
    }
    attrs.update(config.extra_saveload_tests)

    return type(
        f"Test{config.name}FitGenSaveLoad",
        (unittest.TestCase,),
        attrs,
    )


# ---------------------------------------------------------------------------
# Factory — Interface integration
# ---------------------------------------------------------------------------

def _make_fitgen_interface_class(config: FitGenSuiteConfig) -> type:

    @classmethod
    def setUpClass(cls):
        import gc
        import tensorflow as tf
        tf.keras.backend.clear_session()
        gc.collect()
        cls.data, cls.labels = config.make_dataset(seed=99)
        cls.tmp = tempfile.mkdtemp(prefix=f"{config.name.lower()}_fg_iface_")

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp, ignore_errors=True)

    def test_fit_and_generate_end_to_end(self, _cfg=config):
        from GANs.GANInterface import GANInterface
        iface = GANInterface(_cfg.gan_type, save_path=None, prefer_mlx=False)
        iface.fit(
            self.data.copy(),
            _cfg.copy_labels(self.labels),
            **_cfg.fit_kwargs,
        )
        result = _cfg.do_generate(iface, 5)
        _cfg.check_gen_output(self, result, 5)

    def test_fit_save_load_generate_round_trip(self, _cfg=config):
        from GANs.GANInterface import GANInterface
        save_path = os.path.join(self.tmp, "round_trip")
        iface = GANInterface(_cfg.gan_type, save_path=save_path, prefer_mlx=False)
        iface.fit(
            self.data.copy(),
            _cfg.copy_labels(self.labels),
            **_cfg.fit_kwargs,
        )
        iface.save()
        iface2 = GANInterface(_cfg.gan_type, save_path=save_path, prefer_mlx=False)
        iface2.load()
        result = _cfg.do_generate(iface2, 5)
        _cfg.check_gen_output(self, result, 5)

    attrs: dict[str, Any] = {
        "setUpClass":                            setUpClass,
        "tearDownClass":                         tearDownClass,
        "test_fit_and_generate_end_to_end":      test_fit_and_generate_end_to_end,
        "test_fit_save_load_generate_round_trip": test_fit_save_load_generate_round_trip,
    }
    attrs.update(config.extra_interface_tests)

    return type(
        f"Test{config.name}FitGenInterface",
        (unittest.TestCase,),
        attrs,
    )


# ---------------------------------------------------------------------------
# Generate and register all test classes
# ---------------------------------------------------------------------------

_FITGEN_FACTORIES = [
    _make_fitgen_contract_class,
    _make_fitgen_saveload_class,
    _make_fitgen_interface_class,
]

for _cfg in _FITGEN_CONFIGS:
    for _factory in _FITGEN_FACTORIES:
        _cls = _factory(_cfg)
        globals()[_cls.__name__] = _cls

del _cfg, _factory, _cls  # prevent loop variables from leaking into module globals


# ---------------------------------------------------------------------------
# Direct execution
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    unittest.main(verbosity=2)
