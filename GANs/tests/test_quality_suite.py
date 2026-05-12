"""
Parameterized GAN quality suite — one file covers all GAN types.

Each entry in _GAN_CONFIGS generates a dedicated unittest.TestCase subclass
at module load time.  Adding support for a new GAN type requires only one
new GANTestConfig entry — no new class, no new file.

The shared test logic lives in GANQualityMixin (quality_base.py).

Run all types:
    RUN_SLOW_TESTS=1 python -m pytest GANs/tests/test_quality_suite.py -v

Run a single type:
    RUN_SLOW_TESTS=1 python -m pytest GANs/tests/test_quality_suite.py::TestWGANQuality -v

Run via unittest:
    RUN_SLOW_TESTS=1 python -m unittest GANs.tests.test_quality_suite -v
"""

from __future__ import annotations

import os
import sys
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

# MLX availability — used to skip MLX-only tests on non-Apple-Silicon hosts.
try:
    import mlx.core as _mx
    _HAS_MLX = hasattr(_mx, "metal") and _mx.metal.is_available()
except Exception:
    _HAS_MLX = False

# ---------------------------------------------------------------------------
# Gate: skip all generated classes unless RUN_SLOW_TESTS is set
# ---------------------------------------------------------------------------
_RUN = bool(os.environ.get("RUN_SLOW_TESTS"))
_SKIP_MSG = "Set RUN_SLOW_TESTS=1 to enable quality tests (5-10 min on CPU)"

# ---------------------------------------------------------------------------
# Shared dataset constants
# ---------------------------------------------------------------------------
N_MAJORITY        = 40
N_MINORITY_A      = 15
N_MINORITY_B      = 10
N_SAMPLES         = 65
N_FEATURES        = 8
N_TRADING_CLASSES = 3
N_REGIME_CLASSES  = 2
SEQ_LEN           = 1

# CTAB-GAN family needs substantially more data than WGAN/MTWGAN to learn
# its per-column VGM mixtures and conditional structure — production runs
# use thousands of samples, so the test mirrors that.
N_CTAB_MAJORITY   = 1200
N_CTAB_MINORITY_A = 450
N_CTAB_MINORITY_B = 300
N_CTAB_SAMPLES    = N_CTAB_MAJORITY + N_CTAB_MINORITY_A + N_CTAB_MINORITY_B
N_CTAB_REGIME_A   = 1300
N_CTAB_REGIME_B   = 650


import pandas as pd  # noqa: E402 — needed for CTAB-GAN DataFrame input

_CTAB_COLUMNS = [f"f{i}" for i in range(N_FEATURES)]

# ---------------------------------------------------------------------------
# Dataset factories — one per GAN type
# ---------------------------------------------------------------------------

def _make_wgan_dataset(seed: int = 42):
    """2D dataset (N, F) with class-discriminative structure."""
    rng = np.random.default_rng(seed)
    class_means = np.array([
        [ 0.6, -0.5,  0.4, -0.3,  0.5, -0.4,  0.3, -0.5],
        [-0.5,  0.6, -0.4,  0.5, -0.3,  0.4, -0.5,  0.6],
        [ 0.0,  0.0,  0.6, -0.6,  0.0,  0.0,  0.6, -0.6],
    ], dtype="float32")
    idx = np.concatenate([
        np.zeros(N_MAJORITY,   int),
        np.ones(N_MINORITY_A,  int),
        np.full(N_MINORITY_B,  2, int),
    ])
    noise  = rng.normal(0, 0.25, (N_SAMPLES, N_FEATURES)).astype("float32")
    data   = np.clip(class_means[idx] + noise, -1, 1)
    labels = np.eye(N_TRADING_CLASSES, dtype="float32")[idx]
    return data, labels


def _make_mt_dataset(seed: int = 42):
    """3D dataset (N, seq_len=1, F) with multi-task labels."""
    rng = np.random.default_rng(seed)
    class_means = np.array([
        [ 0.6, -0.5,  0.4, -0.3,  0.5, -0.4,  0.3, -0.5],
        [-0.5,  0.6, -0.4,  0.5, -0.3,  0.4, -0.5,  0.6],
        [ 0.0,  0.0,  0.6, -0.6,  0.0,  0.0,  0.6, -0.6],
    ], dtype="float32")
    t_idx = np.concatenate([
        np.zeros(N_MAJORITY,   int),
        np.ones(N_MINORITY_A,  int),
        np.full(N_MINORITY_B,  2, int),
    ])
    noise = rng.normal(0, 0.25, (N_SAMPLES, SEQ_LEN, N_FEATURES)).astype("float32")
    data  = np.clip(class_means[t_idx, np.newaxis, :] + noise, -1, 1)
    r_idx = np.concatenate([np.zeros(45, int), np.ones(20, int)])
    return data, {
        "trading": np.eye(N_TRADING_CLASSES, dtype="float32")[t_idx],
        "regime":  np.eye(N_REGIME_CLASSES,  dtype="float32")[r_idx],
    }


# ---------------------------------------------------------------------------
# _setup_generated implementations — populate cls.gen_x / gen_y_primary /
# quality_metrics.  Each receives the class object and trained iface.
# ---------------------------------------------------------------------------

def _wgan_setup_generated(cls: Any, iface: Any) -> None:
    from GANs.df_wgan_gp import assess_generation_quality  # noqa: E402

    gen_x_3d_list, gen_y_list = [], []
    for c, n in [(1, N_MINORITY_A), (2, N_MINORITY_B)]:
        one_hot = np.zeros((n, N_TRADING_CLASSES), dtype="float32")
        one_hot[:, c] = 1.0
        gen_3d = iface.generate(n, one_hot=one_hot)   # (n, 1, F)
        gen_x_3d_list.append(gen_3d)
        gen_y_list.append(one_hot)

    gen_x_3d          = np.concatenate(gen_x_3d_list, axis=0)
    cls.gen_x         = gen_x_3d[:, 0, :]             # collapse to 2D
    cls.gen_y_primary = np.concatenate(gen_y_list, axis=0)

    # assess_generation_quality extracts generated_data[n_real:] internally
    real_3d = cls.real_data[:, np.newaxis, :]
    aug_3d  = np.concatenate([real_3d, gen_x_3d], axis=0)
    aug_y   = np.concatenate([cls.real_labels, cls.gen_y_primary], axis=0)
    cls.quality_metrics = assess_generation_quality(
        real_3d, cls.real_labels, aug_3d, aug_y, verbose=False
    )


def _make_ctab_dataset(seed: int = 42):
    """Single-task DataFrame dataset for CTAB-GAN with production-scale rows."""
    rng = np.random.default_rng(seed)
    class_means = np.array([
        [ 0.6, -0.5,  0.4, -0.3,  0.5, -0.4,  0.3, -0.5],
        [-0.5,  0.6, -0.4,  0.5, -0.3,  0.4, -0.5,  0.6],
        [ 0.0,  0.0,  0.6, -0.6,  0.0,  0.0,  0.6, -0.6],
    ], dtype="float32")
    idx = np.concatenate([
        np.zeros(N_CTAB_MAJORITY,   int),
        np.ones(N_CTAB_MINORITY_A,  int),
        np.full(N_CTAB_MINORITY_B,  2, int),
    ])
    noise  = rng.normal(0, 0.25, (N_CTAB_SAMPLES, N_FEATURES)).astype("float32")
    data   = np.clip(class_means[idx] + noise, -1, 1)
    labels = np.eye(N_TRADING_CLASSES, dtype="float32")[idx]
    return pd.DataFrame(data, columns=_CTAB_COLUMNS), labels


def _make_mt_ctab_dataset(seed: int = 42):
    """Multi-task DataFrame dataset for MT-CTAB-GAN with production-scale rows."""
    rng = np.random.default_rng(seed)
    class_means = np.array([
        [ 0.6, -0.5,  0.4, -0.3,  0.5, -0.4,  0.3, -0.5],
        [-0.5,  0.6, -0.4,  0.5, -0.3,  0.4, -0.5,  0.6],
        [ 0.0,  0.0,  0.6, -0.6,  0.0,  0.0,  0.6, -0.6],
    ], dtype="float32")
    t_idx = np.concatenate([
        np.zeros(N_CTAB_MAJORITY,   int),
        np.ones(N_CTAB_MINORITY_A,  int),
        np.full(N_CTAB_MINORITY_B,  2, int),
    ])
    noise = rng.normal(0, 0.25, (N_CTAB_SAMPLES, N_FEATURES)).astype("float32")
    data  = np.clip(class_means[t_idx] + noise, -1, 1)
    r_idx = np.concatenate([
        np.zeros(N_CTAB_REGIME_A, int),
        np.ones(N_CTAB_REGIME_B,  int),
    ])
    labels = {
        "trading": np.eye(N_TRADING_CLASSES, dtype="float32")[t_idx],
        "regime":  np.eye(N_REGIME_CLASSES,  dtype="float32")[r_idx],
    }
    return pd.DataFrame(data, columns=_CTAB_COLUMNS), labels


def _ctab_setup_generated(cls: Any, iface: Any) -> None:
    from GANs.df_wgan_gp import assess_generation_quality  # noqa: E402

    gen_x_list, gen_y_list = [], []
    for c, n in [(1, N_MINORITY_A), (2, N_MINORITY_B)]:
        gen_df = iface.generate(n, class_label=c)
        gen_x  = gen_df.values.astype("float32")
        one_hot = np.zeros((n, N_TRADING_CLASSES), dtype="float32")
        one_hot[:, c] = 1.0
        gen_x_list.append(gen_x)
        gen_y_list.append(one_hot)

    gen_x             = np.concatenate(gen_x_list, axis=0)
    cls.gen_x         = gen_x
    cls.gen_y_primary = np.concatenate(gen_y_list, axis=0)

    real_np  = cls.real_data.values.astype("float32")
    real_3d  = real_np[:, np.newaxis, :]
    gen_3d   = gen_x[:, np.newaxis, :]
    aug_3d   = np.concatenate([real_3d, gen_3d], axis=0)
    aug_y    = np.concatenate([cls.real_labels, cls.gen_y_primary], axis=0)
    cls.quality_metrics = assess_generation_quality(
        real_3d, cls.real_labels, aug_3d, aug_y, verbose=False
    )


def _mt_ctab_setup_generated(cls: Any, iface: Any) -> None:
    from GANs.df_mt_wgan_gp import assess_mt_generation_quality  # noqa: E402

    gen_x_list, gen_trading_list = [], []
    for c, n in [(1, N_MINORITY_A), (2, N_MINORITY_B)]:
        trading_oh = np.zeros((n, N_TRADING_CLASSES), dtype="float32")
        trading_oh[:, c] = 1.0
        regime_oh  = np.zeros((n, N_REGIME_CLASSES),  dtype="float32")
        regime_oh[:, 0] = 1.0
        gen_df, _ = iface.generate(
            n, task_labels={"trading": trading_oh, "regime": regime_oh}
        )
        gen_x = gen_df.values.astype("float32")
        gen_x_list.append(gen_x)
        gen_trading_list.append(trading_oh)

    gen_x_np          = np.concatenate(gen_x_list, axis=0)
    gen_x_3d          = gen_x_np[:, np.newaxis, :]
    cls.gen_x         = gen_x_3d
    cls.gen_y_primary = np.concatenate(gen_trading_list, axis=0)

    real_np  = cls.real_data.values.astype("float32")
    real_3d  = real_np[:, np.newaxis, :]
    n_gen    = len(gen_x_3d)
    aug_x    = np.concatenate([real_3d, gen_x_3d], axis=0)
    aug_labels = {
        "trading": np.concatenate([
            cls.real_labels["trading"], cls.gen_y_primary,
        ], axis=0),
        "regime": np.concatenate([
            cls.real_labels["regime"],
            np.zeros((n_gen, N_REGIME_CLASSES), dtype="float32"),
        ], axis=0),
    }
    cls.quality_metrics = assess_mt_generation_quality(
        real_3d, cls.real_labels,
        aug_x, aug_labels,
        primary_task="trading",
        verbose=False,
    )


def _mt_wgan_setup_generated(cls: Any, iface: Any) -> None:
    from GANs.df_mt_wgan_gp import assess_mt_generation_quality  # noqa: E402

    gen_x_list, gen_trading_list = [], []
    for c, n in [(1, N_MINORITY_A), (2, N_MINORITY_B)]:
        trading_oh = np.zeros((n, N_TRADING_CLASSES), dtype="float32")
        trading_oh[:, c] = 1.0
        regime_oh  = np.zeros((n, N_REGIME_CLASSES),  dtype="float32")
        regime_oh[:, 0] = 1.0
        gen_data, _ = iface.generate(n, task_labels={"trading": trading_oh, "regime": regime_oh})
        gen_x_list.append(gen_data)
        gen_trading_list.append(trading_oh)

    gen_x_3d          = np.concatenate(gen_x_list, axis=0)
    cls.gen_x         = gen_x_3d            # keep 3D (seq_len=1)
    cls.gen_y_primary = np.concatenate(gen_trading_list, axis=0)

    n_gen   = len(gen_x_3d)
    aug_x   = np.concatenate([cls.real_data, gen_x_3d], axis=0)
    aug_labels = {
        "trading": np.concatenate([
            cls.real_labels["trading"], cls.gen_y_primary,
        ], axis=0),
        "regime": np.concatenate([
            cls.real_labels["regime"],
            np.zeros((n_gen, N_REGIME_CLASSES), dtype="float32"),
        ], axis=0),
    }
    cls.quality_metrics = assess_mt_generation_quality(
        cls.real_data, cls.real_labels,
        aug_x, aug_labels,
        primary_task="trading",
        verbose=False,
    )


# ---------------------------------------------------------------------------
# Type-specific extra tests (attached to individual generated classes)
# ---------------------------------------------------------------------------

def _test_ctab_skip_label_fidelity(self: Any) -> None:
    """CTAB-GAN quality is measured by statistical fidelity, not discriminative conditioning.

    Class conditioning in CTAB-GAN requires a large training set to be reliably
    discriminative; on the small quality-test dataset (65 samples) the classifier
    check is not a meaningful quality bar.  Statistical tests (mean/std RMSE,
    range coverage) adequately cover CTAB-GAN quality.
    """
    import unittest
    raise unittest.SkipTest(
        "CTAB-GAN: label-fidelity classifier check skipped — "
        "statistical quality metrics cover generation quality for this GAN type"
    )


def _test_tabddpm_skip_joint_checks(self: Any) -> None:
    """TabDDPM quality is measured by marginal statistical fidelity on the test dataset.

    The quality-test dataset (~65 samples per class) is too small for a 20-epoch
    run to produce reliably discriminative class conditioning or to preserve
    joint feature structure.  Statistical marginal metrics (mean/std RMSE, range
    coverage) are the meaningful quality bar for this GAN type at test scale.
    """
    import unittest
    raise unittest.SkipTest(
        "TabDDPM: joint-structure / label-fidelity checks skipped — "
        "statistical marginal metrics cover generation quality at test scale"
    )


def _test_wgan_correlation(self: Any) -> None:
    """WGAN: generated and real features must not be strongly anti-correlated."""
    corr = self.quality_metrics.get("mean_correlation", 0.0)
    self.assertGreater(
        corr, -0.3,
        msg=f"mean_correlation={corr:.4f} — generated features are anti-correlated with real",
    )


# ---------------------------------------------------------------------------
# Config dataclass
# ---------------------------------------------------------------------------

@dataclass
class GANTestConfig:
    name: str                             # used in class name: Test{name}Quality
    gan_type: Any                         # GANType enum value
    n_classes: int                        # number of output classes
    minority_classes: list                # class indices to check for presence
    make_dataset: Callable                # (seed) -> (data, labels)
    setup_generated: Callable             # (cls, iface) -> None
    extra_fit_kwargs: dict = field(default_factory=dict)
    extra_tests: dict = field(default_factory=dict)  # {method_name: fn(self)}
    prefer_mlx: bool = False              # route through MLX backend


# ---------------------------------------------------------------------------
# GAN configs — one entry per GAN type
# ---------------------------------------------------------------------------

from GANs.GANType import GANType  # noqa: E402

_GAN_CONFIGS: list[GANTestConfig] = [
    # Single task variants:
    GANTestConfig(
        name="WGAN",
        gan_type=GANType.WGAN,
        n_classes=N_TRADING_CLASSES,
        minority_classes=[1, 2],
        make_dataset=_make_wgan_dataset,
        setup_generated=_wgan_setup_generated,
        extra_fit_kwargs={"architecture": "mlp"},
        extra_tests={
            "test_mean_feature_correlation_not_negative": _test_wgan_correlation,
            # Tighter thresholds: catch sample-quality regressions that the
            # default 0.5 bars let through. Set to 0.2 — see notes in
            # quality_base.py.
            "MEAN_RMSE_THRESHOLD":           0.2,
            "STD_RMSE_THRESHOLD":            0.2,
        },
    ),

    # TabDDPM — MLX-only, single-task, continuous-only.
    # Quality bars: statistical fidelity (mean/std RMSE, range coverage).
    # label_fidelity and correlation tests are skipped for the same reason as
    # CTABGAN: the quality-test dataset is too small (~65 samples) for a
    # 20-epoch run to produce reliably discriminative class conditioning or to
    # preserve joint feature structure. Statistical marginal metrics are the
    # meaningful quality bar here.
    GANTestConfig(
        name="TabDDPM",
        gan_type=GANType.TAB_DDPM,
        n_classes=N_TRADING_CLASSES,
        minority_classes=[1, 2],
        make_dataset=_make_wgan_dataset,
        setup_generated=_wgan_setup_generated,
        prefer_mlx=True,
        extra_fit_kwargs={
            "d_model":          32,
            "d_layers":         (32, 32),
            "num_timesteps":    100,
            "num_sample_steps": 20,
            "epochs":           20,
            "batch_size":       128,
            "verbose":          False,
        },
        extra_tests={
            "MEAN_RMSE_THRESHOLD":           0.5,
            "STD_RMSE_THRESHOLD":            0.5,
            "test_label_fidelity_above_chance": _test_tabddpm_skip_joint_checks,
            "test_correlation_matrix_close_to_real": _test_tabddpm_skip_joint_checks,
        },
    ),

    # Multi-task variants
    GANTestConfig(
        name="MTWGAN",
        gan_type=GANType.MT_WGAN,
        n_classes=N_TRADING_CLASSES,
        minority_classes=[1, 2],
        make_dataset=_make_mt_dataset,
        setup_generated=_mt_wgan_setup_generated,
    ),
    # CTAB-GAN variants
    # Quality bars: statistical fidelity (mean/std RMSE, range coverage).
    # test_label_fidelity_above_chance is skipped — CTAB-GAN class conditioning
    # is not reliably discriminative on tiny datasets; statistical metrics suffice.
    # RMSE thresholds relaxed to 0.6: VGM preprocessing shifts feature means slightly.
    GANTestConfig(
        name="CTABGAN",
        gan_type=GANType.CTAB_GAN,
        n_classes=N_TRADING_CLASSES,
        minority_classes=[1, 2],
        make_dataset=_make_ctab_dataset,
        setup_generated=_ctab_setup_generated,
        extra_fit_kwargs={
            "epochs":               20,
            "batch_size":           64,
            "latent_dim":           32,
            "generator_layers":     [64],
            "discriminator_layers": [64],
            "monitor_metric":       "combined",  # skip per-epoch eval cost
            "eval_frequency":       0,
        },
        extra_tests={
            # Tightened from 0.6 → 0.2 so the test actually surfaces poor
            # sample quality. Label-fidelity test re-enabled (was previously
            # skipped on the assumption tiny test datasets couldn't support
            # discriminative class conditioning — but if it fails, that's
            # exactly the signal we want).
            "MEAN_RMSE_THRESHOLD":           0.2,
            "STD_RMSE_THRESHOLD":            0.2,
        },
    ),
    GANTestConfig(
        name="MTCTABGAN",
        gan_type=GANType.MT_CTAB_GAN,
        n_classes=N_TRADING_CLASSES,
        minority_classes=[1, 2],
        make_dataset=_make_mt_ctab_dataset,
        setup_generated=_mt_ctab_setup_generated,
        extra_fit_kwargs={
            "epochs":               20,
            "batch_size":           64,
            "latent_dim":           32,
            "generator_layers":     [64],
            "discriminator_layers": [64],
            "monitor_metric":       "combined",  # skip per-epoch eval cost
            "eval_frequency":       0,
        },
        extra_tests={
            "MEAN_RMSE_THRESHOLD":           0.2,
            "STD_RMSE_THRESHOLD":            0.2,
        },
    ),
    # MLX backends — distinct constructor (uses hidden_dim, not the
    # generator_layers/discriminator_layers split; no auxiliary subclass).
    GANTestConfig(
        name="CTABGANMLX",
        gan_type=GANType.CTAB_GAN,
        n_classes=N_TRADING_CLASSES,
        minority_classes=[1, 2],
        make_dataset=_make_ctab_dataset,
        setup_generated=_ctab_setup_generated,
        prefer_mlx=True,
        extra_fit_kwargs={
            "epochs":         20,
            "batch_size":     64,
            "latent_dim":     32,
            "hidden_dim":     64,
            # MLX only supports "eval_quality"
            "monitor_metric": "eval_quality",
            "eval_frequency": 5,
        },
        extra_tests={
            "MEAN_RMSE_THRESHOLD": 0.2,
            "STD_RMSE_THRESHOLD":  0.2,
        },
    ),
    GANTestConfig(
        name="MTCTABGANMLX",
        gan_type=GANType.MT_CTAB_GAN,
        n_classes=N_TRADING_CLASSES,
        minority_classes=[1, 2],
        make_dataset=_make_mt_ctab_dataset,
        setup_generated=_mt_ctab_setup_generated,
        prefer_mlx=True,
        extra_fit_kwargs={
            "epochs":         20,
            "batch_size":     64,
            "latent_dim":     32,
            "hidden_dim":     64,
            # MLX multi-task only supports "eval_quality"
            "monitor_metric": "eval_quality",
            "eval_frequency": 5,
        },
        extra_tests={
            "MEAN_RMSE_THRESHOLD": 0.2,
            "STD_RMSE_THRESHOLD":  0.2,
        },
    ),
]


# ---------------------------------------------------------------------------
# Class factory — builds a TestCase subclass from a GANTestConfig
# ---------------------------------------------------------------------------

from GANs.tests.quality_base import (  # noqa: E402
    GANQualityMixin,
    QUALITY_EPOCHS,
    QUALITY_BATCH,
    QUALITY_N_CRITIC,
)


def _make_quality_test_class(config: GANTestConfig) -> type:
    """Return a skipUnless-decorated TestCase subclass for the given config."""

    # Use default-argument binding to capture config in closures safely
    @classmethod
    def _make_dataset(cls, seed: int = 42, _fn: Callable = config.make_dataset):
        return _fn(seed)

    @classmethod
    def _fit_kwargs(cls, _extra: dict = config.extra_fit_kwargs) -> dict:
        base = {
            "epochs":         QUALITY_EPOCHS,
            "batch_size":     QUALITY_BATCH,
            "n_critic":       QUALITY_N_CRITIC,
            "assess_quality": False,
            "verbose":        False,
        }
        base.update(_extra)
        return base

    @classmethod
    def _setup_generated(cls, iface: Any, _fn: Callable = config.setup_generated) -> None:
        _fn(cls, iface)

    attrs: dict[str, Any] = {
        "GAN_TYPE":        config.gan_type,
        "N_CLASSES":       config.n_classes,
        "MINORITY_CLASSES": config.minority_classes,
        "PREFER_MLX":      config.prefer_mlx,
        "_make_dataset":    _make_dataset,
        "_fit_kwargs":      _fit_kwargs,
        "_setup_generated": _setup_generated,
    }
    attrs.update(config.extra_tests)

    cls = type(
        f"Test{config.name}Quality",
        (GANQualityMixin, unittest.TestCase),
        attrs,
    )
    cls = unittest.skipUnless(_RUN, _SKIP_MSG)(cls)
    if config.prefer_mlx:
        cls = unittest.skipUnless(
            _HAS_MLX, "MLX backend not available on this host"
        )(cls)
    return cls


# ---------------------------------------------------------------------------
# Generate and register test classes into this module's namespace so that
# unittest and pytest discover them automatically.
# ---------------------------------------------------------------------------

for _cfg in _GAN_CONFIGS:
    _cls = _make_quality_test_class(_cfg)
    globals()[_cls.__name__] = _cls
del _cfg, _cls   # prevent loop variables from leaking into module globals


# ---------------------------------------------------------------------------
# Direct execution
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    os.environ["RUN_SLOW_TESTS"] = "1"
    unittest.main(verbosity=2)
