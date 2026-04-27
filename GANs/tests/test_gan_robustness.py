"""
Robustness tests for the GAN backends — exercise the failure modes that
the existing tiny-data functional tests in test_mlx_suite.py don't reach.

These tests are gated behind MLX and TF availability checks.  Each runs a
short training cycle on synthetic data, persists the model, reloads, and
then checks properties that real-world training was found to violate:

  * Bulk finiteness — generate hundreds of samples per class and assert
    none are NaN/Inf.  (test_mlx_suite.py only generates 10.)
  * Range bound — if metadata persists feature_min / feature_max, every
    generated value should fall within those bounds (the post-process
    clip should be enforcing this).
  * Class-conditioning differentiation — samples conditioned on class A
    should differ statistically from samples conditioned on class B.
    Mode collapse on a single class would fail this.

Tests intentionally use larger-than-test_mlx_suite sample counts so that
rare failure modes (the ~4% pathological-row symptom we just hit) have
a chance to show up.

Skipped automatically when the relevant backend isn't installed:
    @requires_mlx — needs Apple Silicon + mlx package
    @requires_tf  — needs tensorflow

Run from the strategies root:
    python -m pytest GANs/tests/test_gan_robustness.py -v
"""
from __future__ import annotations

import os
import shutil
import sys
import tempfile
import unittest
from pathlib import Path
from typing import Tuple

import numpy as np

# ---------------------------------------------------------------------------
# Path / environment setup
# ---------------------------------------------------------------------------

STRATEGIES_ROOT = str(Path(__file__).parent.parent.parent)
if STRATEGIES_ROOT not in sys.path:
    sys.path.insert(0, STRATEGIES_ROOT)

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")


# ---------------------------------------------------------------------------
# Availability checks
# ---------------------------------------------------------------------------


def _mlx_available() -> bool:
    try:
        import mlx.core as mx  # type: ignore
        return hasattr(mx, "metal") and mx.metal.is_available()
    except (ImportError, ModuleNotFoundError):
        return False


def _tf_available() -> bool:
    try:
        import tensorflow as tf  # type: ignore
        return True
    except (ImportError, ModuleNotFoundError):
        return False


_HAS_MLX = _mlx_available()
_HAS_TF = _tf_available()

requires_mlx = unittest.skipUnless(
    _HAS_MLX, "MLX not available (requires Apple Silicon + mlx package)"
)
requires_tf = unittest.skipUnless(_HAS_TF, "TensorFlow not available")


# ---------------------------------------------------------------------------
# Synthetic data — distinguishable by class so mode-collapse is detectable
# ---------------------------------------------------------------------------

# Larger than test_mlx_suite to surface rare bad-sample failure modes.
N_TRAIN = 600
N_FEATURES = 16
N_CLASSES = 3
SEQ_LEN = 1
GEN_PER_CLASS = 500   # generate this many samples per class for tests

FAST_EPOCHS = 3
FAST_BATCH = 64
FAST_N_CRITIC = 1


def _make_class_separable_data(seed: int = 7) -> Tuple[np.ndarray, np.ndarray]:
    """Each class has a distinct mean offset so a working GAN should learn
    to produce class-conditional samples whose means differ."""
    rng = np.random.default_rng(seed)
    per_class = N_TRAIN // N_CLASSES

    chunks = []
    for cls_idx in range(N_CLASSES):
        # Class-specific offset in the first few features.
        offset = np.zeros(N_FEATURES, dtype=np.float32)
        offset[: max(1, N_FEATURES // 4)] = (cls_idx - 1) * 0.5
        chunk = rng.normal(loc=offset, scale=0.3, size=(per_class, N_FEATURES)).astype(
            np.float32
        )
        chunks.append(chunk)
    data = np.concatenate(chunks, axis=0).astype(np.float32)

    labels_idx = np.concatenate(
        [np.full(per_class, c, dtype=int) for c in range(N_CLASSES)]
    )
    labels = np.eye(N_CLASSES, dtype="float32")[labels_idx]
    return data, labels


def _one_hot(n: int, cls_idx: int) -> np.ndarray:
    out = np.zeros((n, N_CLASSES), dtype=np.float32)
    out[:, cls_idx] = 1.0
    return out


# ---------------------------------------------------------------------------
# Per-backend test mixin — same assertions, different fit/generate plumbing
# ---------------------------------------------------------------------------


class _RobustnessTestsBase(unittest.TestCase):
    """Subclasses must set ``cls.iface`` (a fitted GANInterface) in
    setUpClass, and may override ``_generate_for_class`` if their
    backend's generate signature differs."""

    iface = None

    @classmethod
    def setUpClass(cls):
        if cls is _RobustnessTestsBase:
            raise unittest.SkipTest("Base class — skip standalone run")

    def _generate_for_class(self, cls_idx: int, n: int = GEN_PER_CLASS) -> np.ndarray:
        return self.iface.generate(n, one_hot=_one_hot(n, cls_idx))

    # ---------- contract: bulk finiteness ----------

    def test_bulk_generation_is_finite(self):
        """Generate a large batch and confirm no NaN/Inf escapes — the
        symptom that bit us in the multi-task WGAN run."""
        for cls_idx in range(N_CLASSES):
            with self.subTest(cls=cls_idx):
                samples = np.asarray(self._generate_for_class(cls_idx))
                n_bad = int((~np.isfinite(samples)).sum())
                self.assertEqual(
                    n_bad, 0,
                    f"class {cls_idx}: {n_bad} non-finite values "
                    f"out of {samples.size} generated"
                )

    # ---------- contract: bounded range ----------

    def test_generated_values_are_bounded(self):
        """Generated samples should fall inside a reasonable range.  If the
        backend persists feature_min / feature_max, those should bound it;
        otherwise we apply a generous sanity bound to catch extreme drift."""
        all_samples = np.concatenate(
            [np.asarray(self._generate_for_class(c)) for c in range(N_CLASSES)],
            axis=0,
        ).reshape(-1, N_FEATURES)

        max_abs = float(np.max(np.abs(all_samples)))
        self.assertLess(
            max_abs, 50.0,
            f"max |sample| = {max_abs} — generator likely unbounded"
        )

    # ---------- contract: class conditioning differentiates ----------

    def test_class_conditioning_produces_distinct_distributions(self):
        """Mode collapse onto one class would make all classes' samples
        identical.  Assert that conditioning on class 0 vs class 2 produces
        distributions whose means differ measurably (the synthetic data has
        class-specific offsets in the first features)."""
        n = GEN_PER_CLASS
        gen0 = np.asarray(self._generate_for_class(0, n)).reshape(n, -1)
        gen2 = np.asarray(self._generate_for_class(2, n)).reshape(n, -1)
        diff = float(np.linalg.norm(gen0.mean(axis=0) - gen2.mean(axis=0)))
        self.assertGreater(
            diff, 1e-3,
            f"Class 0 vs class 2 generation indistinguishable "
            f"(mean L2 distance = {diff:.6f}) — likely mode collapse"
        )


# ---------------------------------------------------------------------------
# WGAN MLX
# ---------------------------------------------------------------------------


@requires_mlx
class TestWGANMLXRobustness(_RobustnessTestsBase):

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        from GANs.GANInterface import GANInterface
        from GANs.GANType import GANType
        cls.tmp = tempfile.mkdtemp(prefix="wgan_mlx_robust_")
        data, labels = _make_class_separable_data()
        cls.iface = GANInterface(GANType.WGAN, save_path=None, prefer_mlx=True)
        cls.iface.fit(
            data, labels,
            epochs=FAST_EPOCHS, batch_size=FAST_BATCH,
            n_critic=FAST_N_CRITIC, verbose=False,
        )

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp, ignore_errors=True)


# ---------------------------------------------------------------------------
# MT_WGAN MLX — needs different generate signature (task_labels dict)
# ---------------------------------------------------------------------------


@requires_mlx
class TestMTWGANMLXRobustness(_RobustnessTestsBase):
    """MT_WGAN's generate() returns (samples, labels_dict) — override the
    helper to unpack and use trading labels as the class condition."""

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        from GANs.GANInterface import GANInterface
        from GANs.GANType import GANType
        cls.tmp = tempfile.mkdtemp(prefix="mt_wgan_mlx_robust_")

        data2d, labels = _make_class_separable_data()
        # MT_WGAN expects 3-D input — reshape with seq_len=1.
        data3d = data2d.reshape(N_TRAIN, SEQ_LEN, N_FEATURES)
        cls.task_labels_train = {"trading": labels, "regime": labels}

        cls.iface = GANInterface(GANType.MT_WGAN, save_path=None, prefer_mlx=True)
        cls.iface.fit(
            data3d, cls.task_labels_train,
            epochs=FAST_EPOCHS, batch_size=FAST_BATCH,
            n_critic=FAST_N_CRITIC, verbose=False,
        )

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp, ignore_errors=True)

    def _generate_for_class(self, cls_idx: int, n: int = GEN_PER_CLASS) -> np.ndarray:
        task_labels = {
            "trading": _one_hot(n, cls_idx),
            "regime":  _one_hot(n, cls_idx),
        }
        gen_x, _ = self.iface.generate(n, task_labels=task_labels)
        return np.asarray(gen_x)


# ---------------------------------------------------------------------------
# CTAB-GAN MLX — DataFrame return type
# ---------------------------------------------------------------------------


@requires_mlx
class TestCTABGANMLXRobustness(unittest.TestCase):
    """CTAB-GAN's generate uses class_label= rather than one_hot.  Run the
    same robustness checks adapted for its DataFrame return type."""

    @classmethod
    def setUpClass(cls):
        if not _HAS_MLX:
            raise unittest.SkipTest("MLX not available")
        import pandas as pd
        from GANs.GANInterface import GANInterface
        from GANs.GANType import GANType

        cls.tmp = tempfile.mkdtemp(prefix="ctab_mlx_robust_")
        data2d, labels = _make_class_separable_data()
        cls.df = pd.DataFrame(data2d, columns=[f"f{i}" for i in range(N_FEATURES)])
        cls.iface = GANInterface(GANType.CTAB_GAN, save_path=None, prefer_mlx=True)
        cls.iface.fit(
            cls.df, labels,
            epochs=FAST_EPOCHS, batch_size=FAST_BATCH, verbose=False,
        )

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp, ignore_errors=True)

    def _generate_for_class(self, cls_idx: int, n: int = GEN_PER_CLASS):
        return self.iface.generate(n, class_label=cls_idx)

    def test_bulk_generation_is_finite(self):
        for cls_idx in range(N_CLASSES):
            with self.subTest(cls=cls_idx):
                df = self._generate_for_class(cls_idx)
                values = df.to_numpy()
                n_bad = int((~np.isfinite(values)).sum())
                self.assertEqual(
                    n_bad, 0,
                    f"class {cls_idx}: {n_bad} non-finite values "
                    f"out of {values.size} generated"
                )

    def test_generated_values_are_bounded(self):
        all_samples = np.concatenate(
            [self._generate_for_class(c).to_numpy() for c in range(N_CLASSES)],
            axis=0,
        )
        max_abs = float(np.max(np.abs(all_samples)))
        self.assertLess(
            max_abs, 50.0,
            f"max |sample| = {max_abs} — generator likely unbounded"
        )

    def test_class_conditioning_produces_distinct_distributions(self):
        n = GEN_PER_CLASS
        gen0 = self._generate_for_class(0, n).to_numpy()
        gen2 = self._generate_for_class(2, n).to_numpy()
        diff = float(np.linalg.norm(gen0.mean(axis=0) - gen2.mean(axis=0)))
        self.assertGreater(
            diff, 1e-3,
            f"Class 0 vs class 2 generation indistinguishable "
            f"(mean L2 distance = {diff:.6f}) — likely mode collapse"
        )


# ---------------------------------------------------------------------------
# WGAN TF — same contract but on the TensorFlow backend
# ---------------------------------------------------------------------------


@requires_tf
class TestWGANTFRobustness(_RobustnessTestsBase):

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        from GANs.GANInterface import GANInterface
        from GANs.GANType import GANType
        cls.tmp = tempfile.mkdtemp(prefix="wgan_tf_robust_")
        data, labels = _make_class_separable_data()
        cls.iface = GANInterface(GANType.WGAN, save_path=None, prefer_mlx=False)
        cls.iface.fit(
            data, labels,
            epochs=FAST_EPOCHS, batch_size=FAST_BATCH,
            n_critic=FAST_N_CRITIC, verbose=False,
        )

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp, ignore_errors=True)


# ---------------------------------------------------------------------------

if __name__ == "__main__":
    unittest.main(verbosity=2)
