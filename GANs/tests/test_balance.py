"""
Tests for ``GANs.balance.balance_multi_task``.

These tests don't need a real GAN — they mock the interface so we can
control what ``generate()`` returns and inspect what ``task_labels``
each round was called with.  The properties under test are:

  * Targets are computed correctly from per-task ratios.
  * Each round's target task label is one-hot for the chosen class.
  * Other-task labels are drawn from a deficit-weighted distribution
    (NOT majority class — that's the bug we're fixing).
  * The loop terminates when every (task, class) deficit hits zero.
  * Tabular (DataFrame) and sequential (ndarray) backends both work.
"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path
from typing import Dict, List
from unittest.mock import MagicMock

import numpy as np
import pandas as pd

STRATEGIES_ROOT = str(Path(__file__).parent.parent.parent)
if STRATEGIES_ROOT not in sys.path:
    sys.path.insert(0, STRATEGIES_ROOT)

# Stub freqtrade so the GANs module imports cleanly.
_ft_stub = MagicMock()
sys.modules.setdefault("freqtrade", _ft_stub)
sys.modules.setdefault("freqtrade.persistence", _ft_stub)
sys.modules.setdefault("freqtrade.strategy", _ft_stub)

from GANs.balance import balance_multi_task, balance_single_task  # noqa: E402
from GANs.GANType import GANType  # noqa: E402


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _one_hot(class_indices: np.ndarray, num_classes: int) -> np.ndarray:
    return np.eye(num_classes, dtype=np.float32)[class_indices.astype(int)]


def _build_imbalanced_labels(
    n: int,
    num_classes: int,
    majority_idx: int,
    majority_pct: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """One-hot labels with ``majority_pct`` of rows at ``majority_idx``,
    remainder split evenly across the other classes."""
    n_majority = int(n * majority_pct)
    n_minority = (n - n_majority) // (num_classes - 1)
    indices = np.full(n, majority_idx, dtype=int)
    cursor = n_majority
    for c in range(num_classes):
        if c == majority_idx:
            continue
        indices[cursor:cursor + n_minority] = c
        cursor += n_minority
    rng.shuffle(indices)
    return _one_hot(indices, num_classes)


class _MockInterface:
    """Mock GANInterface that records every generate() call and returns
    canned data of the requested batch size."""

    def __init__(self, sample_shape, return_dataframe: bool = False):
        self.sample_shape = sample_shape
        self.return_dataframe = return_dataframe
        self.calls: List[Dict] = []

    def generate(self, n: int, *, task_labels: Dict[str, np.ndarray]):
        self.calls.append({"n": n, "task_labels": {
            k: v.copy() for k, v in task_labels.items()
        }})
        if self.return_dataframe:
            data = pd.DataFrame(
                np.zeros((n, self.sample_shape[-1]), dtype=np.float32),
                columns=[f"f{i}" for i in range(self.sample_shape[-1])],
            )
        else:
            data = np.zeros((n, *self.sample_shape[1:]), dtype=np.float32)
        return data, task_labels


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestBalanceMultiTask(unittest.TestCase):

    def setUp(self):
        self.rng = np.random.default_rng(42)

    def test_returns_originals_when_no_targets(self):
        labels = {"trading": _one_hot(np.array([0, 1, 2, 1, 1]), 3)}
        data = np.zeros((5, 1, 4), dtype=np.float32)
        iface = _MockInterface(sample_shape=(0, 1, 4))
        out_data, out_labels = balance_multi_task(
            iface, data, labels, target_ratios={"trading": 0.0},
            log=lambda *_: None, debug_log=lambda *_: None,
        )
        # Pass-through.
        self.assertEqual(len(iface.calls), 0)
        self.assertEqual(out_data.shape, data.shape)
        np.testing.assert_array_equal(out_labels["trading"], labels["trading"])

    def test_scalar_target_ratio_broadcasts_to_all_tasks(self):
        """``target_ratios=0.5`` should broadcast to every task in labels.
        This is the ergonomic shortcut for ratio sweeps — without it,
        callers have to spell out the same number per task."""
        n = 600
        labels = {
            "trading": _build_imbalanced_labels(n, 3, 1, 0.8, self.rng),
            "regime":  _build_imbalanced_labels(n, 3, 1, 0.8, self.rng),
        }
        data = np.zeros((n, 1, 4), dtype=np.float32)
        iface = _MockInterface(sample_shape=(0, 1, 4))

        out_data, out_labels = balance_multi_task(
            iface, data, labels,
            target_ratios=0.5,  # scalar — should hit every task
            batch_size=200, max_rounds=20,
            log=lambda *_: None, debug_log=lambda *_: None,
        )

        # Both tasks should have augmented to ~50% of the majority count
        # for each minority class.  Allow some slack for rounding.
        for task in ("trading", "regime"):
            idx = np.argmax(out_labels[task], axis=1)
            counts = np.bincount(idx, minlength=3)
            majority = int(counts.max())
            target = int(majority * 0.5)
            for c in range(3):
                self.assertGreaterEqual(
                    int(counts[c]), target,
                    f"task {task!r} class {c} count {int(counts[c])} < target {target}"
                )

    def test_target_class_one_hot_per_round(self):
        """Each round's target_task label should be one-hot for one class."""
        n = 600
        labels = {
            "trading": _build_imbalanced_labels(n, 3, 1, 0.8, self.rng),
            "regime":  _build_imbalanced_labels(n, 3, 1, 0.8, self.rng),
        }
        data = np.zeros((n, 1, 4), dtype=np.float32)
        iface = _MockInterface(sample_shape=(0, 1, 4))

        balance_multi_task(
            iface, data, labels,
            target_ratios={"trading": 0.8, "regime": 0.8},
            batch_size=100, max_rounds=50,
            log=lambda *_: None, debug_log=lambda *_: None,
        )

        self.assertGreater(len(iface.calls), 0)
        for round_idx, call in enumerate(iface.calls):
            tl = call["task_labels"]
            # Every batch's labels must be one-hot (not soft, not zero rows).
            for task, oh in tl.items():
                row_sums = oh.sum(axis=1)
                self.assertTrue(
                    np.allclose(row_sums, 1.0),
                    f"task {task!r} labels are not one-hot in round "
                    f"{round_idx}: row_sums={row_sums[:5]}"
                )

    def test_other_task_labels_track_deficits_not_majority(self):
        """The bug we're fixing: when augmenting trading=0, the regime
        labels in the same batch should NOT be biased toward regime's
        majority class; they should be biased toward regime's minority
        classes (which still have deficits)."""
        n = 600
        labels = {
            "trading": _build_imbalanced_labels(n, 3, 1, 0.8, self.rng),
            "regime":  _build_imbalanced_labels(n, 3, 1, 0.8, self.rng),
        }
        data = np.zeros((n, 1, 4), dtype=np.float32)
        iface = _MockInterface(sample_shape=(0, 1, 4))

        balance_multi_task(
            iface, data, labels,
            target_ratios={"trading": 0.8, "regime": 0.8},
            batch_size=100, max_rounds=50,
            log=lambda *_: None, debug_log=lambda *_: None,
        )

        # Find the FIRST call where target_task was trading.  At that
        # point regime still has its full minority deficit, so the
        # regime labels in this call should be biased AWAY from
        # regime's majority class (=1).
        first_trading_call = None
        for call in iface.calls:
            tl = call["task_labels"]
            trading_target_class = int(np.argmax(tl["trading"][0]))
            # All rows in this batch share a target class — that's the
            # "target task" for the round.  Find a round where trading is
            # the target (every row has the same class) but regime isn't.
            trading_uniform = np.all(np.argmax(tl["trading"], axis=1) == trading_target_class)
            regime_classes = np.argmax(tl["regime"], axis=1)
            regime_uniform = np.all(regime_classes == regime_classes[0])
            if trading_uniform and not regime_uniform:
                first_trading_call = call
                break

        self.assertIsNotNone(
            first_trading_call,
            "Expected at least one round where trading was the target "
            "and regime labels varied — got none."
        )

        # Of regime labels in this batch, fewer than half should be
        # majority class (=1).  The old "all-majority" bug would have
        # 100% regime=1; the deficit-driven sampling should put most
        # weight on regime=0 and regime=2.
        regime_majority_frac = float(np.mean(
            np.argmax(first_trading_call["task_labels"]["regime"], axis=1) == 1
        ))
        self.assertLess(
            regime_majority_frac, 0.5,
            f"Regime majority class fraction was {regime_majority_frac:.2%} "
            f"in a batch where regime had a large deficit — labels are "
            f"still biased toward majority (the original bug)."
        )

    def test_loop_terminates_when_targets_met(self):
        """If targets are easily achievable, the loop should exit before
        max_rounds, not just stop because of the cap."""
        n = 600
        labels = {
            "trading": _build_imbalanced_labels(n, 3, 1, 0.8, self.rng),
        }
        data = np.zeros((n, 1, 4), dtype=np.float32)
        iface = _MockInterface(sample_shape=(0, 1, 4))

        out_data, out_labels = balance_multi_task(
            iface, data, labels,
            target_ratios={"trading": 0.8},
            batch_size=10_000, max_rounds=20,
            log=lambda *_: None, debug_log=lambda *_: None,
        )

        # With a single task and a generous batch_size, we should
        # converge in ~2 rounds (one per minority class) — well under
        # the cap.
        self.assertLessEqual(len(iface.calls), 5)

        # Final per-class counts should all be ≥ target.
        idx = np.argmax(out_labels["trading"], axis=1)
        counts = np.bincount(idx, minlength=3)
        majority = int(counts.max())
        target = int(majority * 0.8)
        for c in range(3):
            self.assertGreaterEqual(
                int(counts[c]), target,
                f"trading class {c} count {int(counts[c])} < target {target}"
            )

    def test_dataframe_backend_concat(self):
        """Tabular backends return DataFrames — concat must use pd.concat,
        not np.concatenate.  A tabular CTAB-GAN-style flow shouldn't crash."""
        n = 100
        labels = {
            "trading": _build_imbalanced_labels(n, 3, 1, 0.7, self.rng),
        }
        data = pd.DataFrame(
            np.zeros((n, 4), dtype=np.float32),
            columns=[f"f{i}" for i in range(4)],
        )
        iface = _MockInterface(sample_shape=(0, 4), return_dataframe=True)

        out_data, out_labels = balance_multi_task(
            iface, data, labels,
            target_ratios={"trading": 0.7},
            batch_size=50, max_rounds=20,
            log=lambda *_: None, debug_log=lambda *_: None,
        )

        self.assertIsInstance(out_data, pd.DataFrame)
        self.assertGreater(len(out_data), n)
        # Column structure preserved.
        self.assertEqual(list(out_data.columns), list(data.columns))


# ---------------------------------------------------------------------------
# Single-task balance — covers the WGAN / CTAB_GAN flows that used to
# live as separate ~150-LOC methods inside BaseNNStrategy.
# ---------------------------------------------------------------------------


class _MockSingleTaskInterface:
    """Mock GANInterface for single-task balance tests.

    Records every generate() call (n + kwargs) and returns canned data
    of the requested batch size in the format the configured gan_type
    would produce in production:
      * GANType.WGAN      → 3-D ndarray (n, 1, F), expects ``one_hot=``
      * GANType.CTAB_GAN  → DataFrame (n, F),       expects ``class_label=``
    """

    def __init__(self, gan_type, num_features: int):
        self.gan_type = gan_type
        self.num_features = num_features
        self.calls: List[Dict] = []

    def generate(self, n: int, **kwargs):
        self.calls.append({"n": n, "kwargs": {
            k: (v.copy() if isinstance(v, np.ndarray) else v)
            for k, v in kwargs.items()
        }})
        if self.gan_type == GANType.WGAN:
            # WGAN returns (n, 1, F).
            return np.zeros((n, 1, self.num_features), dtype=np.float32)
        if self.gan_type == GANType.CTAB_GAN:
            return pd.DataFrame(
                np.zeros((n, self.num_features), dtype=np.float32),
                columns=[f"f{i}" for i in range(self.num_features)],
            )
        raise NotImplementedError(self.gan_type)


class TestBalanceSingleTaskWGAN(unittest.TestCase):
    """WGAN path: 3-D generate output, one_hot conditioning."""

    def setUp(self):
        self.rng = np.random.default_rng(7)

    def test_returns_originals_when_ratio_is_zero(self):
        labels = np.array([0, 1, 2, 1, 1], dtype=np.int64)
        data = np.zeros((5, 4), dtype=np.float32)
        iface = _MockSingleTaskInterface(GANType.WGAN, num_features=4)

        out_data, out_labels = balance_single_task(
            iface, data, labels, target_ratio=0.0,
            log=lambda *_: None, debug_log=lambda *_: None,
        )
        self.assertEqual(len(iface.calls), 0)
        self.assertEqual(out_data.shape, data.shape)
        np.testing.assert_array_equal(out_labels, labels)

    def test_class_indices_input_returns_class_indices_output(self):
        """When labels are 1-D class indices, output must also be 1-D."""
        labels = np.repeat([0, 1, 2], [80, 10, 10]).astype(np.int64)
        self.rng.shuffle(labels)
        data = np.zeros((100, 4), dtype=np.float32)
        iface = _MockSingleTaskInterface(GANType.WGAN, num_features=4)

        out_data, out_labels = balance_single_task(
            iface, data, labels, target_ratio=0.5,
            log=lambda *_: None, debug_log=lambda *_: None,
        )
        # Output shape preserved: 1-D → 1-D.
        self.assertEqual(out_labels.ndim, 1)
        self.assertEqual(out_data.shape[1], 4)
        # 80 majority (class 0) → 50% target → 40 each for classes 1, 2.
        # Started with 10 each, so we need +30 each.
        self.assertEqual(len(iface.calls), 2)
        for call in iface.calls:
            self.assertIn("one_hot", call["kwargs"])
            oh = call["kwargs"]["one_hot"]
            # All rows in a single call share the target class one-hot.
            row_sums = oh.sum(axis=1)
            self.assertTrue(np.allclose(row_sums, 1.0))
            class_picked = int(np.argmax(oh[0]))
            self.assertTrue(np.all(np.argmax(oh, axis=1) == class_picked))

    def test_one_hot_input_returns_one_hot_output(self):
        labels_idx = np.repeat([0, 1, 2], [80, 10, 10]).astype(np.int64)
        self.rng.shuffle(labels_idx)
        labels_oh = _one_hot(labels_idx, 3)
        data = np.zeros((100, 4), dtype=np.float32)
        iface = _MockSingleTaskInterface(GANType.WGAN, num_features=4)

        out_data, out_labels = balance_single_task(
            iface, data, labels_oh, target_ratio=0.5,
            log=lambda *_: None, debug_log=lambda *_: None,
        )
        # Output shape preserved: 2-D → 2-D one-hot.
        self.assertEqual(out_labels.ndim, 2)
        self.assertEqual(out_labels.shape[1], 3)
        row_sums = out_labels.sum(axis=1)
        self.assertTrue(np.allclose(row_sums, 1.0))

    def test_wgan_seq_dim_squeezed_into_2d_output(self):
        """WGAN's (n, 1, F) generate output must be squeezed to (n, F)
        before concatenating with the 2-D real data — otherwise we'd
        get a shape mismatch crash."""
        labels = np.repeat([0, 1, 2], [60, 20, 20]).astype(np.int64)
        data = np.zeros((100, 4), dtype=np.float32)
        iface = _MockSingleTaskInterface(GANType.WGAN, num_features=4)

        out_data, _ = balance_single_task(
            iface, data, labels, target_ratio=0.5,
            log=lambda *_: None, debug_log=lambda *_: None,
        )
        self.assertEqual(out_data.ndim, 2)
        self.assertEqual(out_data.shape[1], 4)


class TestBalanceSingleTaskCTAB(unittest.TestCase):
    """CTAB_GAN path: DataFrame generate output, class_label conditioning."""

    def setUp(self):
        self.rng = np.random.default_rng(7)

    def test_dataframe_input_returns_dataframe(self):
        labels = np.repeat([0, 1, 2], [60, 20, 20]).astype(np.int64)
        self.rng.shuffle(labels)
        data = pd.DataFrame(
            np.zeros((100, 4), dtype=np.float32),
            columns=[f"f{i}" for i in range(4)],
        )
        iface = _MockSingleTaskInterface(GANType.CTAB_GAN, num_features=4)

        out_data, out_labels = balance_single_task(
            iface, data, labels, target_ratio=0.5,
            log=lambda *_: None, debug_log=lambda *_: None,
        )
        self.assertIsInstance(out_data, pd.DataFrame)
        self.assertEqual(list(out_data.columns), list(data.columns))
        # Each generate call should pass class_label, not one_hot.
        self.assertGreater(len(iface.calls), 0)
        for call in iface.calls:
            self.assertIn("class_label", call["kwargs"])
            self.assertIsInstance(call["kwargs"]["class_label"], int)

    def test_per_class_target_ratio_dict(self):
        """target_ratio={0: 1.0, 2: 0.4} only augments classes 0 and 2,
        ignoring 1 — even if 1 is the minority."""
        labels = np.repeat([0, 1, 2], [50, 30, 20]).astype(np.int64)
        data = pd.DataFrame(
            np.zeros((100, 4), dtype=np.float32),
            columns=[f"f{i}" for i in range(4)],
        )
        iface = _MockSingleTaskInterface(GANType.CTAB_GAN, num_features=4)

        out_data, out_labels = balance_single_task(
            iface, data, labels,
            target_ratio={0: 1.0, 2: 0.4},
            log=lambda *_: None, debug_log=lambda *_: None,
        )
        # Class 0 was already at the majority count — no need to augment.
        # Class 2 had 20, target 0.4 * 50 = 20 — also no need.
        # Class 1 isn't in the ratio dict, so it's ignored.
        # → no calls expected.
        self.assertEqual(len(iface.calls), 0)


def _mlx_available() -> bool:
    try:
        import mlx.core as mx  # type: ignore
        return hasattr(mx, "metal") and mx.metal.is_available()
    except (ImportError, ModuleNotFoundError):
        return False


@unittest.skipUnless(_mlx_available(), "MLX not available (requires Apple Silicon + mlx)")
class TestBalanceSeedingDeterminism(unittest.TestCase):
    """The ``seed`` arg makes synthetic generation reproducible: MLX backends
    draw the latent z from the global MLX RNG, so seeding it yields identical
    synthetic rows across runs (the prerequisite for byte-validating the GAN
    augmentation path)."""

    class _MLXGenInterface:
        """WGAN-style stub whose generate() draws from mx.random.normal, so its
        output is RNG-driven and seeding actually has an effect."""

        def __init__(self, num_features: int):
            self.gan_type = GANType.WGAN
            self.num_features = num_features

        def generate(self, n: int, **kwargs):
            import mlx.core as mx
            z = mx.random.normal((n, 1, self.num_features))
            mx.eval(z)
            return np.array(z, dtype=np.float32)

    def _run(self, seed):
        labels = np.repeat([0, 1, 2], [80, 10, 10]).astype(np.int64)
        data = np.zeros((100, 4), dtype=np.float32)
        iface = self._MLXGenInterface(num_features=4)
        out_data, _ = balance_single_task(
            iface, data, labels, target_ratio=0.5,
            log=lambda *_: None, debug_log=lambda *_: None, seed=seed,
        )
        return np.asarray(out_data)

    def test_same_seed_is_reproducible(self):
        np.testing.assert_array_equal(self._run(42), self._run(42))

    def test_no_seed_is_nondeterministic(self):
        # Unseeded generation must differ run-to-run (otherwise the seed arg
        # isn't what's producing the determinism above).
        self.assertFalse(np.array_equal(self._run(None), self._run(None)))


if __name__ == "__main__":
    unittest.main(verbosity=2)
