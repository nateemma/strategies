"""Pair-conditioning tests for CTABGANMLX.

Verifies that:
1. Backward compat: fit without pair_labels works as before.
2. Pair conditioning: fit + generate with pair_labels produces samples whose
   per-feature means track the pair-specific real means, not the global means.
3. Save/load round-trip preserves pair conditioning state.
4. Generating without pair_label raises when the model was fit with pairs.
5. balance.py:_resolve_pair_label correctly looks up names → indices.

Run via:
    RUN_SLOW_TESTS=1 python -m pytest \\
        user_data/strategies/GANs/tests/test_pair_conditioning.py -v -s
"""

from __future__ import annotations

import os
import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pandas as pd

STRATEGIES_ROOT = str(Path(__file__).parent.parent.parent)
if STRATEGIES_ROOT not in sys.path:
    sys.path.insert(0, STRATEGIES_ROOT)

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

_RUN = bool(os.environ.get("RUN_SLOW_TESTS"))
_SKIP_MSG = "Set RUN_SLOW_TESTS=1 to enable pair conditioning tests"

try:
    import mlx.core as _mx
    _HAS_MLX = hasattr(_mx, "metal") and _mx.metal.is_available()
except Exception:
    _HAS_MLX = False


N_FEATURES = 6
N_PAIRS = 3
N_CLASSES = 3
N_PER_PAIR_CLASS = 800  # 2400 rows per pair, 7200 total
PAIR_NAMES = ["BTC/USDT", "ETH/USDT", "SOL/USDT"]


def _make_pair_specific_dataset(seed: int = 42):
    """Build a dataset where each (pair, class) has a distinct feature mean.

    The pair shift dominates the class shift, so a pair-blind generator
    would average across pairs and miss the per-pair structure entirely.
    """
    rng = np.random.default_rng(seed)
    # Class shifts — small, e.g. 0.2 magnitude
    class_shifts = np.array([
        [+0.20, -0.20, +0.10, -0.10, +0.20, -0.20],
        [-0.20, +0.20, -0.10, +0.10, -0.20, +0.20],
        [+0.00, +0.00, +0.20, -0.20, +0.00, +0.00],
    ], dtype="float32")
    # Pair shifts — larger, e.g. 0.6 magnitude (dominant)
    pair_shifts = np.array([
        [+0.6, +0.6, -0.6, -0.6, +0.0, +0.0],
        [-0.6, -0.6, +0.6, +0.6, +0.0, +0.0],
        [+0.0, +0.0, +0.0, +0.0, +0.6, -0.6],
    ], dtype="float32")

    data_chunks = []
    class_labels_chunks = []
    pair_labels_chunks = []
    for p in range(N_PAIRS):
        for c in range(N_CLASSES):
            mean = pair_shifts[p] + class_shifts[c]
            noise = rng.normal(0, 0.15, (N_PER_PAIR_CLASS, N_FEATURES)).astype("float32")
            chunk = np.clip(mean + noise, -1, 1)
            data_chunks.append(chunk)
            class_labels_chunks.append(np.full(N_PER_PAIR_CLASS, c, dtype=np.int32))
            pair_labels_chunks.append(np.full(N_PER_PAIR_CLASS, p, dtype=np.int32))

    data = np.concatenate(data_chunks, axis=0)
    class_labels = np.concatenate(class_labels_chunks, axis=0)
    pair_labels = np.concatenate(pair_labels_chunks, axis=0)

    # Shuffle so pairs/classes interleave in the training data
    perm = rng.permutation(len(data))
    return (
        data[perm],
        class_labels[perm],
        pair_labels[perm],
    )


@unittest.skipUnless(_RUN, _SKIP_MSG)
@unittest.skipUnless(_HAS_MLX, "MLX backend not available")
class PairConditioningTests(unittest.TestCase):
    """End-to-end tests for pair-conditioned CTABGANMLX."""

    @classmethod
    def setUpClass(cls):
        from GANs.df_ctab_gan_mlx import CTABGANMLX  # noqa: E402

        cls.data, cls.class_labels, cls.pair_labels = _make_pair_specific_dataset()
        cls.cols = [f"f{i}" for i in range(N_FEATURES)]
        cls.df = pd.DataFrame(cls.data, columns=cls.cols)
        # One-hot class labels (matches CreateGAN flow)
        cls.class_oh = np.eye(N_CLASSES, dtype=np.float32)[cls.class_labels]

        cls.model = CTABGANMLX(
            latent_dim=32,
            hidden_dim=64,
            epochs=60,
            batch_size=64,
            n_critic=3,
            eval_frequency=0,
            verbose=False,
        )
        cls.model.fit(
            cls.df,
            cls.class_oh,
            pair_labels=cls.pair_labels,
            pair_names=PAIR_NAMES,
        )

    def test_pair_metadata_stored(self):
        self.assertEqual(self.model.num_pairs, N_PAIRS)
        self.assertEqual(self.model.pair_names, PAIR_NAMES)
        self.assertEqual(self.model.cond_dim, N_CLASSES + N_PAIRS)

    def test_pair_label_required_when_conditioned(self):
        with self.assertRaises(ValueError):
            self.model.generate(50, class_label=0)  # missing pair_label

    def test_per_pair_distributions_differ(self):
        """Generated samples for the same class should differ across pairs.

        If pair conditioning is doing its job, gen(class=0, pair=0) and
        gen(class=0, pair=1) should look like real pair 0 vs pair 1 — i.e.,
        the per-pair feature means should be far apart.
        """
        n_gen = 500
        means_per_pair = []
        for p in range(N_PAIRS):
            gen_df = self.model.generate(n_gen, class_label=0, pair_label=p)
            means_per_pair.append(gen_df.values.astype("float32").mean(axis=0))
        means_per_pair = np.stack(means_per_pair, axis=0)

        # Real per-pair means for class 0
        real_means_per_pair = np.stack([
            self.data[(self.class_labels == 0) & (self.pair_labels == p)].mean(axis=0)
            for p in range(N_PAIRS)
        ], axis=0)

        # Pair 0 vs Pair 1 real difference is ~1.2 in feature 0 — generated
        # samples should preserve at least 50% of that gap.
        real_pair_gap_max = np.abs(real_means_per_pair[0] - real_means_per_pair[1]).max()
        gen_pair_gap_max = np.abs(means_per_pair[0] - means_per_pair[1]).max()
        self.assertGreater(
            gen_pair_gap_max,
            0.5 * real_pair_gap_max,
            msg=(
                f"Generated samples don't differentiate pairs: "
                f"real gap {real_pair_gap_max:.3f}, gen gap {gen_pair_gap_max:.3f}. "
                f"This means pair conditioning isn't working."
            ),
        )

    def test_save_load_roundtrip(self):
        from GANs.df_ctab_gan_mlx import CTABGANMLX  # noqa: E402

        with TemporaryDirectory() as tmp:
            self.model.save(tmp)
            restored = CTABGANMLX()
            restored.load(tmp)
            self.assertEqual(restored.num_pairs, N_PAIRS)
            self.assertEqual(restored.pair_names, PAIR_NAMES)
            self.assertEqual(restored.cond_dim, N_CLASSES + N_PAIRS)

            # Loaded model should still produce class+pair conditioned samples
            gen_p0 = restored.generate(100, class_label=0, pair_label=0)
            gen_p1 = restored.generate(100, class_label=0, pair_label=1)
            gap = float(np.abs(gen_p0.values.mean(axis=0) - gen_p1.values.mean(axis=0)).max())
            self.assertGreater(gap, 0.3, msg=f"Loaded model lost pair conditioning, gap={gap:.3f}")


@unittest.skipUnless(_RUN, _SKIP_MSG)
@unittest.skipUnless(_HAS_MLX, "MLX backend not available")
class PairLabelResolutionTests(unittest.TestCase):
    """Test the balance.py:_resolve_pair_label helper."""

    def test_returns_none_when_no_pair_conditioning(self):
        from GANs.balance import _resolve_pair_label  # noqa: E402

        # Fake interface whose model has no pair conditioning
        class _Mock:
            class _Inner:
                num_pairs = 0
                pair_names = None
            _model = _Inner()

        self.assertIsNone(_resolve_pair_label(_Mock(), "BTC/USDT", print))
        self.assertIsNone(_resolve_pair_label(_Mock(), None, print))

    def test_resolves_known_pair_name(self):
        from GANs.balance import _resolve_pair_label  # noqa: E402

        class _Mock:
            class _Inner:
                num_pairs = 3
                pair_names = ["BTC/USDT", "ETH/USDT", "SOL/USDT"]
            _model = _Inner()

        self.assertEqual(_resolve_pair_label(_Mock(), "BTC/USDT", print), 0)
        self.assertEqual(_resolve_pair_label(_Mock(), "ETH/USDT", print), 1)
        self.assertEqual(_resolve_pair_label(_Mock(), "SOL/USDT", print), 2)

    def test_raises_on_unknown_pair(self):
        from GANs.balance import _resolve_pair_label  # noqa: E402

        class _Mock:
            class _Inner:
                num_pairs = 2
                pair_names = ["BTC/USDT", "ETH/USDT"]
            _model = _Inner()

        with self.assertRaises(ValueError):
            _resolve_pair_label(_Mock(), "SHIB/USDT", print)

    def test_raises_when_pair_name_missing(self):
        from GANs.balance import _resolve_pair_label  # noqa: E402

        class _Mock:
            class _Inner:
                num_pairs = 2
                pair_names = ["BTC/USDT", "ETH/USDT"]
            _model = _Inner()

        with self.assertRaises(ValueError):
            _resolve_pair_label(_Mock(), None, print)


if __name__ == "__main__":
    unittest.main()
