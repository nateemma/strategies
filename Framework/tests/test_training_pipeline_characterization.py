"""Characterization fixtures for the training data-prep pipeline.

Purpose: make a future TrainingEngine extraction validatable in SECONDS instead
of the days a real retrain/backtest takes. These golden fixtures pin the exact
deterministic boundary that feeds ``classifier.train()`` — the outputs of:

    prepare_training_data(norm=False)   # split + window + one-hot
    -> preprocess_training_data         # GAN-off pass-through
    -> get_training_class_weights
    -> seeded shuffle (train_model)

Scaling is deliberately excluded (norm=False): it belongs to FeatureNormalizer,
which is verified separately by the scaler-artifact diff. GAN augmentation is
disabled (gan_type=NONE -> enhance/preprocess pass through) because it is the
last, separately-extracted concern and is stochastic; the deterministic core
orchestration is what a TrainingEngine refactor must preserve.

The strategy object is built via __new__ (no freqtrade runtime) with a fixed
config, fed a fixed synthetic dataframe + labels, so the whole thing is
self-contained — no market data, no saved scalers, no model fitting.

To regenerate the goldens after a DELIBERATE behavior change:
    REGEN_TRAINING_FIXTURE=1 .venv/bin/python -m pytest <thisfile>
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pytest
from sklearn.utils import shuffle

_STRATEGIES_ROOT = Path(__file__).resolve().parent.parent.parent
for _sub in ("", "Framework", "utils"):
    _p = str(_STRATEGIES_ROOT / _sub) if _sub else str(_STRATEGIES_ROOT)
    if _p not in sys.path:
        sys.path.insert(0, _p)

FIXTURE_DIR = Path(__file__).resolve().parent / "fixtures"
SEED = 12345
N_ROWS = 600
N_FEATURES = 8
SEQ_LEN = 16


def _mlx_available() -> bool:
    try:
        import mlx.core as mx  # type: ignore
        return hasattr(mx, "metal") and mx.metal.is_available()
    except (ImportError, ModuleNotFoundError):
        return False


def _fixed_inputs():
    """Deterministic synthetic (dataframe, labels) — stands in for one pair of
    already-normalized features + tri-state labels."""
    import pandas as pd

    rng = np.random.RandomState(SEED)
    cols = [f"f{i:02d}" for i in range(N_FEATURES)]
    data = rng.standard_normal((N_ROWS, N_FEATURES)).astype(np.float32)
    df = pd.DataFrame(data, columns=cols)
    labels = rng.randint(0, 3, size=N_ROWS).astype(np.int64)
    return df, labels


def _build_strategy(tensor_method: int):
    """BaseNNStrategy configured for the deterministic, GAN-off, no-scaler path."""
    from Framework.BaseNNStrategy import BaseNNStrategy
    from Framework.BaseStrategy import GANType
    from utils.DataframeUtils import DataframeUtils

    s = BaseNNStrategy.__new__(BaseNNStrategy)
    s.seq_len = SEQ_LEN
    s.tensor_method = tensor_method
    s.TRAIN_DATA_SPLIT = 0.8
    s.gan_type = GANType.NONE
    s.gan_augment = False
    s.augment_training_data = False
    s.shuffle_train_data = True
    s.use_markov_smoothing = False
    s.classifier = None
    s.dataframeUtils = DataframeUtils()
    return s


def _run_pipeline(tensor_method: int) -> dict:
    """Replicate train_model's data-transform sequence up to (not incl.)
    classifier.train(), returning the boundary arrays as numpy."""
    s = _build_strategy(tensor_method)
    df, labels = _fixed_inputs()

    tsr_train, tsr_test, train_labels, test_labels = s.prepare_training_data(
        [df], [labels], norm=False, pair_names=["TEST/USDT"]
    )
    tsr_train, tsr_test, train_labels, test_labels = s.preprocess_training_data(
        df, tsr_train, tsr_test, train_labels, test_labels
    )
    class_weights = s.get_training_class_weights(
        train_labels=train_labels, validation_labels=test_labels
    )

    # Markov transition matrix — deterministic given the held-out labels;
    # exercises _labels_to_class_indices + _compute_markov_transition_matrix
    # (the markov branch train_model runs when use_markov_smoothing is on).
    markov = s._compute_markov_transition_matrix(
        s._labels_to_class_indices(test_labels), num_classes=3
    )

    if s.shuffle_train_data:
        tsr_train = np.asarray(tsr_train)
        train_labels = np.asarray(train_labels)
        tsr_train, train_labels = shuffle(tsr_train, train_labels, random_state=42)

    return {
        "tsr_train": np.asarray(tsr_train),
        "tsr_test": np.asarray(tsr_test),
        "train_labels": np.asarray(train_labels),
        "test_labels": np.asarray(test_labels),
        "class_weights": np.asarray(class_weights, dtype=np.float64),
        "markov_matrix": np.asarray(markov, dtype=np.float64),
    }


def _fixture_path(tensor_method: int) -> Path:
    return FIXTURE_DIR / f"training_pipeline_single_task_m{tensor_method}.npz"


# tensor_method governs the label-offset branch in prepare_training_data, so each
# method gets its own golden. method 0 (numpy) runs everywhere; method 3 (MLX) is
# the Apple-Silicon production path, guarded on mlx availability.
_METHODS = [pytest.param(0, id="numpy")]
if _mlx_available():
    _METHODS.append(pytest.param(3, id="mlx"))


@pytest.mark.parametrize("tensor_method", _METHODS)
def test_training_pipeline_matches_golden(tensor_method):
    out = _run_pipeline(tensor_method)
    path = _fixture_path(tensor_method)

    if os.environ.get("REGEN_TRAINING_FIXTURE") or not path.exists():
        FIXTURE_DIR.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(path, **out)
        pytest.skip(f"(re)generated golden fixture {path.name}")

    golden = np.load(path)
    for key, value in out.items():
        ref = golden[key]
        assert value.shape == ref.shape, f"{key}: shape {value.shape} != {ref.shape}"
        assert value.dtype == ref.dtype, f"{key}: dtype {value.dtype} != {ref.dtype}"
        assert np.array_equal(value, ref), f"{key}: values diverged from golden"


def test_pipeline_is_deterministic():
    """Two runs of the same config must be bit-identical (guards against hidden
    nondeterminism that would make the golden fixtures meaningless)."""
    a = _run_pipeline(0)
    b = _run_pipeline(0)
    for key in a:
        assert np.array_equal(a[key], b[key]), f"{key}: non-deterministic across runs"
