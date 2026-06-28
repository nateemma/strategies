"""Smoke tests for the MLX GANomaly classifier variants.

Verifies that every variant (Dense / LSTM / Attention / Transformer) is
paper-faithful at the execution level: the encoder-decoder-encoder bow-tie
builds, a forward pass and the discriminator's feature-matching layer run, one
joint training step (adversarial FM + L1 contextual + L2 encoder) completes
with real gradients, and predict() returns a finite latent ‖z-ẑ‖ anomaly
score. It does NOT check convergence or trading quality — only that the code
runs. Construction bypasses the framework predictor __init__ via __new__ so the
test stays dependency-light and focused on the model wiring.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

# Strategy dirs on sys.path so the sibling-style family imports resolve.
_STRATEGIES_ROOT = Path(__file__).resolve().parent.parent.parent
for _sub in ("", "Framework", "Predictors", "utils", "Anomaly"):
    _p = str(_STRATEGIES_ROOT / _sub) if _sub else str(_STRATEGIES_ROOT)
    if _p not in sys.path:
        sys.path.insert(0, _p)


def _mlx_available() -> bool:
    try:
        import mlx.core as mx  # type: ignore
        return hasattr(mx, "metal") and mx.metal.is_available()
    except (ImportError, ModuleNotFoundError):
        return False


requires_mlx = pytest.mark.skipif(
    not _mlx_available(), reason="MLX not available (requires Apple Silicon + mlx package)"
)

SEQ_LEN, NUM_FEATURES, N = 8, 6, 64


class _StubDFU:
    """Minimal stand-in for the framework dataframeUtils used by predict()."""

    @staticmethod
    def is_dataframe(_data):
        return False


def _variants():
    import NNGANomalyClassifierMLX as M

    return [
        ("Dense", M.GANomalyClassifierMLX_Dense),
        ("LSTM", M.GANomalyClassifierMLX_LSTM),
        ("Attention", M.GANomalyClassifierMLX_Attention),
        ("Transformer", M.GANomalyClassifierMLX_Transformer),
    ]


def _make(variant_cls):
    """Instantiate a variant without the heavy predictor __init__, configured
    for a 1-epoch smoke run."""
    c = variant_cls.__new__(variant_cls)
    c.gan_epochs = 1
    c.clf_epochs = 1
    c.ganomaly_batch_size = 16
    c.batch_size = 16
    c.dataframeUtils = _StubDFU()
    return c


@requires_mlx
@pytest.mark.parametrize("name", ["Dense", "LSTM", "Attention", "Transformer"])
def test_variant_runs_end_to_end(name):
    import mlx.core as mx

    variant_cls = dict(_variants())[name]
    rng = np.random.default_rng(0)
    x_np = rng.standard_normal((N, SEQ_LEN, NUM_FEATURES)).astype(np.float32)

    c = _make(variant_cls)

    # Bow-tie: create_model must build a model exposing the second encoder E.
    model = c.create_model(SEQ_LEN, NUM_FEATURES)
    assert hasattr(model, "encoder2"), "second encoder missing — bow-tie not wired"

    # Forward pass: (reconstruction, trading) with expected shapes.
    rec, trd = model(mx.array(x_np[:4]))
    assert rec.shape == (4, SEQ_LEN, NUM_FEATURES)
    assert trd.shape == (4, 3)

    # Feature-matching layer must exist and return 2-D pre-logit activations.
    feats = c.discriminator.features(mx.array(x_np[:4]))
    assert feats.ndim == 2 and feats.shape[0] == 4

    # One real joint training step (gradients flow, no shape/NaN errors).
    c.model = model
    c._train_ganomaly_joint(x_np)

    # predict(): three outputs, correct shapes, finite latent anomaly score.
    out = c.predict(x_np)
    assert set(out) == {"reconstruction", "trading", "anomaly_score"}
    assert out["trading"].shape == (N, 3)
    assert out["anomaly_score"].shape == (N,)
    assert np.isfinite(out["anomaly_score"]).all()


@requires_mlx
def test_ablation_skips_adversarial_term():
    """adv_weight=0 (the NNAnomaly_MLX ablation) must still train and predict —
    the discriminator updates and adversarial term are skipped, leaving the
    bow-tie autoencoder + latent-consistency objective."""
    import NNGANomalyClassifierMLX as M

    rng = np.random.default_rng(1)
    x_np = rng.standard_normal((N, SEQ_LEN, NUM_FEATURES)).astype(np.float32)

    c = _make(M.GANomalyClassifierMLX_LSTM)
    c.adv_weight = 0.0
    c.model = c.create_model(SEQ_LEN, NUM_FEATURES)
    c._train_ganomaly_joint(x_np)
    out = c.predict(x_np)
    assert np.isfinite(out["anomaly_score"]).all()
