"""Warm-start transfer gate — blocking for Arm A (T008, T009).

A warm-start that loads nothing is behaviourally identical to Arm B, and would be reported
as a warm-start result. These assertions exist so that cannot happen silently.
"""
from __future__ import annotations

import pathlib
import sys

import mlx.core as mx
import numpy as np
import pytest
from mlx.utils import tree_flatten

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[3]))

from research.labelfree import warmstart as W                           # noqa: E402
from research.labelfree.policy import ClassifierEncoderPolicy           # noqa: E402

NF, FILT = 24, 32


def _classifier_like_weights(seed=7):
    """Weights shaped exactly like the real classifier's encoder, plus its decode path."""
    mx.random.seed(seed)
    p = ClassifierEncoderPolicy(n_features=NF, num_filters=FILT)
    w = {k: v for k, v in tree_flatten(p.parameters())
         if k.split(".")[0] in W.ENCODER_PREFIXES}
    w["bottleneck.weight"] = mx.zeros((FILT, FILT))   # decode path — must NOT transfer
    w["out.weight"] = mx.zeros((3, FILT))
    return w


def test_transfer_moves_encoder_and_only_encoder():
    p = ClassifierEncoderPolicy(n_features=NF, num_filters=FILT)
    moved = W.load_encoder(p, _classifier_like_weights())
    assert moved, "nothing transferred"
    assert all(m.split(".")[0] in W.ENCODER_PREFIXES for m in moved)
    assert not any(m.split(".")[0] in W.EXCLUDED_PREFIXES for m in moved), \
        "the classification head was inherited — only the representation may transfer"


def test_loaded_weights_differ_from_random_init():
    """The check that makes a silent no-op load impossible."""
    mx.random.seed(123)
    p = ClassifierEncoderPolicy(n_features=NF, num_filters=FILT)
    assert not W.differs_from_random(p, NF, seed=123), "fixture is not a true control"
    W.load_encoder(p, _classifier_like_weights(seed=99))
    assert W.differs_from_random(p, NF, seed=123), \
        "post-load weights match random init — the warm-start did nothing"


def test_shape_mismatch_raises_rather_than_partially_loading():
    p = ClassifierEncoderPolicy(n_features=NF, num_filters=FILT)
    bad = _classifier_like_weights()
    bad["lstm.Wx"] = mx.zeros((999, 7))
    with pytest.raises(ValueError, match="shape"):
        W.load_encoder(p, bad)


def test_empty_source_raises():
    p = ClassifierEncoderPolicy(n_features=NF, num_filters=FILT)
    with pytest.raises(ValueError, match="no encoder tensors"):
        W.load_encoder(p, {"bottleneck.weight": mx.zeros((FILT, FILT))})


def test_learning_rate_rule_is_enforced_not_merely_configured():
    W.check_learning_rates(1e-4, 1e-3)                    # 10x apart — fine
    with pytest.raises(ValueError, match="encoder_lr"):
        W.check_learning_rates(1e-3, 1e-3)                # equal — rejected
    with pytest.raises(ValueError, match="encoder_lr"):
        W.check_learning_rates(5e-4, 1e-3)                # only 2x — rejected
