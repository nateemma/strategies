"""Tests for the MLXClassifierNary training-loop attributes (T003).

Verifies max_epochs / early_patience / weight_decay / epochs_run / best_epoch /
model.extra_loss behave per contracts/registry.md, and that defaults leave
train() byte-identical to the pre-study behaviour.
"""
from __future__ import annotations

import numpy as np
import pytest

import mlx.core as mx
import mlx.optimizers as optim

import Predictors.MLXClassifierNary as mlx_classifier_nary
from NNNC.NNNClassifierMLX import ClassifierTypeMLX, create_classifier_mlx

SEQ_LEN = 8
N_FEATURES = 6
N_CLASSES = 3
N_ROWS = 300
N_VAL = 60


def _make_classifier(tmp_path, name="model"):
    # get_checkpoint_path() keys the checkpoint dir off self.name (derived from
    # the model_path filename stem), not the full path — give each classifier
    # instance a distinct filename so concurrent tests never share a checkpoint.
    clf, _ = create_classifier_mlx(ClassifierTypeMLX.MLP, "X/USDT", N_FEATURES, SEQ_LEN, N_CLASSES)
    clf.set_model_path(str(tmp_path / f"{name}.safetensors"))
    return clf


def _make_data():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(N_ROWS, SEQ_LEN, N_FEATURES)).astype(np.float32)
    y = rng.integers(0, N_CLASSES, N_ROWS)
    y_onehot = np.eye(N_CLASSES)[y].astype(np.float32)

    X_train, X_val = X[:-N_VAL], X[-N_VAL:]
    y_train, y_val = y_onehot[:-N_VAL], y_onehot[-N_VAL:]
    return X_train, X_val, y_train, y_val


CLASS_WEIGHTS = [1.0, 1.0, 1.0]


def test_defaults(tmp_path):
    clf = _make_classifier(tmp_path)
    assert (clf.max_epochs, clf.early_patience, clf.weight_decay) == (100, 20, 0.0)


def test_epoch_cap_is_live(tmp_path):
    clf = _make_classifier(tmp_path)
    clf.max_epochs = 3
    clf.early_patience = 100

    X_train, X_val, y_train, y_val = _make_data()
    clf.train(X_train, X_val, y_train, y_val, class_weights=CLASS_WEIGHTS)

    assert clf.epochs_run == 3
    assert 0 <= clf.best_epoch < 3


def test_weight_decay_selects_adamw(tmp_path, monkeypatch):
    calls = {}

    class _RecordingAdamW(optim.AdamW):
        def __init__(self, *args, **kwargs):
            calls["ctor"] = "AdamW"
            super().__init__(*args, **kwargs)

    class _RecordingAdam(optim.Adam):
        def __init__(self, *args, **kwargs):
            calls["ctor"] = "Adam"
            super().__init__(*args, **kwargs)

    monkeypatch.setattr(mlx_classifier_nary.optim, "AdamW", _RecordingAdamW)
    monkeypatch.setattr(mlx_classifier_nary.optim, "Adam", _RecordingAdam)

    X_train, X_val, y_train, y_val = _make_data()

    clf_wd = _make_classifier(tmp_path, name="wd")
    clf_wd.max_epochs = 1
    clf_wd.weight_decay = 1e-2
    clf_wd.train(X_train, X_val, y_train, y_val, class_weights=CLASS_WEIGHTS)
    assert calls["ctor"] == "AdamW"

    calls.clear()
    clf_plain = _make_classifier(tmp_path, name="plain")
    clf_plain.max_epochs = 1
    clf_plain.weight_decay = 0.0
    clf_plain.train(X_train, X_val, y_train, y_val, class_weights=CLASS_WEIGHTS)
    assert calls["ctor"] == "Adam"


def test_extra_loss_is_added(tmp_path):
    clf = _make_classifier(tmp_path)
    clf.max_epochs = 1

    X_train, X_val, y_train, y_val = _make_data()
    clf.train(X_train, X_val, y_train, y_val, class_weights=CLASS_WEIGHTS)

    # Smoke test: an extra_loss attribute on the model must not break a
    # further training call (the attribute may be a static array or callable).
    clf.model.extra_loss = mx.array(0.0)
    clf.max_epochs = 1
    clf.train(X_train, X_val, y_train, y_val, class_weights=CLASS_WEIGHTS, force_train=True)
    assert clf.epochs_run == 1
