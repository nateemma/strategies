"""TensorFlow port of NNMT's default model (NNMT/NNMTClassifierTF.py, Predictors/TFPortClassifierMultiTask.py):
same outputs as the MLX model from the same weights, the MLX loop's constants, train / save / reload /
retrain-by-seed, and the strategy's classifier overrides reach it. MLX comparisons skip where MLX is absent."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

STRAT = Path(__file__).resolve().parents[2]
for p in (STRAT, STRAT / "NNMT", STRAT / "NNNC"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from NNMT_LR_RI_TF import NNMT_LR_RI_TF  # noqa: E402  (first: NNMTStrategy's Keras import sets GPU memory growth, which TF refuses once ops have run)
from NNMT.NNMTClassifierTF import NNMTClassifierTF_LSTM_Raw  # noqa: E402
from Predictors import TFPortClassifierMultiTask as TFMT  # noqa: E402

try:
    import mlx.core as mx
except ModuleNotFoundError:
    mx = None
needs_mlx = pytest.mark.skipif(mx is None, reason="MLX not installed")


@needs_mlx
def test_task_constants_match_mlx():
    from Predictors import MLXClassifierMultiTask as M

    assert TFMT.TASK_NAMES == M.TASK_NAMES and TFMT.TASK_WEIGHTS == M.TASK_WEIGHTS


@needs_mlx
@pytest.mark.parametrize("F", [8, 32])
def test_forward_matches_mlx_on_the_same_weights(tmp_path, F):
    from NNMT.NNMTClassifierMLX import NNMTClassifierMLX_LSTM_Raw

    mm = NNMTClassifierMLX_LSTM_Raw("SOL/USDT", 16, F).create_model(16, F)
    mm.eval()
    mm.save_weights(str(tmp_path / "w.safetensors"))
    tm = NNMTClassifierTF_LSTM_Raw("SOL/USDT", 16, F).create_model(16, F)
    NNMTClassifierTF_LSTM_Raw.load_mlx_weights(tm, tmp_path / "w.safetensors")
    X = np.random.default_rng(F).normal(size=(256, 16, F)).astype(np.float32)
    a, b = mm(mx.array(X)), tm(X, training=False)
    assert set(a) == set(b) == set(TFMT.TASK_NAMES)
    assert max(np.abs(np.array(a[t]) - np.asarray(b[t])).max() for t in a) < 1e-5


def test_strategy_overrides_reach_the_classifier():
    s = NNMT_LR_RI_TF.__new__(NNMT_LR_RI_TF)
    s._CLASSIFIER_TASK_WEIGHTS, s._CLASSIFIER_MAX_EPOCHS = {"trading": 1.0, "risk": 1.0}, 7
    clf = s.get_classifier(s.classifier_type, "SOL/USDT", 16, 8)
    assert isinstance(clf, NNMTClassifierTF_LSTM_Raw)
    assert clf.max_epochs == 7 and clf._task_weights()["trading"] == 0.5


def _toy(n=1100, F=5):
    rng = np.random.default_rng(0)
    X = rng.normal(size=(n, 16, F)).astype(np.float32)
    ys = {}
    for i, t in enumerate(TFMT.TASK_NAMES):
        v = X[:, -1, i % F]
        ys[t] = np.eye(3, dtype=np.float32)[(v > -0.4).astype(int) + (v > 0.4).astype(int)]
    ys["profit"][:50] = 0.0  # unlabelled rows: left out of that head's metrics
    return X, ys


def _train(tmp_path, tag):
    import keras

    keras.utils.set_random_seed(11)
    X, ys = _toy()
    c = NNMTClassifierTF_LSTM_Raw("SOL/USDT", 16, X.shape[-1])
    c.max_epochs = 3
    c.set_model_path(str(tmp_path / tag / f"{tag}.keras"))
    c.train(X[:900], X[900:], {t: y[:900] for t, y in ys.items()}, {t: y[900:] for t, y in ys.items()}, force_train=True)
    return c, X


def test_trains_saves_reloads_and_retrains_identically_by_seed(tmp_path):
    a, X = _train(tmp_path, "a")
    pa = a.predict(X[900:])
    assert set(pa) == set(TFMT.TASK_NAMES) and all(v.shape == (200, 3) for v in pa.values())
    r = NNMTClassifierTF_LSTM_Raw("SOL/USDT", 16, X.shape[-1])
    r.set_model_path(a.model_path)
    pr = r.predict(X[900:])
    assert all(np.array_equal(pr[t], pa[t]) for t in pa)
    b, _ = _train(tmp_path, "b")
    pb = b.predict(X[900:])
    assert all(np.array_equal(pb[t], pa[t]) for t in pa)
