"""TensorFlow port of NNNC's default model (NNNC/NNNClassifierTF.py, Predictors/TFPort*).

The port must (1) compute what the MLX model computes from the same weights, (2) train with the MLX loop's
optimizer, loss and metrics, (3) import and key identically on a host without MLX, and (4) train, save,
reload and retrain-by-seed like any classifier here. Tests comparing against MLX skip where it is absent.
"""
from __future__ import annotations

import subprocess
import sys
import textwrap
from pathlib import Path

import numpy as np
import pytest

STRAT = Path(__file__).resolve().parents[2]
for p in (STRAT, STRAT / "NNNC", STRAT / "NNMT"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import tensorflow as tf  # noqa: E402

from NNNClassifierTF import NNNClassifierTF_LSTM_Raw  # noqa: E402
from Predictors.TFPortBasePredictor import MLXAdam, clip_by_global_norm, focal_loss  # noqa: E402
from utils import CustomMetricNP  # noqa: E402

try:
    import mlx.core as mx
    import mlx.nn as mnn
    import mlx.optimizers as moptim
except ModuleNotFoundError:
    mx = None
needs_mlx = pytest.mark.skipif(mx is None, reason="MLX not installed")
CONFIG = {"exchange": {"pair_whitelist": ["SOL/USDT"]}, "timeframe": "15m"}


def _clf(F=24):
    c = NNNClassifierTF_LSTM_Raw("SOL/USDT", 16, F)
    c.num_classes = 3
    return c


@needs_mlx
@pytest.mark.parametrize("F", [8, 24, 32])
def test_forward_matches_mlx_on_the_same_weights(tmp_path, F):
    from NNNClassifierMLX import NNNClassifierMLX_LSTM_Raw

    m = NNNClassifierMLX_LSTM_Raw("SOL/USDT", 16, F)
    m.num_classes = 3
    mm = m.create_model(16, F)
    mm.eval()
    mm.save_weights(str(tmp_path / "w.safetensors"))
    tm = _clf(F).create_model(16, F)
    NNNClassifierTF_LSTM_Raw.load_mlx_weights(tm, tmp_path / "w.safetensors")
    X = np.random.default_rng(F).normal(size=(256, 16, F)).astype(np.float32)
    assert np.abs(np.array(mm(mx.array(X))) - np.asarray(tm(X, training=False))).max() < 1e-5
    assert tm.count_params() == sum(v.size for k, v in __import__("mlx.utils").utils.tree_flatten(mm.parameters())
                                    if k.split(".")[0] in ("lstm", "ln_lstm", "bottleneck", "out"))


@needs_mlx
@pytest.mark.parametrize("wd", [0.0, 0.01])
def test_optimizer_and_clip_match_mlx(wd):
    from Predictors.MLXClassifierNary import _clip_grads_by_global_norm

    rng = np.random.default_rng(0)
    p0, g = rng.normal(size=(5, 3)).astype(np.float32), rng.normal(size=(5, 3)).astype(np.float32) * 3
    lin = mnn.Linear(3, 5, bias=False)
    lin.weight = mx.array(p0)
    opt = moptim.AdamW(learning_rate=1e-3, weight_decay=wd) if wd else moptim.Adam(learning_rate=1e-3)
    var = tf.Variable(p0)
    tfo = MLXAdam([var], 1e-3, wd)
    for _ in range(3):
        cg, n_mlx = _clip_grads_by_global_norm({"weight": mx.array(g)}, 1.0)
        opt.update(lin, cg)
        mx.eval(lin.parameters())
        tg, n_tf = clip_by_global_norm([tf.constant(g)], 1.0)
        tfo.apply(tg)
        assert abs(float(n_mlx.item()) - float(n_tf)) < 1e-4
    assert np.abs(np.array(lin.weight) - var.numpy()).max() < 1e-6


@needs_mlx
def test_focal_loss_and_metrics_match_mlx():
    from utils.CustomLossMLX import multi_class_focal_loss_mlx
    from utils.CustomMetricMLX import compute_val_metrics

    rng = np.random.default_rng(1)
    y = np.eye(3, dtype=np.float32)[rng.integers(0, 3, 500)]
    p = rng.dirichlet([1, 1, 1], 500).astype(np.float32)
    w = rng.uniform(0.5, 1.5, 500).astype(np.float32)
    for gamma, alpha in ((0.0, [1 / 3] * 3), (1.0, [0.6, 0.1, 0.3])):
        a = float(multi_class_focal_loss_mlx(gamma, alpha)(mx.array(y), mx.array(p), mx.array(w)).item())
        b = float(focal_loss(gamma, alpha)(tf.constant(y), tf.constant(p), tf.constant(w)))
        assert abs(a - b) < 1e-5 * max(1.0, abs(a))
    ref, got = compute_val_metrics(mx.array(y), mx.array(p)), CustomMetricNP.compute_val_metrics(y, p)
    assert ref.keys() == got.keys() and all(abs(ref[k] - got[k]) < 1e-6 for k in ref)


def test_twin_keys_and_imports_are_the_same_without_mlx():
    """A host without MLX imports the twins and resolves the same model keys (so converted weights load there)."""
    code = textwrap.dedent(f"""
        import importlib, importlib.abc, json, sys
        class Block(importlib.abc.MetaPathFinder):
            def find_spec(self, name, path, target=None):
                if name == "mlx" or name.startswith("mlx."):
                    raise ModuleNotFoundError(f"No module named {{name!r}}", name=name)
        if sys.argv[1] == "block":
            sys.meta_path.insert(0, Block())
        for p in ({str(STRAT)!r}, {str(STRAT / 'NNNC')!r}, {str(STRAT / 'NNMT')!r}):
            sys.path.insert(0, p)
        from Framework import ModelKey as MK
        out = {{}}
        for n in ("NNNC_CS_ARO_TF", "NNNC_A_Raw_Open_TF", "NNMT_LR_RI_TF"):
            c = getattr(importlib.import_module(n), n)
            out[n] = MK.key_hash(MK.resolve(c, {CONFIG!r}))
        print(json.dumps(out))
    """)
    run = lambda mode: subprocess.run([sys.executable, "-c", code, mode], capture_output=True, text=True,  # noqa: E731
                                      cwd=STRAT, timeout=600)
    blocked, plain = run("block"), run("plain")
    assert blocked.returncode == 0, blocked.stderr[-2000:]
    assert plain.returncode == 0, plain.stderr[-2000:]
    keys = [__import__("json").loads(r.stdout.strip().splitlines()[-1]) for r in (blocked, plain)]
    assert keys[0] == keys[1]
    assert keys[0]["NNNC_CS_ARO_TF"] == keys[0]["NNNC_A_Raw_Open_TF"]  # the sizer is post-model


def test_twin_differs_from_mlx_model_only_in_framework():
    from Framework import ModelKey as MK
    from NNNC_A_Raw_Open import NNNC_A_Raw_Open
    from NNNC_A_Raw_Open_TF import NNNC_A_Raw_Open_TF

    a, b = MK.resolve(NNNC_A_Raw_Open, CONFIG), MK.resolve(NNNC_A_Raw_Open_TF, CONFIG)
    assert MK.key_hash(a) != MK.key_hash(b)
    assert {k for k in a["attributes"] if a["attributes"][k] != b["attributes"].get(k)} == {"classifier_type", "tensor_method"}
    assert {k for k in a["methods"] if a["methods"][k] != b["methods"].get(k)} == {"get_classifier"}
    assert {k: v for k, v in a.items() if k not in ("attributes", "methods")} == \
           {k: v for k, v in b.items() if k not in ("attributes", "methods")}


def _toy(n=1200, F=6, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, 16, F)).astype(np.float32)
    cls = (X[:, -1, 0] > 0.3).astype(int) * 2 + ((X[:, -1, 0] > -0.3) & (X[:, -1, 0] <= 0.3)).astype(int)
    return X, np.eye(3, dtype=np.float32)[cls]


def _train(tmp_path, tag):
    X, y = _toy()
    c = _clf(X.shape[-1])
    c.max_epochs, c.early_patience, c.train_seed = 4, 2, 7
    c.set_model_path(str(tmp_path / tag / f"{tag}.keras"))
    c.train(X[:1000], X[1000:], y[:1000], y[1000:], force_train=True)
    return c, X


def test_trains_saves_reloads_and_retrains_identically_by_seed(tmp_path):
    a, X = _train(tmp_path, "a")
    assert Path(a.model_path).is_file() and a.epochs_run >= 1
    pa = a.predict(X[1000:])
    assert pa.shape == (200, 3) and np.allclose(pa.sum(axis=1), 1.0, atol=1e-5)
    r = _clf(X.shape[-1])
    r.set_model_path(a.model_path)
    assert np.array_equal(r.predict(X[1000:]), pa)  # reload = same predictions
    b, _ = _train(tmp_path, "b")
    assert np.array_equal(b.predict(X[1000:]), pa)  # same seed = same weights
