# pragma pylint: disable=C0103, C0114, C0115, C0116, C0301
"""TensorFlow port of the MLX classifier machinery (MLXBasePredictor + the MLX training loop's pieces).

The MLX models (NNNClassifierMLX, NNMTClassifierMLX) train with their own hand-written loop, which
differs from the Keras classifiers' ``model.fit`` path (Adam WITHOUT bias correction, global-norm
clipping, non-finite batch skipping, MCC-monitored checkpointing). The ``TFPort*`` classes re-implement
that loop in TensorFlow so a TensorFlow twin trains the way the MLX model does, on hosts without MLX.
This module holds what both ports share; it imports neither MLX nor any strategy code.

Weights are saved as a whole Keras model (``.keras``), at the path the model key hands the classifier.
Seeding follows MLXBasePredictor: ``MLX_SEED`` (default 42) seeds python, numpy and TensorFlow at import
(evaluation/run_folds.py sets it per seed), and op determinism is on, so a seed retrains identically.
"""

from __future__ import annotations

import logging
import os
import sys
import traceback
from pathlib import Path
from typing import Optional

import numpy as np

sys.path.append(str(Path(__file__).parent))
sys.path.append(str(Path(__file__).parent.parent))

import keras  # noqa: E402
import tensorflow as tf  # noqa: E402

from Predictors.BasePredictor import BasePredictor  # noqa: E402

log = logging.getLogger(__name__)

SEED = int(os.environ.get("MLX_SEED", "42"))
keras.utils.set_random_seed(SEED)
tf.config.experimental.enable_op_determinism()


def uniform_init(scale: float):
    """MLX's initialiser for nn.Linear / nn.LSTM: U(-scale, scale)."""
    return keras.initializers.RandomUniform(-scale, scale)


# --- MLX weights -> Keras layers (for converting trained MLX models; numpy only, no MLX needed) ---

_ST_DTYPES = {"F32": np.float32, "F16": np.float16, "F64": np.float64, "I32": np.int32, "I64": np.int64}


def read_safetensors(path) -> dict:
    """{name: numpy array} from a .safetensors file (the format MLX's save_weights writes)."""
    import json
    import struct

    raw = Path(path).read_bytes()
    (n,) = struct.unpack("<Q", raw[:8])
    header = json.loads(raw[8:8 + n])
    body = raw[8 + n:]
    out = {}
    for name, h in header.items():
        if name == "__metadata__":
            continue
        if h["dtype"] not in _ST_DTYPES:
            raise ValueError(f"{path}: {name} has dtype {h['dtype']}, which this reader does not handle")
        a, b = h["data_offsets"]
        out[name] = np.frombuffer(body[a:b], dtype=_ST_DTYPES[h["dtype"]]).reshape(h["shape"]).astype(np.float32)
    return out


def set_mlx_lstm(layer, w: dict, prefix: str) -> None:
    """mlx.nn.LSTM (Wx (4H, D), Wh (4H, H), bias (4H,); gates i, f, g, o) -> keras LSTM (kernel (D, 4H),
    recurrent (H, 4H), bias (4H,); gates i, f, c, o: the same order)."""
    layer.set_weights([w[f"{prefix}.Wx"].T, w[f"{prefix}.Wh"].T, w[f"{prefix}.bias"]])


def set_mlx_linear(layer, w: dict, prefix: str) -> None:
    """mlx.nn.Linear (weight (out, in), bias) -> keras Dense (kernel (in, out), bias)."""
    layer.set_weights([w[f"{prefix}.weight"].T, w[f"{prefix}.bias"]])


def set_mlx_layernorm(layer, w: dict, prefix: str) -> None:
    """mlx.nn.LayerNorm (weight, bias) -> keras LayerNormalization (gamma, beta)."""
    layer.set_weights([w[f"{prefix}.weight"], w[f"{prefix}.bias"]])


def filter_nonfinite_rows(tensor: np.ndarray, labels: dict, label: str = "training"):
    """Drop rows where the input or any label array has NaN/Inf (as the MLX loops' _filter_nonfinite_rows).
    ``labels`` is {name: array}; single-task callers pass one entry."""
    n = len(tensor)
    mask = np.isfinite(tensor).all(axis=tuple(range(1, tensor.ndim))) if tensor.ndim > 1 else np.isfinite(tensor)
    for arr in labels.values():
        mask &= np.isfinite(arr).all(axis=tuple(range(1, arr.ndim))) if arr.ndim > 1 else np.isfinite(arr)
    kept = int(mask.sum())
    if kept == n:
        return tensor, labels
    print(f"    Filtered {n - kept}/{n} ({100.0 * (n - kept) / n:.2f}%) non-finite rows from {label} data")
    if kept == 0:
        raise RuntimeError(f"All {label} rows contained NaN/Inf — nothing to train on.")
    return tensor[mask], {k: v[mask] for k, v in labels.items()}


def focal_loss(gamma: float, alpha_vector):
    """utils.CustomLossMLX.multi_class_focal_loss_mlx in TensorFlow (same clip, same x10 scale)."""
    alpha = tf.constant(np.asarray(alpha_vector, dtype=np.float32))

    def fn(y_true, y_pred, sample_weights=None):
        y_pred = tf.clip_by_value(y_pred, 1e-7, 1.0 - 1e-7)
        ce = -y_true * tf.math.log(y_pred)
        p_t = y_true * y_pred + (1.0 - y_true) * (1.0 - y_pred)
        per = tf.reduce_sum(tf.pow(1.0 - p_t, gamma) * (y_true * alpha) * ce, axis=-1)
        if sample_weights is not None:
            per = per * sample_weights
        return 10.0 * tf.reduce_mean(per)

    return fn


class MLXAdam:
    """mlx.optimizers.Adam / AdamW as MLX applies them: bias_correction False, eps 1e-8, AdamW's decay
    decoupled (p *= 1 - lr * wd before the Adam step). Keras's Adam always bias-corrects."""

    def __init__(self, variables, learning_rate: float, weight_decay: float = 0.0,
                 betas=(0.9, 0.999), eps: float = 1e-8):
        self.vars = list(variables)
        self.lr = tf.Variable(float(learning_rate), trainable=False, dtype=tf.float32)
        self.wd, (self.b1, self.b2), self.eps = float(weight_decay), betas, eps
        self.m = [tf.Variable(tf.zeros_like(v), trainable=False) for v in self.vars]
        self.v = [tf.Variable(tf.zeros_like(v), trainable=False) for v in self.vars]

    @property
    def learning_rate(self) -> float:
        return float(self.lr.numpy())

    @learning_rate.setter
    def learning_rate(self, value: float) -> None:
        self.lr.assign(float(value))

    @tf.function(reduce_retracing=True)
    def apply(self, grads):
        for p, g, m, v in zip(self.vars, grads, self.m, self.v):
            if self.wd > 0:
                p.assign(p * (1.0 - self.lr * self.wd))
            m.assign(self.b1 * m + (1.0 - self.b1) * g)
            v.assign(self.b2 * v + (1.0 - self.b2) * tf.square(g))
            p.assign_sub(self.lr * m / (tf.sqrt(v) + self.eps))


def clip_by_global_norm(grads, max_norm: float):
    """The MLX loops' _clip_grads_by_global_norm: scale = min(max_norm / (norm + 1e-6), 1)."""
    norm = tf.sqrt(tf.add_n([tf.reduce_sum(tf.square(g)) for g in grads]))
    coef = tf.minimum(max_norm / (norm + 1e-6), 1.0)
    return [g * coef for g in grads], norm


def batch_indices(n: int, batch_size: int, shuffle: bool = True):
    """The MLX loops' batching: one numpy permutation per epoch, consecutive slices."""
    idx = np.random.permutation(n) if shuffle else np.arange(n)
    for start in range(0, n, batch_size):
        yield idx[start:start + batch_size]


class TFPortBasePredictor(BasePredictor):
    """MLXBasePredictor's lifecycle (paths, save/load, state queries) for a Keras model."""

    model: Optional[keras.Model] = None
    is_trained: bool = False
    model_path: str = ""
    model_ext: str = "keras"
    name: str = ""
    root_dir: str = ""
    checkpoint_path: str = "/tmp/tf_port_model.keras"

    seq_len: int = 8
    num_features: int = 64
    batch_size: int = 256

    default_max_epochs: int = 256
    num_epochs: int = default_max_epochs
    default_learning_rate: float = 0.0001
    learning_rate: float = default_learning_rate

    clean_data_required: bool = False
    model_per_pair: bool = False
    new_model: bool = False

    requires_dataframes: bool = False
    prescale_dataframe: bool = True
    single_prediction: bool = False
    combine_models: bool = False

    dataframeUtils = None

    def __init__(self, pair: str, seq_len: int, num_features: int, tag: str = ""):
        super().__init__()
        self.loaded_from_file = False
        self.seq_len = seq_len
        self.num_features = num_features
        if self.dataframeUtils is None:
            from utils.DataframeUtils import DataframeUtils

            self.dataframeUtils = DataframeUtils()
        pair_suffix = ("_" + pair.split("/")[0]) if self.model_per_pair else ""
        tag_suffix = ("_" + tag) if tag else ""
        self.category = self.__class__.__name__
        self.name = self.category + pair_suffix + tag_suffix

    # --- paths ---
    def set_model_path(self, path: str):
        if path.endswith(".safetensors"):
            path = path[: -len(".safetensors")] + ".keras"
        elif path.endswith(".h5"):
            path = path[:-3] + ".keras"
        self.model_path = path
        self.name = path.split("/")[-1].split(".")[0]
        self.root_dir = os.path.dirname(path)
        os.makedirs(self.root_dir, exist_ok=True)
        self.checkpoint_path = self.get_checkpoint_path()

    def get_model_root_dir(self) -> str:
        return self.root_dir

    def get_model_path(self) -> str:
        return self.model_path

    def get_checkpoint_path(self) -> str:
        d = f"/tmp/{self.name}/"
        os.makedirs(d, exist_ok=True)
        return os.path.join(d, "checkpoint.weights.h5")

    # --- setters ---
    def set_combine_models(self, flag: bool):
        self.combine_models = flag

    def set_target_column(self, target_column: str):
        pass

    def set_num_epochs(self, num_epochs: int | None = None):
        self.num_epochs = num_epochs if num_epochs is not None else self.default_max_epochs

    def set_learning_rate(self, rate: float | None = None):
        self.learning_rate = max(rate, 1e-5) if rate is not None else self.default_learning_rate

    def set_batch_size(self, batch_size: int):
        self.batch_size = batch_size

    def create_model(self, seq_len: int, num_features: int) -> Optional[keras.Model]:
        log.warning("create_model() should be defined by the subclass")
        return None

    # --- save / load ---
    def save(self, path: str = ""):
        if path:
            self.model_path = path
        if self.model is None:
            log.warning("save() called but model is None")
            return
        os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
        print(f"    Saving TensorFlow model to: {self.model_path}")
        self.model.save(self.model_path)

    def load(self, path: str = "") -> keras.Model | None:
        if path:
            self.model_path = path
        if not os.path.exists(self.model_path):
            print(f"    TensorFlow model not found ({self.model_path})...")
            self.new_model = True
            self.is_trained = False
            return None
        print(f"    Loading existing TensorFlow model ({self.model_path})...")
        try:
            model = keras.models.load_model(self.model_path, compile=False)
        except Exception as e:
            print(f"    Traceback: {traceback.format_exc()}")
            raise RuntimeError(f"Failed to load TensorFlow model from {self.model_path}. Error: {e}") from e
        self.is_trained, self.new_model, self.loaded_from_file = True, False, True
        self.model = model
        return model

    def update_model_weights(self):
        cp = self.get_checkpoint_path()
        if os.path.exists(cp) and self.model is not None:
            self.model.load_weights(cp)

    # --- state ---
    def model_exists(self) -> bool:
        return os.path.exists(self.model_path)

    def model_is_trained(self) -> bool:
        return bool(self.model is not None and self.is_trained and not (self.combine_models and self.new_model))

    def new_model_created(self) -> bool:
        return TFPortBasePredictor.new_model

    def nearest_power_of_2(self, n: int) -> int:
        return 2 ** n.bit_length()

    def backtest(self, data):
        return self.predict(data)

    # --- shared by the ports ---
    def _tensor(self, data) -> np.ndarray:
        """A DataFrame -> (N, seq_len, F) float32 tensor; an array (numpy or MLX) -> float32 numpy."""
        if self.dataframeUtils.is_dataframe(data):
            data = self.dataframeUtils.df_to_tensor(data, self.seq_len, method=3)
        return np.asarray(data, dtype=np.float32)

    def _train_loop(self, X, labels: dict, weights, loss_fn, validate, *, monitor_key: str,
                    grad_clip_norm: float, max_epochs: int, early_patience: int, weight_decay: float = 0.0):
        """The MLX loops' epoch loop: shuffled batches, global-norm clip, non-finite batches skipped
        (50 in a row abort to the best checkpoint), then per epoch `validate()` -> (metrics, val_loss,
        log line); save-best on `monitor_key` (max), ReduceLROnPlateau x0.1 after 8 flat epochs (floor
        1e-6), EarlyStopping after `early_patience`. Restores the best weights. `loss_fn(preds, y, w)`
        maps the model output and the batch's labels dict to a scalar. Returns (epochs_run, best_epoch)."""
        plateau_patience, plateau_factor, max_nan = 8, 0.1, 50
        variables = self.model.trainable_variables
        opt = MLXAdam(variables, self.learning_rate, weight_decay)
        X_tf = tf.constant(X)
        y_tf = {k: tf.constant(v) for k, v in labels.items()}
        w_tf = tf.constant(weights) if weights is not None else None

        @tf.function(reduce_retracing=True)
        def grad_step(idx):
            xb = tf.gather(X_tf, idx)
            yb = {k: tf.gather(v, idx) for k, v in y_tf.items()}
            wb = tf.gather(w_tf, idx) if w_tf is not None else None
            with tf.GradientTape() as tape:
                loss = loss_fn(self.model(xb, training=True), yb, wb)
            grads = tape.gradient(loss, variables)
            grads = [g if g is not None else tf.zeros_like(v) for g, v in zip(grads, variables)]
            clipped, norm = clip_by_global_norm(grads, grad_clip_norm)
            return loss, clipped, norm

        checkpoint = self.get_checkpoint_path()
        best, best_epoch, no_improve, plateau, nan_run, aborted = -np.inf, 0, 0, 0, 0, False
        epoch = -1
        print(f"    batch_size: {self.batch_size}  max_epochs: {max_epochs}")
        print(f"    monitor: {monitor_key} (max)  grad clip (global L2 norm): {grad_clip_norm}")
        for epoch in range(max_epochs):
            losses, clips = [], 0
            for idx in batch_indices(len(X), self.batch_size):
                loss, grads, norm = grad_step(tf.constant(idx, dtype=tf.int32))
                loss_v, norm_v = float(loss), float(norm)
                if not (np.isfinite(loss_v) and np.isfinite(norm_v)):
                    nan_run += 1
                    if nan_run <= 3:
                        print(f"    WARNING: non-finite update at epoch {epoch + 1} (loss={loss_v}, "
                              f"grad_norm={norm_v}; consecutive: {nan_run}); skipping")
                    if nan_run >= max_nan:
                        print(f"    ABORT: {max_nan} consecutive non-finite batches. Falling back to best checkpoint.")
                        aborted = True
                        break
                    continue
                nan_run = 0
                clips += norm_v > grad_clip_norm
                opt.apply(grads)
                losses.append(loss_v)
            if aborted:
                break
            metrics, val_loss, line = validate()
            train_loss = float(np.mean(losses)) if losses else float("nan")
            print(f"    Epoch {epoch + 1:3d}/{max_epochs} — loss: {train_loss:.4f}  val_loss: {val_loss:.4f}  "
                  f"{line}{f'  clips:{clips}' if clips else ''}")
            if metrics[monitor_key] > best:
                best, best_epoch, no_improve, plateau = metrics[monitor_key], epoch, 0, 0
                self.model.save_weights(checkpoint)
                print(f"    ✓ Best model saved  ({monitor_key}: {best:.4f})")
            else:
                no_improve += 1
                plateau += 1
            if plateau >= plateau_patience:
                new_lr = max(opt.learning_rate * plateau_factor, 1e-6)
                if new_lr < opt.learning_rate:
                    opt.learning_rate = new_lr
                    print(f"    ReduceLROnPlateau → lr={new_lr:.2e}")
                plateau = 0
            if no_improve >= early_patience:
                print(f"    EarlyStopping at epoch {epoch + 1} (no improvement for {early_patience} epochs)")
                break
        if os.path.exists(checkpoint):
            print(f"    Restoring best weights from {checkpoint}")
            self.model.load_weights(checkpoint)
        return (epoch if aborted else epoch + 1), best_epoch

    def _forward(self, X: np.ndarray, batch: int = 65536):
        """Inference in chunks (training=False: dropout off), numpy out; a dict for multi-output models."""
        outs = [self.model(tf.constant(X[i:i + batch]), training=False) for i in range(0, len(X), batch)] or \
               [self.model(tf.constant(X), training=False)]
        if isinstance(outs[0], dict):
            return {k: np.concatenate([np.asarray(o[k]) for o in outs]) for k in outs[0]}
        return np.concatenate([np.asarray(o) for o in outs])
