# MLX Multi-Task Classifier base class.
#
# Mirrors ClassifierKerasMultiTask.py API exactly, but the training loop
# (per-task focal losses, summed with per-task weights, monitored on
# val_trading_mcc with EarlyStopping + ReduceLROnPlateau + ModelCheckpoint)
# is implemented manually in MLX rather than via Keras compile/fit.
#
# Subclasses override create_model() to return an mlx.nn.Module whose
# forward pass accepts (batch, seq_len, num_features) and returns a dict
# of six softmax outputs, one per task:
#   {"trading": ..., "regime": ..., "risk": ...,
#    "momentum": ..., "flow": ..., "profit": ...}

from __future__ import annotations

import os
import logging
import sys
import warnings
from contextlib import contextmanager
from pathlib import Path
from typing import Dict, Optional, Tuple, Union

import numpy as np
import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from mlx.utils import tree_flatten, tree_map

sys.path.append(str(Path(__file__).parent))
sys.path.append(str(Path(__file__).parent.parent))

from ClassifierMLX import ClassifierMLX
from CustomLossMLX import multi_class_focal_loss_mlx
from CustomMetricMLX import compute_val_metrics

log = logging.getLogger(__name__)
warnings.simplefilter(action="ignore", category=FutureWarning)


# Tasks predicted by every multi-task model in fixed order.
# ``trading`` is the primary task; the others act as auxiliary regularisers
# during training to encourage a more general representation.
TASK_NAMES: Tuple[str, ...] = (
    "trading",
    "regime",
    "risk",
    "momentum",
    "flow",
    "profit",
)

# Per-task loss weights — must sum to 1.0 after normalisation.
# These mirror the values in ClassifierKerasMultiTask.compile_model().
_TASK_WEIGHTS_RAW: Dict[str, float] = {
    "trading":  2.0,
    "regime":   1.0,
    "risk":     1.0,
    "momentum": 1.0,
    "flow":     1.0,
    "profit":   2.0,
}
_TASK_WEIGHT_TOTAL = sum(_TASK_WEIGHTS_RAW.values())
TASK_WEIGHTS: Dict[str, float] = {
    name: weight / _TASK_WEIGHT_TOTAL for name, weight in _TASK_WEIGHTS_RAW.items()
}


# -----------------------------------------------------------------------
# Internal helpers
# -----------------------------------------------------------------------


def _filter_nonfinite_rows(
    tensor: np.ndarray,
    labels_dict: Dict[str, np.ndarray],
    label: str = "training",
) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
    """
    Drop rows where the input tensor or any task's labels contain NaN/Inf.

    Pathological rows most often come from WGAN-augmented data where the
    generator's output isn't fully bounded — they cause forward passes to
    produce finite losses but NaN gradients, which corrupts the model.
    Filtering once up front is much cheaper than catching each bad batch
    later in the training loop.
    """
    n = len(tensor)

    # Reduce over every dimension except the leading "samples" axis.
    feature_axes = tuple(range(1, tensor.ndim))
    finite_mask = np.isfinite(tensor).all(axis=feature_axes) if feature_axes \
        else np.isfinite(tensor)

    for task, lbls in labels_dict.items():
        if lbls.ndim == 1:
            finite_mask &= np.isfinite(lbls)
        else:
            label_axes = tuple(range(1, lbls.ndim))
            finite_mask &= np.isfinite(lbls).all(axis=label_axes)

    n_kept = int(finite_mask.sum())
    n_dropped = n - n_kept

    if n_dropped == 0:
        return tensor, labels_dict

    pct = 100.0 * n_dropped / n
    print(
        f"    Filtered {n_dropped}/{n} ({pct:.2f}%) non-finite rows from "
        f"{label} data"
    )

    if n_kept == 0:
        raise RuntimeError(
            f"All {label} rows contained NaN/Inf — nothing to train on. "
            f"Check the upstream data pipeline (most likely the GAN generator "
            f"is producing unbounded outputs)."
        )

    return tensor[finite_mask], {t: lbl[finite_mask] for t, lbl in labels_dict.items()}


def _clip_grads_by_global_norm(grads, max_norm: float):
    """
    Scale every gradient tensor so the global L2 norm is at most ``max_norm``.

    Standard "global-norm" gradient clipping (the same flavour TensorFlow's
    ``clip_by_global_norm`` and PyTorch's ``clip_grad_norm_`` use).  Returns
    the clipped gradient tree and the original (pre-clip) global norm so the
    caller can log spikes.
    """
    flat = tree_flatten(grads)
    total_sq = mx.zeros(())
    for _, g in flat:
        total_sq = total_sq + mx.sum(g * g)
    total_norm = mx.sqrt(total_sq)
    # clip_coef = min(max_norm / (total_norm + eps), 1.0)
    clip_coef = mx.minimum(
        mx.array(max_norm) / (total_norm + mx.array(1e-6)),
        mx.array(1.0),
    )
    clipped = tree_map(lambda g: g * clip_coef, grads)
    return clipped, total_norm


def _batch_iter_multi(
    X: Union[np.ndarray, mx.array],
    y_dict: Dict[str, Union[np.ndarray, mx.array]],
    batch_size: int,
    shuffle: bool = True,
):
    """
    Yield (X_batch, y_batch_dict) mini-batches for multi-task training.
    Same shuffling / batching logic as _batch_iter in ClassifierMLXNary,
    but slices every task's label array with the same indices so all
    six heads see consistent rows.
    """
    n = len(X)
    idx_np = np.random.permutation(n) if shuffle else np.arange(n)

    X_mx = mx.array(X, dtype=mx.float32) if not isinstance(X, mx.array) else X
    y_mx = {
        task: (mx.array(arr, dtype=mx.float32) if not isinstance(arr, mx.array) else arr)
        for task, arr in y_dict.items()
    }

    for start in range(0, n, batch_size):
        b_idx_np = idx_np[start : start + batch_size]
        b_idx = mx.array(b_idx_np)
        yield X_mx[b_idx], {task: arr[b_idx] for task, arr in y_mx.items()}


# -----------------------------------------------------------------------


class ClassifierMLXMultiTask(ClassifierMLX):
    """
    MLX multi-task classifier.  Mirrors ClassifierKerasMultiTask.py:
      - Same constructor + train/predict signatures
      - Same per-task class-weight, focal-loss, and task-weight logic
      - Manual EarlyStopping + ReduceLROnPlateau + ModelCheckpoint
      - predict() returns a dict of per-task softmax probabilities
    """

    clean_data_required: bool = False
    learning_rate: float = 1e-4

    # Internal state for per-task class weighting
    class_weights: Dict[str, list]

    def __init__(self, pair, seq_len, num_features, tag=""):
        super().__init__(pair, seq_len, num_features, tag)
        self.name = self.__class__.__name__
        self.class_weights = {}

    # -----------------------------------------------------------------------
    # Model creation (subclasses override)
    # -----------------------------------------------------------------------

    def create_model(self, seq_len: int, num_features: int) -> Optional[nn.Module]:
        """Return an mlx.nn.Module emitting a dict of six softmax outputs."""
        log.warning("create_model() should be defined by the subclass")
        return None

    # -----------------------------------------------------------------------
    # Per-task loss configuration (mirrors ClassifierKerasMultiTask)
    # -----------------------------------------------------------------------

    def calculate_alpha(self, weights) -> list:
        total = sum(weights) if weights is not None else 0.0
        if total > 0:
            return [w / total for w in weights]
        return [1.0 / 3] * 3

    def calculate_gamma(self, alpha) -> float:
        alpha = np.asarray(alpha)
        min_a, max_a = float(np.min(alpha)), float(np.max(alpha))
        if min_a == 0:
            return 2.0
        ratio = max_a / min_a
        if ratio < 1.2:
            return 0.0
        elif ratio < 2.0:
            return 0.5
        elif ratio < 5.0:
            return 1.0
        else:
            return 1.5

    def _build_task_loss_fns(
        self, class_weights: Optional[Dict[str, list]]
    ) -> Dict[str, "callable"]:
        """Build a per-task focal-loss function with task-specific alpha/gamma.

        ``class_weights[task]`` may arrive as a Python list, a numpy array,
        or be missing entirely depending on the caller.  We normalise via
        an explicit ``is not None`` check rather than truthiness so a
        numpy array doesn't trigger the "ambiguous truth value" error.
        """
        default_alpha = [1.0 / 3, 1.0 / 3, 1.0 / 3]
        loss_fns: Dict[str, callable] = {}
        for task in TASK_NAMES:
            raw = class_weights.get(task) if class_weights else None
            if raw is None:
                raw = default_alpha
            alpha = self.calculate_alpha(raw)
            gamma = self.calculate_gamma(alpha)
            loss_fns[task] = multi_class_focal_loss_mlx(
                gamma=gamma, alpha_vector=alpha
            )
        return loss_fns

    # -----------------------------------------------------------------------
    # Training
    # -----------------------------------------------------------------------

    def train(
        self,
        df_train_norm,
        df_test_norm,
        train_results: Dict[str, np.ndarray],
        test_results: Dict[str, np.ndarray],
        force_train: bool = False,
        class_weights: Optional[Dict[str, list]] = None,
    ):
        """
        Train the model.  Signature identical to ClassifierKerasMultiTask.train().

        train_results / test_results : dict {task: one-hot (N, 3) ndarray}
        class_weights                : optional {task: [w0, w1, w2]}
        """
        if not isinstance(train_results, dict) or not isinstance(test_results, dict):
            raise ValueError(
                "ClassifierMLXMultiTask.train requires train_results and "
                "test_results to be dicts keyed by task name "
                f"(got {type(train_results)} and {type(test_results)})"
            )
        missing = [t for t in TASK_NAMES if t not in train_results]
        if missing:
            raise ValueError(
                f"train_results is missing tasks: {missing} "
                f"(expected {list(TASK_NAMES)})"
            )

        # --- lazy load existing model ---
        if self.model is None:
            self.model = self.load()

        # --- early-exit if already trained ---
        if (
            self.model is not None
            and self.model_is_trained()
            and not force_train
            and not self.new_model_created()
        ):
            return

        # --- convert DataFrames → numpy tensors ---
        if self.dataframeUtils.is_dataframe(df_train_norm):
            df_train = df_train_norm.copy()
            df_test = df_test_norm.copy()
            train_tensor = self.dataframeUtils.df_to_tensor(
                df_train, self.seq_len, method=3
            )
            test_tensor = self.dataframeUtils.df_to_tensor(
                df_test, self.seq_len, method=3
            )
        else:
            train_tensor = np.asarray(df_train_norm)
            test_tensor = np.asarray(df_test_norm)

        # Ensure all label arrays are float32 numpy
        train_labels_np: Dict[str, np.ndarray] = {
            task: np.asarray(train_results[task], dtype=np.float32)
            for task in TASK_NAMES
        }
        test_labels_np: Dict[str, np.ndarray] = {
            task: np.asarray(test_results[task], dtype=np.float32)
            for task in TASK_NAMES
        }

        # Drop rows containing NaN/Inf before training begins.  This is a
        # one-shot pass over the whole tensor, vastly cheaper than catching
        # bad batches in the training loop — and it eliminates the failure
        # mode where pathological rows (typically from GAN-augmented data
        # with unbounded generator outputs) corrupt the training run.
        train_tensor, train_labels_np = _filter_nonfinite_rows(
            train_tensor, train_labels_np, label="training"
        )
        test_tensor, test_labels_np = _filter_nonfinite_rows(
            test_tensor, test_labels_np, label="validation"
        )

        # --- create model if missing ---
        if self.model is None:
            self.num_features = train_tensor.shape[-1]
            self.model = self.create_model(self.seq_len, self.num_features)
            if self.model is None:
                print("    ERR: model not created")
                return

            self.model.input_shape = (None, self.seq_len, self.num_features)

            # Build default per-task class weights from the validation labels
            if class_weights is None:
                class_weights = {
                    task: list(self._calculate_class_weights(test_labels_np[task]))
                    for task in TASK_NAMES
                }
            self.class_weights = class_weights

            total_params = sum(
                p.size for _, p in tree_flatten(self.model.trainable_parameters())
            )
            print(f"    MLX multi-task model created.  Parameters: {total_params:,}")

        # --- build per-task loss functions and global task weights ---
        loss_fns = self._build_task_loss_fns(self.class_weights)
        task_weight_keys = list(TASK_NAMES)
        task_weight_arr = mx.array(
            [TASK_WEIGHTS[t] for t in task_weight_keys], dtype=mx.float32
        )
        print(
            f"    task weights (normalized, sum={float(mx.sum(task_weight_arr)):.3f}): "
            f"{TASK_WEIGHTS}"
        )

        # --- build optimizer ---
        optimizer = optim.Adam(learning_rate=self.learning_rate)

        # --- forward + loss ---
        def forward_loss(model, X, y_dict):
            preds = model(X)  # dict of softmax outputs
            total = mx.zeros(())
            for i, task in enumerate(task_weight_keys):
                total = total + task_weight_arr[i] * loss_fns[task](
                    y_dict[task], preds[task]
                )
            return total

        loss_and_grad = nn.value_and_grad(self.model, forward_loss)

        # --- training hyper-params ---
        max_epochs = 100
        early_patience = 20
        plateau_patience = 8
        plateau_factor = 0.1
        # Gradient clipping global L2 norm.  Without it, an unlucky batch can
        # produce gradients large enough to spike Adam's running averages and
        # cascade the model weights to NaN — exactly what symptom #21 looked
        # like in early runs (smooth descent → sudden NaN with no recovery).
        grad_clip_norm = 1.0
        # Match ClassifierKerasMultiTask: monitor val_trading_mcc, maximise.
        monitor_mode = "max"
        # monitor_key = "val_trading_mcc"
        # monitor_key = "val_trading_f1_class_2"  # F1 peaks at epoch 1 (untrained, high recall); 2026-05-15 run showed save-best stuck at epoch 1 while precision improved through epoch 21
        monitor_key = "val_trading_precision"
        checkpoint_path = self.get_checkpoint_path()

        best_metric = -np.inf
        no_improve_cnt = 0
        plateau_cnt = 0
        current_lr = float(self.learning_rate)
        # Bail out if the optimizer state itself becomes contaminated (NaN
        # leak across many consecutive batches).  Restart-from-checkpoint is
        # not currently wired up — we just stop and rely on the most recent
        # best-weights save.
        max_consecutive_nan_batches = 50

        print(f"\n    Training MLX multi-task model: {self.name}")
        print(f"    train: {train_tensor.shape}  test: {test_tensor.shape}")
        print(f"    batch_size: {self.batch_size}  max_epochs: {max_epochs}")
        print(f"    monitor: {monitor_key} ({monitor_mode})")
        print(f"    grad clip (global L2 norm): {grad_clip_norm}")

        consecutive_nan_batches = 0
        aborted = False

        for epoch in range(max_epochs):
            # ---- training phase ----
            self.model.train()
            epoch_losses = []
            epoch_clips = 0  # how many batches in this epoch had grads clipped

            for X_batch, y_batch_dict in _batch_iter_multi(
                train_tensor, train_labels_np, self.batch_size, shuffle=True
            ):
                loss_val, grads = loss_and_grad(self.model, X_batch, y_batch_dict)
                loss_value = float(loss_val.item())

                # We have to check BOTH the loss and the gradient norm before
                # applying anything.  A finite loss can still produce NaN/Inf
                # gradients — e.g. a softmax probability close to its clip
                # bound gives a finite cross-entropy term but a derivative
                # that overflows.  Applying poisoned gradients corrupts every
                # parameter (and Adam's running averages) and from there the
                # next forward pass produces NaN forever.
                clipped_grads, total_norm = _clip_grads_by_global_norm(
                    grads, grad_clip_norm
                )
                norm_value = float(total_norm.item())

                if not (np.isfinite(loss_value) and np.isfinite(norm_value)):
                    consecutive_nan_batches += 1
                    if consecutive_nan_batches <= 3:
                        cause = []
                        if not np.isfinite(loss_value):
                            cause.append(f"loss={loss_value}")
                        if not np.isfinite(norm_value):
                            cause.append(f"grad_norm={norm_value}")
                        print(
                            f"    WARNING: non-finite update at epoch {epoch+1} "
                            f"({', '.join(cause)}; consecutive: "
                            f"{consecutive_nan_batches}); skipping"
                        )
                    if consecutive_nan_batches >= max_consecutive_nan_batches:
                        print(
                            f"    ABORT: {max_consecutive_nan_batches} consecutive "
                            f"non-finite batches — optimizer state is contaminated. "
                            f"Falling back to best checkpoint."
                        )
                        aborted = True
                        break
                    continue

                consecutive_nan_batches = 0

                if norm_value > grad_clip_norm:
                    epoch_clips += 1

                optimizer.update(self.model, clipped_grads)
                mx.eval(self.model.parameters(), optimizer.state)
                epoch_losses.append(loss_value)

            if aborted:
                break

            train_loss = float(np.mean(epoch_losses)) if epoch_losses else float("nan")

            # ---- validation phase ----
            self.model.eval()
            val_X = mx.array(test_tensor, dtype=mx.float32)
            val_y = {
                task: mx.array(test_labels_np[task], dtype=mx.float32)
                for task in TASK_NAMES
            }

            with mx.no_grad() if hasattr(mx, "no_grad") else _nullctx():
                val_preds = self.model(val_X)
                val_loss = mx.zeros(())
                for i, task in enumerate(task_weight_keys):
                    val_loss = val_loss + task_weight_arr[i] * loss_fns[task](
                        val_y[task], val_preds[task]
                    )
                val_loss = val_loss.item()

            mx.eval(*[val_preds[t] for t in TASK_NAMES])

            # Per-task metrics: trading gets the full set, others just MCC.
            per_task_metrics: Dict[str, dict] = {}
            for task in TASK_NAMES:
                per_task_metrics[task] = compute_val_metrics(
                    val_y[task], val_preds[task], target_class=2, num_classes=3
                )

            metrics_flat = self._flatten_metrics(per_task_metrics)
            current_metric = metrics_flat[monitor_key]

            clip_note = f"  clips:{epoch_clips}" if epoch_clips else ""
            print(
                f"    Epoch {epoch+1:3d}/{max_epochs} — "
                f"loss: {train_loss:.4f}  val_loss: {val_loss:.4f}  "
                f"val_trading_precision: {metrics_flat['val_trading_precision']:.4f}  "
                f"val_trading_f1_class_2: {metrics_flat['val_trading_f1_class_2']:.4f}  "
                f"val_trading_mcc: {metrics_flat['val_trading_mcc']:.4f}  "
                f"val_avg_mcc: {metrics_flat['val_avg_mcc']:.4f}"
                f"{clip_note}"
            )

            # ---- ModelCheckpoint (save best) ----
            improved = (
                monitor_mode == "max" and current_metric > best_metric
            ) or (monitor_mode == "min" and current_metric < best_metric)

            if improved:
                best_metric = current_metric
                no_improve_cnt = 0
                plateau_cnt = 0
                self.model.save_weights(checkpoint_path)
                print(f"    ✓ Best model saved  ({monitor_key}: {best_metric:.4f})")
            else:
                no_improve_cnt += 1
                plateau_cnt += 1

            # ---- ReduceLROnPlateau ----
            if plateau_cnt >= plateau_patience:
                new_lr = max(current_lr * plateau_factor, 1e-6)
                if new_lr < current_lr:
                    current_lr = new_lr
                    optimizer.learning_rate = current_lr
                    print(f"    ReduceLROnPlateau → lr={current_lr:.2e}")
                plateau_cnt = 0

            # ---- EarlyStopping ----
            if no_improve_cnt >= early_patience:
                print(
                    f"    EarlyStopping at epoch {epoch+1} "
                    f"(no improvement for {early_patience} epochs)"
                )
                break

        # ---- restore best weights ----
        if os.path.exists(checkpoint_path):
            print(f"    Restoring best weights from {checkpoint_path}")
            self.model.load_weights(checkpoint_path)
            mx.eval(self.model.parameters())

        self.save()
        self.is_trained = True

    # -----------------------------------------------------------------------
    # Prediction
    # -----------------------------------------------------------------------

    def predict(self, data) -> Dict[str, np.ndarray]:
        """
        Returns a dict of per-task softmax probabilities, identical shape
        to ClassifierKerasMultiTask.predict().
        """
        if self.model is None:
            self.model = self.load()

        if self.model is None:
            raise RuntimeError(
                f"CRITICAL: No MLX multi-task model found for {self.name} at "
                f"{self.model_path}. Ensure training completed successfully."
            )

        if self.dataframeUtils.is_dataframe(data):
            tensor = self.dataframeUtils.df_to_tensor(data, self.seq_len, method=3)
        else:
            tensor = np.asarray(data)

        self.model.eval()
        X = mx.array(tensor, dtype=mx.float32)
        preds = self.model(X)
        mx.eval(*[preds[t] for t in TASK_NAMES])
        return {task: np.asarray(preds[task]) for task in TASK_NAMES}

    # -----------------------------------------------------------------------
    # Class-weight + metric helpers
    # -----------------------------------------------------------------------

    def _calculate_class_weights(self, labels: np.ndarray) -> np.ndarray:
        """labels: one-hot (N, num_classes)"""
        class_counts = np.sum(labels, axis=0)
        weights = np.where(class_counts > 0, 1.0 / class_counts, 0.0)
        s = weights.sum()
        if s > 0:
            weights = weights / s
        return weights

    def _flatten_metrics(
        self, per_task: Dict[str, dict]
    ) -> Dict[str, float]:
        """
        Convert {task: {val_precision, val_f1_class_2, val_mcc, val_accuracy}}
        into a flat dict keyed by ``val_<task>_<metric>``, plus an aggregate
        ``val_avg_mcc`` matching the AverageMCCCallback in the Keras path.
        """
        flat: Dict[str, float] = {}
        mccs = []
        for task, m in per_task.items():
            flat[f"val_{task}_precision"]  = m["val_precision"]
            flat[f"val_{task}_f1_class_2"] = m["val_f1_class_2"]
            flat[f"val_{task}_mcc"]        = m["val_mcc"]
            flat[f"val_{task}_accuracy"]   = m["val_accuracy"]
            mccs.append(m["val_mcc"])
        flat["val_avg_mcc"] = float(np.mean(mccs)) if mccs else 0.0
        return flat


# Null context manager (fallback if mx.no_grad() doesn't exist in this version)
@contextmanager
def _nullctx():
    yield
