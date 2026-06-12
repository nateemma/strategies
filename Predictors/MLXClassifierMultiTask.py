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
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from mlx.utils import tree_flatten, tree_map

sys.path.append(str(Path(__file__).parent))
sys.path.append(str(Path(__file__).parent.parent))

from Predictors.MLXBaseClassifier import MLXBaseClassifier
from utils.CustomLossMLX import multi_class_focal_loss_mlx
from utils.CustomMetricMLX import compute_val_metrics

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
    "trading": 2.0,
    "regime": 1.0,
    "risk": 1.0,
    "momentum": 1.0,
    "flow": 1.0,
    "profit": 2.0,
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
    finite_mask = (
        np.isfinite(tensor).all(axis=feature_axes)
        if feature_axes
        else np.isfinite(tensor)
    )

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
        f"    Filtered {n_dropped}/{n} ({pct:.2f}%) non-finite rows from {label} data"
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
    Same shuffling / batching logic as _batch_iter in MLXClassifierNary,
    but slices every task's label array with the same indices so all
    six heads see consistent rows.
    """
    n = len(X)
    idx_np = np.random.permutation(n) if shuffle else np.arange(n)

    X_mx = mx.array(X, dtype=mx.float32) if not isinstance(X, mx.array) else X
    y_mx = {
        task: (
            mx.array(arr, dtype=mx.float32) if not isinstance(arr, mx.array) else arr
        )
        for task, arr in y_dict.items()
    }

    for start in range(0, n, batch_size):
        b_idx_np = idx_np[start : start + batch_size]
        b_idx = mx.array(b_idx_np)
        yield X_mx[b_idx], {task: arr[b_idx] for task, arr in y_mx.items()}


# -----------------------------------------------------------------------


class MLXClassifierMultiTask(MLXBaseClassifier):
    """
    MLX multi-task classifier.  Mirrors ClassifierKerasMultiTask.py:
      - Same constructor + train/predict signatures
      - Same per-task class-weight, focal-loss, and task-weight logic
      - Manual EarlyStopping + ReduceLROnPlateau + ModelCheckpoint
      - predict() returns a dict of per-task softmax probabilities
    """

    clean_data_required: bool = False
    learning_rate: float = 1e-4
    # Max training epochs. Early-stopping (early_patience=20 below) cuts
    # off sooner if no improvement, so bumping this is safe — it just
    # raises the ceiling when the model is still improving at the limit.
    # Override via class attribute on the classifier instance (set in
    # strategy.get_classifier or by the _apply_classifier_overrides hook).
    max_epochs: int = 100

    # Internal state for per-task class weighting
    class_weights: Dict[str, list]

    # Optional per-instance override of the module-level TASK_WEIGHTS.
    # Strategies that want to shift loss budget across tasks (e.g. heavily
    # downweight auxiliary heads to let trading dominate) can set this
    # attribute on the classifier instance in their get_classifier() method.
    # When None, the module default TASK_WEIGHTS applies. Values are
    # normalised to sum to 1.0 at training time, same as the defaults.
    task_weights_override: Optional[Dict[str, float]]

    # Entropy penalty applied to the softmax outputs during training.
    # Adds  ``λ_t * mean(H(softmax_t))``  to the loss for each task t,
    # where H is Shannon entropy. Pushes the model toward sharper
    # (more confident) per-bar predictions — the strategy's filter
    # mechanism benefits from sharp softmax over diffuse high-entropy
    # outputs even at the same MCC.
    #
    # Accepts either a float or a dict:
    #   * float — same penalty across all six task heads (legacy form).
    #   * Dict[str, float] keyed by task name (subset of TASK_NAMES) —
    #     per-task penalty so pressure can be concentrated on heads the
    #     strategy actually depends on (typically trading + profit).
    #     Missing keys default to 0.0 (no penalty on that task).
    #
    # Default 0.0 disables. Useful values 0.01-0.5 per task.
    entropy_penalty_weight: Union[float, Dict[str, float]] = 0.0

    def __init__(self, pair, seq_len, num_features, tag=""):
        super().__init__(pair, seq_len, num_features, tag)
        self.name = self.__class__.__name__
        self.class_weights = {}
        self.task_weights_override = None

    def _task_weights(self) -> Dict[str, float]:
        """Per-instance task weights, normalised to sum to 1.0.

        Returns ``task_weights_override`` if set (renormalised), otherwise
        the module-level ``TASK_WEIGHTS`` default.
        """
        if self.task_weights_override is None:
            return TASK_WEIGHTS
        raw = dict(self.task_weights_override)
        total = sum(raw.values())
        if total <= 0:
            return TASK_WEIGHTS
        return {k: v / total for k, v in raw.items()}

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
            loss_fns[task] = multi_class_focal_loss_mlx(gamma=gamma, alpha_vector=alpha)
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
                "MLXClassifierMultiTask.train requires train_results and "
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
        active_task_weights = self._task_weights()
        task_weight_arr = mx.array(
            [active_task_weights[t] for t in task_weight_keys], dtype=mx.float32
        )
        weight_source = (
            "instance override"
            if self.task_weights_override is not None
            else "module default"
        )
        print(
            f"    task weights ({weight_source}, normalized, "
            f"sum={float(mx.sum(task_weight_arr)):.3f}): {active_task_weights}"
        )

        # --- build optimizer ---
        optimizer = optim.Adam(learning_rate=self.learning_rate)

        # --- forward + loss ---
        # Resolve entropy_penalty_weight (float or dict) into a per-task
        # dict so the loss inner-loop can index by task name directly.
        ew_raw = getattr(self, "entropy_penalty_weight", 0.0)
        if isinstance(ew_raw, dict):
            entropy_per_task: Dict[str, float] = {
                t: float(ew_raw.get(t, 0.0)) for t in TASK_NAMES
            }
        else:
            entropy_per_task = {t: float(ew_raw) for t in TASK_NAMES}
        entropy_active = any(v > 0.0 for v in entropy_per_task.values())
        if entropy_active:
            # Show only non-zero values for log compactness.
            nz = {t: v for t, v in entropy_per_task.items() if v > 0}
            print(
                f"    entropy penalty (per-task): {nz} "
                f"(pushes those softmax heads toward lower entropy / "
                f"sharper predictions)"
            )

        def forward_loss(model, X, y_dict):
            preds = model(X)  # dict of softmax outputs
            total = mx.zeros(())
            for i, task in enumerate(task_weight_keys):
                total = total + task_weight_arr[i] * loss_fns[task](
                    y_dict[task], preds[task]
                )
            # Entropy regularization: penalize high-entropy (diffuse)
            # softmax outputs on selected task heads. Per-task weighting
            # lets the strategy concentrate pressure on the heads that
            # drive filter behavior (typically trading + profit) without
            # collapsing aux heads. eps inside log avoids -inf when the
            # model predicts a class with probability 0.
            if entropy_active:
                eps = 1e-8
                for task in TASK_NAMES:
                    lam = entropy_per_task[task]
                    if lam <= 0.0:
                        continue
                    p = preds[task]
                    h = -mx.sum(p * mx.log(p + eps), axis=-1)  # per-bar entropy
                    total = total + lam * mx.mean(h)
            return total

        loss_and_grad = nn.value_and_grad(self.model, forward_loss)

        # --- training hyper-params ---
        # Read from the instance attribute (class default 100) so strategies
        # can override by setting clf.max_epochs in get_classifier().
        max_epochs = int(getattr(self, "max_epochs", 100))
        early_patience = 20
        plateau_patience = 8
        plateau_factor = 0.1
        # Gradient clipping global L2 norm.  Without it, an unlucky batch can
        # produce gradients large enough to spike Adam's running averages and
        # cascade the model weights to NaN — exactly what symptom #21 looked
        # like in early runs (smooth descent → sudden NaN with no recovery).
        # 2026-05-16: tightened from 1.0 → 0.5 after Fix B1 (balance_multi_task
        # ignoring collateral) more than doubled training set size to ~1.3M rows
        # with much higher class diversity, which surfaced NaN cascades from
        # epoch 44 onward with grad_clip_norm=1.0. The cascade let the model
        # collapse into degenerate prediction modes (trading→all-Sell,
        # profit→all-Hold) despite an improving val_trading_mcc. Tighter clip
        # caps per-batch gradient impact so the optimizer state stays stable.
        grad_clip_norm = 0.5
        # Match ClassifierKerasMultiTask: monitor val_trading_mcc, maximise.
        monitor_mode = "max"
        # monitor_key = "val_trading_mcc"           # empirical winner — robust to class imbalance and degenerate predictions
        # monitor_key = "val_trading_precision"   # 2026-05-15: drifts toward predicting class 1 (Hold/neutral) for most tasks; macro avg satisfied trivially
        # monitor_key = "val_trading_f1_class_2"  # 2026-05-15: peaks at epoch 1 (untrained, high recall); save-best traps the untrained model
        monitor_key = "val_avg_mcc_x_conf"  # 2026-05-23: composite (avg_mcc × mean_max_prob). Picked later epochs with marginally higher conf but worse backtest — the model's epoch-to-epoch conf variation was too small (~0.585→0.622) for the multiplicative weighting to be a useful signal.
        # Sharpness pressure now applied via entropy penalty in the loss
        # (see below) rather than via monitor selection.
        # monitor_key = "val_avg_mcc"
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
                            f"    WARNING: non-finite update at epoch {epoch + 1} "
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

            # Mean max-class probability across tasks — measures softmax
            # sharpness. Two models with equal MCC can have very different
            # confidence distributions (LSTM heads ~ sharp, Attention heads
            # ~ diffuse). The strategy's filter benefits from sharper
            # outputs, so we add a confidence-weighted monitor below.
            max_probs_per_task: List[float] = []
            for task in TASK_NAMES:
                max_probs_per_task.append(
                    float(mx.mean(mx.max(val_preds[task], axis=-1)).item())
                )
            mean_max_prob = float(np.mean(max_probs_per_task))

            metrics_flat = self._flatten_metrics(per_task_metrics)
            metrics_flat["val_mean_max_prob"] = mean_max_prob
            # Composite: rewards sharper softmax at equal MCC. Picks the
            # epoch that's both accurate AND confident, biasing best-model
            # selection toward the calibration profile the strategy uses.
            metrics_flat["val_avg_mcc_x_conf"] = (
                metrics_flat["val_avg_mcc"] * mean_max_prob
            )
            current_metric = metrics_flat[monitor_key]

            clip_note = f"  clips:{epoch_clips}" if epoch_clips else ""
            print(
                f"    Epoch {epoch + 1:3d}/{max_epochs} — "
                f"loss: {train_loss:.4f}  val_loss: {val_loss:.4f}  "
                f"val_trading_precision: {metrics_flat['val_trading_precision']:.4f}  "
                f"val_trading_f1_class_2: {metrics_flat['val_trading_f1_class_2']:.4f}  "
                f"val_trading_mcc: {metrics_flat['val_trading_mcc']:.4f}  "
                f"val_profit_mcc: {metrics_flat['val_profit_mcc']:.4f}  "
                f"val_avg_mcc: {metrics_flat['val_avg_mcc']:.4f}  "
                f"conf: {metrics_flat['val_mean_max_prob']:.4f}  "
                f"avg_mcc×conf: {metrics_flat['val_avg_mcc_x_conf']:.4f}"
                f"{clip_note}"
            )

            # ---- ModelCheckpoint (save best) ----
            improved = (monitor_mode == "max" and current_metric > best_metric) or (
                monitor_mode == "min" and current_metric < best_metric
            )

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
                    f"    EarlyStopping at epoch {epoch + 1} "
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

    def _flatten_metrics(self, per_task: Dict[str, dict]) -> Dict[str, float]:
        """
        Convert {task: {val_precision, val_f1_class_2, val_mcc, val_accuracy}}
        into a flat dict keyed by ``val_<task>_<metric>``, plus an aggregate
        ``val_avg_mcc`` matching the AverageMCCCallback in the Keras path.
        """
        flat: Dict[str, float] = {}
        mccs = []
        for task, m in per_task.items():
            flat[f"val_{task}_precision"] = m["val_precision"]
            flat[f"val_{task}_f1_class_2"] = m["val_f1_class_2"]
            flat[f"val_{task}_mcc"] = m["val_mcc"]
            flat[f"val_{task}_accuracy"] = m["val_accuracy"]
            mccs.append(m["val_mcc"])
        flat["val_avg_mcc"] = float(np.mean(mccs)) if mccs else 0.0
        return flat


# Null context manager (fallback if mx.no_grad() doesn't exist in this version)
@contextmanager
def _nullctx():
    yield
