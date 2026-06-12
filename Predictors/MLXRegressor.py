# MLX regressor base class.
# Functionally parallel to utils/ClassifierKerasLinear.py (which despite its
# name is a regressor — its own docstring acknowledges as much). Mirrors the
# training-loop scaffolding of MLXClassifierNary but with:
#   - linear single-value output (Dense(1)) instead of softmax
#   - MSE loss instead of focal loss
#   - val_loss (min) instead of val_mcc (max) for early stopping
#   - no class-weight machinery (regression doesn't use it)
#
# Subclasses override create_model() to return an mlx.nn.Module whose
# forward pass accepts (batch, seq_len, num_features) and returns
# either (batch, 1) or (batch,) continuous predictions.

from __future__ import annotations

import os
import logging
import sys
import warnings
from contextlib import contextmanager
from pathlib import Path
from typing import Optional, Tuple, Union

import numpy as np
import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from mlx.utils import tree_flatten, tree_map

sys.path.append(str(Path(__file__).parent))
sys.path.append(str(Path(__file__).parent.parent))

from Predictors.MLXBasePredictor import MLXBasePredictor
from Predictors.BaseRegressor import BaseRegressor

log = logging.getLogger(__name__)
warnings.simplefilter(action="ignore", category=FutureWarning)


def _filter_nonfinite_rows(
    tensor: np.ndarray,
    targets: np.ndarray,
    label: str = "training",
) -> Tuple[np.ndarray, np.ndarray]:
    """Drop rows where the input tensor or targets contain NaN/Inf.

    Same purpose as the equivalent helper in MLXClassifierNary — pathological
    rows from upstream data corrupt training. Filtering up-front is cheaper
    than catching bad batches in the loop.
    """
    n = len(tensor)
    feature_axes = tuple(range(1, tensor.ndim))
    finite_mask = (
        np.isfinite(tensor).all(axis=feature_axes)
        if feature_axes
        else np.isfinite(tensor)
    )
    if targets.ndim == 1:
        finite_mask &= np.isfinite(targets)
    else:
        target_axes = tuple(range(1, targets.ndim))
        finite_mask &= np.isfinite(targets).all(axis=target_axes)

    n_kept = int(finite_mask.sum())
    n_dropped = n - n_kept
    if n_dropped == 0:
        return tensor, targets

    pct = 100.0 * n_dropped / n
    print(
        f"    Filtered {n_dropped}/{n} ({pct:.2f}%) non-finite rows from {label} data"
    )
    if n_kept == 0:
        raise RuntimeError(f"All {label} rows contained NaN/Inf — nothing to train on.")
    return tensor[finite_mask], targets[finite_mask]


def _clip_grads_by_global_norm(grads, max_norm: float):
    flat = tree_flatten(grads)
    total_sq = mx.zeros(())
    for _, g in flat:
        total_sq = total_sq + mx.sum(g * g)
    total_norm = mx.sqrt(total_sq)
    clip_coef = mx.minimum(
        mx.array(max_norm) / (total_norm + mx.array(1e-6)),
        mx.array(1.0),
    )
    clipped = tree_map(lambda g: g * clip_coef, grads)
    return clipped, total_norm


def _batch_iter(
    X: Union[np.ndarray, mx.array],
    y: Union[np.ndarray, mx.array],
    batch_size: int,
    shuffle: bool = True,
):
    n = len(X)
    idx = np.random.permutation(n) if shuffle else np.arange(n)
    y_mx = mx.array(y, dtype=mx.float32) if not isinstance(y, mx.array) else y
    X_mx = mx.array(X, dtype=mx.float32) if not isinstance(X, mx.array) else X
    for start in range(0, n, batch_size):
        b_idx = mx.array(idx[start : start + batch_size])
        yield X_mx[b_idx], y_mx[b_idx]


class MLXRegressor(MLXBasePredictor, BaseRegressor):
    """MLX linear regressor: single continuous output, MSE loss."""

    clean_data_required: bool = False

    # Training-loop knobs as class attributes so strategies / subclasses can
    # override per-instance (e.g. set max_epochs = 300 on a concrete regressor)
    # without changing the framework default for everyone else.
    max_epochs: int = 100
    early_patience: int = 20
    plateau_patience: int = 8

    # Diagnostic: prediction horizon (bars). When > 0, the per-epoch line
    # also reports ρ_shortcut = ρ(val_preds[H:], -test_targets[:-H]) — high
    # correlation here means predictions track "current state" (target shifted
    # back by H), which can score well on the actual target due to a
    # half-period coincidence even when the model isn't really forecasting.
    horizon: int = 0

    # -----------------------------------------------------------------------

    def create_model(self, seq_len: int, num_features: int) -> Optional[nn.Module]:
        print("    WARNING: create_model() should be defined by the subclass")
        return None

    # -----------------------------------------------------------------------
    # Training
    # -----------------------------------------------------------------------

    def train(
        self,
        df_train_norm,
        df_test_norm,
        train_results,
        test_results,
        force_train: bool = False,
        class_weights=None,  # accepted (and ignored) for call-site compatibility
        **kwargs,
    ):
        """Train the regressor with MSE loss + manual early-stopping/plateau-LR."""

        if self.model is None:
            self.model = self.load()
        else:
            print("    Model already exists")

        if (
            self.model is not None
            and self.model_is_trained()
            and not force_train
            and not self.new_model_created()
        ):
            return

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
            train_tensor = np.array(df_train_norm)
            test_tensor = np.array(df_test_norm)

        train_targets = np.array(train_results, dtype=np.float32).reshape(-1)
        test_targets = np.array(test_results, dtype=np.float32).reshape(-1)

        train_tensor, train_targets = _filter_nonfinite_rows(
            train_tensor, train_targets, label="training"
        )
        test_tensor, test_targets = _filter_nonfinite_rows(
            test_tensor, test_targets, label="validation"
        )

        def _tgt_stats(arr, label):
            print(
                f"    target stats — {label}: "
                f"mean={arr.mean():.4f} std={arr.std():.4f} "
                f"min={arr.min():.4f} max={arr.max():.4f} "
                f"q05={np.quantile(arr, 0.05):.4f} q95={np.quantile(arr, 0.95):.4f}"
            )

        _tgt_stats(train_targets, "train")
        _tgt_stats(test_targets, " test")

        # Linear baseline — fits a Ridge regression on flattened (seq_len * features)
        # against the same targets. If LSTM R² ≈ Ridge R², the model architecture
        # isn't the bottleneck (features/target are). Diagnostic-only; result is
        # printed and discarded.
        try:
            from sklearn.linear_model import Ridge
            from sklearn.metrics import r2_score
            from scipy.stats import spearmanr

            n_tr = train_tensor.shape[0]
            n_te = test_tensor.shape[0]
            X_tr_flat = np.asarray(train_tensor).reshape(n_tr, -1)
            X_te_flat = np.asarray(test_tensor).reshape(n_te, -1)

            ridge = Ridge(alpha=1.0)
            ridge.fit(X_tr_flat, train_targets)
            pred_tr = ridge.predict(X_tr_flat)
            pred_te = ridge.predict(X_te_flat)

            r2_tr = r2_score(train_targets, pred_tr)
            r2_te = r2_score(test_targets, pred_te)
            sp_tr = float(spearmanr(pred_tr, train_targets).correlation)
            sp_te = float(spearmanr(pred_te, test_targets).correlation)
            print(
                f"    Linear (Ridge) baseline — "
                f"train: R²={r2_tr:.4f} ρ={sp_tr:.4f} "
                f"std={pred_tr.std():.4f} range=[{pred_tr.min():.4f}, {pred_tr.max():.4f}]  |  "
                f"test: R²={r2_te:.4f} ρ={sp_te:.4f} "
                f"std={pred_te.std():.4f} range=[{pred_te.min():.4f}, {pred_te.max():.4f}]"
            )
        except Exception as e:
            print(f"    Linear baseline skipped: {e}")

        if self.model is None:
            self.num_features = train_tensor.shape[-1]
            self.model = self.create_model(self.seq_len, self.num_features)
            if self.model is None:
                print("    ERR: model not created")
                return
            self.model.input_shape = (None, self.seq_len, self.num_features)
            total_params = sum(
                p.size for k, p in tree_flatten(self.model.trainable_parameters())
            )
            print(f"    MLX regressor created. Parameters: {total_params:,}")

        def mse_loss(y_true: mx.array, y_pred: mx.array) -> mx.array:
            # y_pred may be (B,) or (B,1); flatten to (B,) for shape-stable mean.
            y_pred_flat = y_pred.reshape(-1)
            y_true_flat = y_true.reshape(-1)
            diff = y_pred_flat - y_true_flat
            return mx.mean(diff * diff)

        optimizer = optim.Adam(learning_rate=self.learning_rate)

        def forward_loss(model, X, y):
            preds = model(X)
            return mse_loss(y, preds)

        loss_and_grad = nn.value_and_grad(self.model, forward_loss)

        max_epochs = int(self.max_epochs)
        early_patience = int(self.early_patience)
        plateau_patience = int(self.plateau_patience)
        plateau_factor = 0.1
        grad_clip_norm = 1.0
        monitor_mode = "min"
        monitor_key = "val_loss"
        checkpoint_path = self.get_checkpoint_path()

        best_metric = np.inf
        best_epoch = 0
        no_improve_cnt = 0
        plateau_cnt = 0
        current_lr = float(self.learning_rate)
        max_consecutive_nan_batches = 50

        print(f"\n    Training MLX regressor: {self.name}")
        print(f"    train: {train_tensor.shape}  test: {test_tensor.shape}")
        print(f"    batch_size: {self.batch_size}  max_epochs: {max_epochs}")

        consecutive_nan_batches = 0
        aborted = False

        for epoch in range(max_epochs):
            self.model.train()
            epoch_losses = []
            epoch_clips = 0

            for X_batch, y_batch in _batch_iter(
                train_tensor, train_targets, self.batch_size, shuffle=True
            ):
                loss_val, grads = loss_and_grad(self.model, X_batch, y_batch)
                loss_value = float(loss_val.item())
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
                            f"({', '.join(cause)}; consecutive: {consecutive_nan_batches}); skipping"
                        )
                    if consecutive_nan_batches >= max_consecutive_nan_batches:
                        print(
                            f"    ABORT: {max_consecutive_nan_batches} consecutive non-finite "
                            f"batches — falling back to best checkpoint."
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

            self.model.eval()
            val_X = mx.array(test_tensor, dtype=mx.float32)
            val_y = mx.array(test_targets, dtype=mx.float32)
            with mx.no_grad() if hasattr(mx, "no_grad") else _nullctx():
                val_preds = self.model(val_X)
                val_loss = float(mse_loss(val_y, val_preds).item())
            mx.eval(val_preds)

            current_metric = val_loss
            clip_note = f"  clips:{epoch_clips}" if epoch_clips else ""
            val_preds_np = np.array(val_preds).reshape(-1)
            try:
                from scipy.stats import spearmanr

                spearman = float(spearmanr(val_preds_np, test_targets).correlation)
                # Shortcut diagnostic: ρ between predictions and the target
                # shifted backward by horizon. If ρ_shortcut is close to ρ
                # (or higher), the model is correlating with "current state"
                # — i.e., regurgitating an input feature whose value at time
                # i is structurally similar to the target at time i due to
                # the shift. Only meaningful when horizon > 0 is set on the
                # regressor (typically by the strategy).
                h_diag = int(self.horizon)
                if h_diag > 0 and len(test_targets) > h_diag:
                    spearman_shortcut = float(
                        spearmanr(
                            val_preds_np[h_diag:],
                            -test_targets[:-h_diag],
                        ).correlation
                    )
                    shortcut_note = f"  ρ_shortcut={spearman_shortcut:.4f}"
                else:
                    shortcut_note = ""
            except Exception:
                spearman = float("nan")
                shortcut_note = ""
            pred_note = (
                f"  ρ={spearman:.4f}{shortcut_note}"
                f"  pred: mean={val_preds_np.mean():.4f} "
                f"std={val_preds_np.std():.4f} "
                f"min={val_preds_np.min():.4f} max={val_preds_np.max():.4f}"
            )
            print(
                f"    Epoch {epoch + 1:3d}/{max_epochs} — "
                f"loss: {train_loss:.6f}  val_loss: {val_loss:.6f}{clip_note}{pred_note}"
            )

            improved = (monitor_mode == "max" and current_metric > best_metric) or (
                monitor_mode == "min" and current_metric < best_metric
            )

            if improved:
                best_metric = current_metric
                best_epoch = epoch + 1
                no_improve_cnt = 0
                plateau_cnt = 0
                self.model.save_weights(checkpoint_path)
                print(f"    ✓ Best model saved  ({monitor_key}: {best_metric:.6f})")
            else:
                no_improve_cnt += 1
                plateau_cnt += 1

            if plateau_cnt >= plateau_patience:
                new_lr = max(current_lr * plateau_factor, 1e-6)
                if new_lr < current_lr:
                    current_lr = new_lr
                    optimizer.learning_rate = current_lr
                    print(f"    ReduceLROnPlateau → lr={current_lr:.2e}")
                plateau_cnt = 0

            if no_improve_cnt >= early_patience:
                print(
                    f"    EarlyStopping at epoch {epoch + 1} "
                    f"(no improvement for {early_patience} epochs)"
                )
                break

        last_epoch = epoch + 1
        abort_note = "  ABORTED" if aborted else ""
        print(
            f"    Training summary: best_epoch={best_epoch}  last_epoch={last_epoch}  "
            f"best_{monitor_key}={best_metric:.6f}{abort_note}"
        )

        if os.path.exists(checkpoint_path):
            print(f"    Restoring best weights from {checkpoint_path}")
            self.model.load_weights(checkpoint_path)
            mx.eval(self.model.parameters())

        self.save()
        self.is_trained = True

    # -----------------------------------------------------------------------
    # Prediction
    # -----------------------------------------------------------------------

    def predict(self, data) -> np.ndarray:
        """Returns continuous predictions, shape (N,)."""
        if self.model is None:
            self.model = self.load()
        if self.model is None:
            raise RuntimeError(
                f"CRITICAL: No MLX model found for {self.name} at {self.model_path}. "
                "Ensure training completed successfully."
            )

        if self.dataframeUtils.is_dataframe(data):
            tensor = self.dataframeUtils.df_to_tensor(data, self.seq_len, method=3)
        else:
            tensor = np.array(data)

        self.model.eval()
        X = mx.array(tensor, dtype=mx.float32)
        preds = self.model(X)
        mx.eval(preds)
        return np.array(preds).reshape(-1)


@contextmanager
def _nullctx():
    yield
