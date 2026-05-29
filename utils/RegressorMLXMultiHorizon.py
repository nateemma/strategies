"""
Multi-output (multi-horizon) regressor — subclass of RegressorMLXLinear.

Single MLX model with `num_horizons` output heads, jointly trained on H
target columns. Used by NNPredict_MLX_MultiHorizon strategies to predict
the same input at multiple forward horizons simultaneously, then combine
the heads at signal-time for ensemble-style filtering.

Same training scaffolding as the single-output parent (early stopping,
gradient clipping, plateau LR reduction) but:
  - targets kept as (N, H) instead of flattened to (-1)
  - predictions returned as (N, H) instead of (N,)
  - diagnostic spearman computed per-horizon and averaged
  - Ridge baseline skipped (would need per-horizon fits — adds little)
"""

from __future__ import annotations

import os
import sys
from contextlib import contextmanager
from pathlib import Path
from typing import Optional

import numpy as np
import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from mlx.utils import tree_flatten

sys.path.append(str(Path(__file__).parent))

from RegressorMLXLinear import (  # noqa: E402
    RegressorMLXLinear,
    _filter_nonfinite_rows,
    _clip_grads_by_global_norm,
    _batch_iter,
)


@contextmanager
def _nullctx():
    yield


class RegressorMLXMultiHorizon(RegressorMLXLinear):
    """Multi-output MLX regressor — targets shape (N, H), model output (B, H)."""

    clean_data_required: bool = False

    # Number of horizons (output dimensions). Subclass / strategy sets this.
    num_horizons: int = 3

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
        class_weights=None,
        **kwargs,
    ):
        """Train the multi-horizon regressor. Targets are 2-D (N, H)."""

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
            train_tensor = self.dataframeUtils.df_to_tensor(df_train, self.seq_len, method=3)
            test_tensor = self.dataframeUtils.df_to_tensor(df_test, self.seq_len, method=3)
        else:
            train_tensor = np.array(df_train_norm)
            test_tensor = np.array(df_test_norm)

        # NOTE: no .reshape(-1) — keep targets as (N, H).
        train_targets = np.asarray(train_results, dtype=np.float32)
        test_targets = np.asarray(test_results, dtype=np.float32)
        if train_targets.ndim == 1:
            train_targets = train_targets.reshape(-1, 1)
        if test_targets.ndim == 1:
            test_targets = test_targets.reshape(-1, 1)

        train_tensor, train_targets = _filter_nonfinite_rows(
            train_tensor, train_targets, label="training"
        )
        test_tensor, test_targets = _filter_nonfinite_rows(
            test_tensor, test_targets, label="validation"
        )

        h = train_targets.shape[1]
        print(
            f"    multi-horizon targets: shape={train_targets.shape}  "
            f"per-horizon mean={train_targets.mean(axis=0).round(4).tolist()}  "
            f"std={train_targets.std(axis=0).round(4).tolist()}"
        )

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
            print(f"    MLX multi-horizon regressor created. Parameters: {total_params:,}")

        def mse_loss(y_true: mx.array, y_pred: mx.array) -> mx.array:
            # Both (B, H). MSE averaged across batch and horizons.
            diff = y_pred - y_true
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
        checkpoint_path = self.get_checkpoint_path()

        best_metric = np.inf
        best_epoch = 0
        no_improve_cnt = 0
        plateau_cnt = 0
        current_lr = float(self.learning_rate)

        print(f"\n    Training MLX multi-horizon regressor: {self.name}")
        print(
            f"    train: {train_tensor.shape}  test: {test_tensor.shape}  "
            f"num_horizons: {h}"
        )
        print(f"    batch_size: {self.batch_size}  max_epochs: {max_epochs}")

        for epoch in range(max_epochs):
            self.model.train()
            epoch_losses = []
            epoch_clips = 0

            for X_batch, y_batch in _batch_iter(
                train_tensor, train_targets, self.batch_size, shuffle=True
            ):
                loss_val, grads = loss_and_grad(self.model, X_batch, y_batch)
                loss_value = float(loss_val.item())
                clipped_grads, total_norm = _clip_grads_by_global_norm(grads, grad_clip_norm)
                norm_value = float(total_norm.item())

                if not (np.isfinite(loss_value) and np.isfinite(norm_value)):
                    continue

                if norm_value > grad_clip_norm:
                    epoch_clips += 1

                optimizer.update(self.model, clipped_grads)
                mx.eval(self.model.parameters(), optimizer.state)
                epoch_losses.append(loss_value)

            train_loss = float(np.mean(epoch_losses)) if epoch_losses else float("nan")

            # Validation
            self.model.eval()
            val_X = mx.array(test_tensor, dtype=mx.float32)
            val_y = mx.array(test_targets, dtype=mx.float32)
            with mx.no_grad() if hasattr(mx, "no_grad") else _nullctx():
                val_preds = self.model(val_X)
                val_loss = float(mse_loss(val_y, val_preds).item())
            mx.eval(val_preds)

            # Per-horizon ρ for diagnostics
            try:
                from scipy.stats import spearmanr
                val_preds_np = np.asarray(val_preds)  # (N, H)
                rhos = []
                for j in range(h):
                    r = spearmanr(val_preds_np[:, j], test_targets[:, j]).correlation
                    rhos.append(float(r) if np.isfinite(r) else 0.0)
                rho_str = " ".join(f"ρ{j}={r:+.3f}" for j, r in enumerate(rhos))
            except Exception:
                rho_str = ""

            clip_note = f"  clips:{epoch_clips}" if epoch_clips else ""
            print(
                f"    Epoch {epoch+1:>3}/{max_epochs} — "
                f"loss: {train_loss:.6f}  val_loss: {val_loss:.6f}  "
                f"{rho_str}{clip_note}"
            )

            current_metric = val_loss
            if current_metric < best_metric:
                best_metric = current_metric
                best_epoch = epoch + 1
                no_improve_cnt = 0
                plateau_cnt = 0
                self.model.save_weights(checkpoint_path)
                print(f"    ✓ Best model saved  (val_loss: {best_metric:.6f})")
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
                    f"    EarlyStopping at epoch {epoch+1} "
                    f"(no improvement for {early_patience} epochs)"
                )
                break

        last_epoch = epoch + 1
        print(
            f"    Training summary: best_epoch={best_epoch}  last_epoch={last_epoch}  "
            f"best_val_loss={best_metric:.6f}"
        )

        # Restore best checkpoint
        if os.path.exists(checkpoint_path):
            print(f"    Restoring best weights from {checkpoint_path}")
            self.model.load_weights(checkpoint_path)
            mx.eval(self.model.parameters())
        self.is_trained = True

    # -----------------------------------------------------------------------
    # Prediction
    # -----------------------------------------------------------------------

    def predict(self, data) -> np.ndarray:
        """Returns continuous predictions, shape (N, num_horizons).

        Unlike the single-output parent which flattens to (N,), the
        multi-horizon variant keeps both dimensions so the caller can
        combine the heads.
        """
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
        out = np.asarray(preds)
        if out.ndim == 1:
            out = out.reshape(-1, 1)
        return out
