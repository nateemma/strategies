# MLX N-ary Classifier base class.
# Mirrors ClassifierKerasNary.py API exactly.
#
# Implements the full training loop (EarlyStopping, ReduceLROnPlateau,
# ModelCheckpoint) that Keras provides automatically — here done manually.
#
# Subclasses should override create_model() to return an mlx.nn.Module
# whose forward pass (__call__) accepts (batch, seq_len, num_features) and
# returns (batch, num_classes) softmax probabilities.

from __future__ import annotations

import os
import logging
import sys
import warnings
from contextlib import contextmanager
from pathlib import Path
from typing import Union, Optional, Dict, Tuple

import numpy as np
import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from mlx.utils import tree_flatten

sys.path.append(str(Path(__file__).parent))
sys.path.append(str(Path(__file__).parent.parent))

from ClassifierMLX import ClassifierMLX
from CustomLossMLX import multi_class_focal_loss_mlx
from CustomMetricMLX import compute_val_metrics

log = logging.getLogger(__name__)
warnings.simplefilter(action="ignore", category=FutureWarning)


# -----------------------------------------------------------------------
# Internal helpers
# -----------------------------------------------------------------------


def _batch_iter(X: Union[np.ndarray, mx.array], y: Union[np.ndarray, mx.array], batch_size: int, shuffle: bool = True):
    """Yield (mx.array X_batch, mx.array y_batch) mini-batches."""
    n = len(X)
    idx = np.random.permutation(n) if shuffle else np.arange(n)
    # Pre-convert labels to mx.array once if they aren't already
    y_mx = mx.array(y, dtype=mx.float32) if not isinstance(y, mx.array) else y
    # Pre-convert data to mx.array once if it isn't already
    # (though method=3 already does this, this helps other methods)
    X_mx = mx.array(X, dtype=mx.float32) if not isinstance(X, mx.array) else X

    for start in range(0, n, batch_size):
        b_idx_np = idx[start : start + batch_size]
        # Convert batch indices to mx.array for compatible indexing
        b_idx = mx.array(b_idx_np) 
        yield X_mx[b_idx], y_mx[b_idx]


# -----------------------------------------------------------------------


class ClassifierMLXNary(ClassifierMLX):
    """
    MLX N-ary (multi-class softmax) classifier.
    Mirrors ClassifierKerasNary.py:
      - Same constructor + train/predict signatures
      - Same class-weight and focal-loss logic
      - Manual EarlyStopping + ReduceLROnPlateau + ModelCheckpoint
    """

    clean_data_required: bool = False
    num_classes: int = 3

    # Internal state for class weighting
    class_weights: list = []
    class_weight_dict: dict = {}

    # -----------------------------------------------------------------------
    # Model creation (subclasses override)
    # -----------------------------------------------------------------------

    def create_model(self, seq_len: int, num_features: int) -> nn.Module | None:
        """Return an mlx.nn.Module. Subclasses must override this."""
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
        class_weights=None,
    ):
        """
        Train the model.  Signature identical to ClassifierKerasNary.train().

        df_train_norm / df_test_norm : pandas DataFrame *or* numpy array
                                       (already normalised + tensorised)
        train_results / test_results : one-hot numpy array (N, num_classes)
        force_train                  : ignore is_trained flag
        class_weights                : optional per-class weight array
        """
        # --- lazy load existing model ---
        if self.model is None:
            self.model = self.load()
        else:
            print("    Model already exists")

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
            if self.clean_data_required:
                df1 = df_train_norm.copy()
                df1["%labels"] = train_results
                df1 = df1[df1["%labels"] < 0.1]
                df_train = df1.drop("%labels", axis=1)

                df2 = df_test_norm.copy()
                df2["%labels"] = test_results
                df2 = df2[df2["%labels"] < 0.1]
                df_test = df2.drop("%labels", axis=1)
            else:
                df_train = df_train_norm.copy()
                df_test = df_test_norm.copy()

            train_tensor = self.dataframeUtils.df_to_tensor(df_train, self.seq_len, method=3)
            test_tensor = self.dataframeUtils.df_to_tensor(df_test, self.seq_len, method=3)
        else:
            # already numpy tensors
            train_tensor = np.array(df_train_norm)
            test_tensor = np.array(df_test_norm)

        # Ensure results are numpy
        train_results_np = np.array(train_results, dtype=np.float32)
        test_results_np = np.array(test_results, dtype=np.float32)

        # --- create model if missing ---
        if self.model is None:
            self.num_features = train_tensor.shape[-1]
            self.model = self.create_model(self.seq_len, self.num_features)
            if self.model is None:
                print("    ERR: model not created")
                return

            # Decorate the MLX model with input_shape for keras compatibility upstream
            self.model.input_shape = (None, self.seq_len, self.num_features)

            if class_weights is None:
                class_weights = self.calculate_class_weights(test_results_np)
            self.set_class_weights(class_weights)

            # Print rough parameter count
            total_params = sum(
                p.size for k, p in tree_flatten(self.model.trainable_parameters())
            )
            print(f"    MLX model created.  Parameters: {total_params:,}")

        # --- build loss function ---
        focal_alpha = self.calculate_alpha(self.class_weights)
        gamma = self.calculate_gamma(focal_alpha)
        loss_fn = multi_class_focal_loss_mlx(gamma=gamma, alpha_vector=focal_alpha)

        # --- build optimizer ---
        optimizer = optim.Adam(learning_rate=self.learning_rate)

        # ---- define per-step loss+grad function ----
        def forward_loss(model, X, y):
            preds = model(X)
            return loss_fn(y, preds)

        loss_and_grad = nn.value_and_grad(self.model, forward_loss)

        # ---- training hyper-params (mirrors ClassifierKerasNary) ----
        max_epochs = 100
        early_patience = 20
        plateau_patience = 8
        plateau_factor = 0.1
        monitor_mode = "max"  # we monitor val_precision
        # monitor_key = "val_precision"
        # monitor_key = "val_f1_class_2"
        monitor_key = "val_mcc"
        checkpoint_path = self.get_checkpoint_path()

        best_metric = -np.inf
        no_improve_cnt = 0
        plateau_cnt = 0
        current_lr = float(self.learning_rate)

        print(f"\n    Training MLX model: {self.name}")
        print(f"    train: {train_tensor.shape}  test: {test_tensor.shape}")
        print(f"    batch_size: {self.batch_size}  max_epochs: {max_epochs}")

        for epoch in range(max_epochs):
            # ---- training phase ----
            self.model.train()
            epoch_losses = []

            for X_batch, y_batch in _batch_iter(
                train_tensor, train_results_np, self.batch_size, shuffle=True
            ):
                loss_val, grads = loss_and_grad(self.model, X_batch, y_batch)
                optimizer.update(self.model, grads)
                mx.eval(self.model.parameters(), optimizer.state)
                epoch_losses.append(loss_val.item())

            train_loss = float(np.mean(epoch_losses))

            # ---- validation phase ----
            self.model.eval()
            val_X = mx.array(test_tensor, dtype=mx.float32)
            val_y = mx.array(test_results_np, dtype=mx.float32)

            with mx.no_grad() if hasattr(mx, "no_grad") else _nullctx():
                val_preds = self.model(val_X)
                val_loss = loss_fn(val_y, val_preds).item()

            mx.eval(val_preds)
            metrics = compute_val_metrics(
                val_y, val_preds, target_class=2, num_classes=self.num_classes
            )
            current_metric = metrics[monitor_key]

            print(
                f"    Epoch {epoch+1:3d}/{max_epochs} — "
                f"loss: {train_loss:.4f}  val_loss: {val_loss:.4f}  "
                f"val_precision: {metrics['val_precision']:.4f}  "
                f"val_f1_class_2: {metrics['val_f1_class_2']:.4f}  "
                f"val_mcc: {metrics['val_mcc']:.4f}"
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

    def predict(self, data) -> np.ndarray:
        """
        Returns raw softmax probabilities, shape (N, num_classes).
        Identical return type to ClassifierKerasNary.predict().
        """
        # lazy load
        if self.model is None:
            self.model = self.load()

        if self.model is None:
            raise RuntimeError(
                f"CRITICAL: No MLX model found for {self.name} at {self.model_path}. "
                "Ensure training completed successfully."
            )

        # accept DataFrame or numpy tensor
        if self.dataframeUtils.is_dataframe(data):
            tensor = self.dataframeUtils.df_to_tensor(data, self.seq_len, method=3)
        else:
            tensor = np.array(data)

        self.model.eval()
        X = mx.array(tensor, dtype=mx.float32)
        preds = self.model(X)
        mx.eval(preds)
        return np.array(preds)

    # -----------------------------------------------------------------------
    # Class-weight helpers (identical logic to ClassifierKerasNary)
    # -----------------------------------------------------------------------

    def calculate_class_weights(self, labels: np.ndarray) -> np.ndarray:
        """labels: one-hot (N, num_classes)"""
        class_counts = np.sum(labels, axis=0)
        weights = np.where(class_counts > 0, 1.0 / class_counts, 0.0)
        weights = weights / weights.sum()
        return weights

    def set_class_weights(self, weights):
        self.class_weights = list(weights)
        self.class_weight_dict = dict(enumerate(self.class_weights))

    def get_class_weights(self) -> list:
        return self.class_weights

    def get_class_weight_dict(self) -> dict:
        return self.class_weight_dict

    def calculate_alpha(self, weights) -> list:
        total = sum(weights)
        if total > 0:
            return [w / total for w in weights]
        n = (
            self.num_classes
            if hasattr(self, "num_classes") and self.num_classes > 0
            else len(weights)
        )
        return [1.0 / n] * n

    def calculate_gamma(self, alpha) -> float:
        alpha = np.array(alpha)
        min_a, max_a = np.min(alpha), np.max(alpha)
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


# Null context manager (fallback if mx.no_grad() doesn't exist in this version)
@contextmanager
def _nullctx():
    yield
