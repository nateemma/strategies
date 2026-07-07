# pragma pylint: disable=C0103, C0114, C0115, C0116, C0301, C0303, C0325, C0411, C0413
# type: ignore
# pylint: disable=import-error

"""
Forecasters for the NNWavelet family.

Unlike NNPredict_Coeff (coefficients are *features* to predict a scalar gain),
these regressors are MULTI-OUTPUT: they predict the future coefficient *vector*
and then reconstruct the gain from it (inverse wavelet transform), returning the
last reconstructed value as the predicted future gain.

Putting the reconstruction inside predict() lets the strategy reuse
NNPredictStrategy.get_predictions() unchanged — it still receives a 1-D array of
gains to z-score into buy/sell signals.

Two backends, same interface:
  * WaveletMLXRegressor  — MLX multi-output MLP (the neural predictor)
  * WaveletRidgeRegressor — sklearn multi-output Ridge (linear floor for A/B)

The strategy injects `wavelet` (a seeded utils.Wavelets transform) and `n_coeffs`
(the output dimension) onto the instance in get_classifier() before training.
"""

import os
import sys
from enum import Enum
from pathlib import Path
from typing import Optional

import numpy as np
import joblib
from sklearn.linear_model import Ridge

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim

sys.path.append(str(Path(__file__).parent))
sys.path.append(str(Path(__file__).parent.parent))

from Predictors.MLXRegressor import (
    MLXRegressor,
    _batch_iter,
    _clip_grads_by_global_norm,
)
from Predictors.BaseRegressor import BaseRegressor
from DataframeUtils import DataframeUtils


# ---------------------------------------------------------------------------
# Shared reconstruction: predicted coefficient rows -> predicted gain (1-D)
# ---------------------------------------------------------------------------
def reconstruct_gains(wavelet, coeff_rows: np.ndarray) -> np.ndarray:
    """Inverse-transform each predicted coefficient row and take the last
    reconstructed value (the gain at the forecast horizon), mirroring
    TS_Wavelet.predict_data. `wavelet` must already be seeded (its coeff_slices
    populated by a prior get_coeffs/coeff_to_array call) so array_to_coeff can
    rebuild the coefficient structure."""
    if wavelet is None:
        raise RuntimeError("reconstruct_gains: wavelet not set on the regressor")
    coeff_rows = np.nan_to_num(np.asarray(coeff_rows, dtype=float))
    out = np.zeros(len(coeff_rows), dtype=np.float32)
    for i in range(len(coeff_rows)):
        coeffs = wavelet.array_to_coeff(coeff_rows[i])
        vals = wavelet.get_values(coeffs)
        out[i] = float(vals[-1])
    return np.nan_to_num(out)


# ---------------------------------------------------------------------------
# MLX multi-output MLP
# ---------------------------------------------------------------------------
class MultiOutputMLP(nn.Module):
    """Flattens (seq_len, num_features) -> dense -> coefficient vector."""

    def __init__(self, input_dim: int, output_dim: int, hidden=(256, 128)):
        super().__init__()
        layers = []
        d = input_dim
        for h in hidden:
            layers += [nn.Linear(d, h), nn.ReLU()]
            d = h
        layers.append(nn.Linear(d, output_dim))
        self.layers = layers

    def __call__(self, x: mx.array) -> mx.array:
        x = x.reshape(x.shape[0], -1)  # accept 3-D (B, seq, feat) or 2-D input
        for layer in self.layers:
            x = layer(x)
        return x


class WaveletMLXRegressor(MLXRegressor):
    """Multi-output MLX MLP: predicts the future coefficient vector, then
    reconstructs the gain. Inherits MLXRegressor's full training loop (batching,
    best-checkpoint on val MSE, early stop, safetensors save/load), which is
    already multi-output-safe (the MSE loss flattens both operands)."""

    # injected by the strategy in get_classifier() before train()/predict()
    wavelet = None
    n_coeffs: int = 0

    def create_model(self, seq_len: int, num_features: int) -> Optional[nn.Module]:
        if not self.n_coeffs:
            raise RuntimeError(
                "WaveletMLXRegressor: n_coeffs not set — the strategy must assign "
                "it in get_classifier() before training."
            )
        return MultiOutputMLP(seq_len * num_features, int(self.n_coeffs))

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
        """Multi-output training loop. MLXRegressor.train() can't be reused
        because it flattens the target to 1-D; everything else (batching, grad
        clip, best-checkpoint on val MSE, early stop) is mirrored here for the
        (N, C) coefficient target."""
        if self.model is None:
            self.model = self.load()
        if (
            self.model is not None
            and self.model_is_trained()
            and not force_train
            and not self.new_model_created()
        ):
            return

        def _tensor(d):
            if self.dataframeUtils.is_dataframe(d):
                return np.asarray(
                    self.dataframeUtils.df_to_tensor(d.copy(), self.seq_len, method=3)
                )
            return np.asarray(d)

        def _drop_nonfinite(t, y):
            m = np.isfinite(t).all(axis=tuple(range(1, t.ndim)))
            m &= np.isfinite(y).all(axis=1) if y.ndim > 1 else np.isfinite(y)
            return t[m], y[m]

        train_tensor, y_tr = _drop_nonfinite(
            _tensor(df_train_norm), np.asarray(train_results, dtype=np.float32)
        )
        test_tensor, y_te = _drop_nonfinite(
            _tensor(df_test_norm), np.asarray(test_results, dtype=np.float32)
        )

        if self.model is None:
            self.model = self.create_model(self.seq_len, self.num_features)
            self.model.input_shape = (None, self.seq_len, self.num_features)
        mx.eval(self.model.parameters())

        def loss_fn(model, X, y):
            return mx.mean(mx.square(model(X) - y))

        loss_and_grad = nn.value_and_grad(self.model, loss_fn)
        optimizer = optim.Adam(learning_rate=self.learning_rate)

        max_epochs = int(self.max_epochs)
        patience = int(self.early_patience)
        checkpoint = self.get_checkpoint_path()
        best, best_epoch, no_improve = np.inf, 0, 0

        print(f"\n    Training Wavelet MLX MLP: {self.name}  out_dim={self.n_coeffs}")
        print(
            f"    train: {train_tensor.shape}  test: {test_tensor.shape}  "
            f"batch_size: {self.batch_size}  max_epochs: {max_epochs}"
        )

        val_X = mx.array(test_tensor, dtype=mx.float32)
        val_y = mx.array(y_te, dtype=mx.float32)

        for epoch in range(max_epochs):
            self.model.train()
            losses = []
            for Xb, yb in _batch_iter(train_tensor, y_tr, self.batch_size, shuffle=True):
                loss, grads = loss_and_grad(self.model, Xb, yb)
                grads, _ = _clip_grads_by_global_norm(grads, 1.0)
                optimizer.update(self.model, grads)
                mx.eval(self.model.parameters(), optimizer.state)
                losses.append(float(loss.item()))

            self.model.eval()
            val_preds = self.model(val_X)
            val_loss = float(mx.mean(mx.square(val_preds - val_y)).item())
            mx.eval(val_preds)

            if val_loss < best - 1e-9:
                best, best_epoch, no_improve = val_loss, epoch, 0
                self.model.save_weights(checkpoint)
            else:
                no_improve += 1

            if epoch == 0 or (epoch + 1) % 10 == 0:
                tr = float(np.mean(losses)) if losses else float("nan")
                print(
                    f"    Epoch {epoch + 1:3d}/{max_epochs} — train_mse={tr:.6f} "
                    f"val_mse={val_loss:.6f}  best={best:.6f}@{best_epoch + 1}"
                )
            if no_improve >= patience:
                print(f"    Early stop at epoch {epoch + 1} (no improvement {patience})")
                break

        if os.path.exists(checkpoint):
            self.model.load_weights(checkpoint)
            mx.eval(self.model.parameters())
        self.save()
        self.is_trained = True

    def predict(self, data) -> np.ndarray:
        if self.model is None:
            self.model = self.load()
        if self.model is None:
            raise RuntimeError(
                f"CRITICAL: No MLX model found for {self.name} at {self.model_path}."
            )

        if self.dataframeUtils.is_dataframe(data):
            tensor = self.dataframeUtils.df_to_tensor(data, self.seq_len, method=3)
        else:
            tensor = np.array(data)

        self.model.eval()
        preds = self.model(mx.array(tensor, dtype=mx.float32))
        mx.eval(preds)
        return reconstruct_gains(self.wavelet, np.array(preds))


# ---------------------------------------------------------------------------
# sklearn multi-output Ridge (linear floor)
# ---------------------------------------------------------------------------
class WaveletRidgeRegressor(BaseRegressor):
    """Multi-output Ridge on flattened (seq_len, num_features) inputs against the
    future coefficient matrix, then reconstruct. Same interface the framework
    expects (RidgeRegressor's, but 2-D target)."""

    is_trained: bool = False
    model = None
    name: str = ""
    model_path: str = ""
    model_per_pair: bool = False
    new_model: bool = False
    clean_data_required: bool = False
    requires_dataframes: bool = False
    prescale_dataframe: bool = False

    alpha: float = 1.0
    wavelet = None
    n_coeffs: int = 0

    def __init__(self, pair: str, seq_len: int, num_features: int, tag: str = ""):
        super().__init__()
        self.pair = pair
        self.seq_len = seq_len
        self.num_features = num_features
        self.tag = tag
        pair_suffix = "_" + pair.split("/")[0] if self.model_per_pair else ""
        tag_suffix = "_" + tag if tag else ""
        self.name = self.__class__.__name__ + pair_suffix + tag_suffix
        self.dataframeUtils = DataframeUtils()

    # --- interface plumbing (mirrors RidgeRegressor) ---
    def set_model_path(self, path: str) -> None:
        for ext in (".keras", ".safetensors"):
            if path.endswith(ext):
                path = path[: -len(ext)] + ".joblib"
                break
        self.model_path = path
        d = os.path.dirname(path)
        if d and not os.path.exists(d):
            os.makedirs(d, exist_ok=True)

    def set_batch_size(self, batch_size: int) -> None:
        return

    def model_exists(self) -> bool:
        return bool(self.model_path) and os.path.exists(self.model_path)

    def model_is_trained(self) -> bool:
        return self.is_trained

    def new_model_created(self) -> bool:
        return self.new_model

    def _to_matrix(self, df) -> np.ndarray:
        if self.dataframeUtils.is_dataframe(df):
            tensor = self.dataframeUtils.df_to_tensor(df.copy(), self.seq_len, method=3)
        else:
            tensor = np.asarray(df)
        n = tensor.shape[0]
        return np.asarray(tensor).reshape(n, -1)

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
        if self.model is None:
            self.model = self.load()
        if (
            self.model is not None
            and self.model_is_trained()
            and not force_train
            and not self.new_model_created()
        ):
            return

        X_tr = self._to_matrix(df_train_norm)
        y_tr = np.asarray(train_results, dtype=np.float32)  # (N, C)
        finite = np.isfinite(X_tr).all(axis=1) & np.isfinite(y_tr).all(axis=1)
        X_tr, y_tr = X_tr[finite], y_tr[finite]

        print(f"\n    Training Wavelet Ridge: {self.name}  alpha={self.alpha}")
        print(f"    train X: {X_tr.shape}  y: {y_tr.shape}")

        self.model = Ridge(alpha=self.alpha)
        self.model.fit(X_tr, y_tr)
        self.model.input_shape = (None, self.seq_len, self.num_features)
        self.save()
        self.is_trained = True

    def predict(self, data) -> np.ndarray:
        if self.model is None:
            self.model = self.load()
        if self.model is None:
            raise RuntimeError(
                f"CRITICAL: No Ridge model found for {self.name} at {self.model_path}."
            )
        X = self._to_matrix(data)
        coeff_rows = self.model.predict(X)  # (N, C)
        return reconstruct_gains(self.wavelet, coeff_rows)

    def save(self) -> None:
        if not self.model_path:
            return
        d = os.path.dirname(self.model_path)
        if d and not os.path.exists(d):
            os.makedirs(d, exist_ok=True)
        joblib.dump(self.model, self.model_path)

    def load(self):
        if not self.model_path or not os.path.exists(self.model_path):
            return None
        try:
            model = joblib.load(self.model_path)
            self.is_trained = True
            return model
        except Exception as e:
            print(f"    WARN: failed to load Ridge model from {self.model_path}: {e}")
            return None


class WaveletRegressorType(Enum):
    MLX = WaveletMLXRegressor
    RIDGE = WaveletRidgeRegressor
