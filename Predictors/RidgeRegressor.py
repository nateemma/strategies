"""
RidgeRegressor — sklearn Ridge regression as a drop-in for the LSTM regressors
in NNPredict_*.

Ridge is fit on flattened (seq_len * num_features) inputs against the
continuous future_gain target. Closed-form solution, no batches, no epochs.
Provides the same train/predict/save/load interface that the framework
expects from a regressor predictor.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import joblib
from sklearn.linear_model import Ridge

sys.path.append(str(Path(__file__).parent))
sys.path.append(str(Path(__file__).parent.parent))

from Predictors.BaseRegressor import BaseRegressor
from DataframeUtils import DataframeUtils


def _filter_nonfinite_rows(
    tensor: np.ndarray, targets: np.ndarray, label: str
) -> Tuple[np.ndarray, np.ndarray]:
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
    print(f"    Filtered {n_dropped}/{n} ({pct:.2f}%) non-finite rows from {label} data")
    if n_kept == 0:
        raise RuntimeError(f"All {label} rows contained NaN/Inf — nothing to train on.")
    return tensor[finite_mask], targets[finite_mask]


class RidgeRegressor(BaseRegressor):
    """sklearn Ridge regression on flattened (seq_len, num_features) tensors.

    Implements the regressor interface called by BaseNNStrategy:
      set_model_path, set_batch_size, model_exists, model_is_trained,
      new_model_created, train, predict, save, load.
    """

    # class-level defaults (instance attrs are set in __init__)
    is_trained: bool = False
    model = None
    name: str = ""
    model_path: str = ""
    model_per_pair: bool = False  # one shared model across all pairs
    new_model: bool = False
    clean_data_required: bool = False
    requires_dataframes: bool = False
    prescale_dataframe: bool = False

    alpha: float = 1.0  # ridge regularization strength

    def __init__(self, pair: str, seq_len: int, num_features: int, tag: str = ""):
        super().__init__()
        self.pair = pair
        self.seq_len = seq_len
        self.num_features = num_features
        self.tag = tag

        category = self.__class__.__name__
        pair_suffix = "_" + pair.split("/")[0] if self.model_per_pair else ""
        tag_suffix = "_" + tag if tag else ""
        self.name = category + pair_suffix + tag_suffix

        self.dataframeUtils = DataframeUtils()

    # ----------------------------------------------------------------
    # Interface methods called by BaseNNStrategy
    # ----------------------------------------------------------------

    def set_model_path(self, path: str) -> None:
        """Framework hands us a .keras / .safetensors path; rewrite to .joblib."""
        for ext in (".keras", ".safetensors"):
            if path.endswith(ext):
                path = path[: -len(ext)] + ".joblib"
                break
        self.model_path = path
        save_dir = os.path.dirname(path)
        if save_dir and not os.path.exists(save_dir):
            os.makedirs(save_dir, exist_ok=True)

    def set_batch_size(self, batch_size: int) -> None:
        # closed-form Ridge doesn't batch; accepted for interface compatibility
        return

    def model_exists(self) -> bool:
        return bool(self.model_path) and os.path.exists(self.model_path)

    def model_is_trained(self) -> bool:
        return self.is_trained

    def new_model_created(self) -> bool:
        return self.new_model

    # ----------------------------------------------------------------
    # Training
    # ----------------------------------------------------------------

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
        if self.model is None:
            self.model = self.load()

        if (
            self.model is not None
            and self.model_is_trained()
            and not force_train
            and not self.new_model_created()
        ):
            return

        if self.dataframeUtils.is_dataframe(df_train_norm):
            train_tensor = self.dataframeUtils.df_to_tensor(
                df_train_norm.copy(), self.seq_len, method=3
            )
            test_tensor = self.dataframeUtils.df_to_tensor(
                df_test_norm.copy(), self.seq_len, method=3
            )
        else:
            train_tensor = np.asarray(df_train_norm)
            test_tensor = np.asarray(df_test_norm)

        train_targets = np.asarray(train_results, dtype=np.float32).reshape(-1)
        test_targets = np.asarray(test_results, dtype=np.float32).reshape(-1)

        train_tensor, train_targets = _filter_nonfinite_rows(
            train_tensor, train_targets, "training"
        )
        test_tensor, test_targets = _filter_nonfinite_rows(
            test_tensor, test_targets, "validation"
        )

        n_tr = train_tensor.shape[0]
        n_te = test_tensor.shape[0]
        X_tr = np.asarray(train_tensor).reshape(n_tr, -1)
        X_te = np.asarray(test_tensor).reshape(n_te, -1)

        print(f"\n    Training Ridge regressor: {self.name}  alpha={self.alpha}")
        print(f"    train: {X_tr.shape}  test: {X_te.shape}")

        self.model = Ridge(alpha=self.alpha)
        self.model.fit(X_tr, train_targets)

        # input_shape attribute satisfies the framework's shape check at
        # BaseNNStrategy.get_predictions (line ~1839).
        self.model.input_shape = (None, self.seq_len, self.num_features)

        try:
            from sklearn.metrics import r2_score
            from scipy.stats import spearmanr

            pred_tr = self.model.predict(X_tr)
            pred_te = self.model.predict(X_te)
            r2_tr = r2_score(train_targets, pred_tr)
            r2_te = r2_score(test_targets, pred_te)
            sp_tr = float(spearmanr(pred_tr, train_targets).correlation)
            sp_te = float(spearmanr(pred_te, test_targets).correlation)
            print(
                f"    Ridge fit — train: R²={r2_tr:.4f} ρ={sp_tr:.4f} "
                f"std={pred_tr.std():.4f} range=[{pred_tr.min():.4f}, {pred_tr.max():.4f}]"
            )
            print(
                f"                test: R²={r2_te:.4f} ρ={sp_te:.4f} "
                f"std={pred_te.std():.4f} range=[{pred_te.min():.4f}, {pred_te.max():.4f}]"
            )
        except Exception as e:
            print(f"    Ridge metrics skipped: {e}")

        self.save()
        self.is_trained = True

    # ----------------------------------------------------------------
    # Prediction
    # ----------------------------------------------------------------

    def predict(self, data) -> np.ndarray:
        if self.model is None:
            self.model = self.load()
        if self.model is None:
            raise RuntimeError(
                f"CRITICAL: No Ridge model found for {self.name} at {self.model_path}. "
                "Ensure training completed successfully."
            )

        if self.dataframeUtils.is_dataframe(data):
            tensor = self.dataframeUtils.df_to_tensor(data, self.seq_len, method=3)
        else:
            tensor = np.asarray(data)

        n = tensor.shape[0]
        X = np.asarray(tensor).reshape(n, -1)
        preds = self.model.predict(X)
        return np.asarray(preds, dtype=np.float32).reshape(-1)

    # ----------------------------------------------------------------
    # Persistence
    # ----------------------------------------------------------------

    def save(self) -> None:
        if not self.model_path:
            print(f"    WARN: no model_path set for {self.name}")
            return
        save_dir = os.path.dirname(self.model_path)
        if save_dir and not os.path.exists(save_dir):
            os.makedirs(save_dir, exist_ok=True)
        joblib.dump(self.model, self.model_path)

    def load(self) -> Optional[Ridge]:
        if not self.model_path or not os.path.exists(self.model_path):
            return None
        try:
            model = joblib.load(self.model_path)
            # restore input_shape if missing (older saves)
            if not hasattr(model, "input_shape"):
                model.input_shape = (None, self.seq_len, self.num_features)
            self.is_trained = True
            return model
        except Exception as e:
            print(f"    WARN: failed to load Ridge model from {self.model_path}: {e}")
            return None
