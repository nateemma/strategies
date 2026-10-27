# pragma pylint: disable=C0103, C0114, C0115, C0116, C0301
"""TensorFlow port of Predictors/MLXClassifierNary.py (single-task softmax classifier).

Same inputs, labels, class weights, focal loss (alpha / gamma from the class weights), optimizer (MLX
Adam, AdamW when weight_decay > 0), clipping (global norm 1.0), monitor (val_mcc), patience and
P&L-magnitude weighting as the MLX classifier; see TFPortBasePredictor for the shared loop.
Subclasses build a Keras model in create_model() whose output is (batch, num_classes) softmax.
"""

from __future__ import annotations

import numpy as np
import tensorflow as tf

from Predictors.BaseClassifier import BaseClassifier
from Predictors.TFPortBasePredictor import TFPortBasePredictor, filter_nonfinite_rows, focal_loss
from utils.CustomMetricNP import compute_val_metrics


class TFPortClassifierNary(TFPortBasePredictor, BaseClassifier):
    train_seed: int | None = None
    max_epochs: int = 100
    early_patience: int = 20
    weight_decay: float = 0.0
    epochs_run: int = 0
    best_epoch: int = 0
    pnl_loss_alpha: float | None = None
    class_weights: list = []
    class_weight_dict: dict = {}
    num_classes: int = 3

    def train(self, df_train_norm, df_test_norm, train_results, test_results, force_train: bool = False,
              class_weights=None, sample_weights=None):
        """Signature and behaviour of MLXClassifierNary.train()."""
        if self.train_seed is not None:
            import keras

            keras.utils.set_random_seed(int(self.train_seed))
        if self.model is None:
            self.model = self.load()
        if self.model is not None and self.model_is_trained() and not force_train and not self.new_model_created():
            return

        if self.dataframeUtils.is_dataframe(df_train_norm) and self.clean_data_required:
            df_train_norm = df_train_norm[np.asarray(train_results) < 0.1]
            df_test_norm = df_test_norm[np.asarray(test_results) < 0.1]
        train_x, test_x = self._tensor(df_train_norm), self._tensor(df_test_norm)
        train_x, tr = filter_nonfinite_rows(train_x, {"y": np.asarray(train_results, dtype=np.float32)}, "training")
        test_x, te = filter_nonfinite_rows(test_x, {"y": np.asarray(test_results, dtype=np.float32)}, "validation")
        train_y, test_y = tr["y"], te["y"]

        train_w = self._pnl_weights(sample_weights, len(train_x))

        if self.model is None:
            self.num_features = train_x.shape[-1]
            self.model = self.create_model(self.seq_len, self.num_features)
            if self.model is None:
                print("    ERR: model not created")
                return
            if class_weights is None:
                class_weights = self.calculate_class_weights(test_y)
            self.set_class_weights(class_weights)
            print(f"    TensorFlow model created.  Parameters: {self.model.count_params():,}")

        alpha = self.calculate_alpha(self.class_weights)
        loss = focal_loss(self.calculate_gamma(alpha), alpha)
        def validate():
            preds = self._forward(test_x)
            m = compute_val_metrics(test_y, preds, target_class=2, num_classes=self.num_classes)
            line = (f"val_precision: {m['val_precision']:.4f}  val_f1_class_2: {m['val_f1_class_2']:.4f}  "
                    f"val_mcc: {m['val_mcc']:.4f}  val_conf: {m['val_confidence']:.4f}  "
                    f"val_conf_x_mcc: {m['val_confidence_x_mcc']:.4f}")
            return m, float(loss(tf.constant(test_y), tf.constant(preds))), line

        print(f"\n    Training TensorFlow model: {self.name}")
        print(f"    train: {train_x.shape}  test: {test_x.shape}")
        self.epochs_run, self.best_epoch = self._train_loop(
            train_x, {"y": train_y}, train_w, lambda p, y, w: loss(y["y"], p, w), validate,
            monitor_key="val_mcc", grad_clip_norm=1.0, max_epochs=int(self.max_epochs),
            early_patience=int(self.early_patience), weight_decay=float(self.weight_decay))
        self.save()
        self.is_trained = True

    def _pnl_weights(self, sample_weights, n: int):
        """MLXClassifierNary's optional P&L-magnitude loss weighting (pnl_loss_alpha)."""
        alpha = getattr(self, "pnl_loss_alpha", None)
        if sample_weights is None or not alpha:
            return None
        mag = np.asarray(sample_weights, dtype=np.float64)
        if len(mag) != n:
            print(f"    WARNING: pnl sample_weights len {len(mag)} != train rows {n}; disabling P&L weighting")
            return None
        base = np.nanmedian(mag)
        mag = np.where(np.isnan(mag), base if np.isfinite(base) else 0.0, mag)
        cap = np.quantile(mag, 0.95)
        mag = np.clip(mag, 0.0, cap if cap > 0 else None)
        mean = float(mag.mean())
        mag = mag / mean if mean > 0 else np.ones_like(mag)
        return ((1.0 - float(alpha)) + float(alpha) * mag).astype(np.float32)

    def predict(self, data) -> np.ndarray:
        """Softmax probabilities, (N, num_classes), as MLXClassifierNary.predict()."""
        if self.model is None:
            self.model = self.load()
        if self.model is None:
            raise RuntimeError(f"CRITICAL: No TensorFlow model found for {self.name} at {self.model_path}.")
        return self._forward(self._tensor(data))

    # --- class weights (MLXClassifierNary's logic) ---
    def calculate_class_weights(self, labels: np.ndarray) -> np.ndarray:
        counts = np.sum(labels, axis=0)
        w = np.where(counts > 0, 1.0 / np.maximum(counts, 1e-12), 0.0)
        return w / w.sum()

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
        n = self.num_classes if self.num_classes > 0 else len(weights)
        return [1.0 / n] * n

    def calculate_gamma(self, alpha) -> float:
        alpha = np.asarray(alpha)
        lo, hi = np.min(alpha), np.max(alpha)
        if lo == 0:
            return 2.0
        r = hi / lo
        return 0.0 if r < 1.2 else 0.5 if r < 2.0 else 1.0 if r < 5.0 else 1.5
