# pragma pylint: disable=C0103, C0114, C0115, C0116, C0301
"""TensorFlow port of Predictors/MLXClassifierMultiTask.py (six 3-way softmax heads).

Same per-task class weights and focal losses, task weights (module default or task_weights_override),
entropy penalty (dropped on zero-weight tasks), optimizer (MLX Adam), clipping (global norm 0.5) and
monitor (val_trading_mcc, or the instance's monitor_key) as the MLX classifier; see TFPortBasePredictor
for the shared loop. Subclasses build a Keras model whose output is {task: (batch, 3) softmax}.
"""

from __future__ import annotations

from typing import Dict, Optional, Union

import numpy as np
import tensorflow as tf

from Predictors.BaseClassifier import BaseClassifier
from Predictors.TFPortBasePredictor import TFPortBasePredictor, filter_nonfinite_rows, focal_loss
from utils.CustomMetricNP import compute_val_metrics

# Copies of MLXClassifierMultiTask's constants (that module imports MLX); NNMT/tests/test_tf_port_nnmt.py
# checks they are equal where MLX is installed.
TASK_NAMES = ("trading", "regime", "risk", "momentum", "flow", "profit")
_TASK_WEIGHTS_RAW = {"trading": 2.0, "regime": 1.0, "risk": 1.0, "momentum": 1.0, "flow": 1.0, "profit": 2.0}
TASK_WEIGHTS = {k: v / sum(_TASK_WEIGHTS_RAW.values()) for k, v in _TASK_WEIGHTS_RAW.items()}


class TFPortClassifierMultiTask(TFPortBasePredictor, BaseClassifier):
    clean_data_required: bool = False
    learning_rate: float = 1e-4
    max_epochs: int = 100
    entropy_penalty_weight: Union[float, Dict[str, float]] = 0.0

    def __init__(self, pair, seq_len, num_features, tag=""):
        super().__init__(pair, seq_len, num_features, tag)
        self.name = self.__class__.__name__
        self.class_weights = {}
        self.task_weights_override = None

    def _task_weights(self) -> Dict[str, float]:
        if self.task_weights_override is None:
            return TASK_WEIGHTS
        raw = dict(self.task_weights_override)
        total = sum(raw.values())
        return {k: v / total for k, v in raw.items()} if total > 0 else TASK_WEIGHTS

    def calculate_alpha(self, weights) -> list:
        total = sum(weights) if weights is not None else 0.0
        return [w / total for w in weights] if total > 0 else [1.0 / 3] * 3

    def calculate_gamma(self, alpha) -> float:
        alpha = np.asarray(alpha)
        lo, hi = float(np.min(alpha)), float(np.max(alpha))
        if lo == 0:
            return 2.0
        r = hi / lo
        return 0.0 if r < 1.2 else 0.5 if r < 2.0 else 1.0 if r < 5.0 else 1.5

    def _calculate_class_weights(self, labels: np.ndarray) -> np.ndarray:
        counts = np.sum(labels, axis=0)
        w = np.where(counts > 0, 1.0 / np.maximum(counts, 1e-12), 0.0)
        return w / w.sum() if w.sum() > 0 else w

    def _build_task_loss_fns(self, class_weights: Optional[Dict[str, list]]):
        fns = {}
        for task in TASK_NAMES:
            raw = class_weights.get(task) if class_weights else None
            alpha = self.calculate_alpha(raw if raw is not None else [1.0 / 3] * 3)
            fns[task] = focal_loss(self.calculate_gamma(alpha), alpha)
        return fns

    def train(self, df_train_norm, df_test_norm, train_results: Dict[str, np.ndarray],
              test_results: Dict[str, np.ndarray], force_train: bool = False,
              class_weights: Optional[Dict[str, list]] = None):
        """Signature and behaviour of MLXClassifierMultiTask.train()."""
        if not isinstance(train_results, dict) or not isinstance(test_results, dict):
            raise ValueError("train_results and test_results must be dicts keyed by task name")
        missing = [t for t in TASK_NAMES if t not in train_results]
        if missing:
            raise ValueError(f"train_results is missing tasks: {missing} (expected {list(TASK_NAMES)})")
        if self.model is None:
            self.model = self.load()
        if self.model is not None and self.model_is_trained() and not force_train and not self.new_model_created():
            return

        train_x, test_x = self._tensor(df_train_norm), self._tensor(df_test_norm)
        train_x, train_y = filter_nonfinite_rows(
            train_x, {t: np.asarray(train_results[t], dtype=np.float32) for t in TASK_NAMES}, "training")
        test_x, test_y = filter_nonfinite_rows(
            test_x, {t: np.asarray(test_results[t], dtype=np.float32) for t in TASK_NAMES}, "validation")

        if self.model is None:
            self.num_features = train_x.shape[-1]
            self.model = self.create_model(self.seq_len, self.num_features)
            if self.model is None:
                print("    ERR: model not created")
                return
            if class_weights is None:
                class_weights = {t: list(self._calculate_class_weights(test_y[t])) for t in TASK_NAMES}
            self.class_weights = class_weights
            print(f"    TensorFlow multi-task model created.  Parameters: {self.model.count_params():,}")

        loss_fns = self._build_task_loss_fns(self.class_weights)
        tw = self._task_weights()
        print(f"    task weights ({'instance override' if self.task_weights_override is not None else 'module default'}, "
              f"normalized, sum={sum(tw.get(t, 0.0) for t in TASK_NAMES):.3f}): {tw}")
        ew = getattr(self, "entropy_penalty_weight", 0.0)
        ent = {t: float(ew.get(t, 0.0)) for t in TASK_NAMES} if isinstance(ew, dict) else {t: float(ew) for t in TASK_NAMES}
        dropped = {t: v for t, v in ent.items() if v > 0.0 and float(tw.get(t, 0.0)) <= 0.0}
        if dropped:
            print(f"    entropy penalty DROPPED for zero-weight task(s) {dropped}")
            ent = {t: (0.0 if t in dropped else v) for t, v in ent.items()}
        if any(v > 0 for v in ent.values()):
            print(f"    entropy penalty (per-task): { {t: v for t, v in ent.items() if v > 0} }")

        def supervised(preds, y):
            return tf.add_n([tw[t] * loss_fns[t](y[t], preds[t]) for t in TASK_NAMES])

        def loss_fn(preds, y, _w):
            total = supervised(preds, y)
            for t in TASK_NAMES:
                if ent[t] > 0.0:
                    p = preds[t]
                    total = total + ent[t] * tf.reduce_mean(-tf.reduce_sum(p * tf.math.log(p + 1e-8), axis=-1))
            return total

        def validate():
            preds = self._forward(test_x)
            per = {}
            for t in TASK_NAMES:
                lab = test_y[t].sum(axis=-1) > 0  # rows with no label for a task are left out of its metrics
                per[t] = compute_val_metrics(test_y[t][lab], preds[t][lab], target_class=2, num_classes=3)
            flat = self._flatten_metrics(per)
            flat["val_mean_max_prob"] = float(np.mean([np.mean(np.max(preds[t], axis=-1)) for t in TASK_NAMES]))
            flat["val_avg_mcc_x_conf"] = flat["val_avg_mcc"] * flat["val_mean_max_prob"]
            val_loss = float(supervised({t: tf.constant(preds[t]) for t in TASK_NAMES},
                                        {t: tf.constant(test_y[t]) for t in TASK_NAMES}))
            line = (f"val_trading_precision: {flat['val_trading_precision']:.4f}  "
                    f"val_trading_mcc: {flat['val_trading_mcc']:.4f}  val_profit_mcc: {flat['val_profit_mcc']:.4f}  "
                    f"[reg: {flat['val_regime_mcc']:.3f} risk: {flat['val_risk_mcc']:.3f} "
                    f"mom: {flat['val_momentum_mcc']:.3f} flow: {flat['val_flow_mcc']:.3f}]  "
                    f"val_avg_mcc: {flat['val_avg_mcc']:.4f}  conf: {flat['val_mean_max_prob']:.4f}")
            return flat, val_loss, line

        monitor = getattr(self, "monitor_key", None) or "val_trading_mcc"
        print(f"\n    Training TensorFlow multi-task model: {self.name}")
        print(f"    train: {train_x.shape}  test: {test_x.shape}")
        self._train_loop(train_x, train_y, None, loss_fn, validate, monitor_key=monitor, grad_clip_norm=0.5,
                         max_epochs=int(getattr(self, "max_epochs", 100)), early_patience=20)
        self.save()
        self.is_trained = True

    def predict(self, data) -> Dict[str, np.ndarray]:
        """{task: (N, 3) softmax}, as MLXClassifierMultiTask.predict()."""
        if self.model is None:
            self.model = self.load()
        if self.model is None:
            raise RuntimeError(f"CRITICAL: No TensorFlow multi-task model found for {self.name} at {self.model_path}.")
        out = self._forward(self._tensor(data))
        return {t: out[t] for t in TASK_NAMES}

    def _flatten_metrics(self, per_task: Dict[str, dict]) -> Dict[str, float]:
        flat = {}
        for t, m in per_task.items():
            for k in ("precision", "f1_class_2", "mcc", "accuracy"):
                flat[f"val_{t}_{k}"] = m[f"val_{k}"]
        flat["val_avg_mcc"] = float(np.mean([m["val_mcc"] for m in per_task.values()])) if per_task else 0.0
        return flat
