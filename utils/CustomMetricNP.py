"""NumPy versions of utils/CustomMetricMLX.py's validation metrics (same formulas, same 1e-7 guards).

Used by the TensorFlow ports of the MLX classifiers (Predictors/TFPort*), which must not import MLX.
y_true is one-hot (N, C), y_pred softmax (N, C); argmax picks the class.
"""

import numpy as np


def _classes(y_true, y_pred):
    return np.argmax(np.asarray(y_true), axis=-1), np.argmax(np.asarray(y_pred), axis=-1)


def per_class_f1(y_true, y_pred, target_class: int = 2) -> float:
    t, p = _classes(y_true, y_pred)
    tp = np.sum((p == target_class) & (t == target_class))
    fp = np.sum((p == target_class) & (t != target_class))
    fn = np.sum((p != target_class) & (t == target_class))
    precision, recall = tp / (tp + fp + 1e-7), tp / (tp + fn + 1e-7)
    return float(2.0 * precision * recall / (precision + recall + 1e-7))


def precision_class(y_true, y_pred, target_class: int = 2) -> float:
    t, p = _classes(y_true, y_pred)
    tp = np.sum((p == target_class) & (t == target_class))
    fp = np.sum((p == target_class) & (t != target_class))
    return float(tp / (tp + fp + 1e-7))


def mcc(y_true, y_pred, num_classes: int = 3) -> float:
    t, p = _classes(y_true, y_pred)
    cm = np.zeros((num_classes, num_classes), dtype=np.float64)
    np.add.at(cm, (t, p), 1.0)
    s, c = cm.sum(), np.trace(cm)
    p_k, t_k = cm.sum(axis=1), cm.sum(axis=0)
    return float((c * s - np.dot(p_k, t_k)) / np.sqrt((s * s - np.dot(p_k, p_k)) * (s * s - np.dot(t_k, t_k)) + 1e-7))


def accuracy(y_true, y_pred) -> float:
    t, p = _classes(y_true, y_pred)
    return float(np.mean(p == t))


def confidence_non_hold(y_pred, hold_class: int = 1) -> float:
    y_pred = np.asarray(y_pred)
    keep = np.argmax(y_pred, axis=-1) != hold_class
    return float(np.max(y_pred, axis=-1)[keep].mean()) if keep.any() else 0.0


def compute_val_metrics(y_true, y_pred, target_class: int = 2, num_classes: int = 3) -> dict:
    """CustomMetricMLX.compute_val_metrics: same keys, same values."""
    m = mcc(y_true, y_pred, num_classes)
    conf = confidence_non_hold(y_pred, hold_class=1)
    return {
        "val_precision": precision_class(y_true, y_pred, target_class),
        "val_f1_class_2": per_class_f1(y_true, y_pred, target_class),
        "val_mcc": m,
        "val_accuracy": accuracy(y_true, y_pred),
        "val_confidence": conf,
        "val_confidence_x_mcc": conf * m,
    }
