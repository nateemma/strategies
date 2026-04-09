"""
MLX equivalents of CustomMetric.py.
Metrics are plain functions (not stateful classes) — computed per validation epoch
over the full validation set at once.
"""

import mlx.core as mx
import numpy as np


# ---------------------------------------------------------
# Per-class F1 (port of PerClassF1 from CustomMetric.py)
# ---------------------------------------------------------

def per_class_f1_mlx(y_true: mx.array, y_pred: mx.array,
                     target_class: int = 2, num_classes: int = 3) -> float:
    """
    Compute F1 score for a single class.

    Args:
        y_true:       One-hot labels  (N, num_classes)
        y_pred:       Softmax probs   (N, num_classes)
        target_class: Which class to evaluate (default: 2 = buy signal)
        num_classes:  Total number of classes

    Returns:
        F1 score for target_class as a Python float.
    """
    pred_classes = mx.argmax(y_pred, axis=-1)   # (N,)
    true_classes = mx.argmax(y_true, axis=-1)   # (N,)

    tp = mx.sum((pred_classes == target_class) & (true_classes == target_class)).item()
    fp = mx.sum((pred_classes == target_class) & (true_classes != target_class)).item()
    fn = mx.sum((pred_classes != target_class) & (true_classes == target_class)).item()

    precision = tp / (tp + fp + 1e-7)
    recall    = tp / (tp + fn + 1e-7)
    f1 = 2.0 * precision * recall / (precision + recall + 1e-7)
    return float(f1)


# ---------------------------------------------------------
# Precision for a single class
# ---------------------------------------------------------

def precision_class_mlx(y_true: mx.array, y_pred: mx.array,
                         target_class: int = 2) -> float:
    """Precision for target_class."""
    pred_classes = mx.argmax(y_pred, axis=-1)
    true_classes = mx.argmax(y_true, axis=-1)

    tp = mx.sum((pred_classes == target_class) & (true_classes == target_class)).item()
    fp = mx.sum((pred_classes == target_class) & (true_classes != target_class)).item()
    return float(tp / (tp + fp + 1e-7))


# ---------------------------------------------------------
# MCC (port of MCCMetric / MultiClassMCCMetric)
# ---------------------------------------------------------

def mcc_mlx(y_true: mx.array, y_pred: mx.array, num_classes: int = 3) -> float:
    """
    Multi-class Matthews Correlation Coefficient.
    Formula: (c*s - sum_k p_k*t_k) / sqrt((s²-sum p²)(s²-sum t²))
    """
    pred_classes = mx.argmax(y_pred, axis=-1)
    true_classes = mx.argmax(y_true, axis=-1)

    # Build confusion matrix
    pred_np = np.array(pred_classes)
    true_np = np.array(true_classes)
    cm = np.zeros((num_classes, num_classes), dtype=np.float64)
    for t, p in zip(true_np, pred_np):
        cm[t, p] += 1.0

    s = cm.sum()
    c = np.trace(cm)
    p_k = cm.sum(axis=1)
    t_k = cm.sum(axis=0)

    numerator = c * s - np.dot(p_k, t_k)
    denominator = np.sqrt(
        (s * s - np.dot(p_k, p_k)) * (s * s - np.dot(t_k, t_k)) + 1e-7
    )
    return float(numerator / denominator)


# ---------------------------------------------------------
# Accuracy (for quick sanity-check logging)
# ---------------------------------------------------------

def accuracy_mlx(y_true: mx.array, y_pred: mx.array) -> float:
    pred_classes = mx.argmax(y_pred, axis=-1)
    true_classes = mx.argmax(y_true, axis=-1)
    return float(mx.mean(pred_classes == true_classes).item())


# ---------------------------------------------------------
# Compute all validation metrics at once
# ---------------------------------------------------------

def compute_val_metrics(y_true: mx.array, y_pred: mx.array,
                        target_class: int = 2, num_classes: int = 3) -> dict:
    """
    Returns a dict of all metrics used during training.
    Key names mirror the Keras metric names so that monitoring logic is identical.
    """
    return {
        "val_precision":   precision_class_mlx(y_true, y_pred, target_class),
        "val_f1_class_2":  per_class_f1_mlx(y_true, y_pred, target_class, num_classes),
        "val_mcc":         mcc_mlx(y_true, y_pred, num_classes),
        "val_accuracy":    accuracy_mlx(y_true, y_pred),
    }
