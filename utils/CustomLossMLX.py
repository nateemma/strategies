"""
MLX equivalents of custom loss functions from CustomLoss.py.
Uses mlx.core operations instead of TensorFlow/Keras backend.
"""

import mlx.core as mx
import numpy as np


# ---------------------------------------------------------
# Multi-class focal loss (port of multi_class_focal_loss from CustomLoss.py)
# ---------------------------------------------------------

def multi_class_focal_loss_mlx(gamma: float = 0.5, alpha_vector=None):
    """
    Multi-class focal loss with per-class alpha weighting.
    Direct port of multi_class_focal_loss() from CustomLoss.py.

    Args:
        gamma:        Focusing parameter. 0.0 = standard cross-entropy.
        alpha_vector: Per-class weights as a list or numpy array.
                      Should be normalised (sum to 1.0).

    Returns:
        A loss function  loss(y_true, y_pred) -> scalar mx.array.
        y_true: one-hot (batch, num_classes)  float32
        y_pred: softmax probs (batch, num_classes) float32
    """
    if alpha_vector is None:
        alpha_vector = [1.0 / 3, 1.0 / 3, 1.0 / 3]

    # Materialise as constant MLX array so it lives in the compute graph
    alpha_mx = mx.array(np.array(alpha_vector, dtype=np.float32))

    def focal_loss_fixed(
        y_true: mx.array, y_pred: mx.array, sample_weights: mx.array = None
    ) -> mx.array:
        epsilon = 1e-7

        # Guard against log(0) / division by zero
        y_pred = mx.clip(y_pred, epsilon, 1.0 - epsilon)

        # Cross-entropy term: -(y_true * log(y_pred))
        cross_entropy = -y_true * mx.log(y_pred)

        # Modulating factor: (1 - p_t)^gamma
        p_t = y_true * y_pred + (1.0 - y_true) * (1.0 - y_pred)
        modulating_factor = mx.power(1.0 - p_t, gamma)

        # Per-class alpha weighting
        alpha_factor = y_true * alpha_mx

        # Focal loss, summed over classes → per-sample loss
        fl = modulating_factor * alpha_factor * cross_entropy
        per_sample = mx.sum(fl, axis=-1)  # (batch,)

        # Optional per-sample weighting (e.g. P&L-magnitude weighting). None ⇒
        # unchanged. Weights should have mean ≈ 1 so the loss scale is preserved.
        if sample_weights is not None:
            per_sample = per_sample * sample_weights

        # Mean over batch  (x10 matches the Keras version)
        return 10.0 * mx.mean(per_sample)

    return focal_loss_fixed


# ---------------------------------------------------------
# Standard cross-entropy loss (convenience wrapper)
# ---------------------------------------------------------

def categorical_cross_entropy_mlx(y_true: mx.array, y_pred: mx.array) -> mx.array:
    """Standard categorical cross-entropy. y_true one-hot, y_pred softmax."""
    epsilon = 1e-7
    y_pred = mx.clip(y_pred, epsilon, 1.0 - epsilon)
    return -mx.mean(mx.sum(y_true * mx.log(y_pred), axis=-1))
