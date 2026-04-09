"""
Contains custom loss functions for use with keras models
"""


import tensorflow as tf
import keras
from tensorflow.keras import backend as K


# ---------------------------------------------------------

# Standard focal loss

@keras.saving.register_keras_serializable()
def focal_loss_softmax(y_true, y_pred, gamma=2.0, alpha=0.25):
    """
    Focal loss for softmax activation (multiple outputs)
    Handles class imbalance by down-weighting easy examples
    """
    y_true = tf.cast(y_true, tf.float32)
    epsilon = tf.keras.backend.epsilon()
    y_pred = tf.keras.backend.clip(y_pred, epsilon, 1.0 - epsilon)

    # Calculate cross entropy
    cross_entropy = -y_true * tf.math.log(y_pred)

    # Calculate focal weight: (1 - p_t)^gamma
    # For softmax, p_t is the predicted probability of the true class
    p_t = tf.reduce_sum(y_true * y_pred, axis=-1, keepdims=True)
    focal_weight = tf.math.pow(1 - p_t, gamma)

    # Apply focal weight and alpha
    loss = alpha * focal_weight * cross_entropy

    return tf.reduce_mean(tf.reduce_sum(loss, axis=-1))


# --------------------------------------------------------------


@keras.saving.register_keras_serializable()
def multi_class_focal_loss(gamma=0.5, alpha_vector=None):
    """
    Multi-class focal loss that supports per-class alpha weighting.
    """

    if alpha_vector is None:
        alpha_vector = [0.3, 0.3, 0.3]  # Fallback to no alpha weighting

    alpha_vector = K.constant(alpha_vector)

    def focal_loss_fixed(y_true, y_pred):
        # Convert y_true to one-hot encoding if it's not already
        if K.ndim(y_true) == K.ndim(y_pred):
            y_true_one_hot = y_true  # Already one-hot
        else:
            y_true_one_hot = K.one_hot(
                K.cast(y_true, "int32"), num_classes=K.shape(y_pred)[-1]
            )

        # Clip y_pred to prevent log(0)
        y_pred = K.clip(y_pred, K.epsilon(), 1.0 - K.epsilon())

        # Calculate cross-entropy (CE) loss
        cross_entropy = -y_true_one_hot * K.log(y_pred)

        # Calculate modulating factor (1 - pt)^gamma
        p_t = y_true_one_hot * y_pred + (1 - y_true_one_hot) * (1 - y_pred)
        modulating_factor = K.pow(1.0 - p_t, gamma)

        # Apply alpha weighting (element-wise multiplication by alpha vector)
        alpha_factor = y_true_one_hot * alpha_vector

        # Calculate Focal Loss
        fl = modulating_factor * alpha_factor * cross_entropy

        return 10.0 * K.sum(fl, axis=-1)

    return focal_loss_fixed


# ---------------------------------------------------------
# Range / magnitude regularisation losses
# ---------------------------------------------------------

@keras.saving.register_keras_serializable()
class RangePenaltyLoss(tf.keras.losses.Loss):
    """Serializable version of range penalty loss."""

    def __init__(
        self, target_range=2.0, penalty_weight=0.2, name="range_penalty_loss", **kwargs
    ):
        super().__init__(name=name, **kwargs)
        self.target_range = target_range
        self.penalty_weight = penalty_weight

    def call(self, y_true, y_pred):
        # Standard MSE loss
        mse_loss = tf.keras.losses.MeanSquaredError()(y_true, y_pred)

        # Range penalty: penalize relative to how little of the range is used
        pred_range = tf.reduce_max(y_pred) - tf.reduce_min(y_pred)
        range_utilization = (
            pred_range / self.target_range
        )  # 0.0 = no range used, 1.0 = full range used
        range_penalty = tf.maximum(
            0.0, 1.0 - range_utilization
        )  # Higher penalty for lower utilization

        # Combine losses
        total_loss = mse_loss + self.penalty_weight * range_penalty

        return total_loss

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "target_range": self.target_range,
                "penalty_weight": self.penalty_weight,
            }
        )
        return config


# ---------------------------------------------------------


@keras.saving.register_keras_serializable()
class RangeRegularizedLoss(tf.keras.losses.Loss):

    def __init__(
        self,
        target_std=0.05,
        std_penalty_weight=0.5,
        target_mean=0.02,  # <-- The mean of your scaled y_true
        mean_penalty_weight=0.1,  # <-- The weight for the mean penalty
        err_weight=1.0,
        name="range_loss",
        **kwargs,
    ):
        super().__init__(name=name, **kwargs)
        self.target_std = tf.constant(target_std, dtype=tf.float32)
        self.std_penalty_weight = tf.constant(std_penalty_weight, dtype=tf.float32)
        self.target_mean = tf.constant(
            target_mean, dtype=tf.float32
        )  # <-- Initialize as constant
        self.mean_penalty_weight = tf.constant(
            mean_penalty_weight, dtype=tf.float32
        )  # <-- Initialize as constant
        self.err_weight = tf.constant(err_weight, dtype=tf.float32)

    def call(self, y_true, y_pred):
        # 1. Base Loss: Mean Absolute Error (MAE)
        # err_loss = tf.keras.losses.MeanAbsoluteError()(y_true, y_pred)
        err_loss = tf.keras.losses.MeanSquaredError()(y_true, y_pred)

        # 2. STD Regularizer (ensures the predicted range is wide enough)
        pred_mean = tf.reduce_mean(y_pred)
        pred_variance = tf.reduce_mean(tf.square(y_pred - pred_mean))
        pred_std = tf.sqrt(pred_variance + 1e-8)

        # 3. Mean Penalty
        # Penalizes the squared difference between the predicted batch mean and the true target mean.
        mean_penalty = tf.square(pred_mean - self.target_mean)

        # 4. Range Penalty: Penalize when predicted STD is below target STD
        std_penalty = tf.maximum(0.0, self.target_std - pred_std)

        # 5. Total Loss Combination
        total_loss = (
            (self.err_weight * err_loss)
            + (self.std_penalty_weight * std_penalty)
            + (self.mean_penalty_weight * mean_penalty)
        )  # <-- Add the penalty

        return total_loss

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "target_std": self.target_std.numpy().item(),
                "std_penalty_weight": self.std_penalty_weight.numpy().item(),
                "target_mean": self.target_mean.numpy().item(),
                "mean_penalty_weight": self.mean_penalty_weight.numpy().item(),
                "err_weight": self.err_weight.numpy().item(),
            }
        )
        return config

# ---------------------------------------------------------


@keras.saving.register_keras_serializable()
class MagnitudeEnforcingLoss(tf.keras.losses.Loss):
    """Loss that penalizes mean errors that are too close to zero (prevents collapse to mean)
    Note that this does not use any batch-related variables so should constitent across batches
    """

    def __init__(
        self,
        min_magnitude=0.01,  # e.g., 1% profit/loss for a penalty threshold
        magnitude_penalty_weight=5.0,  # High weight to force magnitude
        name="mag_loss",
        **kwargs,
    ):
        super().__init__(name=name, **kwargs)
        self.min_magnitude = tf.constant(min_magnitude, dtype=tf.float32)
        self.magnitude_penalty_weight = tf.constant(
            magnitude_penalty_weight, dtype=tf.float32
        )

    def call(self, y_true, y_pred):
        # 1. Base Loss: Mean Absolute Error (MAE)
        # This measures how close the prediction is to the actual value.
        err_loss = tf.keras.losses.MeanAbsoluteError()(y_true, y_pred)

        # 2. Magnitude Penalty (Stable and Batch-Independent)
        # Calculate the absolute value of the predictions.
        abs_pred = tf.abs(y_pred)

        # Penalize if the prediction magnitude is below the threshold (min_magnitude).
        # We only want to penalize predictions that are "too safe" (near zero).
        magnitude_penalty = tf.maximum(0.0, self.min_magnitude - abs_pred)

        # Calculate the mean penalty across the batch.
        mean_magnitude_penalty = tf.reduce_mean(magnitude_penalty)

        # 3. Total Loss Combination
        total_loss = err_loss + (self.magnitude_penalty_weight * mean_magnitude_penalty)

        return total_loss

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "min_magnitude": self.min_magnitude.numpy().item(),
                "magnitude_penalty_weight": self.magnitude_penalty_weight.numpy().item(),
            }
        )
        return config

# ---------------------------------------------------------
# Additional loss helpers consolidated from classifier modules
# ---------------------------------------------------------

@keras.saving.register_keras_serializable()
def focal_loss_sigmoid(y_true, y_pred, gamma=3.0, alpha=0.75):
    """Binary focal loss for sigmoid outputs."""
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)

    epsilon = tf.keras.backend.epsilon()
    y_pred = tf.clip_by_value(y_pred, epsilon, 1.0 - epsilon)

    p_t = y_true * y_pred + (1 - y_true) * (1 - y_pred)
    modulating_factor = tf.pow(1 - p_t, gamma)
    alpha_factor = y_true * alpha + (1 - y_true) * (1 - alpha)

    focal_loss = -alpha_factor * modulating_factor * tf.math.log(p_t)
    return tf.reduce_mean(focal_loss)


@keras.saving.register_keras_serializable()
def dice_loss(y_true, y_pred):
    """Dice loss for imbalanced binary classification."""
    smooth = 1e-6
    y_true_f = tf.keras.backend.flatten(y_true)
    y_pred_f = tf.keras.backend.flatten(y_pred)
    intersection = tf.keras.backend.sum(y_true_f * y_pred_f)
    return 1 - (
        (2.0 * intersection + smooth)
        / (tf.keras.backend.sum(y_true_f) + tf.keras.backend.sum(y_pred_f) + smooth)
    )


@keras.saving.register_keras_serializable()
def relative_mse_loss(y_true, y_pred):
    """MSE computed on relative error."""
    epsilon = 1e-8
    relative_error = (y_true - y_pred) / (tf.abs(y_true) + epsilon)
    return tf.reduce_mean(relative_error ** 2)


@keras.saving.register_keras_serializable()
def relative_log_loss(y_true, y_pred):
    """Relative error loss with logarithmic compression."""
    epsilon = 1e-8
    relative_error = tf.abs(y_true - y_pred) / (tf.abs(y_true) + epsilon)
    loss = tf.reduce_mean(tf.math.log(1 + relative_error))
    return tf.maximum(epsilon, loss)


@keras.saving.register_keras_serializable()
def range_penalty_loss(y_true, y_pred, target_range=2.0, penalty_weight=0.1):
    """Functional variant of RangePenaltyLoss."""
    mse_loss = tf.keras.losses.MeanSquaredError()(y_true, y_pred)
    pred_range = tf.reduce_max(y_pred) - tf.reduce_min(y_pred)
    range_utilization = pred_range / target_range
    range_penalty = tf.maximum(0.0, 1.0 - range_utilization)
    return mse_loss + penalty_weight * range_penalty


@keras.saving.register_keras_serializable()
def unitary_range_penalty_loss(y_true, y_pred):
    """Convenience helper for [-1, 1] ranges."""
    return range_penalty_loss(y_true, y_pred, target_range=2.0, penalty_weight=1.0)


@keras.saving.register_keras_serializable()
def scaled_huber_loss(y_true, y_pred, delta=0.01, scale=1000):
    """Huber loss scaled to maintain gradient magnitude for small ranges."""
    huber = tf.keras.losses.Huber(delta=delta)
    return huber(y_true, y_pred) * scale


@keras.saving.register_keras_serializable()
def scaled_err_loss(y_true, y_pred, scale=10):
    """Mean absolute error with a configurable scaling factor."""
    mae = tf.keras.losses.MeanAbsoluteError()
    return mae(y_true, y_pred) * scale


@keras.saving.register_keras_serializable()
def scaled_mse_loss(y_true, y_pred, scale=100):
    """Mean squared error with a configurable scaling factor."""
    mse = tf.keras.losses.MeanSquaredError()
    return mse(y_true, y_pred) * scale
