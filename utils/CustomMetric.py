"""
Collection of custom metrics for use with keras models
"""


import tensorflow as tf
import keras


from tensorflow.keras.metrics import Precision, Recall

# ----------------------------------------------------------

class MinorityF1(tf.keras.metrics.Metric):
    """
    Calculates the average F1 score for the minority classes (0 and 2).
    This should work for any 3 class problem where class 1 is dominant and classes 0 and 2 are minority.
    This is necessary because tf.keras.metrics.F1Score does not support class_id.
    """

    def __init__(self, name="minority_f1", **kwargs):
        super(MinorityF1, self).__init__(name=name, **kwargs)

        # Initialize Precision and Recall for Class 0
        self.precision_0 = Precision(class_id=0, name="precision_0")
        self.recall_0 = Recall(class_id=0, name="recall_0")

        # Initialize Precision and Recall for Class 2
        self.precision_2 = Precision(class_id=2, name="precision_2")
        self.recall_2 = Recall(class_id=2, name="recall_2")

    def update_state(self, y_true, y_pred, sample_weight=None):
        # Update state for all four underlying metrics
        self.precision_0.update_state(y_true, y_pred, sample_weight)
        self.recall_0.update_state(y_true, y_pred, sample_weight)
        self.precision_2.update_state(y_true, y_pred, sample_weight)
        self.recall_2.update_state(y_true, y_pred, sample_weight)

    def result(self):
        # 1. Calculate F1 for Sell (F1_0)
        p_0 = self.precision_0.result()
        r_0 = self.recall_0.result()
        f1_0 = 2 * (p_0 * r_0) / (p_0 + r_0 + tf.keras.backend.epsilon())

        # 2. Calculate F1 for Buy (F1_2)
        p_2 = self.precision_2.result()
        r_2 = self.recall_2.result()
        f1_2 = 2 * (p_2 * r_2) / (p_2 + r_2 + tf.keras.backend.epsilon())

        # 3. Return the Macro F1 of the two trade classes
        return (f1_0 + f1_2) / 2

    def reset_states(self):
        # Reset the state for all four underlying metrics
        self.precision_0.reset_states()
        self.recall_0.reset_states()
        self.precision_2.reset_states()
        self.recall_2.reset_states()


# ----------------------------------------------------------

class Class2F1(tf.keras.metrics.Metric):
    """
    Calculates the average F1 score for class 2 only
    This is necessary because tf.keras.metrics.F1Score does not support class_id.
    """

    def __init__(self, name="class2_f1", **kwargs):
        super(Class2F1, self).__init__(name=name, **kwargs)

        # Initialize Precision and Recall for Class 2
        self.precision_2 = Precision(class_id=2, name="precision_2")
        self.recall_2 = Recall(class_id=2, name="recall_2")

    def update_state(self, y_true, y_pred, sample_weight=None):
        # Update state for all  underlying metrics
        self.precision_2.update_state(y_true, y_pred, sample_weight)
        self.recall_2.update_state(y_true, y_pred, sample_weight)

    def result(self):

        p_2 = self.precision_2.result()
        r_2 = self.recall_2.result()
        f1_2 = 2 * (p_2 * r_2) / (p_2 + r_2 + tf.keras.backend.epsilon())

        return f1_2

    def reset_states(self):
        # Reset the state for all  underlying metrics
        self.precision_2.reset_states()
        self.recall_2.reset_states()


# ---------------------------------------------------------


# Custom weighted F1 metric using class weights
@keras.saving.register_keras_serializable(package="ClassifierKeras")
class WeightedF1Score(keras.metrics.F1Score):
    def __init__(self, class_weights=None, name="weighted_f1_score", **kwargs):
        # Remove 'average' from kwargs if present to avoid duplicate argument
        kwargs.pop("average", None)
        super(WeightedF1Score, self).__init__(average="weighted", name=name, **kwargs)
        self.class_weights = class_weights

    def update_state(self, y_true, y_pred, sample_weight=None):
        # Apply class weights as sample weights if provided
        if self.class_weights is not None and sample_weight is None:
            # Convert one-hot to class indices for weighting
            y_true_classes = tf.argmax(y_true, axis=-1)
            sample_weight = tf.gather(self.class_weights, y_true_classes)

        super().update_state(y_true, y_pred, sample_weight)

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "class_weights": self.class_weights,
            }
        )
        return config


# ---------------------------
# WEIGHTED F1 METRIC
# ---------------------------
@keras.saving.register_keras_serializable()
class WeightedF1(tf.keras.metrics.Metric):
    def __init__(self, num_classes=4, class_weights=None, name="weighted_f1", **kwargs):
        super().__init__(name=name, **kwargs)
        self.num_classes = num_classes
        self.class_weights = tf.constant(
            class_weights if class_weights is not None else [1.0] * num_classes,
            dtype=tf.float32,
        )
        self.tp = self.add_weight(
            shape=(num_classes,), initializer="zeros", dtype=tf.float32
        )
        self.fp = self.add_weight(
            shape=(num_classes,), initializer="zeros", dtype=tf.float32
        )
        self.fn = self.add_weight(
            shape=(num_classes,), initializer="zeros", dtype=tf.float32
        )

    def update_state(self, y_true, y_pred, sample_weight=None):
        del sample_weight
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.argmax(y_pred, axis=-1)
        y_pred = tf.one_hot(y_pred, depth=self.num_classes, dtype=tf.float32)

        tp = tf.reduce_sum(y_true * y_pred, axis=0)
        fp = tf.reduce_sum((1 - y_true) * y_pred, axis=0)
        fn = tf.reduce_sum(y_true * (1 - y_pred), axis=0)

        self.tp.assign_add(tp)
        self.fp.assign_add(fp)
        self.fn.assign_add(fn)

    def result(self):
        precision = self.tp / (self.tp + self.fp + 1e-7)
        recall = self.tp / (self.tp + self.fn + 1e-7)
        f1 = 2 * precision * recall / (precision + recall + 1e-7)
        return tf.reduce_sum(f1 * self.class_weights) / tf.reduce_sum(
            self.class_weights
        )

    def reset_states(self):
        for var in [self.tp, self.fp, self.fn]:
            var.assign(tf.zeros_like(var))


# ---------------------------------------------------------

# Returns the F1 score for the specified class


@keras.saving.register_keras_serializable()
class PerClassF1(tf.keras.metrics.Metric):
    def __init__(self, num_classes, target_class=2, name="class_f1", **kwargs):
        super().__init__(name=name, **kwargs)
        self.num_classes = num_classes
        self.target_class = target_class  # Track specific class (default: class 2)
        self.true_positives = self.add_weight(
            name="true_positives", shape=(num_classes,), initializer="zeros"
        )
        self.false_positives = self.add_weight(
            name="false_positives", shape=(num_classes,), initializer="zeros"
        )
        self.false_negatives = self.add_weight(
            name="false_negatives", shape=(num_classes,), initializer="zeros"
        )

    def update_state(self, y_true, y_pred, sample_weight=None):
        del sample_weight

        # Handle y_true - convert to integer indices if it's one-hot encoded
        if len(y_true.shape) > 1:
            # If y_true is one-hot encoded, convert to sparse indices
            y_true = tf.argmax(y_true, axis=-1, output_type=tf.int32)
        else:
            # If y_true is already sparse, just cast to int32
            y_true = tf.cast(y_true, tf.int32)

        # Handle y_pred - convert to integer indices
        y_pred = tf.argmax(y_pred, axis=-1, output_type=tf.int32)

        # One-hot encode both with same depth
        y_true_onehot = tf.one_hot(y_true, depth=self.num_classes, dtype=tf.float32)
        y_pred_onehot = tf.one_hot(y_pred, depth=self.num_classes, dtype=tf.float32)

        # Counts
        tp = tf.reduce_sum(y_true_onehot * y_pred_onehot, axis=0)
        fp = tf.reduce_sum((1 - y_true_onehot) * y_pred_onehot, axis=0)
        fn = tf.reduce_sum(y_true_onehot * (1 - y_pred_onehot), axis=0)
        # # Counts
        # tp = tf.reduce_sum(y_true * y_pred, axis=0)
        # fp = tf.reduce_sum((1 - y_true) * y_pred, axis=0)
        # fn = tf.reduce_sum(y_true * (1 - y_pred), axis=0)

        # Update weights
        self.true_positives.assign_add(tf.cast(tp, self.dtype))
        self.false_positives.assign_add(tf.cast(fp, self.dtype))
        self.false_negatives.assign_add(tf.cast(fn, self.dtype))

    def result(self):
        precision = self.true_positives / (
            self.true_positives + self.false_positives + 1e-7
        )
        recall = self.true_positives / (
            self.true_positives + self.false_negatives + 1e-7
        )
        f1 = 2 * precision * recall / (precision + recall + 1e-7)
        # Return F1 score for the target class only (scalar)
        return f1[self.target_class]

    def reset_states(self):
        for var in self.variables:
            var.assign(tf.zeros_like(var))


# ---------------------------------------------------------


@keras.saving.register_keras_serializable()
def f1_micro_metric(y_true, y_pred):
    """Micro-averaged F1 metric for binary outputs."""
    y_true = tf.keras.backend.cast(y_true, "float32")
    y_pred = tf.keras.backend.cast(y_pred, "float32")

    y_pred = tf.keras.backend.round(y_pred)

    true_positives = tf.keras.backend.sum(y_true * y_pred)
    false_positives = tf.keras.backend.sum((1 - y_true) * y_pred)
    false_negatives = tf.keras.backend.sum(y_true * (1 - y_pred))

    precision = true_positives / (
        true_positives + false_positives + tf.keras.backend.epsilon()
    )
    recall = true_positives / (
        true_positives + false_negatives + tf.keras.backend.epsilon()
    )

    f1 = 2 * (precision * recall) / (
        precision + recall + tf.keras.backend.epsilon()
    )
    return f1


# ---------------------------------------------------------


# Matthews Correlation Coefficient (MCC) Metric
@keras.saving.register_keras_serializable()
class MCCMetric(keras.metrics.Metric):
    def __init__(self, name="mcc", **kwargs):
        super(MCCMetric, self).__init__(name=name, **kwargs)
        self.tp = self.add_weight(name="tp", initializer="zeros")
        self.tn = self.add_weight(name="tn", initializer="zeros")
        self.fp = self.add_weight(name="fp", initializer="zeros")
        self.fn = self.add_weight(name="fn", initializer="zeros")

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_true = tf.cast(y_true, tf.bool)
        y_pred = tf.cast(tf.greater_equal(y_pred, 0.5), tf.bool)

        self.tp.assign_add(
            tf.reduce_sum(tf.cast(tf.logical_and(y_true, y_pred), tf.float32))
        )
        self.tn.assign_add(
            tf.reduce_sum(
                tf.cast(
                    tf.logical_and(tf.logical_not(y_true), tf.logical_not(y_pred)),
                    tf.float32,
                )
            )
        )
        self.fp.assign_add(
            tf.reduce_sum(
                tf.cast(tf.logical_and(tf.logical_not(y_true), y_pred), tf.float32)
            )
        )
        self.fn.assign_add(
            tf.reduce_sum(
                tf.cast(tf.logical_and(y_true, tf.logical_not(y_pred)), tf.float32)
            )
        )

    def result(self):
        numerator = (self.tp * self.tn) - (self.fp * self.fn)
        denominator = tf.sqrt(
            (self.tp + self.fp)
            * (self.tp + self.fn)
            * (self.tn + self.fp)
            * (self.tn + self.fn)
        )
        return numerator / (denominator + tf.keras.backend.epsilon())

    def reset_state(self):
        self.tp.assign(0)
        self.tn.assign(0)
        self.fp.assign(0)
        self.fn.assign(0)


# ---------------------------------------------------------
# Multi-class MCC Metric
# ---------------------------------------------------------

@keras.saving.register_keras_serializable()
class MultiClassMCCMetric(keras.metrics.Metric):
    """
    Multi-class Matthews Correlation Coefficient (MCC) metric.
    Computes MCC for multi-class classification problems.
    """
    def __init__(self, num_classes=3, name="mcc", **kwargs):
        super(MultiClassMCCMetric, self).__init__(name=name, **kwargs)
        self.num_classes = num_classes
        # Store confusion matrix counts for each class
        self.confusion_matrix = self.add_weight(
            name="confusion_matrix",
            shape=(num_classes, num_classes),
            initializer="zeros",
            dtype=tf.float32
        )

    def update_state(self, y_true, y_pred, sample_weight=None):
        # Convert one-hot to class indices
        y_true = tf.argmax(y_true, axis=-1)
        y_pred = tf.argmax(y_pred, axis=-1)
        
        # Build confusion matrix
        cm = tf.math.confusion_matrix(
            y_true, y_pred, num_classes=self.num_classes, dtype=tf.float32
        )
        
        if sample_weight is not None:
            # Apply sample weights if provided
            cm = cm * tf.cast(sample_weight, tf.float32)
        
        self.confusion_matrix.assign_add(cm)

    def result(self):
        """
        Compute multi-class MCC using the formula:
        MCC = (c * s - sum_k p_k * t_k) / sqrt((s^2 - sum_k p_k^2) * (s^2 - sum_k t_k^2))
        where:
        - c = sum of diagonal elements (correct predictions)
        - s = sum of all elements (total samples)
        - p_k = sum of row k (predicted class k)
        - t_k = sum of column k (true class k)
        """
        cm = self.confusion_matrix
        
        # Total samples
        s = tf.reduce_sum(cm)
        
        # Sum of diagonal (correct predictions)
        c = tf.reduce_sum(tf.linalg.diag_part(cm))
        
        # Sum of each row (predicted class k)
        p_k = tf.reduce_sum(cm, axis=1)
        
        # Sum of each column (true class k)
        t_k = tf.reduce_sum(cm, axis=0)
        
        # Compute numerator: c * s - sum_k (p_k * t_k)
        numerator = c * s - tf.reduce_sum(p_k * t_k)
        
        # Compute denominator: sqrt((s^2 - sum_k p_k^2) * (s^2 - sum_k t_k^2))
        s_squared = s * s
        p_k_squared_sum = tf.reduce_sum(p_k * p_k)
        t_k_squared_sum = tf.reduce_sum(t_k * t_k)
        
        denominator = tf.sqrt(
            (s_squared - p_k_squared_sum) * (s_squared - t_k_squared_sum) + tf.keras.backend.epsilon()
        )
        
        mcc = numerator / denominator
        return mcc

    def reset_state(self):
        self.confusion_matrix.assign(tf.zeros_like(self.confusion_matrix))


# ---------------------------------------------------------
# Average MCC Callback
# ---------------------------------------------------------

@keras.saving.register_keras_serializable()
class AverageMCCCallback(keras.callbacks.Callback):
    """
    Callback that computes and logs the average MCC across all tasks.
    Adds 'val_avg_mcc' and 'avg_mcc' to the training history.
    """
    def __init__(self, task_names=None, **kwargs):
        super().__init__(**kwargs)
        self.task_names = task_names or ["trading", "regime", "risk", "momentum", "flow", "profit"]
    
    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        
        # Compute average training MCC
        train_mccs = []
        for task in self.task_names:
            mcc_key = f"{task}_mcc"
            if mcc_key in logs:
                train_mccs.append(float(logs[mcc_key]))
        
        if train_mccs:
            avg_train_mcc = sum(train_mccs) / len(train_mccs)
            logs["avg_mcc"] = avg_train_mcc
        
        # Compute average validation MCC
        val_mccs = []
        for task in self.task_names:
            val_mcc_key = f"val_{task}_mcc"
            if val_mcc_key in logs:
                val_mccs.append(float(logs[val_mcc_key]))
        
        if val_mccs:
            avg_val_mcc = sum(val_mccs) / len(val_mccs)
            logs["val_average_mcc"] = avg_val_mcc
            # Ensure it's a float that Keras can track
            if hasattr(logs, '__setitem__'):
                logs["val_average_mcc"] = float(avg_val_mcc)
    
    def get_config(self):
        """Returns the configuration of the callback."""
        config = super().get_config()
        config.update({
            "task_names": self.task_names
        })
        return config
    
    @classmethod
    def from_config(cls, config):
        """Creates a callback from its configuration."""
        task_names = config.pop("task_names", None)
        return cls(task_names=task_names, **config)
