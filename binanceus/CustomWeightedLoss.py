import tensorflow as tf
import keras
import keras.backend as K
import numpy as np
from enum import Enum, auto





class CustomWeightedLoss(tf.keras.losses.Loss):

    # enum of available custom loss functions
    class WeightedLossType(Enum):
        WEIGHTED_CATEGORICAL = auto()
        CATEGORICAL_FOCAL = auto()
        F1_MACRO = auto()
        F1_MICRO = auto()
        F1_BETA = auto()

    def __init__(self, loss_type, class_weights, name="CustomWeightedLoss"):
        super().__init__(name=name)
        self.loss_type = loss_type
        self.class_weights = class_weights

    def get_config(self):
        # Return the configuration of the loss class
        return {"loss_type": self.loss_type.value, "class_weights": self.class_weights, "name": self.name}

    #-------------------------------
    # different custom loss variants
    def f1_micro_loss(self, y_true, y_pred):
        # calculate true positives
        tp = K.sum(K.cast(y_true * y_pred, 'float'), axis=None)
        # calculate false positives
        fp = K.sum(K.cast((1 - y_true) * y_pred, 'float'), axis=None)
        # calculate false negatives
        fn = K.sum(K.cast(y_true * (1 - y_pred), 'float'), axis=None)
        # calculate precision
        p = tp / (tp + fp + K.epsilon())
        # calculate recall
        r = tp / (tp + fn + K.epsilon())
        # calculate micro f1 score
        f1 = 2 * p * r / (p + r + K.epsilon())

        # multiply f1 score by self.class_weights
        weighted_f1 = f1 * self.class_weights

        # return the weighted micro f1 score
        return 1 - weighted_f1

    def f1_macro(self, y_true, y_pred):
        # calculate true positives
        tp = K.sum(K.cast(y_true * y_pred, 'float'), axis=0)
        # calculate false positives
        fp = K.sum(K.cast((1 - y_true) * y_pred, 'float'), axis=0)
        # calculate false negatives
        fn = K.sum(K.cast(y_true * (1 - y_pred), 'float'), axis=0)
        # calculate precision
        p = tp / (tp + fp + K.epsilon())
        # calculate recall
        r = tp / (tp + fn + K.epsilon())
        # calculate macro f1 score
        f1 = 2 * p * r / (p + r + K.epsilon())

        # multiply f1 score by self.class_weights
        weighted_f1 = f1 * self.class_weights

        # return the average macro f1 score across all classes
        return K.mean(weighted_f1)

    def f1_macro_loss(self, y_true, y_pred):
        return 1 - self.f1_macro(y_true, y_pred)

    # this version should be differentiable
    def f1_macro_diff(self, y_true, y_pred):

        # compatible with tf <=2.11
        # calculate true positives
        tp = K.sum(K.cast(y_true * y_pred, 'float'), axis=0)
        # calculate false positives
        fp = K.sum(K.cast((1 - y_true) * y_pred, 'float'), axis=0)
        # calculate false negatives
        fn = K.sum(K.cast(y_true * (1 - y_pred), 'float'), axis=0)
        # calculate precision
        p = tp / (tp + fp + K.epsilon())
        # calculate recall
        r = tp / (tp + fn + K.epsilon())

        # Calculate the weighted precision and recall
        w_p = p * self.class_weights
        w_r = r * self.class_weights

        # calculate macro f1 score
        f1 = 2 * w_p * w_r / (w_p + w_r + K.epsilon())

        f1_mean = K.mean(f1)

        # loss = tf.keras.losses.binary_crossentropy(y_true, y_pred) * (1 - w_p) + w_r
        loss = tf.keras.losses.binary_crossentropy(y_true, y_pred) * f1_mean

        # return loss
        return 1 - f1_mean

    def f1_weighted_loss(self, y_true, y_pred):
        ground_positives = K.sum(y_true, axis=0) + K.epsilon()  # = TP + FN
        pred_positives = K.sum(y_pred, axis=0) + K.epsilon()  # = TP + FP
        true_positives = K.sum(y_true * y_pred, axis=0) + K.epsilon()  # = TP

        precision = true_positives / pred_positives
        recall = true_positives / ground_positives
        # both = 1 if ground_positives == 0 or pred_positives == 0

        f1 = 2 * (precision * recall) / (precision + recall + K.epsilon())

        weighted_f1 = f1 * ground_positives / K.sum(ground_positives)
        weighted_f1 = K.sum(weighted_f1)

        return 1 - weighted_f1

    def f1_beta_loss(self, y_true, y_pred):
        # beta>1 adds weight to recall. beta<1 adds weight to precision
        beta = 2.0

        # convert labels to one-hot vectors
        y_true = K.one_hot(K.cast(y_true, 'int32'), num_classes=K.int_shape(y_pred)[-1])
        # clip predictions to avoid log(0)
        y_pred = K.clip(y_pred, K.epsilon(), 1 - K.epsilon())
        # calculate true positives, false positives and false negatives
        tp = K.sum(y_true * y_pred, axis=-1)
        fp = K.sum((1 - y_true) * y_pred, axis=-1)
        fn = K.sum(y_true * (1 - y_pred), axis=-1)
        # calculate precision and recall
        p = tp / (tp + fp + K.epsilon())
        r = tp / (tp + fn + K.epsilon())
        # calculate F1 beta score
        bb = beta * beta
        fbeta_score = (1 + bb) * (p * r) / (bb * p + r + K.epsilon())
        # return the negative F1 beta score as the loss
        return 1 - fbeta_score

    def weighted_categorical_loss(self, y_true, y_pred):

        # Clip the prediction value to prevent NaN's and Inf's
        epsilon = K.epsilon()
        y_pred = K.clip(y_pred, epsilon, 1. - epsilon)

        # Calculate Cross Entropy
        ce_loss = -y_true * K.log(y_pred)

        # # Calculate cross-entropy loss
        # ce_loss = K.categorical_crossentropy(y_true, y_pred, from_logits=False)

        # Apply class self.class_weights
        weighted_ce_loss = ce_loss * self.class_weights

        return weighted_ce_loss

    def categorical_focal_loss(self, y_true, y_pred):

        gamma = 2.0

        # Clip the prediction value to prevent NaN's and Inf's
        epsilon = K.epsilon()
        y_pred = K.clip(y_pred, epsilon, 1. - epsilon)

        # Calculate Cross Entropy
        cross_entropy = -y_true * K.log(y_pred)

        # Calculate Focal Loss
        loss = self.class_weights * K.pow(1 - y_pred, gamma) * cross_entropy

        # Compute mean loss in mini_batch
        return K.mean(K.sum(loss, axis=-1))

    # ---------------------


    def call(self, y_true, y_pred):
        if self.loss_type == CustomWeightedLoss.WeightedLossType.WEIGHTED_CATEGORICAL:
            return self.weighted_categorical_loss(y_true, y_pred)
        elif self.loss_type == CustomWeightedLoss.WeightedLossType.CATEGORICAL_FOCAL:
            return self.categorical_focal_loss(y_true, y_pred)
        elif self.loss_type == CustomWeightedLoss.WeightedLossType.F1_MACRO:
            return self.f1_macro_loss(y_true, y_pred)
        elif self.loss_type == CustomWeightedLoss.WeightedLossType.F1_MICRO:
            return self.f1_micro_loss(y_true, y_pred)
        elif self.loss_type == CustomWeightedLoss.WeightedLossType.F1_BETA:
            return self.f1_beta_loss(y_true, y_pred)
        else:
            print(f"    Unknown loss type: {self.loss_type}")

        return None
