import tensorflow as tf

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
        tp = tf.keras.backend.sum(tf.keras.backend.cast(y_true * y_pred, 'float'), axis=None)
        # calculate false positives
        fp = tf.keras.backend.sum(tf.keras.backend.cast((1 - y_true) * y_pred, 'float'), axis=None)
        # calculate false negatives
        fn = tf.keras.backend.sum(tf.keras.backend.cast(y_true * (1 - y_pred), 'float'), axis=None)
        # calculate precision
        p = tp / (tp + fp + tf.keras.backend.epsilon())
        # calculate recall
        r = tp / (tp + fn + tf.keras.backend.epsilon())
        # calculate micro f1 score
        f1 = 2 * p * r / (p + r + tf.keras.backend.epsilon())

        # multiply f1 score by self.class_weights
        weighted_f1 = f1 * self.class_weights

        # return the weighted micro f1 score
        return 1 - weighted_f1

    def f1_macro(self, y_true, y_pred):
        # calculate true positives
        tp = tf.keras.backend.sum(tf.keras.backend.cast(y_true * y_pred, 'float'), axis=0)
        # calculate false positives
        fp = tf.keras.backend.sum(tf.keras.backend.cast((1 - y_true) * y_pred, 'float'), axis=0)
        # calculate false negatives
        fn = tf.keras.backend.sum(tf.keras.backend.cast(y_true * (1 - y_pred), 'float'), axis=0)
        # calculate precision
        p = tp / (tp + fp + tf.keras.backend.epsilon())
        # calculate recall
        r = tp / (tp + fn + tf.keras.backend.epsilon())
        # calculate macro f1 score
        f1 = 2 * p * r / (p + r + tf.keras.backend.epsilon())

        # multiply f1 score by self.class_weights
        weighted_f1 = f1 * self.class_weights

        # return the average macro f1 score across all classes
        return tf.keras.backend.mean(weighted_f1)

    def f1_macro_loss(self, y_true, y_pred):
        return 1 - self.f1_macro(y_true, y_pred)

    # this version should be differentiable
    def f1_macro_diff(self, y_true, y_pred):

        # compatible with tf <=2.11
        # calculate true positives
        tp = tf.keras.backend.sum(tf.keras.backend.cast(y_true * y_pred, 'float'), axis=0)
        # calculate false positives
        fp = tf.keras.backend.sum(tf.keras.backend.cast((1 - y_true) * y_pred, 'float'), axis=0)
        # calculate false negatives
        fn = tf.keras.backend.sum(tf.keras.backend.cast(y_true * (1 - y_pred), 'float'), axis=0)
        # calculate precision
        p = tp / (tp + fp + tf.keras.backend.epsilon())
        # calculate recall
        r = tp / (tp + fn + tf.keras.backend.epsilon())

        # Calculate the weighted precision and recall
        w_p = p * self.class_weights
        w_r = r * self.class_weights

        # calculate macro f1 score
        f1 = 2 * w_p * w_r / (w_p + w_r + tf.keras.backend.epsilon())

        f1_mean = tf.keras.backend.mean(f1)

        # loss = tf.keras.losses.binary_crossentropy(y_true, y_pred) * (1 - w_p) + w_r
        loss = tf.keras.losses.binary_crossentropy(y_true, y_pred) * f1_mean

        # return loss
        return 1 - f1_mean

    def f1_weighted_loss(self, y_true, y_pred):
        ground_positives = tf.keras.backend.sum(y_true, axis=0) + tf.keras.backend.epsilon()  # = TP + FN
        pred_positives = tf.keras.backend.sum(y_pred, axis=0) + tf.keras.backend.epsilon()  # = TP + FP
        true_positives = tf.keras.backend.sum(y_true * y_pred, axis=0) + tf.keras.backend.epsilon()  # = TP

        precision = true_positives / pred_positives
        recall = true_positives / ground_positives
        # both = 1 if ground_positives == 0 or pred_positives == 0

        f1 = 2 * (precision * recall) / (precision + recall + tf.keras.backend.epsilon())

        weighted_f1 = f1 * ground_positives / tf.keras.backend.sum(ground_positives)
        weighted_f1 = tf.keras.backend.sum(weighted_f1)

        return 1 - weighted_f1

    def f1_beta_loss(self, y_true, y_pred):
        # beta>1 adds weight to recall. beta<1 adds weight to precision
        beta = 2.0

        # convert labels to one-hot vectors
        y_true = tf.keras.backend.one_hot(tf.keras.backend.cast(y_true, 'int32'), num_classes=tf.keras.backend.int_shape(y_pred)[-1])
        # clip predictions to avoid log(0)
        y_pred = tf.keras.backend.clip(y_pred, tf.keras.backend.epsilon(), 1 - tf.keras.backend.epsilon())
        # calculate true positives, false positives and false negatives
        tp = tf.keras.backend.sum(y_true * y_pred, axis=-1)
        fp = tf.keras.backend.sum((1 - y_true) * y_pred, axis=-1)
        fn = tf.keras.backend.sum(y_true * (1 - y_pred), axis=-1)
        # calculate precision and recall
        p = tp / (tp + fp + tf.keras.backend.epsilon())
        r = tp / (tp + fn + tf.keras.backend.epsilon())
        # calculate F1 beta score
        bb = beta * beta
        fbeta_score = (1 + bb) * (p * r) / (bb * p + r + tf.keras.backend.epsilon())
        # return the negative F1 beta score as the loss
        return 1 - fbeta_score

    def weighted_categorical_loss(self, y_true, y_pred):

        # Clip the prediction value to prevent NaN's and Inf's
        epsilon = tf.keras.backend.epsilon()
        y_pred = tf.keras.backend.clip(y_pred, epsilon, 1. - epsilon)

        # Calculate Cross Entropy
        ce_loss = -y_true * tf.keras.backend.log(y_pred)

        # # Calculate cross-entropy loss
        # ce_loss = tf.keras.backend.categorical_crossentropy(y_true, y_pred, from_logits=False)

        # Apply class self.class_weights
        weighted_ce_loss = ce_loss * self.class_weights

        return weighted_ce_loss

    def categorical_focal_loss(self, y_true, y_pred):

        gamma = 2.0

        # Clip the prediction value to prevent NaN's and Inf's
        epsilon = tf.keras.backend.epsilon()
        y_pred = tf.keras.backend.clip(y_pred, epsilon, 1. - epsilon)

        # Calculate Cross Entropy
        cross_entropy = -y_true * tf.keras.backend.log(y_pred)

        # Calculate Focal Loss
        loss = self.class_weights * tf.keras.backend.pow(1 - y_pred, gamma) * cross_entropy

        # Compute mean loss in mini_batch
        return tf.keras.backend.mean(tf.keras.backend.sum(loss, axis=-1))

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
