# base class that implements Neural Network Binary Classifier.
# This class implements a keras classifier that provides a binary result (i.e. 0/1)

# subclasses should override the create_model() method


import numpy as np
import pandas as pd

pd.options.mode.chained_assignment = None  # default='warn'

# Strategy specific imports, files must reside in same folder as strategy
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent))

import logging
import warnings

log = logging.getLogger(__name__)
# log.setLevel(logging.DEBUG)
warnings.simplefilter(action="ignore", category=pd.errors.PerformanceWarning)

warnings.filterwarnings(
    "ignore", message="The objective has been evaluated at this point before."
)

warnings.filterwarnings(
    "ignore", category=UserWarning, module="keras.src.saving.saving_lib"
)

import random

import os
from sklearn.utils.class_weight import compute_class_weight

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "1"
os.environ["TF_DETERMINISTIC_OPS"] = "1"

import tensorflow as tf

# import tensorflow_addons as tfa  # Commented out due to compatibility issues

seed = 42
os.environ["PYTHONHASHSEED"] = str(seed)
random.seed(seed)
tf.random.set_seed(seed)
np.random.seed(seed)

tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.WARN)

# import keras
from keras import layers


from Predictors.KerasBaseClassifier import KerasBaseClassifier
from utils.CustomLoss import focal_loss_sigmoid, focal_loss_softmax, dice_loss
from utils.CustomMetric import f1_micro_metric

# Standalone loss functions and metrics are imported from shared modules


class KerasClassifierBinary(KerasBaseClassifier):
    clean_data_required = False

    # Loss function selection - moved to class level
    use_focal_loss = False  # Set to True for focal loss, False for BCE (switched to BCE due to training issues)
    use_dice_loss = False  # Set to True for dice loss (alternative to focal)

    # def enable_focal_loss(self):
    #     self.use_focal_loss = True
    #     self.use_dice_loss = False
    #     return

    # create model - subclasses should overide this
    def create_model(self, seq_len, num_features):

        model = None

        print("    WARNING: create_model() should be defined by the subclass")

        # create a more robust model for binary classification
        model = tf.keras.Sequential(name=self.name)

        # NOTE: don't use relu with LSTMs, cannot use GPU if you do (much slower). Use tanh

        # First LSTM layer - increased capacity
        model.add(
            layers.LSTM(
                256,
                return_sequences=True,
                activation="tanh",
                input_shape=(seq_len, num_features),
            )
        )
        model.add(layers.Dropout(rate=0.3))

        # Second LSTM layer - increased capacity
        model.add(layers.LSTM(128, return_sequences=True, activation="tanh"))
        model.add(layers.Dropout(rate=0.3))

        # Third LSTM layer
        model.add(layers.LSTM(64, return_sequences=False, activation="tanh"))
        model.add(layers.Dropout(rate=0.3))

        # Dense layers for feature extraction - increased capacity
        model.add(layers.Dense(64, activation="relu"))
        model.add(layers.Dropout(rate=0.2))

        model.add(layers.Dense(32, activation="relu"))
        model.add(layers.Dropout(rate=0.1))

        # last layer is a binary decision - do not change
        # Use proper initialization to avoid all-zero predictions
        model.add(
            layers.Dense(
                1,
                activation="sigmoid",
                kernel_initializer="glorot_normal",
                bias_initializer="zeros",
            )
        )  # Neutral bias since we're using class weights

        return model

    def load(self, path=""):
        """
        Override load method to calculate optimal threshold when model is loaded
        """
        # Call parent load method
        model = super().load(path)

        # If model was loaded successfully, get optimal threshold
        if model is not None:
            try:
                # DEBUG: Print model loading info
                print("    🔍 DEBUG: Model loading:")
                print(f"      Model type: {type(model)}")
                print(
                    f"      Model name: {model.name if hasattr(model, 'name') else 'unknown'}"
                )
                print(
                    f"      Model layers: {len(model.layers) if hasattr(model, 'layers') else 'unknown'}"
                )

                # Try to get threshold from model attributes
                if hasattr(model, "optimal_threshold"):
                    self.optimal_threshold = model.optimal_threshold
                    print(
                        f"    Loaded optimal threshold from model: {self.optimal_threshold:.3f}"
                    )
                else:
                    # Use default threshold for imbalanced data
                    self.optimal_threshold = 0.25  # Default for imbalanced data
                    print(
                        f"    Using default optimal threshold: {self.optimal_threshold:.3f}"
                    )
                    print(
                        "    Note: For best results, retrain model to calculate data-specific threshold"
                    )
            except Exception as e:
                print(f"    Warning: Could not load optimal threshold: {e}")
                self.optimal_threshold = 0.5  # Fallback to default

        return model

    def compile_model(self, model):

        # Use lower learning rate for focal loss to prevent gradient issues
        if self.use_focal_loss:
            learning_rate = 0.001  # Lower learning rate for focal loss
            print(f"    Using lower learning rate ({learning_rate}) for focal loss")
        else:
            learning_rate = 0.005  # Lower learning rate for BCE to prevent collapse
            print(
                f"    Using lower learning rate ({learning_rate}) for BCE to prevent training collapse"
            )

        optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate, clipnorm=1.0)

        # Check if the model has 1 or 2 output neurons to determine the appropriate loss
        output_shape = model.output_shape[-1]

        if output_shape == 1:
            # Binary classification with sigmoid activation
            if self.use_focal_loss:
                model.compile(
                    optimizer=optimizer,
                    loss=focal_loss_sigmoid,
                    metrics=["accuracy", "precision", "recall", f1_micro_metric],
                )
                print("    🎯 Using focal loss (better for imbalanced data)")
            elif self.use_dice_loss:
                model.compile(
                    optimizer=optimizer,
                    loss=dice_loss,
                    metrics=["accuracy", "precision", "recall", f1_micro_metric],
                )
                print("    🎯 Using dice loss (alternative for imbalanced data)")
            else:
                model.compile(
                    optimizer=optimizer,
                    loss="binary_crossentropy",
                    metrics=["accuracy", "precision", "recall", f1_micro_metric],
                )
                print("    🎯 Using binary crossentropy (standard loss)")
        elif output_shape == 2:
            # 2-class classification with softmax activation
            if self.use_focal_loss:
                model.compile(
                    optimizer=optimizer,
                    loss=focal_loss_softmax,
                    metrics=["accuracy", "precision", "recall", f1_micro_metric],
                )
                print("    🎯 Using focal loss (better for imbalanced data)")
            else:
                model.compile(
                    optimizer=optimizer,
                    loss="categorical_crossentropy",
                    metrics=["accuracy", "precision", "recall", f1_micro_metric],
                )
                print("    🎯 Using categorical crossentropy (standard loss)")
        else:
            print(f"    WARN: Unexpected output shape: {output_shape}")
            model.compile(
                optimizer=optimizer,
                loss="binary_crossentropy",
                metrics=["accuracy", "precision", "recall", f1_micro_metric],
            )

        return model

    # update training using the suplied (normalised) dataframe. Training is cumulative
    # the 'labels' args should contain 0.0 for normal results, '1.0' for anomalies (buy or sell)
    def train(
        self,
        df_train_norm,
        df_test_norm,
        train_results,
        test_results,
        force_train=False,
    ):

        # lazy loading because params can change up to this point
        if self.model is None:
            # load saved model if present
            self.model = self.load()

        # print(f'is_trained:{self.is_trained} force_train:{force_train}')

        # if model is already trained, and caller is not requesting a re-train, then just return
        if (
            (self.model is not None)
            and self.model_is_trained()
            and (not force_train)
            and (not self.new_model_created())
        ):
            # print(f"    Not training. is_trained:{self.is_trained} force_train:{force_train} new_model:{self.new_model}")
            print("    Model is already trained")
            return

        if self.model is None:
            self.model = self.create_model(self.seq_len, self.num_features)
            if self.model is None:
                print("    ERR: model not created")
                return
            self.model = self.compile_model(self.model)
            self.model.summary()

            # Print model output information
            output_shape = self.model.output_shape
            print(f"    Model output shape: {output_shape}")
            print(f"    Model output type: {type(self.model.output)}")

        if self.dataframeUtils.is_dataframe(df_train_norm):
            # remove rows with positive labels?!
            if self.clean_data_required:
                df1 = df_train_norm.copy()
                df1["%labels"] = train_results
                df1 = df1[(df1["%labels"] < 0.1)]
                df_train = df1.drop("%labels", axis=1)

                df2 = df_train_norm.copy()
                df2["%labels"] = train_results
                df2 = df2[(df2["%labels"] < 0.1)]
                df_test = df2.drop("%labels", axis=1)
            else:
                df_train = df_train_norm.copy()
                df_test = df_test_norm.copy()

            train_tensor = self.dataframeUtils.df_to_tensor(
                df_train, self.seq_len, method=1
            )
            test_tensor = self.dataframeUtils.df_to_tensor(
                df_test, self.seq_len, method=1
            )
        else:
            # already in tensor format
            train_tensor = df_train_norm.copy()
            test_tensor = df_test_norm.copy()

        monitor_field = "val_f1_micro_metric"  # Monitor F1 score instead of loss
        monitor_mode = "max"  # Maximize F1 score
        # monitor_field = 'val_loss'  # Monitor F1 score instead of loss
        # monitor_mode = "min"  # Maximize F1 score

        min_delta = 0.0001
        early_patience = 8  # Reduced patience to catch collapse earlier
        plateau_patience = 4  # Reduced patience for learning rate reduction

        # callback to control early exit on plateau of results
        early_callback = tf.keras.callbacks.EarlyStopping(
            monitor=monitor_field,
            mode=monitor_mode,
            patience=early_patience,
            min_delta=min_delta,
            restore_best_weights=True,
            verbose=1,
        )

        plateau_callback = tf.keras.callbacks.ReduceLROnPlateau(
            monitor=monitor_field,
            mode=monitor_mode,
            factor=0.1,
            min_delta=min_delta,
            patience=plateau_patience,
            verbose=1,
        )

        # callback to control saving of 'best' model
        # Note that we use validation loss as the metric, not training loss
        self.checkpoint_path = self.get_checkpoint_path()
        checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
            filepath=self.checkpoint_path,
            save_weights_only=True,
            monitor=monitor_field,
            mode=monitor_mode,
            save_best_only=True,
            verbose=1,
        )

        # Add custom callback to monitor prediction variance and detect all-zero predictions
        class PredictionVarianceCallback(tf.keras.callbacks.Callback):
            def __init__(self, classifier_instance):
                super().__init__()
                self.classifier = classifier_instance
                self.zero_prediction_count = 0
                self.fallback_to_bce = False
                self.low_f1_epochs = 0  # Track consecutive epochs with low F1

            def on_epoch_end(self, epoch, logs=None):
                if epoch % 3 == 0:  # Check more frequently
                    # Get predictions on a small sample
                    sample_size = min(1000, len(train_tensor))
                    sample_preds = self.model.predict(
                        train_tensor[:sample_size], verbose=0
                    )
                    # Normalize raw predictions to 1D
                    try:
                        # Use the normalize method from the classifier instance
                        sample_preds = self.classifier._normalize_raw_predictions(
                            sample_preds
                        )
                    except Exception as e:
                        print(f"      ERR: Failed to normalize sample predictions: {e}")
                        # Fallback normalization
                        if sample_preds.shape[-1] == 1:
                            sample_preds = sample_preds.reshape(-1)
                        elif sample_preds.shape[-1] == 2:
                            sample_preds = sample_preds[:, 1]
                        else:
                            sample_preds = sample_preds.reshape(-1)

                    pred_std = np.std(sample_preds)
                    pred_mean = np.mean(sample_preds)
                    pred_min = np.min(sample_preds)
                    pred_max = np.max(sample_preds)

                    # Count predictions close to zero
                    zero_threshold = 0.01
                    zero_count = np.sum(sample_preds < zero_threshold)
                    zero_percentage = (zero_count / len(sample_preds)) * 100

                    print(
                        f"      Epoch {epoch}: pred_mean={pred_mean:.4f}, pred_std={pred_std:.6f}, range=[{pred_min:.4f}, {pred_max:.4f}]"
                    )
                    print(
                        f"      Zero predictions (<{zero_threshold}): {zero_count}/{len(sample_preds)} ({zero_percentage:.1f}%)"
                    )

                    # Detect all-zero predictions
                    if pred_std < 1e-4 or zero_percentage > 95:
                        self.zero_prediction_count += 1
                        print(
                            f"      ⚠️  WARNING: Low prediction variance or all-zero predictions at epoch {epoch}"
                        )
                        print(f"      🔍 DEBUG: All predictions are {pred_mean:.4f}")
                        print(
                            f"      🚨 Zero prediction count: {self.zero_prediction_count}"
                        )

                        # Check if gradients are zero
                        if hasattr(self.model, "layers"):
                            for i, layer in enumerate(self.model.layers):
                                if (
                                    hasattr(layer, "kernel")
                                    and layer.kernel is not None
                                ):
                                    grad_norm = tf.norm(layer.kernel)
                                    print(
                                        f"      Layer {i} ({layer.name}) kernel norm: {grad_norm:.6f}"
                                    )

                    # Check for low F1 score (indicates focal loss issues)
                    if logs and "f1_micro_metric" in logs:
                        f1_score = logs["f1_micro_metric"]
                        if f1_score < 0.01:  # Very low F1 score
                            self.low_f1_epochs += 1
                            print(
                                f"      ⚠️  WARNING: Low F1 score detected: {f1_score:.6f} (epoch {epoch})"
                            )
                        else:
                            self.low_f1_epochs = 0

                    # Suggest fallback to BCE if focal loss is failing
                    if self.low_f1_epochs >= 3 and not self.fallback_to_bce:
                        print(
                            f"      💡 SUGGESTION: Consider switching to BCE loss (low F1 for {self.low_f1_epochs} epochs)"
                        )
                        print(
                            "      🔧 To fix: Set use_focal_loss = False in KerasClassifierBinary"
                        )
                        self.fallback_to_bce = True

                    # Early stopping if model is completely dead
                    if (pred_std < 1e-6 or zero_percentage > 99) and epoch > 3:
                        print(
                            f"      🛑 STOPPING: Model is completely dead at epoch {epoch}"
                        )
                        print(
                            "      💡 Suggestion: Try reducing gamma, increasing alpha, or using BCE loss"
                        )
                        self.model.stop_training = True

                    # Also stop if F1 score is consistently very low
                    if logs and "f1_micro_metric" in logs:
                        f1_score = logs["f1_micro_metric"]
                        if (
                            f1_score < 0.001 and epoch > 5
                        ):  # Very low F1 score for multiple epochs
                            print(
                                f"      🛑 STOPPING: Consistently low F1 score ({f1_score:.6f}) at epoch {epoch}"
                            )
                            print(
                                "      💡 Suggestion: Check class balance and consider focal loss"
                            )
                            self.model.stop_training = True

        variance_callback = PredictionVarianceCallback(self)

        # Calculate class weights to handle imbalance

        # Use original class indices if available (for proper class weight calculation)
        if (
            hasattr(self, "original_train_class_indices")
            and self.original_train_class_indices is not None
        ):
            print("Using original class indices for class weight calculation")
            train_class_indices = self.original_train_class_indices
            unique, counts = np.unique(train_class_indices, return_counts=True)
            print(f"Original class indices - unique: {unique}, counts: {counts}")

            # Use class weights for BCE to handle imbalance
            if self.use_focal_loss:
                print("Using focal loss - no class weights needed")
                class_weights = None
            else:
                # Calculate balanced class weights but cap them to prevent overfitting
                class_weights_arr = compute_class_weight(
                    "balanced", classes=unique, y=train_class_indices
                )
                print(f"    Class weights uncapped: {class_weights_arr}")

                # Allow much stronger class weights for better handling of severe imbalance
                max_weight = min(
                    max(class_weights_arr) / 2.0, 30.0
                )  # Cap to prevent overfitting
                min_weight = min(class_weights_arr)  # Lower weights for majority class

                if max_weight is not None:
                    class_weights_arr = np.clip(
                        class_weights_arr, min_weight, max_weight
                    )
                    print(
                        f"    Class weights capped: min={min_weight}, max={max_weight}"
                    )
                else:
                    print("    Class weights uncapped (raw sklearn values)")

                class_weights = dict(zip(unique, class_weights_arr))
                print(f"Class weights (sklearn, uncapped): {class_weights}")
            print(f"Class distribution: {dict(zip(unique, counts))}")
        else:
            # Fallback to using the provided train_results
            unique, counts = np.unique(train_results, return_counts=True)
            print(f"Unique labels: {unique}, Counts: {counts}")
            print("train_results shape:", np.shape(train_results))
            print("train_results sample:", train_results[:20])
            print(
                "train_results unique/counts:",
                np.unique(train_results, return_counts=True),
            )
            if len(unique) == 2:
                # Handle one-hot encoded labels
                if (
                    hasattr(train_results, "ndim")
                    and train_results.ndim > 1
                    and train_results.shape[1] == 2
                ):
                    labels_flat = np.argmax(train_results, axis=1)
                    print(
                        "Detected one-hot encoded labels. Converted to class indices for class weight calculation."
                    )
                else:
                    labels_flat = np.asarray(train_results).ravel()
                class_weights_arr = compute_class_weight(
                    "balanced", classes=unique, y=labels_flat
                )
                class_weights = dict(zip(unique, class_weights_arr))
                print(f"Class weights (sklearn): {class_weights}")
                print(f"Class distribution: {dict(zip(unique, counts))}")

        callbacks = [
            plateau_callback,
            early_callback,
            checkpoint_callback,
            variance_callback,
        ]

        # if self.dbg_verbose:
        print("")
        print("    training model: {}...".format(self.name))

        # print("    train_tensor:{} test_tensor:{}".format(np.shape(train_tensor), np.shape(test_tensor)))

        # Model weights are saved at the end of every epoch, if it's the best seen so far.
        fhis = self.model.fit(
            train_tensor,
            train_results,
            batch_size=self.batch_size,
            epochs=self.num_epochs,
            callbacks=callbacks,
            validation_data=(test_tensor, test_results),
            class_weight=class_weights,
            verbose=1,
        )

        # Print training history for debugging
        if hasattr(fhis, "history"):
            print(f"    Final training loss: {fhis.history['loss'][-1]:.4f}")
            print(f"    Final validation loss: {fhis.history['val_loss'][-1]:.4f}")
            if "accuracy" in fhis.history:
                print(
                    f"    Final training accuracy: {fhis.history['accuracy'][-1]:.4f}"
                )
                print(
                    f"    Final validation accuracy: {fhis.history['val_accuracy'][-1]:.4f}"
                )

            # Print prediction distribution after training
            print("\n    Testing model predictions on training data...")
            train_preds = self.predict(train_tensor)
            test_preds = self.predict(test_tensor)

            print(
                f"    Training predictions: {np.unique(train_preds, return_counts=True)}"
            )
            print(f"    Test predictions: {np.unique(test_preds, return_counts=True)}")

            # Debug output is now included in predict() method

            # Find optimal threshold for better predictions
            print("\n    Finding optimal threshold...")
            optimal_threshold = self.find_optimal_threshold(test_tensor, test_results)
            print(f"    Optimal threshold: {optimal_threshold:.3f}")

            # Test predictions with optimal threshold
            optimal_predictions = self.predict_with_threshold(
                test_tensor, optimal_threshold
            )
            print(
                f"    Optimal predictions - unique/counts: {np.unique(optimal_predictions, return_counts=True)}"
            )

        # # The model weights (that are considered the best) are loaded into th model.
        # self.update_model_weights()

        self.save()
        self.is_trained = True

        return

    def predict(self, data):

        # lazy loading because params can change up to this point
        if self.model is None:
            # load saved model if present
            self.model = self.load()

        if self.dataframeUtils.is_dataframe(data):
            # convert dataframe to tensor
            df_tensor = self.dataframeUtils.df_to_tensor(data, self.seq_len, method=1)
        else:
            df_tensor = data

        if self.model == None:
            print("    ERR: no model for predictions")
            predictions = np.zeros(np.shape(df_tensor)[0], dtype=float)
            return predictions

        # run the prediction
        preds = self.model.predict(df_tensor, verbose=0)
        preds = np.array(preds)

        # self.analyze_raw_predictions(preds)

        # Calculate dynamic threshold based on current predictions
        dynamic_threshold = self.calculate_dynamic_threshold(preds)

        # Store the dynamic threshold for this prediction
        self.current_dynamic_threshold = dynamic_threshold

        # Normalize raw predictions to 1D
        try:
            preds = self._normalize_raw_predictions(preds)
        except Exception as e:
            print(f"    ERR: Failed to normalize predictions: {e}")
            return np.zeros(df_tensor.shape[0], dtype=float)

        # Return raw predictions (0.0-1.0 range)
        return preds

    def _normalize_raw_predictions(self, raw_predictions):
        """
        Normalize raw predictions to 1D array regardless of output shape

        Args:
            raw_predictions: Raw model outputs (could be 1D or 2D)

        Returns:
            Normalized 1D predictions (0.0-1.0 range)
        """
        # Handle both binary (sigmoid) and 2-class (softmax) outputs
        if raw_predictions.shape[-1] == 1:
            # Binary classification with sigmoid activation
            return raw_predictions.reshape(-1)
        elif raw_predictions.shape[-1] == 2:
            # 2-class classification with softmax activation
            # Take the probability of class 1 (positive class)
            return raw_predictions[:, 1]  # Probability of positive class
        else:
            # Already 1D or unexpected shape, return as is
            return raw_predictions.reshape(-1)

    def raw_predictions_to_signals(self, raw_predictions, threshold=0.5):
        """
        Convert raw predictions (0.0-1.0) to binary signals (0 or 1)

        Args:
            raw_predictions: Raw model outputs (0.0-1.0 range)
            threshold: Threshold for converting to binary signals (default: 0.5)

        Returns:
            Binary signals (0 or 1)
        """
        # Normalize raw predictions to 1D
        normalized_preds = self._normalize_raw_predictions(raw_predictions)

        # Convert to binary signals
        signals = np.where(normalized_preds > threshold, 1.0, 0.0)
        return signals

    def analyze_raw_predictions(self, raw_predictions):
        """
        Analyze raw predictions and provide detailed statistics

        Args:
            raw_predictions: Raw model outputs (0.0-1.0 range)

        Returns:
            Dictionary with analysis results
        """
        # Normalize raw predictions to 1D
        raw_predictions = self._normalize_raw_predictions(raw_predictions)

        # Basic statistics
        total_count = len(raw_predictions)
        min_val = raw_predictions.min()
        max_val = raw_predictions.max()
        mean_val = raw_predictions.mean()
        std_val = raw_predictions.std()
        median_val = np.median(raw_predictions)

        # Non-zero statistics
        non_zero_preds = raw_predictions[raw_predictions > 0]
        non_zero_count = len(non_zero_preds)
        non_zero_mean = non_zero_preds.mean() if non_zero_count > 0 else 0
        non_zero_std = non_zero_preds.std() if non_zero_count > 0 else 0

        # Clipped statistics (0.1-0.9 range) to avoid skew from very low values
        clipped_preds = np.clip(raw_predictions, 0.1, 0.9)
        clipped_mean = np.mean(clipped_preds)
        clipped_std = np.std(clipped_preds)

        # Distribution analysis
        distribution = {
            "< 0.1": np.sum(raw_predictions < 0.1),
            "< 0.2": np.sum(raw_predictions < 0.2),
            "< 0.3": np.sum(raw_predictions < 0.3),
            "< 0.4": np.sum(raw_predictions < 0.4),
            "< 0.5": np.sum(raw_predictions < 0.5),
            "> 0.5": np.sum(raw_predictions > 0.5),
            "> 0.7": np.sum(raw_predictions > 0.7),
            "> 0.9": np.sum(raw_predictions > 0.9),
        }

        # Print analysis
        print("    🔍 Raw Predictions Analysis:")
        print(f"      Total predictions: {total_count}")
        print(f"      Range: {min_val:.6f} - {max_val:.6f}")
        print(f"      Mean: {mean_val:.6f}")
        print(f"      Std: {std_val:.6f}")
        print(f"      Median: {median_val:.6f}")
        print(
            f"      Non-zero count: {non_zero_count} ({non_zero_count / total_count * 100:.1f}%)"
        )
        print(f"      Non-zero mean: {non_zero_mean:.6f}")
        print(f"      Non-zero std: {non_zero_std:.6f}")
        print(f"      Clipped mean (0.1-0.9): {clipped_mean:.6f}")
        print(f"      Clipped std (0.1-0.9): {clipped_std:.6f}")
        print("      Distribution:")
        for key, count in distribution.items():
            print(f"        {key}: {count} ({count / total_count * 100:.1f}%)")
        print(f"      First 10 values: {raw_predictions[:10]}")

        return {
            "total_count": total_count,
            "min": min_val,
            "max": max_val,
            "mean": mean_val,
            "std": std_val,
            "median": median_val,
            "non_zero_count": non_zero_count,
            "non_zero_mean": non_zero_mean,
            "non_zero_std": non_zero_std,
            "clipped_mean": clipped_mean,
            "clipped_std": clipped_std,
            "distribution": distribution,
        }

    def suggest_threshold(self, raw_predictions):
        """
        Suggest a threshold based on prediction distribution analysis

        Args:
            raw_predictions: Raw model outputs (0.0-1.0 range)

        Returns:
            Suggested threshold
        """
        analysis = self.analyze_raw_predictions(raw_predictions)

        # Normalize raw predictions to 1D for clipping
        raw_predictions = self._normalize_raw_predictions(raw_predictions)

        # Clip predictions to 0.1-0.9 range for threshold calculation to avoid skew from very low values
        clipped_predictions = np.clip(raw_predictions, 0.1, 0.9)
        clipped_mean = np.mean(clipped_predictions)

        # Use clipped mean as the suggested threshold, but add a larger bias to be more conservative
        suggested_threshold = (
            clipped_mean + 0.10
        )  # Add larger bias to reduce false positives

        # Clamp to reasonable range
        suggested_threshold = np.clip(suggested_threshold, 0.1, 0.9)

        print(
            f"    💡 Suggested threshold: {suggested_threshold:.3f} (clipped-mean-based)"
        )
        print(f"      Clipped mean (0.1-0.9): {clipped_mean:.3f}")
        print(
            f"      Non-zero mean: {analysis['non_zero_mean']:.3f}, Std: {analysis['non_zero_std']:.3f}"
        )

        return suggested_threshold

    def calculate_dynamic_threshold(self, raw_predictions):
        """
        Calculate a dynamic threshold based on current prediction distribution

        Args:
            raw_predictions: Raw model outputs (0.0-1.0 range)

        Returns:
            Dynamic threshold optimized for current predictions
        """

        # Normalize raw predictions to 1D
        raw_predictions = self._normalize_raw_predictions(raw_predictions)

        # Calculate basic statistics
        mean_val = np.mean(raw_predictions)
        std_val = np.std(raw_predictions)
        median_val = np.median(raw_predictions)

        # Clip predictions to 0.1-0.9 range for threshold calculation to avoid skew from very low values
        clipped_predictions = np.clip(raw_predictions, 0.1, 0.9)
        clipped_mean = np.mean(clipped_predictions)
        clipped_std = np.std(clipped_predictions)

        # Use clipped mean as the dynamic threshold, but add a larger bias to be more conservative
        threshold = clipped_mean + 2.0 * clipped_std

        # Clamp to reasonable range
        threshold = np.clip(threshold, 0.1, 0.9)

        print(f"    🎯 Dynamic threshold: {threshold:.3f} (clipped-mean-based)")
        print(
            f"      Distribution: mean={mean_val:.3f}, std={std_val:.3f}, median={median_val:.3f}"
        )
        print(f"      Clipped mean (0.1-0.9): {clipped_mean:.3f}")

        return threshold

    def find_optimal_threshold(
        self, validation_data, validation_labels, thresholds=None
    ):
        """
        Find optimal threshold for binary classification based on F1 score
        """
        if thresholds is None:
            thresholds = np.arange(0.05, 0.5, 0.025)  # Lower range for better recall

        if self.dataframeUtils.is_dataframe(validation_data):
            val_tensor = self.dataframeUtils.df_to_tensor(
                validation_data, self.seq_len, method=1
            )
        else:
            val_tensor = validation_data

        raw_preds = self.model.predict(val_tensor, verbose=0)
        raw_preds = np.array(raw_preds)

        # Normalize raw predictions to 1D
        try:
            raw_preds = self._normalize_raw_predictions(raw_preds)
        except Exception as e:
            print(f"    ERR: Failed to normalize predictions: {e}")
            return 0.5

        # Handle both 1D and 2D validation labels
        if len(validation_labels.shape) == 2:
            # Convert one-hot encoded labels to class indices
            validation_labels = np.argmax(validation_labels, axis=1)

        best_f1 = 0
        best_threshold = 0.5
        best_recall = 0
        best_recall_threshold = 0.5

        for threshold in thresholds:
            predictions = np.where(raw_preds > threshold, 1.0, 0.0)

            # Calculate metrics
            tp = np.sum((predictions == 1) & (validation_labels == 1))
            fp = np.sum((predictions == 1) & (validation_labels == 0))
            fn = np.sum((predictions == 0) & (validation_labels == 1))

            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = (
                2 * precision * recall / (precision + recall)
                if (precision + recall) > 0
                else 0
            )

            if f1 > best_f1:
                best_f1 = f1
                best_threshold = threshold

            if recall > best_recall:
                best_recall = recall
                best_recall_threshold = threshold

        print(f"    Optimal threshold: {best_threshold:.3f} (F1: {best_f1:.3f})")
        print(
            f"    Best recall threshold: {best_recall_threshold:.3f} (Recall: {best_recall:.3f})"
        )
        print(
            "    Note: Use recall threshold if you want to catch more buy opportunities"
        )

        # Also suggest threshold based on non-zero statistics
        suggested_threshold = self.suggest_threshold(raw_preds)
        print(
            f"    Suggested threshold: {suggested_threshold:.3f} (based on non-zero statistics)"
        )

        # Store the optimal threshold for use in predict() method
        self.optimal_threshold = best_threshold

        # Also store it as a model attribute so it persists with the model
        if hasattr(self, "model") and self.model is not None:
            self.model.optimal_threshold = best_threshold

        return best_threshold

    def predict_with_threshold(self, data, threshold):
        """
        Make predictions with a custom threshold
        """
        if self.model is None:
            self.model = self.load()

        if self.dataframeUtils.is_dataframe(data):
            df_tensor = self.dataframeUtils.df_to_tensor(data, self.seq_len, method=1)
        else:
            df_tensor = data

        if self.model == None:
            print("    ERR: no model for predictions")
            predictions = np.zeros(np.shape(df_tensor)[0], dtype=float)
            return predictions

        # run the prediction
        preds = self.model.predict(df_tensor, verbose=0)
        preds = np.array(preds)

        # Normalize raw predictions to 1D
        try:
            preds = self._normalize_raw_predictions(preds)
        except Exception as e:
            print(f"    ERR: Failed to normalize predictions: {e}")
            predictions = np.zeros(preds.shape[0], dtype=float)
            return predictions

        # Use custom threshold
        predictions = np.where(preds > threshold, 1.0, 0.0)

        return predictions

    def tune_threshold(self, validation_data, validation_labels, method="f1"):
        """
        Tune threshold using different methods
        """
        if self.dataframeUtils.is_dataframe(validation_data):
            val_tensor = self.dataframeUtils.df_to_tensor(
                validation_data, self.seq_len, method=1
            )
        else:
            val_tensor = validation_data

        raw_preds = self.model.predict(val_tensor, verbose=0)
        raw_preds = np.array(raw_preds)

        # Normalize raw predictions to 1D
        try:
            raw_preds = self._normalize_raw_predictions(raw_preds)
        except Exception as e:
            print(f"    ERR: Failed to normalize predictions: {e}")
            return 0.5

        # Handle both 1D and 2D validation labels
        if len(validation_labels.shape) == 2:
            # Convert one-hot encoded labels to class indices
            validation_labels = np.argmax(validation_labels, axis=1)

        thresholds = np.arange(0.1, 0.9, 0.05)
        results = []

        for threshold in thresholds:
            predictions = np.where(raw_preds > threshold, 1.0, 0.0)

            # Calculate metrics
            tp = np.sum((predictions == 1) & (validation_labels == 1))
            fp = np.sum((predictions == 1) & (validation_labels == 0))
            fn = np.sum((predictions == 0) & (validation_labels == 1))
            tn = np.sum((predictions == 0) & (validation_labels == 0))

            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = (
                2 * precision * recall / (precision + recall)
                if (precision + recall) > 0
                else 0
            )
            accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0

            if method == "f1":
                score = f1
            elif method == "precision":
                score = precision
            elif method == "recall":
                score = recall
            elif method == "accuracy":
                score = accuracy
            else:
                score = f1

            results.append(
                {
                    "threshold": threshold,
                    "f1": f1,
                    "precision": precision,
                    "recall": recall,
                    "accuracy": accuracy,
                    "score": score,
                }
            )

        # Find best threshold
        best_result = max(results, key=lambda x: x["score"])

        print(f"    Best threshold ({method}): {best_result['threshold']:.3f}")
        print(
            f"    F1: {best_result['f1']:.3f}, Precision: {best_result['precision']:.3f}, Recall: {best_result['recall']:.3f}"
        )

        return best_result["threshold"]

    def evaluate_model_quality(
        self, train_buy, predict_buy, threshold=0.5, verbose=True
    ):
        """
        Comprehensive evaluation of model quality by comparing train_buy and predict_buy

        Args:
            train_buy: Ground truth labels (0 or 1)
            predict_buy: Model predictions (0 or 1)
            threshold: Threshold for converting raw predictions to binary (if needed)
            verbose: Whether to print detailed results

        Returns:
            Dictionary with comprehensive evaluation metrics
        """
        # Ensure inputs are numpy arrays
        train_buy = np.array(train_buy)
        predict_buy = np.array(predict_buy)

        # If predict_buy contains raw predictions (0.0-1.0), convert to binary
        if (
            predict_buy.dtype == float
            and np.max(predict_buy) <= 1.0
            and np.min(predict_buy) >= 0.0
        ):
            predict_buy = np.where(predict_buy > threshold, 1.0, 0.0)

        # Basic statistics
        total_samples = len(train_buy)
        positive_samples = np.sum(train_buy == 1)
        negative_samples = np.sum(train_buy == 0)
        positive_predictions = np.sum(predict_buy == 1)
        negative_predictions = np.sum(predict_buy == 0)

        # Confusion matrix
        tp = np.sum((predict_buy == 1) & (train_buy == 1))  # True Positives
        fp = np.sum((predict_buy == 1) & (train_buy == 0))  # False Positives
        tn = np.sum((predict_buy == 0) & (train_buy == 0))  # True Negatives
        fn = np.sum((predict_buy == 0) & (train_buy == 1))  # False Negatives

        # Calculate metrics
        accuracy = (tp + tn) / total_samples if total_samples > 0 else 0
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1_score = (
            2 * precision * recall / (precision + recall)
            if (precision + recall) > 0
            else 0
        )

        # Additional metrics
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0  # True Negative Rate
        sensitivity = recall  # True Positive Rate (same as recall)
        balanced_accuracy = (sensitivity + specificity) / 2

        # Matthews Correlation Coefficient (MCC)
        numerator = (tp * tn) - (fp * fn)
        denominator = np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
        mcc = numerator / denominator if denominator > 0 else 0

        # F-beta scores (F2 gives more weight to recall)
        f2_score = (
            5 * precision * recall / (4 * precision + recall)
            if (4 * precision + recall) > 0
            else 0
        )

        # Class distribution analysis
        class_imbalance_ratio = (
            negative_samples / positive_samples
            if positive_samples > 0
            else float("inf")
        )

        # Prediction distribution analysis
        prediction_imbalance_ratio = (
            negative_predictions / positive_predictions
            if positive_predictions > 0
            else float("inf")
        )

        # Store results
        results = {
            "total_samples": total_samples,
            "positive_samples": positive_samples,
            "negative_samples": negative_samples,
            "positive_predictions": positive_predictions,
            "negative_predictions": negative_predictions,
            "class_imbalance_ratio": class_imbalance_ratio,
            "prediction_imbalance_ratio": prediction_imbalance_ratio,
            "tp": tp,
            "fp": fp,
            "tn": tn,
            "fn": fn,
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1_score": f1_score,
            "f2_score": f2_score,
            "specificity": specificity,
            "sensitivity": sensitivity,
            "balanced_accuracy": balanced_accuracy,
            "mcc": mcc,
        }

        if verbose:
            print("\n    📊 MODEL QUALITY EVALUATION:")
            print(f"    {'=' * 50}")
            print("    📈 Basic Statistics:")
            print(f"      Total samples: {total_samples}")
            print(
                f"      Positive samples: {positive_samples} ({positive_samples / total_samples * 100:.1f}%)"
            )
            print(
                f"      Negative samples: {negative_samples} ({negative_samples / total_samples * 100:.1f}%)"
            )
            print(
                f"      Positive predictions: {positive_predictions} ({positive_predictions / total_samples * 100:.1f}%)"
            )
            print(
                f"      Negative predictions: {negative_predictions} ({negative_predictions / total_samples * 100:.1f}%)"
            )
            print(f"      Class imbalance ratio: {class_imbalance_ratio:.2f}:1")
            print(
                f"      Prediction imbalance ratio: {prediction_imbalance_ratio:.2f}:1"
            )

            print("\n    🎯 Confusion Matrix:")
            print(f"      True Positives (TP): {tp}")
            print(f"      False Positives (FP): {fp}")
            print(f"      True Negatives (TN): {tn}")
            print(f"      False Negatives (FN): {fn}")

            print("\n    📊 Performance Metrics:")
            print(f"      Accuracy: {accuracy:.4f}")
            print(f"      Precision: {precision:.4f}")
            print(f"      Recall (Sensitivity): {recall:.4f}")
            print(f"      Specificity: {specificity:.4f}")
            print(f"      F1 Score: {f1_score:.4f}")
            print(f"      F2 Score: {f2_score:.4f}")
            print(f"      Balanced Accuracy: {balanced_accuracy:.4f}")
            print(f"      Matthews Correlation: {mcc:.4f}")

            # Quality assessment
            print("\n    🔍 Quality Assessment:")
            if f1_score > 0.8:
                quality = "Excellent"
            elif f1_score > 0.6:
                quality = "Good"
            elif f1_score > 0.4:
                quality = "Fair"
            elif f1_score > 0.2:
                quality = "Poor"
            else:
                quality = "Very Poor"

            print(f"      Overall Quality: {quality} (F1: {f1_score:.4f})")

            # Specific issues
            if precision < 0.3:
                print("      ⚠️  Low Precision: Model has many false positives")
            if recall < 0.3:
                print("      ⚠️  Low Recall: Model misses many positive cases")
            if abs(class_imbalance_ratio - prediction_imbalance_ratio) > 2:
                print(
                    "      ⚠️  Prediction distribution doesn't match class distribution"
                )
            if mcc < 0.1:
                print(
                    "      ⚠️  Low MCC: Model shows poor correlation with true labels"
                )

            print(f"    {'=' * 50}")

        return results

    def compare_train_predict(
        self,
        train_data,
        train_labels,
        predict_data,
        predict_labels,
        threshold=0.5,
        save_plots=True,
    ):
        """
        Comprehensive comparison between training and prediction performance

        Args:
            train_data: Training data tensor
            train_labels: Training labels
            predict_data: Prediction data tensor
            predict_labels: Prediction labels
            threshold: Threshold for binary conversion
            save_plots: Whether to save comparison plots

        Returns:
            Dictionary with comparison results
        """
        # Get predictions
        train_preds_raw = self.model.predict(train_data, verbose=0)
        predict_preds_raw = self.model.predict(predict_data, verbose=0)

        # Normalize predictions
        train_preds_raw = self._normalize_raw_predictions(train_preds_raw)
        predict_preds_raw = self._normalize_raw_predictions(predict_preds_raw)

        # Convert to binary
        train_preds_binary = np.where(train_preds_raw > threshold, 1.0, 0.0)
        predict_preds_binary = np.where(predict_preds_raw > threshold, 1.0, 0.0)

        # Evaluate both sets
        train_results = self.evaluate_model_quality(
            train_labels, train_preds_binary, verbose=False
        )
        predict_results = self.evaluate_model_quality(
            predict_labels, predict_preds_binary, verbose=False
        )

        # Calculate differences
        comparison = {
            "train": train_results,
            "predict": predict_results,
            "differences": {},
        }

        metrics = ["accuracy", "precision", "recall", "f1_score", "f2_score", "mcc"]
        for metric in metrics:
            train_val = train_results[metric]
            predict_val = predict_results[metric]
            diff = predict_val - train_val
            comparison["differences"][metric] = {
                "train": train_val,
                "predict": predict_val,
                "difference": diff,
                "percent_change": (diff / train_val * 100)
                if train_val != 0
                else float("inf"),
            }

        # Print comparison
        print("\n    🔄 TRAIN vs PREDICT COMPARISON:")
        print(f"    {'=' * 60}")
        print(
            f"    {'Metric':<15} {'Train':<10} {'Predict':<10} {'Diff':<10} {'% Change':<10}"
        )
        print(f"    {'-' * 60}")

        for metric in metrics:
            diff_data = comparison["differences"][metric]
            print(
                f"    {metric:<15} {diff_data['train']:<10.4f} {diff_data['predict']:<10.4f} "
                f"{diff_data['difference']:<10.4f} {diff_data['percent_change']:<10.1f}%"
            )

        # Overall assessment
        print("\n    📊 Overall Assessment:")
        avg_f1_diff = comparison["differences"]["f1_score"]["difference"]
        if abs(avg_f1_diff) < 0.05:
            print(f"      ✅ Model generalizes well (F1 difference: {avg_f1_diff:.4f})")
        elif avg_f1_diff > 0.05:
            print(
                f"      ⚠️  Model may be overfitting (F1 difference: {avg_f1_diff:.4f})"
            )
        else:
            print(
                f"      ⚠️  Model may be underfitting (F1 difference: {avg_f1_diff:.4f})"
            )

        return comparison
