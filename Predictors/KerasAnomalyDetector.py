# Keras classifier for supervised anomaly detection
# Inherits from KerasBasePredictor to follow the existing pattern

import numpy as np
import tensorflow as tf
import keras


from Predictors.KerasBasePredictor import KerasBasePredictor
from Predictors.BaseAnomalyDetector import BaseAnomalyDetector
from utils.CustomLoss import multi_class_focal_loss
from utils.CustomMetric import MinorityF1, MCCMetric


import warnings
import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

# Performance optimizations
# Option 1: Enable mixed precision for faster training
tf.keras.mixed_precision.set_global_policy("mixed_float16")

# Option 2: Optimize GPU memory growth
gpus = tf.config.experimental.list_physical_devices("GPU")
if gpus:
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
    print(f"    GPU memory growth enabled for {len(gpus)} GPU(s)")
else:
    print("    No GPU detected, using CPU")

warnings.filterwarnings(
    "ignore", message="The objective has been evaluated at this point before."
)


warnings.filterwarnings(
    "ignore", category=UserWarning, module="keras.src.saving.saving_lib"
)

########################################################


class KerasAnomalyDetector(KerasBasePredictor, BaseAnomalyDetector):
    """
    Multi-task classifier that predicts:
    - Latent space (for anomaly score)
    - Trading classification (sell/hold/buy)

    The idea is that we do 'normal' anomaly detection with an auto encoder, plus
    we add a trading classification head to the model.
    """

    is_trained = False
    learning_rate = 2e-4

    def __init__(self, pair, seq_len, num_features, tag=""):
        super().__init__(pair, seq_len, num_features, tag)
        self.name = self.__class__.__name__

    def create_model(self, seq_len, num_features):
        """placeholder for create_model"""

        # model must return "reconstruction" and "trading" outputs

        raise NotImplementedError("Subclasses must implement create_model")

        return None

    # Normalize alpha vectors to sum to 1.0 (required for focal loss consistency)
    def normalize_alpha(self, alpha_vec):
        total = sum(alpha_vec)
        if total > 0:
            return [w / total for w in alpha_vec]
        else:
            return [1.0 / len(alpha_vec)] * len(alpha_vec)

    def compile_model(self, model, class_weights=None, loss_weights=None):
        """Compile with task-specific losses and metrics

        Args:
            model: Keras model to compile
            class_weights: Optional class weights for trading loss
            loss_weights: Optional dict of loss weights (e.g., {'reconstruction': 0.0, 'trading': 1.0})
                          If None, defaults to both losses weighted at 1.0
        """

        if class_weights is None:
            # defaults based on prior testing
            trading_alpha = [1.0, 1.0, 1.0]
        else:
            trading_alpha = class_weights["trading"]

        trading_alpha = self.normalize_alpha(trading_alpha)
        print(f"    trading alpha: {trading_alpha}")

        # print(f"    trading alpha: {trading_alpha}")
        trading_loss_fn = multi_class_focal_loss(gamma=1.0, alpha_vector=trading_alpha)

        precision = keras.metrics.Precision(name="precision", class_id=2)

        # NOTE: training is VERY sensitive to the loss functions and weights. trading is most important
        # BUT regression tasks need sufficient weight to learn properly
        # These numbers are derived from observing training data and balancing the valifation loss
        # contributions of each task
        if loss_weights is None:
            task_weights = {
                "reconstruction": 1.0,
                "trading": 1.0,
            }
        else:
            task_weights = loss_weights
        print(f"    task weights: {task_weights}")

        model.compile(
            optimizer=tf.keras.optimizers.Adam(
                # learning_rate=self.learning_rate, clipnorm=0.5  # Reduced clipping
                learning_rate=self.learning_rate
            ),
            loss={
                "reconstruction": "mse",
                "trading": trading_loss_fn,
            },
            loss_weights=task_weights,
            metrics={
                "reconstruction": [],
                "trading": [precision, MinorityF1(), MCCMetric()],
            },
        )
        return model

    # ---------------------------------------------------------

    def train(
        self,
        df_train_norm,
        df_test_norm,
        train_results,
        test_results,
        force_train=False,
        class_weights=None,
    ):
        """Override train method to handle multi-task dictionary inputs with class weights"""

        # lazy loading because params can change up to this point
        if self.model is None:
            # load saved model if present
            self.model = self.load()
        # else:
        #     print("    Model already exists")

        # Check if model architecture has changed (e.g., new heads added)
        architecture_changed = False
        if self.model is not None:
            expected_outputs = ["reconstruction", "trading"]
            current_outputs = list(self.model.output_names)
            if set(current_outputs) != set(expected_outputs):
                print(
                    f"    Model architecture changed: expected {expected_outputs}, got {current_outputs}"
                )
                architecture_changed = True
                # Force retrain by setting model to None
                self.model = None

        # if model is already trained, and caller is not requesting a re-train, then just return
        if (
            (self.model is not None)
            and self.model_is_trained()
            and (not force_train)
            and (not self.new_model_created())
            and (not architecture_changed)
        ):
            # print("    Model is already trained")
            return

        if self.dataframeUtils.is_dataframe(df_train_norm):
            df_train = df_train_norm.copy()
            df_test = df_test_norm.copy()
            train_tensor = self.dataframeUtils.df_to_tensor(df_train, self.seq_len)
            test_tensor = self.dataframeUtils.df_to_tensor(df_test, self.seq_len)
        else:
            # already in tensor format
            train_tensor = df_train_norm.copy()
            test_tensor = df_test_norm.copy()

        # if model does not exist, create and compile it
        if self.model is None:
            # pca_W, pca_M = self.get_initial_pca_matrices(train_tensor)
            # self.model = self.create_model(
            #     self.seq_len, self.num_features, pca_W, pca_M

            self.model = self.create_model(self.seq_len, self.num_features)
            if self.model is None:
                raise Exception("Error creating model")
                return

            # Compile model without class weights
            self.model = self.compile_model(self.model, class_weights=class_weights)
            self.model.summary()

        # Monitor overall validation loss for early stopping (better for multi-task)
        # monitor_field = "val_loss"
        # monitor_mode = "min"
        # monitor_field = "val_trading_loss"
        # monitor_mode = "min"
        # monitor_field = "val_trading_minority_f1"
        monitor_field = "val_trading_mcc"
        monitor_mode = "max"

        # Store monitor configuration for later use in result checking
        self.monitor_field = monitor_field
        self.monitor_mode = monitor_mode

        min_delta = 0.001
        early_patience = 20
        plateau_patience = 8

        # callback to control early exit on plateau of results
        early_callback = keras.callbacks.EarlyStopping(
            monitor=monitor_field,
            mode=monitor_mode,
            patience=early_patience,
            min_delta=min_delta,
            restore_best_weights=True,
            verbose=1,
        )

        plateau_callback = keras.callbacks.ReduceLROnPlateau(
            monitor=monitor_field,
            mode=monitor_mode,
            factor=0.1,
            min_delta=min_delta,
            patience=plateau_patience,
            verbose=1,
        )

        # callback to control saving of 'best' model
        checkpoint_callback = keras.callbacks.ModelCheckpoint(
            filepath=self.get_checkpoint_path(),
            save_weights_only=True,
            monitor=monitor_field,
            mode=monitor_mode,
            save_best_only=True,
            verbose=1,
        )

        callbacks = [plateau_callback, early_callback, checkpoint_callback]

        # Store reference to early stopping callback to access best_epoch later
        self.early_callback = early_callback

        print("")
        print(
            f"    training multi-task model: {self.name}  batch_size:{self.batch_size}"
        )
        print(f"    monitor field: {monitor_field}  monitor mode: {monitor_mode}")

        # Model weights are saved at the end of every epoch, if it's the best seen so far.
        # Use class weights if provided (for the trading classification task)
        fit_kwargs = {
            "batch_size": self.batch_size,
            "epochs": 100,
            "callbacks": callbacks,
            "validation_data": (test_tensor, test_results),
            "verbose": 1,
        }

        # # Use class weights if provided to handle trading class imbalance
        # if class_weights is not None and 'trading' in class_weights:
        #     print(f"    Using class weights for trading: {class_weights['trading']}")
        #     fit_kwargs['class_weight'] = {'trading': class_weights['trading']}
        # else:
        #     print(f"    No weights for trading")

        fhis = self.model.fit(train_tensor, train_results, **fit_kwargs)

        # Examine final training results and print warnings if needed
        self._check_training_results(fhis)

        # reset learning rate
        if force_train:
            self.learning_rate = self.learning_rate * 0.5
            # tf.keras.backend.set_value(self.model.optimizer.learning_rate, self.learning_rate) # keras 2.x
            self.model.optimizer.learning_rate.assign(self.learning_rate)  # keras 3.x

        self.save()
        self.is_trained = True

        return

    def predict(self, data):
        """Override predict method to handle multi-task outputs"""

        # lazy loading because params can change up to this point
        if self.model is None:
            # load saved model if present
            self.model = self.load()

        if self.dataframeUtils.is_dataframe(data):
            # convert dataframe to tensor
            df_tensor = self.dataframeUtils.df_to_tensor(data, self.seq_len)
        else:
            df_tensor = data

        if self.model == None:
            print("    ERR: no model for predictions")
            predictions = np.zeros(np.shape(df_tensor)[0], dtype=float)
            return predictions

        # run the prediction - returns a dictionary of outputs (because model has named outputs)
        preds = self.model.predict(df_tensor, verbose=0)

        # Return all predictions as a dictionary
        return {
            "reconstruction": preds["reconstruction"],
            "trading": preds["trading"],
        }

    def make_window_dataset(self, df, labels, window_size, step):
        """Create windowed dataset from dataframe and labels"""
        if self.dataframeUtils.is_dataframe(df):
            df_tensor = self.dataframeUtils.df_to_tensor(df, window_size, method=1)
        else:
            df_tensor = df

        # Create corresponding labels
        if len(labels.shape) > 1:  # One-hot encoded
            window_labels = labels[window_size - 1 :: step]
        else:  # Scalar labels
            window_labels = labels[window_size - 1 :: step]

        return df_tensor, window_labels

    def _check_training_results(self, history):
        """Check training results and print warnings if training was insufficient"""

        print()

        if history is None or not hasattr(history, "history"):
            print("    WARNING: No training history available to check")
            return

        epochs_trained = len(history.history.get("loss", []))
        print(f"    Training completed: {epochs_trained} epochs")

        # Check if training was stopped early with insufficient epochs
        if epochs_trained < 10:
            print(
                f"    ⚠️  WARNING: Training stopped early after only {epochs_trained} epochs (< 10)"
            )
            print(
                "    Consider increasing early stopping patience or adjusting learning parameters"
            )

        # Get best epoch from the early stopping callback
        monitor_field = getattr(self, "monitor_field", "val_loss")
        monitor_mode = getattr(self, "monitor_mode", "min")

        # Try to get best_epoch from the callback first
        if hasattr(self, "early_callback") and hasattr(
            self.early_callback, "best_epoch"
        ):
            best_epoch_idx = self.early_callback.best_epoch
            if best_epoch_idx is None:
                # Training didn't stop early, use the last epoch
                best_epoch_idx = (
                    len(history.history[list(history.history.keys())[0]]) - 1
                )
        else:
            # Fallback: calculate best epoch from history
            if monitor_field not in history.history:
                print(
                    f"    WARNING: Monitor field '{monitor_field}' not found in training history"
                )
                print(f"    Available metrics: {list(history.history.keys())}")
                val_metrics = [
                    k for k in history.history.keys() if k.startswith("val_")
                ]
                if val_metrics:
                    monitor_field = val_metrics[0]
                    print(
                        f"    Falling back to '{monitor_field}' for best epoch determination"
                    )
                else:
                    print(
                        "    Cannot determine best epoch - no validation metrics found"
                    )
                    return

            monitor_values = history.history[monitor_field]
            if monitor_mode == "min":
                best_epoch_idx = monitor_values.index(min(monitor_values))
            else:  # max
                best_epoch_idx = monitor_values.index(max(monitor_values))

        # Print information about the best epoch
        if monitor_field in history.history:
            best_monitor_value = history.history[monitor_field][best_epoch_idx]
            print(
                f"    Best epoch: {best_epoch_idx + 1} (based on {monitor_field}={best_monitor_value:.4f})"
            )
        else:
            print(f"    Best epoch: {best_epoch_idx + 1}")

        if best_epoch_idx < 10:
            print(
                f"    ⚠️  WARNING: Training stopped early after only {best_epoch_idx + 1} epochs (< 10)"
            )
            print(
                "    Consider increasing early stopping patience or adjusting learning parameters"
            )

        # Check validation trading F1 score
        val_trading_f1_key = "val_trading_minority_f1"
        if val_trading_f1_key in history.history:
            best_val_trading_f1 = history.history[val_trading_f1_key][best_epoch_idx]
            print(
                f"    Best validation trading F1 score: {best_val_trading_f1:.4f} (epoch {best_epoch_idx + 1})"
            )

            if best_val_trading_f1 < 0.2:
                print(
                    f"    ⚠️  WARNING: Very Low validation trading F1 score: {best_val_trading_f1:.4f} (< 0.2)"
                )
                print("    Model is essentially random guessing")
            elif best_val_trading_f1 < 0.3:
                print(
                    f"    ⚠️  WARNING: Low validation trading F1 score: {best_val_trading_f1:.4f} (< 0.3)"
                )
                print(
                    "    Model may not be performing well for trading classification"
                )
            else:
                print(
                    f"    ✓ Validation trading F1 score is good: {best_val_trading_f1:.4f}"
                )
        else:
            print(
                "    WARNING: Could not find validation trading F1 score in training history"
            )
            print(f"    Available metrics: {list(history.history.keys())}")

        # Print summary of all validation metrics from the best epoch
        print("    Best epoch validation metrics:")
        for key, values in history.history.items():
            if key.startswith("val_") and len(values) > 0:
                best_value = values[best_epoch_idx]
                print(f"      {key}: {best_value:.4f}")
