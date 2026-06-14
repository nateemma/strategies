# Multi-task Keras classifier for regime, volatility, and risk prediction
# Inherits from KerasBaseClassifier to follow the existing pattern

import numpy as np
from sklearn.decomposition import PCA
import tensorflow as tf
import keras


from Predictors.KerasBaseClassifier import KerasBaseClassifier
from utils.CustomLoss import multi_class_focal_loss
from utils.CustomMetric import MinorityF1, MCCMetric, AverageMCCCallback


import warnings
import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

# Performance optimizations
# Precision: float32 (Keras/TF default) — mixed_float16 removed (no speedup on
# tensorflow-metal, added fp16 risk; no longer an import-time global side effect).

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


# ---------------------------
# FOCAL LOSS (categorical)
# ---------------------------

keras_focal_loss_fn = tf.keras.losses.CategoricalFocalCrossentropy(
    alpha=0.25,  # Class balancing factor
    gamma=2.0,  # Modulating factor for hard examples
)

strict_focal_loss_fn = tf.keras.losses.CategoricalFocalCrossentropy(
    alpha=0.25,  # Class balancing factor
    gamma=3.0,  # Modulating factor for hard examples
)

soft_focal_loss_fn = tf.keras.losses.CategoricalFocalCrossentropy(
    alpha=0.25,  # Class balancing factor
    gamma=1.0,  # Modulating factor for hard examples
)

# Custom multi-task metrics are imported from utils.CustomMetric

########################################################


class KerasClassifierMultiTask(KerasBaseClassifier):
    """
    Multi-task classifier that predicts:
    - Trading (buy/hold/sell)
    - Market regime  Bull/Bear/Sideways)
    - Risk level (low/medium/high)
    - Momentum (continuous regression)
    - Flow (continuous regression)
    - Profit (continuous regression)

    The idea is that having to balance predictions across multiple tasks, you will end up
    with a more general purpose model with less overfitting to the training data.
    """

    is_trained = False
    clean_data_required = False
    # learning_rate = 5e-5
    learning_rate = 1e-4

    # Optional per-instance override of the default task weights below.
    # Strategies that want to shift loss budget across tasks (e.g. heavily
    # downweight auxiliary heads to let trading dominate) can set this
    # attribute on the classifier instance in their get_classifier() method.
    # When None, the in-method default applies. Values are normalised to
    # sum to 1.0 at compile time, same as the defaults.
    task_weights_override = None

    def __init__(self, pair, seq_len, num_features, tag=""):
        super().__init__(pair, seq_len, num_features, tag)
        self.name = self.__class__.__name__
        self.task_weights_override = None

    def create_model(self, seq_len, num_features):
        """placeholder for create_model"""

        raise NotImplementedError("Subclasses must implement create_model")

        return None

    def calculate_alpha(self, weights):
        total = sum(weights)
        if total > 0:
            alpha = [w / total for w in weights]
        else:
            # oops, check whether self.num_classes exists
            if hasattr(self, "num_classes") and self.num_classes > 0:
                alpha = [1.0 / self.num_classes] * self.num_classes
            else:
                alpha = [1.0 / len(weights)] * len(weights)
        return alpha

    def calculate_gamma(self, alpha):
        alpha = np.array(alpha)
        min_a = np.min(alpha)
        max_a = np.max(alpha)

        if min_a == 0:
            return 2.0  # extremely imbalanced

        ratio = max_a / min_a

        if ratio < 1.2:
            return 0.0
        elif ratio < 2.0:
            return 0.5
        elif ratio < 5.0:
            return 1.0
        else:
            return 1.5

    def compile_model(self, model, class_weights=None):
        """Compile with task-specific losses and metrics"""

        if class_weights is None:
            # defaults based on prior testing
            default_alpha = [0.333, 0.333, 0.333]
            trading_alpha = default_alpha
            risk_alpha = default_alpha
            regime_alpha = default_alpha
            momentum_alpha = default_alpha
            flow_alpha = default_alpha
            profit_alpha = default_alpha
        else:
            trading_alpha = class_weights["trading"]
            risk_alpha = class_weights["risk"]
            regime_alpha = class_weights["regime"]
            momentum_alpha = class_weights["momentum"]
            flow_alpha = class_weights["flow"]
            profit_alpha = class_weights["profit"]
            print(f"    class weights: {class_weights}")

        trading_alpha = self.calculate_alpha(trading_alpha)
        risk_alpha = self.calculate_alpha(risk_alpha)
        regime_alpha = self.calculate_alpha(regime_alpha)
        momentum_alpha = self.calculate_alpha(momentum_alpha)
        flow_alpha = self.calculate_alpha(flow_alpha)
        profit_alpha = self.calculate_alpha(profit_alpha)

        trading_gamma = self.calculate_gamma(trading_alpha)
        risk_gamma = self.calculate_gamma(risk_alpha)
        regime_gamma = self.calculate_gamma(regime_alpha)
        momentum_gamma = self.calculate_gamma(momentum_alpha)
        flow_gamma = self.calculate_gamma(flow_alpha)
        profit_gamma = self.calculate_gamma(profit_alpha)

        # print(f"    trading alpha: {trading_alpha}")
        # Higher gamma for trading to focus more on hard examples (minority classes)
        trading_loss_fn = multi_class_focal_loss(
            gamma=trading_gamma, alpha_vector=trading_alpha
        )
        risk_loss_fn = multi_class_focal_loss(gamma=risk_gamma, alpha_vector=risk_alpha)
        regime_loss_fn = multi_class_focal_loss(
            gamma=regime_gamma, alpha_vector=regime_alpha
        )
        momentum_loss_fn = multi_class_focal_loss(
            gamma=momentum_gamma, alpha_vector=momentum_alpha
        )
        flow_loss_fn = multi_class_focal_loss(gamma=flow_gamma, alpha_vector=flow_alpha)
        profit_loss_fn = multi_class_focal_loss(
            gamma=profit_gamma, alpha_vector=profit_alpha
        )

        precision = keras.metrics.Precision(name="precision", class_id=2)

        # NOTE: training is VERY sensitive to the loss functions and weights. trading is most important
        # BUT regression tasks need sufficient weight to learn properly
        # These numbers are derived from observing training data and balancing the valifation loss
        # contributions of each task
        # Per-instance override beats the default below. See class attribute
        # docstring for the use case (per-variant downweighting of aux tasks).
        if self.task_weights_override is not None:
            task_weights_raw = dict(self.task_weights_override)
            weight_source = "instance override"
        else:
            task_weights_raw = {
                "trading": 2.0,
                "regime": 1.0,
                "risk": 1.0,
                "momentum": 1.0,
                "flow": 1.0,
                "profit": 2.0,
            }
            weight_source = "module default"

        # # Normalize task weights to sum to 1.0 for better interpretability and stable loss magnitude
        # # This preserves relative ratios while making the total loss a weighted average
        total_weight = sum(task_weights_raw.values())
        task_weights = {
            task: weight / total_weight for task, weight in task_weights_raw.items()
        }
        print(
            f"    task weights ({weight_source}, normalized, "
            f"sum={sum(task_weights.values()):.3f}): {task_weights}"
        )
        # print(
        #     f"task_weights_raw: {task_weights_raw}"
        # )
        # task_weights = task_weights_raw

        model.compile(
            optimizer=tf.keras.optimizers.Adam(
                # learning_rate=self.learning_rate, clipnorm=0.5  # Reduced clipping
                learning_rate=self.learning_rate
            ),
            loss={
                "trading": trading_loss_fn,
                "regime": regime_loss_fn,
                "risk": risk_loss_fn,
                # "momentum": RangePenaltyLoss(target_range=2.0, penalty_weight=1.0),
                # "flow": RangePenaltyLoss(target_range=2.0, penalty_weight=1.0),
                # "profit": RangePenaltyLoss(target_range=2.0, penalty_weight=1.0),
                # "momentum": tf.keras.losses.Huber(delta=0.1),
                "flow": flow_loss_fn,
                "momentum": momentum_loss_fn,
                "profit": profit_loss_fn,
            },
            # loss_weights are used to balance the loss between the tasks. trading is most important
            # BUT regression tasks need sufficient weight to learn properly
            # These numbers are derived from observing training data and balancing the valifation loss
            # contributions of each task
            loss_weights=task_weights,
            metrics={
                "trading": [precision, MinorityF1(), MCCMetric()],
                "regime": [MCCMetric()],
                "risk": [MCCMetric()],
                "momentum": [MCCMetric()],
                "flow": [MCCMetric()],
                "profit": [MCCMetric()],
            },
        )
        return model

    # ---------------------------------------------------------

    def get_initial_pca_matrices(
        self, train_tensor, num_filters=128, num_components=32
    ):
        """Get the initial PCA matrices for the training data"""

        print("    Pre-loading PCA estimate")

        # 1a. Build a temporary mini-model for the initial transformation
        input_shape = (self.seq_len, self.num_features)
        temp_input = tf.keras.layers.Input(shape=input_shape)
        temp_resized = tf.keras.layers.Dense(
            num_filters, activation="linear", name="resized_input"
        )(temp_input)

        # NOTE: You must use the actual trained/loaded weights for this Dense layer
        # if you want the initial PCA to be representative.
        # For a first-time run, random weights are fine, but for subsequent runs, use model weights.

        temp_model = tf.keras.Model(inputs=temp_input, outputs=temp_resized)

        # 1b. Get the Resized NumPy Features
        X_train_resized = temp_model.predict(
            train_tensor
        )  # Shape (N_train, seq_len, num_filters)

        # 1c. Flatten and Fit PCA
        X_train_flat = X_train_resized.reshape(-1, num_filters)

        pca = PCA(n_components=num_components, svd_solver="full")
        pca.fit(X_train_flat)

        pca_W = pca.components_.T
        pca_M = pca.mean_.reshape(1, -1)

        return pca_W, pca_M

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
            expected_outputs = [
                "trading",
                "regime",
                "momentum",
                "risk",
                "flow",
                "profit",
            ]
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
        # monitor_field = "val_average_mcc"
        # monitor_field = "val_momentum_mcc"
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

        # Callback to compute average MCC across all tasks
        # Must be first so metric is available when other callbacks check for it
        avg_mcc_callback = AverageMCCCallback(
            task_names=["trading", "regime", "risk", "momentum", "flow", "profit"]
        )

        callbacks = [
            avg_mcc_callback,
            plateau_callback,
            early_callback,
            checkpoint_callback,
        ]

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

        if self.model is None:
            print("    ERR: no model for predictions")
            predictions = np.zeros(np.shape(df_tensor)[0], dtype=float)
            return predictions

        # run the prediction - returns a dictionary of outputs (because model has named outputs)
        preds = self.model.predict(df_tensor, verbose=0)

        # Return all predictions as a dictionary
        return {
            "trading": preds["trading"],
            "regime": preds["regime"],
            "momentum": preds["momentum"],
            "risk": preds["risk"],
            "flow": preds["flow"],
            "profit": preds["profit"],
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

        # Calculate recommended task weights for balanced loss
        self._calculate_recommended_weights(history, best_epoch_idx)

    def _calculate_recommended_weights(self, history, best_epoch_idx):
        """Calculate recommended loss weights for balanced task contributions"""

        # Define the task loss keys in validation history
        task_loss_keys = {
            "trading": "val_trading_loss",
            "regime": "val_regime_loss",
            "risk": "val_risk_loss",
            "momentum": "val_momentum_loss",
            "flow": "val_flow_loss",
            "profit": "val_profit_loss",
        }

        # print()

        # Extract validation losses from the best epoch
        task_losses = {}
        for task_name, loss_key in task_loss_keys.items():
            if loss_key in history.history:
                task_losses[task_name] = history.history[loss_key][best_epoch_idx]

        if len(task_losses) < 2:
            print(
                "    WARNING: Not enough task losses found to calculate recommended weights"
            )
            return

        # Calculate total validation loss and target contribution per task
        total_val_loss = sum(task_losses.values())
        num_tasks = len(task_losses)
        target_contribution = total_val_loss / num_tasks

        # ===== OPTION 1: BALANCED CONTRIBUTION (all tasks contribute equally) =====
        print("\n    Option 1: BALANCED CONTRIBUTION - All tasks contribute equally")
        print(
            f"    (Target: each task contributes ~{target_contribution:.2f} to total loss of {total_val_loss:.2f})"
        )
        print("    ")
        print("    loss_weights = {")

        balanced_weights = {}
        for task_name in ["trading", "regime", "risk", "momentum", "flow", "profit"]:
            if task_name in task_losses:
                current_loss = task_losses[task_name]
                # Calculate weight needed for this task to contribute equally
                recommended_weight = (
                    target_contribution / current_loss if current_loss > 0 else 1.0
                )
                balanced_weights[task_name] = recommended_weight
                print(
                    f'        "{task_name}": {recommended_weight:.1f},  # {current_loss:.4f} * {recommended_weight:.1f} ≈ {current_loss * recommended_weight:.2f}'
                )

        print("    }")

        # ===== OPTION 2: PRIORITIZE HIGH LOSS (focus on worst-performing tasks) =====
        print("\n    Option 2: PRIORITIZE HIGH LOSS - Focus on worst-performing tasks")
        print("    (High loss tasks get higher weights to improve them faster)")
        print("    ")
        print("    loss_weights = {")

        priority_weights = {}
        for task_name in ["trading", "regime", "risk", "momentum", "flow", "profit"]:
            if task_name in task_losses:
                current_loss = task_losses[task_name]
                # Give higher weight to tasks with higher loss
                recommended_weight = (
                    current_loss / target_contribution
                    if target_contribution > 0
                    else 1.0
                )
                priority_weights[task_name] = recommended_weight
                print(
                    f'        "{task_name}": {recommended_weight:.1f},  # {current_loss:.4f} * {recommended_weight:.1f} ≈ {current_loss * recommended_weight:.2f}'
                )

        print("    }")
        print()
