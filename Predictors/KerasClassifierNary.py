# base class that implements Neural Network N-ary Classifier.
# This class implements a keras classifier that provides a n-ary result

# subclasses should override the create_model() method

import numpy as np
import pandas as pd

pd.options.mode.chained_assignment = None  # default='warn'

# Strategy specific imports, files must reside in same folder as strategy
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent))
sys.path.append(str(Path(__file__).parent.parent))

import warnings
import random
import os

import tensorflow as tf
import keras

# log = logging.getLogger(__name__)
# # log = logging.getLogger()
# # log.setLevel(logging.DEBUG)
# log.addFilter(logging.Filter(name='loading'))
# log.addFilter(logging.Filter(name='saving'))
# logging.getLogger('tensorflow').setLevel(logging.WARNING)
# logging.getLogger('keras').setLevel(logging.WARNING)
# logging.getLogger('pickle').setLevel(logging.CRITICAL)

warnings.simplefilter(action="ignore", category=pd.errors.PerformanceWarning)
warnings.filterwarnings(
    "ignore", message="The objective has been evaluated at this point before."
)

warnings.filterwarnings(
    "ignore", category=UserWarning, module="keras.src.saving.saving_lib"
)


os.environ["TF_CPP_MIN_LOG_LEVEL"] = "1"
os.environ["TF_DETERMINISTIC_OPS"] = "1"
# os.environ['PYTHON_PICKLING_LOGLEVEL'] = '0'

# # workaround for memory leak in tensorflow 2.10
# os.environ['TF_RUN_EAGER_OP_AS_FUNCTION'] = '0'


seed = 42
os.environ["PYTHONHASHSEED"] = str(seed)
random.seed(seed)
tf.random.set_seed(seed)
np.random.seed(seed)

# tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.WARN)


from Predictors.KerasBaseClassifier import KerasBaseClassifier
from utils.CustomLoss import multi_class_focal_loss
from utils.CustomMetric import MCCMetric, PerClassF1

# Custom losses and metrics are imported from utils.CustomLoss / utils.CustomMetric

########################################################

# --------------------------------------------------------------


@keras.saving.register_keras_serializable(package="KerasBaseClassifier")
class KerasClassifierNary(KerasBaseClassifier):
    clean_data_required = False
    # num_classes will be set dynamically by create_classifier
    num_classes = 3

    # create model - subclasses should overide this
    def create_model(self, seq_len, num_features):

        model = None

        print("    WARNING: create_model() should be defined by the subclass")

        # create a simple model for illustrative purposes (or to test the framework)
        model = keras.Sequential(name=self.name)

        # NOTE: don't use relu with LSTMs, cannot use GPU if you do (much slower). Use tanh

        # simplest possible model:
        model.add(
            keras.layers.LSTM(
                128,
                return_sequences=True,
                activation="tanh",
                input_shape=(seq_len, num_features),
            )
        )
        model.add(keras.layers.Dropout(rate=0.1))

        # last layer is a nary decision - do not change
        model.add(keras.layers.Dense(self.num_classes, activation="softmax"))

        return model

    def compile_model(self, model):

        weights = self.get_class_weights()
        num_classes = len(weights)

        focal_alpha = self.calculate_alpha(weights)
        gamma = self.calculate_gamma(focal_alpha)

        # print(f"    focal_alpha: {focal_alpha} gamma: {gamma}")

        # Enable mixed precision training for better GPU utilization (2x speedup on modern GPUs)
        use_mixed_precision = True
        if use_mixed_precision:
            try:
                policy = tf.keras.mixed_precision.Policy("mixed_float16")
                tf.keras.mixed_precision.set_global_policy(policy)
                # print("    Mixed precision training enabled (float16/float32)")
            except (AttributeError, RuntimeError):
                # Mixed precision not available - that's okay
                use_mixed_precision = False
                # print("    Mixed precision not available, using float32")

        # Create base optimizer
        base_optimizer = keras.optimizers.Adam(learning_rate=0.0001)

        # Wrap with LossScaleOptimizer for mixed precision if enabled
        if use_mixed_precision:
            optimizer = tf.keras.mixed_precision.LossScaleOptimizer(base_optimizer)
        else:
            optimizer = base_optimizer

        # Use standard cross entropy for now - focal loss is causing overfitting
        # loss = "categorical_crossentropy"
        # loss = "sparse_categorical_crossentropy"
        # loss = focal_loss_softmax
        # loss = categorical_focal_loss(alpha=weights)
        loss = multi_class_focal_loss(gamma=gamma, alpha_vector=focal_alpha)

        # Use standard weighted F1 score instead of custom metric to avoid serialization issues
        # f1_weighted = keras.metrics.F1Score(average="weighted", name="f1_weighted")
        f1_macro = keras.metrics.F1Score(average="macro", name="f1_macro")
        # f1_weighted = WeightedF1(num_classes=num_classes, class_weights=weights)
        f1_class_2 = PerClassF1(
            num_classes=num_classes, target_class=2, name="f1_class_2"
        )

        recall_metric = keras.metrics.Recall(name="recall", class_id=2)
        # accuracy_metric = keras.metrics.Accuracy(name="accuracy")
        precision_metric = keras.metrics.Precision(name="precision", class_id=2)

        metrics = [f1_class_2, precision_metric, MCCMetric()]

        model.compile(
            optimizer=optimizer,
            loss=loss,
            metrics=metrics,
        )

        return model

    # update training using the suplied (normalised) dataframe. Training is cumulative
    # the 'labels' args should contain integres for each class
    def train(
        self,
        df_train_norm,
        df_test_norm,
        train_results,
        test_results,
        force_train=False,
        class_weights=None,
    ):

        # lazy loading because params can change up to this point
        if self.model is None:
            # load saved model if present
            self.model = self.load()
        else:
            print("    Model already exists")

        # print(f'is_trained:{self.is_trained} force_train:{force_train}')

        # print("    train_tensor:{} test_tensor:{}".format(np.shape(train_tensor), np.shape(test_tensor)))

        # if model is already trained, and caller is not requesting a re-train, then just return
        if (
            (self.model is not None)
            and self.model_is_trained()
            and (not force_train)
            and (not self.new_model_created())
        ):
            # print(f"    Not training. is_trained:{self.is_trained} force_train:{force_train} new_model:{self.new_model}")
            # print("    Model is already trained")
            return

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

            train_tensor = self.dataframeUtils.df_to_tensor(df_train, self.seq_len)
            test_tensor = self.dataframeUtils.df_to_tensor(df_test, self.seq_len)
        else:
            # already in tensor format
            train_tensor = df_train_norm.copy()
            test_tensor = df_test_norm.copy()

        # if model does not exist, create and compile it
        if self.model is None:
            self.num_features = np.shape(train_tensor)[-1]
            self.model = self.create_model(self.seq_len, self.num_features)
            if self.model is None:
                print("    ERR: model not created")
                return

            if class_weights is None:
                # calculate class weights based on test_results (not train_results)
                class_weights = self.calculate_class_weights(test_results)

            self.set_class_weights(class_weights)
            self.model = self.compile_model(self.model)
            self.model.summary()

        # monitor_field = 'val_loss'
        # monitor_mode = "min"

        # Increase training epochs and patience for better convergence on rare classes

        # Use class 2 F1 score for monitoring
        # monitor_field = "val_f1_class_2"
        # monitor_field = "val_precision"
        monitor_field = "val_mcc"
        monitor_mode = "max"

        min_delta = 0  # Changed from 0.001 to capture all improvements
        early_patience = 20
        plateau_patience = 8

        # print(f"monitor_field: {monitor_field}, monitor_mode: {monitor_mode}, early_patience: {early_patience}")

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

        # K.set_value(self.model.optimizer.learning_rate, 0.001)

        # if self.dbg_verbose:
        class_weight = self.get_class_weight_dict()
        # print("")
        # print(
        #     f"    training model: {self.name}  batch_size:{self.batch_size} class_weight:{class_weight}"
        # )

        # print("    train_tensor:{} test_tensor:{}".format(np.shape(train_tensor), np.shape(test_tensor)))

        # Model weights are saved at the end of every epoch, if it's the best seen so far.
        fhis = self.model.fit(
            train_tensor,
            train_results,
            batch_size=self.batch_size,
            epochs=100,  # Increased epochs for better convergence
            callbacks=callbacks,
            validation_data=(test_tensor, test_results),
            class_weight=class_weight,  # Enable class weights
            verbose=1,
        )

        # The model weights (that are considered the best) are loaded into th model.
        # Note: don't need to do this if restore_best_weights=True in early_callback
        # self.update_model_weights()

        self.save()
        self.is_trained = True

        # score = self.model.evaluate(test_tensor, test_results, verbose=1)
        # print(f'    Model test score:{score[0]:.3f}  accuracy:{score[1]:.3f}')

        return

    def predict(self, data):

        # lazy loading because params can change up to this point
        if self.model is None:
            # load saved model if present
            self.model = self.load()

        if self.dataframeUtils.is_dataframe(data):
            # convert dataframe to tensor
            df_tensor = self.dataframeUtils.df_to_tensor(data, self.seq_len)
        else:
            df_tensor = data

        # If df_tensor is an MLX array, convert it to numpy for Keras
        try:
            import mlx.core as mx

            if isinstance(df_tensor, mx.array):
                df_tensor = np.array(df_tensor)
        except (ImportError, ModuleNotFoundError):
            pass

        if self.model is None:
            raise RuntimeError(
                f"CRITICAL: No model found for {self.name} at {self.model_path}. "
                "Ensure training completed successfully and custom layers are registered."
            )

        # run the prediction
        preds = self.model.predict(df_tensor, verbose=0)

        # # Using the Max value. This emulates the keras GlobalMaxPooling1D layer
        # # print(f'preds: {np.shape(preds)}')
        # preds = np.max(preds, axis=1)

        # convert softmax result into a nary value
        # predictions = np.argmax(preds, axis=1)  # softmax output

        # return predictions
        return preds

    # -----------------------------------------
    # functions to help with calculating class weights

    class_weights = []
    class_weight_dict = {}

    def calculate_class_weights(self, labels):
        # labels: 1-hot encoded
        class_counts = np.sum(labels, axis=0)
        # avoid division by zero
        class_weights = np.where(class_counts > 0, 1.0 / class_counts, 0.0)
        # normalize to sum to 1
        class_weights = class_weights / class_weights.sum()
        return class_weights

    def set_class_weights(self, weights):

        # print(f'weights: {weights}')
        # self.class_weights = weights / np.max(weights)  # normalise to 0..1 range
        self.class_weights = weights

        # print(f"    Raw class_weights: {self.class_weights}")

        # You can then use the class_weights array as a dictionary for the class_weight argument in Keras
        self.class_weight_dict = dict(enumerate(self.class_weights))

        return

    def get_class_weights(self):
        return self.class_weights

    def get_class_weight_dict(self):
        return self.class_weight_dict

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

    def get_config(self):
        config = super().get_config()
        config["clean_data_required"] = (self.clean_data_required,)
        config["is_trained"] = (self.is_trained,)
        # config['custom_objects'] = self.custom_objects
        # config['optimizer'] = weakref.proxy(self.optimizer)
        return {**config}
