# base class that implements Neural Network Trinary Classifier.
# This class implements a keras classifier that provides a trinary result (nothing, buy, sell)

# subclasses should override the create_model() method
from enum import Enum, auto

import numpy as np
import pandas as pd

pd.options.mode.chained_assignment = None  # default='warn'

# Strategy specific imports, files must reside in same folder as strategy
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent))
sys.path.append(str(Path(__file__).parent.parent))

import logging
import warnings
import random
import os
import weakref

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

# warnings.simplefilter(action='ignore', category=pd.errors.PerformanceWarning)



# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # errors only
# os.environ['TF_DETERMINISTIC_OPS'] = '1'
# os.environ['PYTHON_PICKLING_LOGLEVEL'] = '0'

# # workaround for memory leak in tensorflow 2.10
# os.environ['TF_RUN_EAGER_OP_AS_FUNCTION'] = '0'


# seed = 42
# os.environ['PYTHONHASHSEED'] = str(seed)
# random.seed(seed)
# tf.random.set_seed(seed)
# np.random.seed(seed)

# tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.WARN)

import sklearn

from ClassifierKeras import ClassifierKeras
from CustomWeightedLoss import CustomWeightedLoss
from CustomAdam import CustomAdam


# enum for results
class Result(Enum):
    NOTHING = 0
    BUY = 1
    SELL = 2


@keras.saving.register_keras_serializable(package="ClassifierKeras")
class ClassifierKerasTrinary(ClassifierKeras):
    clean_data_required = False
    custom_objects = {'CustomWeightedLoss': CustomWeightedLoss}

    # create model - subclasses should overide this
    def create_model(self, seq_len, num_features):

        model = None

        print("    WARNING: create_model() should be defined by the subclass")

        # create a simple model for illustrative purposes (or to test the framework)
        model = keras.Sequential(name=self.name)

        # NOTE: don't use relu with LSTMs, cannot use GPU if you do (much slower). Use tanh

        # simplest possible model:
        model.add(keras.layers.LSTM(128, return_sequences=True, activation='tanh', input_shape=(seq_len, num_features)))
        model.add(keras.layers.Dropout(rate=0.1))

        # last layer is a trinary decision - do not change
        model.add(keras.layers.Dense(3, activation='softmax'))

        return model

    def compile_model(self, model):

        optimizer = keras.optimizers.Adam(learning_rate=0.01)
        # optimizer = keras.src.optimizers.Adam(learning_rate=0.01)
        # optimizer = CustomAdam(learning_rate=0.01)
        # workaround for tensorflow issues with Adam:
        # optimizer = keras.optimizers.legacy.Adam(learning_rate=0.01)

        # import tensorflow_addons
        # optimizer = tensorflow_addons.optimizers.AdamW(learning_rate=0.001, weight_decay=0.004)
        # optimizer = tf.train.AdamWOptimizer(learning_rate=0.001, weight_decay=0.004)
        # optimizer = keras.optimizers.SGD(learning_rate=0.01, momentum=0.9, nesterov=True, clipnorm=1.)

        # The loss function makes a big difference, probably because the class distribution (hold/sell/buy) is
        # very unbalanced, i.e. way more holds than buys or sells

        # categorical cross entropy is the 'standard' loss function for categorization problems
        # loss = "categorical_crossentropy"
        # loss = "sparse_categorical_crossentropy"

        # Try some custom loss functions, where we can weight the los based on actual class distribution
        # loss = CustomWeightedLoss(CustomWeightedLoss.WeightedLossType.CATEGORICAL_FOCAL, self.get_class_weights())
        loss = CustomWeightedLoss(CustomWeightedLoss.WeightedLossType.WEIGHTED_CATEGORICAL, self.get_class_weights())
        # print(f"weights: {self.get_class_weights()}")

        buy_p = keras.metrics.Precision(class_id=1)
        metrics = ["categorical_accuracy", buy_p]

        model.compile(
            optimizer=optimizer,
            loss=loss,
            metrics=metrics,
        )

        return model

    # update training using the suplied (normalised) dataframe. Training is cumulative
    # the 'labels' args should contain 0.0 for normal results, '1.0' for buys, 2.0 for sells
    def train(self, df_train_norm, df_test_norm, train_results, test_results, force_train=False):

        # lazy loading because params can change up to this point
        if self.model is None:
            # load saved model if present
            self.model = self.load()

        # print(f'is_trained:{self.is_trained} force_train:{force_train}')

        # print("    train_tensor:{} test_tensor:{}".format(np.shape(train_tensor), np.shape(test_tensor)))

        # if model is already trained, and caller is not requesting a re-train, then just return
        if (self.model is not None) and self.model_is_trained() and (not force_train) and (
                not self.new_model_created()):
            # print(f"    Not training. is_trained:{self.is_trained} force_train:{force_train} new_model:{self.new_model}")
            print("    Model is already trained")
            return

        if self.dataframeUtils.is_dataframe(df_train_norm):
            # remove rows with positive labels?!
            if self.clean_data_required:
                df1 = df_train_norm.copy()
                df1['%labels'] = train_results
                df1 = df1[(df1['%labels'] < 0.1)]
                df_train = df1.drop('%labels', axis=1)

                df2 = df_train_norm.copy()
                df2['%labels'] = train_results
                df2 = df2[(df2['%labels'] < 0.1)]
                df_test = df2.drop('%labels', axis=1)
            else:
                df_train = df_train_norm.copy()
                df_test = df_test_norm.copy()

            train_tensor = self.dataframeUtils.df_to_tensor(df_train, self.seq_len)
            test_tensor = self.dataframeUtils.df_to_tensor(df_test, self.seq_len)
        else:
            # already in tensor format
            train_tensor = df_train_norm.copy()
            test_tensor = df_test_norm.copy()


        # set class weights (used by custom loss and metric functions)
        self.set_class_weights(train_results)

        # if model does not exist, create and compile it
        if self.model is None:
            self.model = self.create_model(self.seq_len, self.num_features)
            if self.model is None:
                print("    ERR: model not created")
                return
            self.model = self.compile_model(self.model)
            self.model.summary()


        # monitor_field = 'val_loss' # assess based on validation loss, rather than training loss

        monitor_mode = "min"
        monitor_field = 'loss'

        min_delta = 0.001
        early_patience = 8
        plateau_patience = 4

        # print(f"monitor_field: {monitor_field}, monitor_mode: {monitor_mode}, early_patience: {early_patience}")

        # callback to control early exit on plateau of results
        early_callback = keras.callbacks.EarlyStopping(
            monitor=monitor_field,
            mode=monitor_mode,
            patience=early_patience,
            min_delta=min_delta,
            restore_best_weights=True,
            verbose=0)

        plateau_callback = keras.callbacks.ReduceLROnPlateau(
            monitor=monitor_field,
            mode=monitor_mode,
            factor=0.1,
            min_delta=min_delta,
            patience=plateau_patience,
            verbose=0)

        # callback to control saving of 'best' model
        # Note that we use validation loss as the metric, not training loss
        checkpoint_callback = keras.callbacks.ModelCheckpoint(
            filepath=self.get_checkpoint_path(),
            save_weights_only=True,
            monitor=monitor_field,
            mode=monitor_mode,
            save_best_only=True,
            verbose=0)

        callbacks = [plateau_callback, early_callback, checkpoint_callback]

        # K.set_value(self.model.optimizer.learning_rate, 0.001)

        # if self.dbg_verbose:
        print("")
        print("    training model: {}...".format(self.name))

        # print("    train_tensor:{} test_tensor:{}".format(np.shape(train_tensor), np.shape(test_tensor)))

        # Model weights are saved at the end of every epoch, if it's the best seen so far.
        fhis = self.model.fit(train_tensor, train_results,
                              batch_size=self.batch_size,
                              epochs=self.num_epochs,
                              callbacks=callbacks,
                              validation_data=(test_tensor, test_results),
                              # class_weight=self.get_class_weight_dict(),
                              verbose=1)

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

        if self.model == None:
            print("    ERR: no model for predictions")
            predictions = np.zeros(np.shape(df_tensor)[0], dtype=float)
            return predictions

        # run the prediction
        preds = self.model.predict(df_tensor, verbose=0)

        # # Using the Max value. This emulates the keras GlobalMaxPooling1D layer
        # # print(f'preds: {np.shape(preds)}')
        # preds = np.max(preds, axis=1)

        # convert softmax result into a trinary value
        predictions = np.argmax(preds, axis=1)  # softmax output

        return predictions

    # -----------------------------------------
    # functions to help with calculating class weights

    class_weights = []
    class_weight_dict = {}

    def set_class_weights(self, label_tensor):
        # Assuming your labels are one-hot encoded, you need to convert them to integers first
        y_train = tf.argmax(label_tensor,
                            axis=-1)  # creates tensor of shape (batch_size, timesteps) with integers 0, 1, or 2
        y_train = tf.reshape(y_train, [-1])  # flatten the tensor to a shape of (batch_size * timesteps,)
        y_train = y_train.numpy()  # convert the tensor to a numpy array

        # Now you can use the compute_class_weight function
        classes = np.unique(y_train)  # This will give you an array of [0, 1, 2]
        self.class_weights = sklearn.utils.class_weight.compute_class_weight(class_weight='balanced',
                                                                             classes=classes,
                                                                             y=y_train)
        # This will give you an array of class weights, such as [0.5, 4.0, 4.0]

        # Hack?: make sure buy is at least as important as sell
        if len(self.class_weights) >= 3:
            self.class_weights[2] = min(self.class_weights[1], self.class_weights[2])

        # normalise so that we can keep the metric range roughly in the 0..1 range

        # self.class_weights = self.class_weights / np.sum(self.class_weights)
        self.class_weights = self.class_weights / np.max(self.class_weights)

        # print(f'class_weights: {self.class_weights}')

        # You can then use the class_weights array as a dictionary for the class_weight argument in Keras
        self.class_weight_dict = dict(enumerate(self.class_weights))

        return

    def get_class_weights(self):
        return self.class_weights

    def get_class_weight_dict(self):
        return self.class_weight_dict
    
    def get_config(self):
        config = super().get_config()
        config['clean_data_required'] = self.clean_data_required,
        config['is_trained'] = self.is_trained,
        config['custom_objects'] = self.custom_objects
        # config['optimizer'] = weakref.proxy(self.optimizer)
        return {**config}

