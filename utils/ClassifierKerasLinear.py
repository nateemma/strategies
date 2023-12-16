# base class that implements Neural Network Binary Classifier.
# This class implements a keras classifier that provides a linear result (i.e. floats)
# subclasses should override the create_model() method


import numpy as np
import numpy
from pandas import DataFrame, Series
import pandas as pd

pd.options.mode.chained_assignment = None  # default='warn'

# Strategy specific imports, files must reside in same folder as strategy
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent))

import logging
import warnings
import random
import os
import tensorflow as tf

log = logging.getLogger(__name__)
# log.setLevel(logging.DEBUG)
warnings.simplefilter(action='ignore', category=pd.errors.PerformanceWarning)


# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
# os.environ['TF_DETERMINISTIC_OPS'] = '1'



# seed = 42
# os.environ['PYTHONHASHSEED'] = str(seed)
# random.seed(seed)
# tf.random.set_seed(seed)
# np.random.seed(seed)

# tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.WARN)

#import keras
# from keras import layers
# from keras import backend as K

import h5py

from DataframeUtils import DataframeUtils
from ClassifierKeras import ClassifierKeras


class ClassifierKerasLinear(ClassifierKeras):
    clean_data_required = False
    combine_models = True

    # create model - subclasses should override this
    def create_model(self, seq_len, num_features):

        print("    WARNING: create_model() should be defined by the subclass")

        model = tf.keras.Sequential()

        # simplest possible model:
        model.add(tf.keras.layers.LSTM(64, return_sequences=True, activation='tanh', input_shape=(seq_len, num_features)))
        model.add(tf.keras.layers.Dropout(rate=0.1))
        model.add(tf.keras.layers.Dense(1, activation='linear'))

        return model

    def compile_model(self, model):
        # optimizer = tf.keras.optimizers.Adam(learning_rate=0.01)
        # workaround for tensorflow issues with Adam:
        optimizer = tf.keras.optimizers.legacy.Adam(learning_rate=self.learning_rate)
        # optimizer = tf.keras.optimizers.SGD(learning_rate=1, momentum=0.9)
        model.compile(metrics=['mae', 'mse'], loss='mse', optimizer=optimizer)

        return model

    # update training using the suplied (normalised) dataframe. Training is cumulative
    def train(self, df_train_norm, df_test_norm, train_results, test_results, force_train=False, save_model=True):

        # lazy loading because params can change up to this point
        if self.model is None:
            # load saved model if present
            # print('    Loading model')
            self.model = self.load()

        # print(f'is_trained:{self.is_trained} force_train:{force_train}')

        new_model = self.new_model_created()

        # print(f"    is_trained:{self.is_trained} force_train:{force_train} new_model:{self.new_model} combine_models:{self.combine_models}")

        # if model is already trained, and caller is not requesting a re-train, then just return
        if (self.model is not None) and \
            (self.model_is_trained()) and \
            (not force_train) and \
            (not self.combine_models) or \
                (
                    self.model_is_trained() and (not new_model)
                ):
            # (not self.combine_models or (not new_model)):
            # print(f"    SKipping training - is_trained:{self.is_trained} force_train:{force_train} new_model:{self.new_model} combine_models:{self.combine_models}")
            # print("    Model is already trained")
            return


        # if model doesn't exist, create it (lazy initialisation)
        if self.model is None:
            self.num_features = np.shape(df_train_norm)[2]
            self.model = self.create_model(self.seq_len, self.num_features)
            if self.model is None:
                print("    ERR: model not created")
                return
            self.model = self.compile_model(self.model)
            self.model.summary()

        # if the input is a dataframe, we can 'clean' it, then convert to tensor format
        # cannot clean a tensor since it doesn't have column headings any more
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

        # set up callbacks
        monitor_field = 'loss'
        monitor_mode = "min"
        early_patience = 8
        plateau_patience = 4

        # callback to control early exit on plateau of results
        early_callback = tf.keras.callbacks.EarlyStopping(
            monitor=monitor_field,
            mode=monitor_mode,
            patience=early_patience,
            min_delta=0.00001,
            restore_best_weights=True,
            verbose=1)

        plateau_callback = tf.keras.callbacks.ReduceLROnPlateau(
            monitor=monitor_field,
            mode=monitor_mode,
            factor=0.1,
            min_delta=0.0001,
            patience=plateau_patience,
            verbose=0)

        # callback to control saving of 'best' model
        # Note that we use validation loss as the metric, not training loss
        checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
            filepath=self.checkpoint_path,
            save_weights_only=True,
            monitor=monitor_field,
            mode=monitor_mode,
            save_best_only=True,
            verbose=0)

        callbacks = [plateau_callback, early_callback, checkpoint_callback]

        # if self.dbg_verbose:
        print("")
        print("    training model: {}...".format(self.name))

        # print("    train_tensor:{} train_results:{}".format(np.shape(train_tensor), np.shape(train_results)))
        # print("    test_tensor:{}  test_results:{}".format(np.shape(test_tensor), np.shape(test_results)))

        # Model weights are saved at the end of every epoch, if it's the best seen so far.
        fhis = self.model.fit(train_tensor, train_results,
                                batch_size=self.batch_size,
                                epochs=self.num_epochs,
                                callbacks=callbacks,
                                validation_data=(test_tensor, test_results),
                                verbose=1)


        # reset learning rate
        if self.combine_models:
            tf.keras.backend.set_value(self.model.optimizer.learning_rate, self.learning_rate)

        # # The model weights (that are considered the best) are loaded into th model.
        # self.update_model_weights()

        if save_model:
            self.save()
        self.is_trained = True

        return

    def predict(self, data):

        # lazy loading because params can change up to this point
        if self.model is None:
            # load saved model if present
            self.model = self.load()

        # print(f'data type:{type(data)}')

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

        # print(f"predict() - preds: {np.shape(preds)}")

        # reshape so that we return just a straight array of predictions
        if preds.ndim == 3:
            predictions = np.squeeze(preds[:, 0, 0])
            # predictions = preds[:, np.shape(preds)[1]-1, 0]
        elif preds.ndim == 2:
            predictions = np.squeeze(preds[:, 0])
        else:
            print(f"    WARN: unexpected dimension ({preds.ndim}) for predictions")
            predictions = np.zeros(np.shape(df_tensor)[0], dtype=float)
        # preds = np.array(preds[:, 0]).reshape(-1, 1)
        # predictions = preds[:, 0]

        return predictions
