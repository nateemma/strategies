# base class that implements Neural Network Binary Classifier.
# This class implements a keras classifier that implements an autoencoder

# subclasses should override the create_model() method


import numpy as np
from pandas import DataFrame, Series
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
warnings.simplefilter(action='ignore', category=pd.errors.PerformanceWarning)

import random

import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
os.environ['TF_DETERMINISTIC_OPS'] = '1'

import tensorflow as tf

seed = 42
os.environ['PYTHONHASHSEED'] = str(seed)
random.seed(seed)
tf.random.set_seed(seed)
np.random.seed(seed)

tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.WARN)

#import keras
from keras import layers

import h5py

from DataframeUtils import DataframeUtils
from ClassifierKeras import ClassifierKeras

class ClassifierKerasEncoder(ClassifierKeras):

    clean_data_required = False
    encoder_layer = 'encoder_output'

    # create model - subclasses should overide this
    def create_model(self, seq_len, num_features):

        model = None
        outer_dim = 64
        inner_dim = 16

        print("    WARNING: create_model() should be defined by the subclass")
        # create a simple model for illustrative purposes (or to test the framework)
        model = tf.keras.Sequential(name=self.name)

        # Encoder
        model.add(layers.Dense(outer_dim, activation='relu', input_shape=(seq_len, num_features)))
        model.add(layers.Dense(2 * outer_dim, activation='relu'))
        model.add(layers.Dense(inner_dim, activation='relu', name=self.encoder_layer))  # name is mandatory

        # Decoder
        model.add(layers.Dense(2 * outer_dim, activation='relu', input_shape=(1, inner_dim)))
        model.add(layers.Dense(outer_dim, activation='relu'))

        model.add(layers.Dense(num_features, activation=None))

        return model

    def compile_model(self, model):

        optimizer = tf.keras.optimizers.Adam(learning_rate=0.01)

        model.compile(metrics=['accuracy', 'mse'], loss='mse', optimizer=optimizer)

        return model

    # update training using the suplied (normalised) dataframe. Training is cumulative
    # the 'labels' args should contain 0.0 for normal results, '1.0' for anomalies (buy or sell)
    def train(self, df_train_norm, df_test_norm, train_results, test_results, force_train=False):

        if self.is_trained and not force_train:
            return

        if self.model is None:
            self.model = self.create_model(self.seq_len, self.num_features)
            if self.model is None:
                print("    ERR: model not created")
                return

            self.model = self.compile_model(self.model)
            self.model.summary()

        if self.dataframeUtils.is_dataframe(df_train):
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

        monitor_field = 'loss'
        monitor_mode = "min"
        early_patience = 4
        plateau_patience = 4

        # callback to control early exit on plateau of results
        early_callback = tf.keras.callbacks.EarlyStopping(
            monitor=monitor_field,
            mode=monitor_mode,
            patience=early_patience,
            min_delta=0.0001,
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

        # print("    train_tensor:{} test_tensor:{}".format(np.shape(train_tensor), np.shape(test_tensor)))

        # Model weights are saved at the end of every epoch, if it's the best seen so far.
        
        # Note that this compares the input tensors to themselves
        fhis = self.model.fit(train_tensor, train_tensor,
                                    batch_size=self.batch_size,
                                    epochs=self.num_epochs,
                                    callbacks=callbacks,
                                    validation_data=(test_tensor, test_tensor),
                                    verbose=1)

        # # The model weights (that are considered the best) are loaded into th model.
        # self.update_model_weights()

        self.save()
        self.is_trained = True

        return


    def predict(self, data):


        # convert to tensor format and run the autoencoder
        # tensor = np.array(df_norm).reshape(df_norm.shape[0], 1, df_norm.shape[1])
        tensor = self.dataframeUtils.df_to_tensor(data, self.seq_len)

        predict_tensor = self.model.predict(tensor, verbose=1)

        # not sure why, but predict sometimes returns an odd length
        if np.shape(predict_tensor)[0] != np.shape(tensor)[0]:
            print("    ERR: prediction length mismatch ({} vs {})".format(len(predict_tensor), np.shape(tensor)[0]))
            predictions = np.zeros(data.shape[0], dtype=float)
        else:
            # get losses by comparing input to output
            msle = tf.keras.losses.msle(predict_tensor, tensor)
            msle = msle[:, 0]

            # mean + stddev method
            # threshold for anomaly scores
            threshold = np.mean(msle.numpy()) + 2.0 * np.std(msle.numpy())

            # anything anomylous results in a '1'
            predictions = np.where(msle > threshold, 1.0, 0.0)

            # # Median Absolute Deviation method
            # threshold = 3.0 # empirical for Dense
            # # threshold = 2.0 # empirical for Conv
            # z_scores = self.mad_score(msle)
            # predictions = np.where(z_scores > threshold, 1.0, 0.0)

            # # Mean Absolute Error (MAE) method
            # t1 = predict_tensor[:, 0, :].reshape(np.shape(predict_tensor)[0], np.shape(predict_tensor)[2])
            # t2 = tensor[:, 0, :].reshape(np.shape(tensor)[0], np.shape(tensor)[2])
            # print("    predict_tensor:{} tensor:{}".format(np.shape(predict_tensor), np.shape(tensor)))

            # mae_loss = np.mean(np.abs(predict_tensor - tensor), axis=1)
            # threshold = np.max(mae_loss)
            # predictions = np.where(mae_loss > threshold, 1.0, 0.0)
            # print("    predictions:{} data:{}".format(np.shape(predictions), predictions))

        return predictions

