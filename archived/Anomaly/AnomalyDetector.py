# class that implements an Anomaly detection autoencoder
# The idea is to train the autoencoder on data that does NOT contain the signal you are looking for (buy/sell)
# Then when the autoencoder tries to predict the transform, anything with unusual error is considered to be an 'anomoly'


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

class AnomalyDetector():

    dbg_ignore_load_failure = False
    dbg_verbose = 1
    first_run = True

    encoder: tf.keras.Model = None
    decoder: tf.keras.Model = None
    autoencoder: tf.keras.Model = None

    # these will be overwritten by the specific autoencoder
    name = "AnomalyDetector"
    latent_dim = 96
    num_features = 128
    checkpoint_path = ""
    model_path = ""
    tag = ""
    is_trained = False
    clean_data_required = False # training data can contain anomalies

    # the following affect training of the model.
    seq_len = 8  # 'depth' of training sequence
    num_epochs = 512  # number of iterations for training
    batch_size = 1024  # batch size for training

    def __init__(self, name, num_features, tag=""):
        super().__init__()

        self.name = name
        self.num_features = num_features
        self.tag = tag
        self.checkpoint_path = self.get_checkpoint_path()
        self.model_path = self.get_model_path()

        # TODO: load saved model if present?!
        if os.path.exists(self.model_path):
            self.autoencoder = self.load()
            if self.autoencoder == None:
                if self.dbg_ignore_load_failure:
                    print("    Restore failed, building new model...")
                    self.build_model()
                else:
                    print("    Failed to load model ({})".format(self.model_path))
        else:
            self.build_model()

        if self.autoencoder == None:
            print("    ERR: model not created!")
        else:
            self.autoencoder.summary()

    # override the build_model function in subclasses
    def build_model(self):

        # Note that these are very 'loose' models. We don't really care about compression etc. we just need to
        # establish a pattern such that an anomaly is more easily detected
        model_type = 0

        if model_type == 0:
            # default autoencoder is a (fairly) simple set of dense layers
            self.autoencoder = tf.keras.Sequential(name=self.name)

            # Encoder
            self.autoencoder.add(layers.Dense(96, activation='elu', input_shape=(1, self.num_features)))
            # self.autoencoder.add(layers.Dropout(rate=0.2))
            self.autoencoder.add(layers.Dense(96, activation='elu'))
            # self.autoencoder.add(layers.Dropout(rate=0.2))
            # self.autoencoder.add(layers.Dense(32, activation='elu'))
            # self.autoencoder.add(layers.Dropout(rate=0.2))
            self.autoencoder.add(layers.Dense(self.latent_dim, activation='elu', name='encoder_output'))

            # Decoder
            self.autoencoder.add(layers.Dense(96, activation='elu', input_shape=(1, self.latent_dim)))
            # self.autoencoder.add(layers.Dropout(rate=0.2))
            self.autoencoder.add(layers.Dense(96, activation='elu'))
            # self.autoencoder.add(layers.Dropout(rate=0.2))
            # self.autoencoder.add(layers.Dense(128, activation='elu'))
            self.autoencoder.add(layers.Dense(self.num_features, activation=None))

        elif model_type == 1:
            # LSTM
            self.autoencoder = tf.keras.Sequential(name=self.name)

            # Encoder
            self.autoencoder.add(layers.Dense(96, activation='elu', input_shape=(1, self.num_features)))
            self.autoencoder.add(layers.Dropout(rate=0.4))
            self.autoencoder.add(layers.LSTM(96, return_sequences=True, activation='elu'))
            # self.autoencoder.add(layers.LSTM(64, return_sequences=True, activation='elu'))
            # self.autoencoder.add(layers.Dropout(rate=0.4))
            # self.autoencoder.add(layers.LSTM(48, return_sequences=True, activation='elu'))
            # self.autoencoder.add(layers.Dropout(rate=0.4))
            self.autoencoder.add(layers.Dense(self.latent_dim, activation='elu', name='encoder_output'))

            # Decoder
            self.autoencoder.add(layers.LSTM(96, return_sequences=True, activation='elu', input_shape=(1, self.latent_dim)))
            self.autoencoder.add(layers.Dropout(rate=0.4))
            # self.autoencoder.add(layers.LSTM(64, return_sequences=True, activation='elu'))
            # self.autoencoder.add(layers.Dropout(rate=0.4))
            # self.autoencoder.add(layers.LSTM(96, return_sequences=True, activation='elu'))
            # self.autoencoder.add(layers.Dropout(rate=0.4))
            self.autoencoder.add(layers.Dense(self.num_features, activation=None))

        elif model_type == 2:
            # Convolutional
            self.autoencoder = tf.keras.Sequential(name=self.name)

            # Encoder
            self.autoencoder.add(layers.Conv1D(filters=64, kernel_size=7, padding="same", input_shape=(1, self.num_features)))
            self.autoencoder.add(layers.Dropout(rate=0.2))
            self.autoencoder.add(layers.Conv1D(filters=32, kernel_size=7, padding="same"))
            self.autoencoder.add(layers.Dropout(rate=0.2))
            self.autoencoder.add(layers.Conv1D(filters=16, kernel_size=7, padding="same", name='encoder_output'))

            # Decoder
            self.autoencoder.add(layers.Conv1DTranspose(filters=16, kernel_size=7, padding="same", input_shape=(1, self.num_features)))
            self.autoencoder.add(layers.Dropout(rate=0.2))
            self.autoencoder.add(layers.Conv1DTranspose(filters=32, kernel_size=7, padding="same"))
            self.autoencoder.add(layers.Dropout(rate=0.2))
            self.autoencoder.add(layers.Conv1DTranspose(filters=64, kernel_size=7, padding="same"))

            self.autoencoder.add(layers.Dense(self.num_features, activation=None))

        else:
            print("ERR: unknown model_type")
            self.autoencoder = None
            return


        # optimizer = tf.keras.optimizers.Adam()
        optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)

        self.autoencoder.compile(metrics=['accuracy', 'mse'], loss='mse', optimizer=optimizer)

        # self.update_model_weights()

        return

    # update training using the supplied (normalised) dataframe. Training is cumulative
    # the 'labels' args should contain 0.0 for normal results, '1.0' for anomalies (buy or sell)
    def train(self, df_train_norm: DataFrame, df_test_norm: DataFrame, train_labels, test_labels, force_train=False):

        if self.is_trained and not force_train:
            return

        # extract non-anomalous data
        df1 = df_train_norm.copy()
        df1['%labels'] = train_labels
        df1 = df1[(df1['%labels'] < 0.1)]
        df_train = df1.drop('%labels', axis=1)

        df2 = df_test_norm.copy()
        df2['%labels'] = test_labels
        df2 = df1[(df1['%labels'] < 0.1)]
        df_test = df2.drop('%labels', axis=1)

        train_tensor = np.array(df_train).reshape(df_train.shape[0], 1, df_train.shape[1])
        test_tensor = np.array(df_test).reshape(df_test.shape[0], 1, df_test.shape[1])


        # if self.dbg_verbose > 0:
        #     min_val = (df_test.min()).min()
        #     max_val = (df_test.max()).max()
        #     print("df_test min: {:.3f} max: {:.3f}".format(min_val, max_val))

        monitor_field = 'loss'
        monitor_mode = "min"

        # if first run, loosen constraints
        if self.first_run:
            self.first_run = False
            early_patience = 32
            plateau_patience = 6
        else:
            early_patience = 12
            plateau_patience = 4

        # callback to control early exit on plateau of results
        early_callback = tf.keras.callbacks.EarlyStopping(
            monitor=monitor_field,
            mode=monitor_mode,
            patience=early_patience,
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
        fhis = self.autoencoder.fit(train_tensor, train_tensor,
                                    batch_size=self.batch_size,
                                    epochs=self.num_epochs,
                                    callbacks=callbacks,
                                    validation_data=(test_tensor, test_tensor),
                                    verbose=1)

        # The model weights (that are considered the best) are loaded into th model.
        self.update_model_weights()

        return

    # evaluate model using the supplied (normalised) dataframe as test data.
    def evaluate(self, df_norm: DataFrame):

        # train_tensor = np.array(df_train).reshape(df_train.shape[0], 1, df_train.shape[1])
        test_tensor = np.array(df_norm).reshape(df_norm.shape[0], 1, df_norm.shape[1])

        print("    Predicting...")
        preds = self.autoencoder.predict(test_tensor, verbose=1)

        print("    Comparing...")
        score = self.autoencoder.evaluate(test_tensor, preds, return_dict=True, verbose=2)
        print("model:{} score:{} ".format(self.name, score))
        # print("tensors equal: ", (test_tensor == preds))

        loss = tf.keras.metrics.mean_squared_error(test_tensor, preds)
        # print("    loss:{} {}".format(np.shape(loss), loss))
        loss = np.array(loss[0])
        print("    loss:")
        print("        sum:{:.3f} min:{:.3f} max:{:.3f} mean:{:.3f} std:{:.3f}".format(loss.sum(),
                                                                                       loss.min(), loss.max(),
                                                                                       loss.mean(), loss.std()))
        return

    # 'recosnstruct' a dataframe by passing it through the autoencoder
    def reconstruct(self, df_norm:DataFrame) -> DataFrame:
        cols = df_norm.columns
        tensor = np.array(df_norm).reshape(df_norm.shape[0], 1, df_norm.shape[1])
        encoded_tensor = self.autoencoder.predict(tensor, verbose=1)
        encoded_array = encoded_tensor.reshape(np.shape(encoded_tensor)[0], np.shape(encoded_tensor)[2])

        return pd.DataFrame(encoded_array, columns=cols)

    # transform supplied (normalised) dataframe into a lower dimension version
    def transform(self, df_norm: DataFrame) -> DataFrame:
        cols = df_norm.columns
        tensor = np.array(df_norm).reshape(df_norm.shape[0], 1, df_norm.shape[1])
        encoded_tensor = self.encoder.predict(tensor, verbose=1)
        encoded_array = encoded_tensor.reshape(np.shape(encoded_tensor)[0], np.shape(encoded_tensor)[2])

        return pd.DataFrame(encoded_array, columns=cols)


    # only need to override/define the predict function
    def predict(self, df_norm: DataFrame):

        # convert to tensor format and
        # run the autoencoder
        tensor = np.array(df_norm).reshape(df_norm.shape[0], 1, df_norm.shape[1])
        predict_tensor = self.autoencoder.predict(tensor, verbose=1)

        # not sure why, but predict sometimes returns an odd length
        if len(predict_tensor) != np.shape(tensor)[0]:
            print("    ERR: prediction length mismatch ({} vs {})".format(len(predict_tensor), np.shape(tensor)[0]))
            predictions = np.zeros(df_norm.shape[0], dtype=float)
        else:
            # get losses by comparing input to output
            msle = tf.keras.losses.msle(predict_tensor, tensor)

            # # mean + stddev method
            # # threshold for anomaly scores
            # threshold = np.mean(msle.numpy()) + 2.0 * np.std(msle.numpy())
            #
            # # anything anomylous results in a '1'
            # predictions = np.where(msle > threshold, 1.0, 0.0)

            # Median Absolute Deviation method
            threshold = 3.0 # empirical for Dense
            # threshold = 2.0 # empirical for Conv
            z_scores = self.mad_score(msle)
            predictions = np.where(z_scores > threshold, 1.0, 0.0)

        return predictions

    # save the full encoder model to the (optional) supplied path
    def save(self, path=""):
        if len(path) == 0:
            path = self.model_path
        print("    saving model to: ", path)
        # self.autoencoder.save(path)
        tf.keras.models.save_model(self.autoencoder, filepath=path)
        return

    # load the full encoder model from the (optional) supplied path. Use this to load a fully trained autoencoder
    def load(self, path=""):
        if len(path) == 0:
            path = self.model_path

        # if model exists, load it
        if os.path.exists(path):
            print("    Loading existing model ({})...".format(path))
            try:
                self.autoencoder = tf.keras.models.load_model(path, compile=False)
                # optimizer = tf.keras.optimizers.Adam()
                optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
                self.autoencoder.compile(metrics=['accuracy', 'mse'], loss='mse', optimizer=optimizer)
                self.is_trained = True

            except Exception as e:
                print("    ", str(e))
                print("    Error loading model from {}. Check whether model format changed".format(path))
        else:
            print("    model not found ({})...".format(path))

        return self.autoencoder

    # get the encoder part of the autoencoder
    def get_encoder(self) -> tf.keras.Model:
        return self.encoder

    def get_checkpoint_path(self):
        # Note that keras expects it to be called 'checkpoint'
        checkpoint_dir = '/tmp'
        model_path = checkpoint_dir + "/" + self.tag + self.name + "/" + "checkpoint"
        return model_path

    def get_model_path(self):
        # path to 'full' model file
        # TODO: include input/output sizes in name, to help prevent mismatches?!
        save_dir = './'
        model_path = save_dir + self.tag + self.name + ".keras"
        return model_path

    def update_model_weights(self):

        # if checkpoint already exists, load the weights
        if os.path.exists(self.checkpoint_path):
            print("    Loading existing model weights ({})...".format(self.checkpoint_path))
            try:
                self.autoencoder.load_weights(self.checkpoint_path)
            except:
                print("    Error loading weights from {}. Check whether model format changed".format(
                    self.checkpoint_path))
        else:
            print("    model not found ({})...".format(self.checkpoint_path))

        return

    # Median Absolute Deviation
    def mad_score(self, points):
        """https://www.itl.nist.gov/div898/handbook/eda/section3/eda35h.htm """
        m = np.median(points)
        ad = np.abs(points - m)
        mad = np.median(ad)

        return 0.6745 * ad / mad

    def model_is_trained(self) -> bool:
        return self.is_trained

    def needs_clean_data(self) -> bool:
        return self.clean_data_required