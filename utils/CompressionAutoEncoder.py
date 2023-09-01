# class that implements an Auto-Encoder for dimensional reduction of a panda dataframe
# This can be used as-is, and can also be sub-classed - just override the build_model function and create
# different encoder and decoder variables

# Note: this is intended to do some pretty drastic compression, and is intended for use with unsupervised
# anomaly detection algorithms/classifiers (which typically struggle with high dimensions)


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
# from keras import layers
# from tf.keras.optimizers import SGD
import h5py

class CompressionAutoEncoder():

    dbg_ignore_load_failure = True
    dbg_verbose = 1


    encoder: tf.keras.Model =  None
    decoder: tf.keras.Model =  None
    autoencoder: tf.keras.Model =  None

    # these will be overwritten by the specific autoencoder
    name = "CompressionAutoEncoder"
    tag = ""
    latent_dim = 16
    num_features = 128
    checkpoint_path = ""
    model_path = ""
    is_trained = False
    clean_data_required = False # training data can contain anomalies

    # the following affect training of the model.
    seq_len = 8  # 'depth' of training sequence
    num_epochs = 512  # number of iterations for training
    batch_size = 1024  # batch size for training

    def __init__(self, num_features, num_dims=16, tag=""):
        super().__init__()
        
        self.name = tag + self.__class__.__name__
        self.num_features = num_features
        self.latent_dim = num_dims
        self.tag = tag
        self.checkpoint_path = self.get_checkpoint_path()
        self.model_path = self.get_model_path()

        # load saved model if present
        if os.path.exists(self.model_path):
            self.autoencoder = self.load()
            if self.autoencoder == None:
                if self.dbg_ignore_load_failure:
                    print("    Restore failed, building new model...")
                    self.build_model(num_features, num_dims)
                else:
                    print("    Failed to load model ({})".format(self.model_path))
        else:
            print("    No saved model found, building new model...")
            self.build_model(num_features, num_dims)

        if self.autoencoder == None:
            print("    ERR: model not created!")
        else:
            self.autoencoder.summary()

    # override the build_model function in subclasses
    def build_model(self, num_features, num_dims):
        # default autoencoder is a (fairly) simple set of dense layers
        self.encoder = tf.keras.Sequential(name="encoder")
        self.encoder.add(tf.keras.layers.Dense(128, activation='relu', input_shape=(1, self.num_features)))
        # self.encoder.add(tf.keras.layers.Dropout(rate=0.1))
        # self.encoder.add(tf.keras.layers.Dense(96, activation='relu'))
        # self.encoder.add(tf.keras.layers.Dropout(rate=0.1))
        self.encoder.add(tf.keras.layers.Dense(1024, activation='relu'))
        # self.encoder.add(tf.keras.layers.Dropout(rate=0.1))
        self.encoder.add(tf.keras.layers.Dense(self.latent_dim, activation='relu', name='encoder_output'))

        self.decoder = tf.keras.Sequential(name="decoder")
        self.decoder.add(tf.keras.layers.Dense(1024, activation='relu', input_shape=(1, self.latent_dim)))
        # self.encoder.add(tf.keras.layers.Dropout(rate=0.1))
        # self.decoder.add(tf.keras.layers.Dense(96, activation='relu'))
        # self.encoder.add(tf.keras.layers.Dropout(rate=0.1))
        # self.decoder.add(tf.keras.layers.Dense(64, activation='relu'))
        # self.encoder.add(tf.keras.layers.Dropout(rate=0.1))
        self.decoder.add(tf.keras.layers.Dense(128, activation='relu'))
        self.decoder.add(tf.keras.layers.Dense(self.num_features, activation=None))

        self.autoencoder = tf.keras.Sequential(name=self.name)
        self.autoencoder.add(self.encoder)
        self.autoencoder.add(self.decoder)

        # optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0,
        #                                   amsgrad=False)

        optimizer = tf.keras.optimizers.SGD(learning_rate=1, momentum=0.9)
        # optimizer = tf.keras.optimizers.Adam(learning_rate=0.1)

        self.autoencoder.compile(metrics=['accuracy', 'mse'], loss='mse', optimizer=optimizer)

        self.update_model_weights()

        return


    # update training using the suplied (normalised) dataframe. Training is cumulative
    def train(self, df_train: DataFrame, df_test: DataFrame, train_labels, test_labels):
        

        train_tensor = np.array(df_train).reshape(df_train.shape[0], 1, df_train.shape[1])
        test_tensor = np.array(df_test).reshape(df_test.shape[0], 1, df_test.shape[1])


        # callback to control early exit on plateau of results
        early_callback = tf.keras.callbacks.EarlyStopping(
            monitor="loss",
            mode="min",
            patience=4,
            verbose=1)

        plateau_callback = tf.keras.callbacks.ReduceLROnPlateau(
            monitor='loss',
            factor=0.1,
            min_delta=0.0001,
            patience=4,
            verbose=0)

        # callback to control saving of 'best' model
        # Note that we use validation loss as the metric, not training loss
        checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
            filepath=self.checkpoint_path,
            save_weights_only=True,
            monitor='loss',
            mode='min',
            save_best_only=True,
            verbose=0)

        # if self.dbg_verbose:
        print("")
        print("    training model: {}...".format(self.name))

        # print("    train_tensor:{} test_tensor:{}".format(np.shape(train_tensor), np.shape(test_tensor)))

        # Model weights are saved at the end of every epoch, if it's the best seen so far.
        fhis = self.autoencoder.fit(train_tensor, train_tensor,
                                    batch_size=self.batch_size,
                                    epochs=self.num_epochs,
                                    callbacks=[plateau_callback, early_callback, checkpoint_callback],
                                    validation_data=(test_tensor, test_tensor),
                                    verbose=1)

        # The model weights (that are considered the best) are loaded into th model.
        self.update_model_weights()

        return


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
            z_scores = self.mad_score(msle)
            predictions = np.where(z_scores > threshold, 1.0, 0.0)

        return predictions

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

    # 'recosnstruct' a dataframe by passing it through the autoencoder (mostly for debug)
    def reconstruct(self, df_norm: DataFrame) -> DataFrame:
        cols = df_norm.columns
        tensor = np.array(df_norm).reshape(df_norm.shape[0], 1, df_norm.shape[1])
        encoded_tensor = self.autoencoder.predict(tensor, verbose=1)
        encoded_array = encoded_tensor.reshape(np.shape(encoded_tensor)[0], np.shape(encoded_tensor)[2])

        return pd.DataFrame(encoded_array, columns=cols)

    # transform supplied (normalised) dataframe into a lower dimension version
    def transform(self, df_norm: DataFrame) -> DataFrame:

        if self.encoder is None:
            self.encoder = self.autoencoder.get_layer("encoder")

        tensor = np.array(df_norm).reshape(df_norm.shape[0], 1, df_norm.shape[1])
        encoded_tensor = self.encoder.predict(tensor, verbose=1)
        encoded_array = encoded_tensor.reshape(np.shape(encoded_tensor)[0], np.shape(encoded_tensor)[2])

        return pd.DataFrame(encoded_array)

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
                # optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
                optimizer = tf.keras.optimizers.SGD(learning_rate=1, momentum=0.9)
                # optimizer = tf.keras.optimizers.Adam(learning_rate=0.1)

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
        curr_class = self.__class__.__name__
        model_path = checkpoint_dir + "/" + self.tag + curr_class + "/" + "checkpoint"
        return model_path

    def get_model_path(self):
        # path to 'full' model file
        # TODO: include input/output sizes in name, to help prevent mismatches?!
        save_dir = './'
        curr_class = self.__class__.__name__
        model_path = save_dir + self.tag + curr_class + ".keras"
        return model_path

    def update_model_weights(self):

        # if checkpoint already exists, load the weights
        if os.path.exists(self.checkpoint_path):
            print("    Loading existing model weights ({})...".format(self.checkpoint_path))
            try:
                self.autoencoder.load_weights(self.checkpoint_path)
            except:
                print("    Error loading weights from {}. Check whether model format changed".format(self.checkpoint_path))
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
        return False