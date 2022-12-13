# class that implements an Anomaly detection autoencoder, based on an LSTM model
# This version uses sequences of data, so do not feed it 'point' data, it must be time ordered


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

import keras
from keras import layers

import h5py

class AnomalyDetector_LSTM():

    dbg_ignore_load_failure = True # creates model if not present
    dbg_verbose = 1
    first_run = True
    # first_run = False # TEMP DBG

    encoder: keras.Model = None
    decoder: keras.Model = None
    autoencoder: keras.Model = None

    # these will be overwritten by the specific autoencoder
    name = "AnomalyDetector"
    inner_dim = 16
    compression_ratio = 2
    outer_dim = compression_ratio * inner_dim
    num_features = 32

    checkpoint_path = ""
    model_path = ""
    # tag = ""
    is_trained = False
    clean_data_required = False # training data can contain anomalies

    # the following affect training of the model.
    seq_len = 8  # 'depth' of training sequence
    # num_epochs = 512  # number of iterations for training
    num_epochs = 256  # number of iterations for training
    # num_epochs = 32 # TEMP FOR DEBUG ONLY
    batch_size = 1024  # batch size for training

    def __init__(self, num_features, pair):
        super().__init__()

        self.num_features = num_features
        self.outer_dim = self.compression_ratio * self.inner_dim
        # use dimensions in name to avoid conflict with other autoencoders
        cname = pair.split("/")[0]
        self.name =  self.__class__.__name__ + "_" + str(self.num_features) + "_" + str(self.inner_dim) + "_" + cname


        self.checkpoint_path = self.get_checkpoint_path()
        self.model_path = self.get_model_path()

        if self.num_features < (self.outer_dim):
            print("WARNING: num_features ({}) less than expected (<{})".format(self.num_features, self.outer_dim))

        # load saved model if present
        if os.path.exists(self.model_path):
            self.autoencoder = self.load()
            if self.autoencoder == None:
                if self.dbg_ignore_load_failure:
                    print("    Restore failed, building new model...")
                    self.build_model()
                else:
                    print("    Failed to load model ({})".format(self.model_path))
        else:
            print("    Model file not found ({}). Creating new model...".format(self.model_path))
            self.build_model()

        if self.autoencoder == None:
            print("    ERR: model not created!")
        else:
            self.autoencoder.summary()

    # override the build_model function in subclasses
    def build_model(self):

        self.autoencoder = keras.Sequential(name=self.name)

        #NOTE: don't use relu with LSTMs, cannot use GPU if you do (much slower)

        # Encoder
        self.autoencoder.add(layers.Bidirectional(
            layers.LSTM(self.outer_dim, return_sequences=True, activation='tanh'), input_shape=(self.seq_len, self.num_features)
        ))
        # self.autoencoder.add(layers.RepeatVector(self.num_features))

        # self.autoencoder.add(layers.Dropout(rate=0.2))
        self.autoencoder.add(layers.Bidirectional(
        layers.LSTM(int(self.outer_dim/2), return_sequences=True, activation='tanh')
        ))
        # self.autoencoder.add(layers.Dropout(rate=0.2))
        self.autoencoder.add(layers.Dense(self.inner_dim, activation='tanh', name='encoder_output'))

        # Decoder
        self.autoencoder.add(layers.Bidirectional(
            layers.LSTM(int(self.outer_dim/2), return_sequences=True, activation='tanh', input_shape=(1, self.inner_dim))
        ))
        # self.autoencoder.add(layers.Dropout(rate=0.2))
        self.autoencoder.add(layers.Bidirectional(
            layers.LSTM(self.outer_dim, return_sequences=True, activation='tanh')
        ))
        # self.autoencoder.add(layers.Dropout(rate=0.2))
        self.autoencoder.add(layers.Dense(self.num_features, activation=None))

        # optimizer = keras.optimizers.Adam()
        optimizer = keras.optimizers.Adam(learning_rate=0.01)

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

        # train_tensor = np.array(df_train).reshape(df_train.shape[0], 1, df_train.shape[1])
        # test_tensor = np.array(df_test).reshape(df_test.shape[0], 1, df_test.shape[1])

        train_tensor = self.df_to_tensor(df_train, self.seq_len)
        test_tensor = self.df_to_tensor(df_test, self.seq_len)


        # if self.dbg_verbose > 0:
        #     min_val = (df_test.min()).min()
        #     max_val = (df_test.max()).max()
        #     print("df_test min: {:.3f} max: {:.3f}".format(min_val, max_val))

        monitor_field = 'loss'
        monitor_mode = "min"

        # # if first run, loosen constraints
        # if self.first_run:
        #     self.first_run = False
        #     early_patience = 8
        #     plateau_patience = 6
        # else:
        #     early_patience = 4
        #     plateau_patience = 4
        early_patience = 4
        plateau_patience = 4

        # callback to control early exit on plateau of results
        early_callback = keras.callbacks.EarlyStopping(
            monitor=monitor_field,
            mode=monitor_mode,
            patience=early_patience,
            min_delta=0.0001,
            verbose=1)

        plateau_callback = keras.callbacks.ReduceLROnPlateau(
            monitor=monitor_field,
            mode=monitor_mode,
            factor=0.1,
            min_delta=0.0001,
            patience=plateau_patience,
            verbose=0)

        # callback to control saving of 'best' model
        # Note that we use validation loss as the metric, not training loss
        checkpoint_callback = keras.callbacks.ModelCheckpoint(
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

        self.save()
        self.is_trained = True

        return

    # evaluate model using the supplied (normalised) dataframe as test data.
    def evaluate(self, df_norm: DataFrame):

        # train_tensor = np.array(df_train).reshape(df_train.shape[0], 1, df_train.shape[1])
        # test_tensor = np.array(df_norm).reshape(df_norm.shape[0], 1, df_norm.shape[1])
        test_tensor = self.df_to_tensor(df_norm, self.seq_len)

        print("    Predicting...")
        preds = self.autoencoder.predict(test_tensor, verbose=1)

        print("    Comparing...")
        score = self.autoencoder.evaluate(test_tensor, preds, return_dict=True, verbose=2)
        print("model:{} score:{} ".format(self.name, score))
        # print("tensors equal: ", (test_tensor == preds))

        loss = tf.keras.metrics.mean_squared_error(test_tensor, preds)
        print("    loss:{} {}".format(np.shape(loss), loss))
        loss = np.array(loss[0])
        print("    loss:")
        print("        sum:{:.3f} min:{:.3f} max:{:.3f} mean:{:.3f} std:{:.3f}".format(loss.sum(),
                                                                                       loss.min(), loss.max(),
                                                                                       loss.mean(), loss.std()))
        return

    # 'recosnstruct' a dataframe by passing it through the autoencoder
    def reconstruct(self, df_norm:DataFrame) -> DataFrame:
        cols = df_norm.columns
        # tensor = np.array(df_norm).reshape(df_norm.shape[0], 1, df_norm.shape[1])
        tensor = self.df_to_tensor(df_norm, self.seq_len)
        encoded_tensor = self.autoencoder.predict(tensor, verbose=1)
        print("    encoded_tensor:{}".format(np.shape(encoded_tensor)))
        encode_array = encoded_tensor[:, 0, :]
        encoded_array = encode_array.reshape(np.shape(encoded_tensor)[0], np.shape(encoded_tensor)[2])
        print("    encoded_array:{}".format(np.shape(encoded_array)))


        return pd.DataFrame(encoded_array, columns=cols)

    # transform supplied (normalised) dataframe into a lower dimension version
    def transform(self, df_norm: DataFrame) -> DataFrame:
        cols = df_norm.columns
        # tensor = np.array(df_norm).reshape(df_norm.shape[0], 1, df_norm.shape[1])
        tensor = self.df_to_tensor(df_norm, self.seq_len)
        encoded_tensor = self.encoder.predict(tensor, verbose=1)
        encoded_array = encoded_tensor.reshape(np.shape(encoded_tensor)[0], np.shape(encoded_tensor)[2])

        return pd.DataFrame(encoded_array, columns=cols)


    # only need to override/define the predict function
    def predict(self, df_norm: DataFrame):

        # convert to tensor format and run the autoencoder
        # tensor = np.array(df_norm).reshape(df_norm.shape[0], 1, df_norm.shape[1])
        tensor = self.df_to_tensor(df_norm, self.seq_len)

        predict_tensor = self.autoencoder.predict(tensor, verbose=1)

        # not sure why, but predict sometimes returns an odd length
        if np.shape(predict_tensor)[0] != np.shape(tensor)[0]:
            print("    ERR: prediction length mismatch ({} vs {})".format(len(predict_tensor), np.shape(tensor)[0]))
            predictions = np.zeros(df_norm.shape[0], dtype=float)
        else:
            # get losses by comparing input to output
            msle = tf.keras.losses.msle(predict_tensor, tensor)
            msle = msle[:, 0]

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
        keras.models.save_model(self.autoencoder, filepath=path)
        return

    # load the full encoder model from the (optional) supplied path. Use this to load a fully trained autoencoder
    def load(self, path=""):
        if len(path) == 0:
            path = self.model_path

        # if model exists, load it
        if os.path.exists(path):
            print("    Loading existing model ({})...".format(path))
            try:
                self.autoencoder = keras.models.load_model(path, compile=False)
                # optimizer = keras.optimizers.Adam()
                optimizer = keras.optimizers.Adam(learning_rate=0.001)
                self.autoencoder.compile(metrics=['accuracy', 'mse'], loss='mse', optimizer=optimizer)
                self.is_trained = True

            except Exception as e:
                print("    ", str(e))
                print("    Error loading model from {}. Check whether model format changed".format(path))
        else:
            print("    model not found ({})...".format(path))

        return self.autoencoder

    # get the encoder part of the autoencoder
    def get_encoder(self) -> keras.Model:
        return self.encoder

    def get_checkpoint_path(self):
        # Note that keras expects it to be called 'checkpoint'
        checkpoint_dir = '/tmp'
        model_path = checkpoint_dir + "/" + self.name + "/" + "checkpoint"
        return model_path

    def get_model_path(self):
        # path to 'full' model file
        # TODO: include input/output sizes in name, to help prevent mismatches?!
        save_dir = self.__class__.__name__ + '/'
        model_path = save_dir + self.name + ".h5"
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


    def df_to_tensor(self, df, seq_len):
        # input format = [nrows, nfeatures], output = [nrows, seq_len, nfeatures]
        # print("df type:", type(df))
        if not isinstance(df, type([np.ndarray, np.array])):
            data = np.array(df)
        else:
            data = df

        nrows = np.shape(data)[0]
        nfeatures = np.shape(data)[1]
        tensor_arr = np.zeros((nrows, seq_len, nfeatures), dtype=float)
        zero_row = np.zeros((nfeatures), dtype=float)
        # tensor_arr = []

        # print("data:{} tensor:{}".format(np.shape(data), np.shape(tensor_arr)))
        # print("nrows:{} nfeatures:{}".format(nrows, nfeatures))

        reverse = True

        # fill the first part (0..seqlen rows), which are only sparsely populated
        for row in range(seq_len):
            for seq in range(seq_len):
                if seq >= (seq_len - row - 1):
                    src_row = (row + seq) - seq_len + 1
                    tensor_arr[row][seq] = data[src_row]
                else:
                    tensor_arr[row][seq] = zero_row
            if reverse:
                tensor_arr[row] = np.flipud(tensor_arr[row])

        # fill the rest
        # print("Data:{}, len:{}".format(np.shape(data), seq_len))
        for row in range(seq_len, nrows):
            tensor_arr[row] = data[(row - seq_len) + 1:row + 1]
            if reverse:
                tensor_arr[row] = np.flipud(tensor_arr[row])

        # print("data: ", data)
        # print("tensor: ", tensor_arr)
        # print("data:{} tensor:{}".format(np.shape(data), np.shape(tensor_arr)))
        return tensor_arr
