# base class that implements an Anomaly detector using keras models
# You should not really use this directly, use one of the specific subclasses instead, depending on the type
# of classifier (linear, binary, encoder etc.)

# specific classifier subclasses (linear etc) should override create_model, compile_model, train and predict
# specific models should then further override create_model


import numpy as np
import pandas as pd
from pandas import DataFrame

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

tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
logging.disable(logging.WARNING)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = '3'

# workaround for memory leak in tensorflow 2.10
os.environ['TF_RUN_EAGER_OP_AS_FUNCTION'] = '0'
mem_fraction = 0.4
config = tf.compat.v1.ConfigProto(device_count={'GPU': 0})
config.gpu_options.allow_growth = True
config.gpu_options.per_process_gpu_memory_fraction = mem_fraction
sess = tf.compat.v1.Session(config=config)
tf.compat.v1.keras.backend.set_session(sess)

seed = 42
os.environ['PYTHONHASHSEED'] = str(seed)
random.seed(seed)
tf.random.set_seed(seed)
np.random.seed(seed)

from DataframeUtils import DataframeUtils


class ClassifierKeras():
    model = None
    is_trained = False
    category = ""
    name = ""
    model_path = ""
    checkpoint_path = "/tmp/model.h5"
    seq_len = 8
    num_features = 64
    encoder_layer = 'encoder_output'
    encoder = None
    num_epochs = 256  # number of iterations for training
    batch_size = 1024  # batch size for training
    clean_data_required = False  # train with positive rows removed
    model_per_pair = False  # set to False to combine across all pairs
    new_model = False  # True if a new model was created this run
    dataframeUtils = None
    requires_dataframes = False  # set to True if classifier takes dataframes rather than tensors
    prescale_dataframe = True  # set to True if algorithms need dataframes to be pre-scaled
    single_prediction = False  # True if algorithm only produces 1 prediction (not entire data array)
    combine_models = False  # True means combine models for all pairs (unless model per pair). False will train only on 1st pair

    # ---------------------------

    # Note: pair is needed because we cannot combine model across pairs because of huge price differences

    def __init__(self, pair, seq_len, num_features, tag=""):
        super().__init__()

        self.loaded_from_file = False

        if self.model_per_pair:
            pair_suffix = "_" + pair.split("/")[0]
        else:
            pair_suffix = ""

        if tag == "":
            tag_suffix = ""
        else:
            tag_suffix = "_" + tag

        self.category = self.__class__.__name__
        self.name = self.category + pair_suffix + tag_suffix

        # self.model_path = self.get_model_path()
        self.set_model_name(self.category, self.name)
        self.seq_len = seq_len
        self.num_features = num_features
        # print("    num_features: ", num_features)

        if self.dataframeUtils is None:
            self.dataframeUtils = DataframeUtils()

    # ---------------------------

    # set model name - this overrides the default naming. This allows the strategy to set the naming convention
    # directory and extension are handled, just need to supply the ategory (e.g. the strat name) and main file name
    # caller will have to take care of adding pair names, tag etc.
    def set_model_name(self, category, model_name):
        root_dir = self.get_model_root_dir()
        save_dir = root_dir + category + '/'
        file_path = save_dir + model_name + ".h5"

        # update tracking vars (need to override defaults)
        self.category = category
        self.model_path = file_path
        self.name = model_name
        # print(f"    Set model path:{self.model_path}")

        return self.model_path

    # ---------------------------
    # sets the combine-Models flag.
    # If True, models will be combined across multiple pairs
    # If False, only first pair is used for training (unless per_pair is specified)
    def set_combine_models(self, combine_models):
        self.combine_models = combine_models

    # ---------------------------

    # create model - subclasses should overide this
    def create_model(self, seq_len, num_features):

        model = None
        outer_dim = 64
        inner_dim = 16

        print("    WARNING: create_model() should be defined by the subclass")
        # create a simple model for illustrative purposes (or to test the framework)
        model = tf.keras.Sequential(name=self.name)

        # Encoder
        model.add(tf.keras.layers.Dense(outer_dim, activation='relu', input_shape=(seq_len, num_features)))
        model.add(tf.keras.layers.Dense(2 * outer_dim, activation='relu'))
        model.add(tf.keras.layers.Dense(inner_dim, activation='relu', name=self.encoder_layer))  # name is mandatory

        # Decoder
        model.add(tf.keras.layers.Dense(2 * outer_dim, activation='relu', input_shape=(1, inner_dim)))
        model.add(tf.keras.layers.Dense(outer_dim, activation='relu'))

        model.add(tf.keras.layers.Dense(num_features, activation=None))
        return model

    # ---------------------------

    # compile the model. This is a distinct function because it can vary by model type
    def compile_model(self, model):

        optimizer = tf.keras.optimizers.Adam(learning_rate=0.01)

        model.compile(metrics=['accuracy', 'mse'], loss='mse', optimizer=optimizer)

        return model

    # ---------------------------

    # update training using the supplied (normalised) dataframe. Training is cumulative
    # the 'labels' args should contain 0.0 for normal results, '1.0' for anomalies (buy or sell)
    def train(self, df_train_norm, df_test_norm, train_results, test_results, force_train=False):

        # lazy loading because params can change up to this point
        if self.model is None:
            # load saved model if present
            self.model = self.load()

        # just return if model has already been trained, unless force_train is set, or this was a new model
        if self.model_is_trained() and (not force_train) and (not self.new_model_created()):
            return

        if self.model is None:
            self.model = self.create_model(self.seq_len, self.num_features)
            if self.model is None:
                print("    ERR: model not created")
                return

            self.model = self.compile_model(self.model)
            self.model.summary()

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
            filepath=self.get_checkpoint_path(),
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
        fhis = self.model.fit(train_tensor, train_tensor,
                              batch_size=self.batch_size,
                              epochs=self.num_epochs,
                              callbacks=callbacks,
                              validation_data=(test_tensor, test_tensor),
                              verbose=0)

        # # The model weights (that are considered the best) are loaded into th model.
        # self.update_model_weights()

        self.save()
        self.is_trained = True

        return

    # ---------------------------

    # run the model prediction against the entire data buffer
    def backtest(self, data):
        # for keras-based models, this is the same thing as running predict(). Here for compatibility with other types
        # print(f"    backtest. Data size:{np.shape(data)}")
        return self.predict(data)

    # ---------------------------

    def predict(self, data):

        # lazy loading because params can change up to this point
        if self.model is None:
            # load saved model if present
            self.model = self.load()

        # convert to tensor format and run the autoencoder
        if self.dataframeUtils.is_dataframe(data):
            # convert dataframe to tensor
            tensor = self.dataframeUtils.df_to_tensor(data, self.seq_len)
        else:
            tensor = data

        predict_tensor = self.model.predict(tensor, verbose=1)

        # not sure why, but predict sometimes returns an odd length
        if np.shape(predict_tensor)[0] != np.shape(tensor)[0]:
            print("    ERR: prediction length mismatch ({} vs {})".format(len(predict_tensor), np.shape(tensor)[0]))
            predictions = np.zeros(np.shape(tensor)[0], dtype=float)
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

    # ---------------------------

    # evaluate model using the supplied (normalised) dataframe as test data.
    def evaluate(self, data):

        # lazy loading because params can change up to this point
        if self.model is None:
            # load saved model if present
            self.model = self.load()

        if self.dataframeUtils.is_dataframe(data):
            # convert dataframe to tensor
            test_tensor = self.dataframeUtils.df_to_tensor(data, self.seq_len)
        else:
            test_tensor = data

        print("    Predicting...")
        preds = self.model.predict(test_tensor, verbose=0)

        print("    Comparing...")
        score = self.model.evaluate(test_tensor, preds, return_dict=True, verbose=0)
        print("model:{} score:{} ".format(self.name, score))

        loss = tf.keras.metrics.mean_squared_error(test_tensor, preds)
        # print("    loss:{} {}".format(np.shape(loss), loss))
        loss = np.array(loss[0])
        print("    loss:")
        print("        sum:{:.3f} min:{:.3f} max:{:.3f} mean:{:.3f} std:{:.3f}".format(loss.sum(),
                                                                                       loss.min(), loss.max(),
                                                                                       loss.mean(), loss.std()))
        return

    # ---------------------------

    # 'recosnstruct' a dataframe by passing it through the model
    def reconstruct(self, df_norm: DataFrame) -> DataFrame:

        # lazy loading because params can change up to this point
        if self.model is None:
            # load saved model if present
            self.model = self.load()

        cols = df_norm.columns
        tensor = self.dataframeUtils.df_to_tensor(df_norm, self.seq_len)
        encoded_tensor = self.model.predict(tensor, verbose=1)
        # print("    encoded_tensor:{}".format(np.shape(encoded_tensor)))
        encode_array = encoded_tensor[:, 0, :]
        encoded_array = encode_array.reshape(np.shape(encoded_tensor)[0], np.shape(encoded_tensor)[2])
        # print("    encoded_array:{}".format(np.shape(encoded_array)))

        return pd.DataFrame(encoded_array, columns=cols)

    # ---------------------------

    # transform supplied (normalised) dataframe into a lower dimension version
    def transform(self, df_norm: DataFrame) -> DataFrame:

        # lazy loading because params can change up to this point
        if self.model is None:
            # load saved model if present
            self.model = self.load()

        if self.encoder is None:
            self.encoder = self.model.get_layer(self.encoder_layer)
        cols = df_norm.columns
        # tensor = np.array(df_norm).reshape(df_norm.shape[0], 1, df_norm.shape[1])
        tensor = self.dataframeUtils.df_to_tensor(df_norm, self.seq_len)
        encoded_tensor = self.encoder.predict(tensor, verbose=1)
        encoded_array = encoded_tensor.reshape(np.shape(encoded_tensor)[0], np.shape(encoded_tensor)[2])

        return pd.DataFrame(encoded_array, columns=cols)

    # ---------------------------

    # returns path to the root directory used for storing models
    def get_model_root_dir(self):
        # set as subdirectory of location of this file (so that it can be included in the repository)
        file_dir = os.path.dirname(str(Path(__file__)))
        root_dir = file_dir + "/models/"
        if not os.path.exists(root_dir):
            os.makedirs(root_dir)
        return root_dir

    # ---------------------------

    # returns path to 'full' model file
    def get_model_path(self):
        root_dir = self.get_model_root_dir()
        save_dir = root_dir + self.category + '/'
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        model_path = save_dir + self.name + ".h5"
        return model_path

    # ---------------------------

    def get_checkpoint_path(self):
        checkpoint_dir = '/tmp' + "/" + self.name + "/"
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
        model_path = checkpoint_dir + "checkpoint.h5"
        return model_path

    # ---------------------------

    def save(self, path=""):

        if len(path) == 0:
            self.model_path = self.get_model_path()
            path = self.model_path
        else:
            self.model_path = path

        print("    saving model to: ", path)
        save_dir = os.path.dirname(path)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        tf.keras.models.save_model(self.model, filepath=path, save_format='h5')
        return

    # ---------------------------

    def load(self, path=""):

        if len(path) == 0:
            self.model_path = self.get_model_path()
            path = self.model_path
        else:
            self.model_path = path

        model = None

        # if model exists, load it
        if os.path.exists(path):
            print("    Loading existing model ({})...".format(path))
            try:
                # check for custom load function (used with custom layers)
                custom_load = getattr(self, "custom_load", None)
                if callable(custom_load):
                    model = self.custom_load(path)
                else:
                    model = tf.keras.models.load_model(path, compile=False)
                self.compile_model(model)
                self.is_trained = True

            except Exception as e:
                print("    ", str(e))
                print("    Error loading model from {}. Check whether model format changed".format(path))
        else:
            print("    model not found ({})...".format(path))
            # flag this as a new model. Note that this is a class global variable because we need to track this
            # across multiple instances (e.g. if we are combining all pairs into one model)
            if self.combine_models:
                ClassifierKeras.new_model = True
            else:
                ClassifierKeras.new_model = False

            self.is_trained = False

        return model

    # ---------------------------

    def model_exists(self) -> bool:
        path = self.get_model_path()
        return os.path.exists(path)

    # ---------------------------

    def model_is_trained(self) -> bool:
        return self.is_trained

    # ---------------------------

    def needs_clean_data(self) -> bool:
        # print("    clean_data_required: ", self.clean_data_required)
        return self.clean_data_required

    # ---------------------------

    def needs_dataframes(self) -> bool:
        return self.requires_dataframes

    # ---------------------------

    def prescale_data(self) -> bool:
        return self.prescale_dataframe

    # ---------------------------

    def new_model_created(self) -> bool:
        return ClassifierKeras.new_model

    # ---------------------------

    def returns_single_prediction(self) -> bool:
        return self.single_prediction

    # ---------------------------

    def update_model_weights(self):

        self.checkpoint_path = self.get_checkpoint_path()

        # if checkpoint already exists, load the weights
        if os.path.exists(self.checkpoint_path):
            print("    Loading model weights ({})...".format(self.checkpoint_path))
            try:
                self.model.load_weights(self.checkpoint_path)
            except:
                print("    Error loading weights from {}. Check whether model format changed".format(
                    self.checkpoint_path))
        else:
            print("    model not found ({})...".format(self.checkpoint_path))

        return

    # ---------------------------

    # Median Absolute Deviation
    def mad_score(self, points):
        """https://www.itl.nist.gov/div898/handbook/eda/section3/eda35h.htm """
        m = np.median(points)
        ad = np.abs(points - m)
        mad = np.median(ad)

        return 0.6745 * ad / mad
