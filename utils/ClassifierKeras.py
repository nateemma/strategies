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
import random
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
os.environ['TF_DETERMINISTIC_OPS'] = '1'
os.environ['TF_RUN_EAGER_OP_AS_FUNCTION'] = '0'

import tensorflow as tf
import keras

from DataframeUtils import DataframeUtils

@keras.saving.register_keras_serializable(package="ClassifierKeras")
class ClassifierKeras():
    model = None
    is_trained = False
    model_path = ""
    model_ext = "keras"
    name = ""
    checkpoint_path = "/tmp/model." + model_ext
    seq_len = 8
    num_features = 64
    encoder_layer = 'encoder_output'
    encoder = None
    default_max_epochs = 256
    num_epochs = default_max_epochs  # number of iterations for training (can be set)
    default_learning_rate = 0.01
    learning_rate = default_learning_rate
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

    def __init__(self, pair, seq_len, num_features, tag=""):
        super().__init__()

        # environment setup
        log = logging.getLogger(__name__)
        # log.setLevel(logging.DEBUG)
        warnings.simplefilter(action='ignore', category=pd.errors.PerformanceWarning)

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

        # class init:

        self.loaded_from_file = False

        self.seq_len = seq_len
        self.num_features = num_features
        # print("    num_features: ", num_features)

        if self.dataframeUtils is None:
            self.dataframeUtils = DataframeUtils()

    
    # ---------------------------
    # needed for serialisation
    def get_config(self):
        config = {
            'model': self.model,
            'is_trained': self.is_trained,
            'model_path': self.model_path,
            'model_ext': self.model_ext,
            'name': self.name,
            'checkpoint_path': self.checkpoint_path,
            'seq_len': self.seq_len,
            'num_features': self.num_features,
            'encoder_layer': self.encoder_layer,
            'encoder': self.encoder,
            'num_epochs': self.num_epochs,
            'batch_size': self.batch_size,
            'clean_data_required': self.clean_data_required ,
            'model_per_pair': self.model_per_pair,
            'new_model': self.new_model,
            'dataframeUtils': self.dataframeUtils,
            'requires_dataframes': self.requires_dataframes,
            'prescale_dataframe': self.prescale_dataframe,
            'single_prediction': self.single_prediction,
            'combine_models': self.combine_models
        }
        return {**config}
    
    # ---------------------------
    # sets the combine-Models flag.
    # If True, models will be combined across multiple pairs
    # If False, only first pair is used for training (unless per_pair is specified)
    def set_combine_models(self, combine_models):
        self.combine_models = combine_models


    # ---------------------------
    
    def set_target_column(self, target_column):
        pass
        return
    

    # ---------------------------
    
    def set_num_epochs(self, num_epochs=None):
        if num_epochs is None:
            self.num_epochs = self.default_max_epochs
        else:
            self.num_epochs = num_epochs
        return
        
    # ---------------------------
    
    def set_learning_rate(self, rate=None):
        if rate is None:
            self.learning_rate = self.default_learning_rate
        else:
            self.learning_rate = rate
        return
    
    # ---------------------------
    # utility function to find the nearest power of 2 greater than the supplied number
    def nearest_power_of_2(self, n):
        return 2**n.bit_length()
    
    # ---------------------------

    # create model - subclasses should overide this
    def create_model(self, seq_len, num_features):

        model = None
        outer_dim = 64
        inner_dim = 16

        print("    WARNING: create_model() should be defined by the subclass")
        # create a simple model for illustrative purposes (or to test the framework)
        model = keras.Sequential(name=self.name)

        # Encoder
        model.add(keras.layers.Dense(outer_dim, activation='relu', input_shape=(seq_len, num_features)))
        model.add(keras.layers.Dense(2 * outer_dim, activation='relu'))
        model.add(keras.layers.Dense(inner_dim, activation='relu', name=self.encoder_layer))  # name is mandatory

        # Decoder
        model.add(keras.layers.Dense(2 * outer_dim, activation='relu', input_shape=(1, inner_dim)))
        model.add(keras.layers.Dense(outer_dim, activation='relu'))

        model.add(keras.layers.Dense(num_features, activation=None))
        return model

    # ---------------------------

    # compile the model. This is a distinct function because it can vary by model type
    def compile_model(self, model):

        optimizer = keras.optimizers.Adam(learning_rate=0.01)
        # workaround for tensorflow issues with Adam:
        # optimizer = keras.optimizers.legacy.Adam(learning_rate=0.01)

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
        early_callback = keras.callbacks.EarlyStopping(
            monitor=monitor_field,
            mode=monitor_mode,
            patience=early_patience,
            min_delta=0.0001,
            restore_best_weights=True,
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
            msle = keras.losses.msle(predict_tensor, tensor)
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
    def evaluate(self, data, results):
        
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
        # preds = self.model.predict(test_tensor, verbose=0)
        preds = self.predict(test_tensor)

        print("    Comparing...")
        # score = self.model.evaluate(test_tensor, preds, return_dict=True, verbose=0)
        # print("    model:{}   score:{} ".format(self.name, score))

        print(f"    predictions:{np.shape(preds)}  results:{np.shape(results)}")

        # rpreds = preds.reshape(-1,1)
        if preds.shape != results.shape:
            print(f"    Dimension mismatch. predictions:{np.shape(preds)} results:{np.shape(results)}")
            return
        
        loss = keras.metrics.mean_squared_error(preds, results)
        if loss.ndim > 1:
        # loss = np.array(loss[0])
            print("    loss: sum:{:.3f} min:{:.3f} max:{:.3f} mean:{:.3f} std:{:.3f}".format(np.sum(loss),
                                                                                            np.min(loss), np.max(loss),
                                                                                            np.mean(loss), np.std(loss)))
        else:
            print("    loss: {}".format(loss))

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

    # set the path for the model (dir + file name)
    # need this because utility function cannot figure out naming if multiple directories are used
    def set_model_path(self, path):
        self.model_path = path
        filename = path.split("/")[-1]
        self.name = filename.split(".")[0]
        self.root_dir = os.path.dirname(path)
        if not os.path.exists(self.root_dir):
            os.makedirs(self.root_dir)
        return

    # returns path to the root directory used for storing models
    def get_model_root_dir(self):

        if len(self.root_dir) > 0:
            return self.root_dir       
        
        print("*** ERR: model path not set ***")

        return ""

    # ---------------------------

    # returns path to 'full' model file
    def get_model_path(self):
        return self.model_path

    # ---------------------------

    def get_checkpoint_path(self):
        checkpoint_dir = '/tmp' + "/" + self.name + "/"
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
        model_path = checkpoint_dir + "checkpoint." + self.model_ext
        return model_path

    # ---------------------------

    def save(self, path=""):

        if len(path) == 0:
            path = self.model_path
        else:
            self.model_path = path

        print("    saving model to: ", path)
        save_dir = os.path.dirname(path)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        keras.models.save_model(self.model, filepath=path, save_format=self.model_ext)
        return

    # ---------------------------

    def load(self, path=""):

        if len(path) == 0:
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
                    model = keras.models.load_model(path, compile=False)
                self.compile_model(model)
                self.is_trained = True
                ClassifierKeras.new_model = False

            except Exception as e:
                print("    ", str(e))
                print("    Error loading model from {}. Check whether model format changed".format(path))
        else:
            print("    model not found ({})...".format(path))
            # flag this as a new model. Note that this is a class global variable because we need to track this
            # across multiple instances (e.g. if we are combining all pairs into one model)
            # if self.combine_models:
            #     ClassifierKeras.new_model = True
            # else:
            #     ClassifierKeras.new_model = False
            ClassifierKeras.new_model = True
            
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
    
