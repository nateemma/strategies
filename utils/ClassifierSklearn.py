# base class that implements an Anomaly detector using sklearn algorithms
# subclasses should override the create_classifier() method


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

# import tensorflow as tf

seed = 42
os.environ['PYTHONHASHSEED'] = str(seed)
random.seed(seed)
# tf.random.set_seed(seed)
np.random.seed(seed)

# tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.WARN)

#import keras
from keras import layers
from sklearn.ensemble import IsolationForest

import h5py
import joblib

from numpy import quantile
from DataframeUtils import DataframeUtils


class ClassifierSklearn():
    model = None
    is_trained = False
    category = ""
    name = ""
    model_path = ""
    checkpoint_path = ""

    use_saved_model = True
    loaded_from_file = False
    contamination = 0.01  # ratio of signals to samples. Used in several algorithms, so saved

    clean_data_required = False  # train with positive rows removed
    model_per_pair = True  # set to False to combine across all pairs
    new_model = False  # True if a new model was created this run

    dataframeUtils = None
    requires_dataframes = True  # set to True if classifier takes dataframes rather than tensors
    prescale_dataframe = False  # set to True if algorithms need dataframes to be pre-scaled
    single_prediction = False  # True if alogorithm only produces 1 prediction (not entire data array)
    use_scores = True # True if model supports scoring of results (ensembles do not)

    def __init__(self, pair, tag=""):
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

        if self.dataframeUtils is None:
            self.dataframeUtils = DataframeUtils()

    # set model name - this overrides the default naming. This allows the strategy to set the naming convention
    # directory and extension are handled, just need to supply the category (e.g. the strat name) and main file name
    # caller will have to take care of adding pair names, tag etc.
    def set_model_name(self, category, model_name):
        root_dir = self.get_model_root_dir()
        save_dir = root_dir + category + '/'
        file_path = save_dir + model_name + ".sav"

        # update tracking vars (need to override defaults)
        self.category = category
        self.model_path = file_path
        self.name = model_name
        # print(f"    Set model path:{self.model_path}")

        return self.model_path

    # create classifier - subclasses should overide this
    def create_classifier(self):

        classifier = None

        print("    ERR: create_classifier() should be defined by the subclass")
        classifier = IsolationForest()  # just so that there is something viable to use

        return classifier

    # fit the model (can be overridden)
    def model_fit(self, df, labels):
        return self.model.fit(df, labels)

    # run predciction (can be overridden)
    def model_predict(self, df):
        return self.model.predict(df)

    # update training using the suplied (normalised) dataframe. Training is cumulative
    # the 'labels' args should contain 0.0 for normal results, '1.0' for anomalies (buy or sell)
    def train(self, df_train_norm: DataFrame, df_test_norm: DataFrame, train_labels, test_labels, force_train=False):

        # NOTE: sklearn algorithms are *not* cumulative or reusable, so need to re-train each time
        # # already trained? Just return
        # if self.is_trained and (force_train == False):
        #     return

        if self.clean_data_required:
            # only use entries that do not have buy/sell signal
            df1 = df_train_norm.copy()
            df1['%labels'] = train_labels
            df1 = df1[(df1['%labels'] < 0.1)]
            labels = df1['%labels']
            df_train = df1.drop('%labels', axis=1)
        else:
            df_train = df_train_norm.copy()
            labels = train_labels

        # calculate contamination rate (using original data, not cleaned)
        # self.contamination = round((train_labels.sum() / np.shape(train_labels)[0]), 3)
        self.contamination = (train_labels.sum() / np.shape(train_labels)[0])

        # if classifier is not yet defined, create it
        if self.model == None:
            self.model = self.create_classifier()

        # TODO: create class each time, with actual ratio of 'anomalies' ?!
        print(f"    fitting classifier: {self.__class__.__name__} contamination: {self.contamination}")
        # self.num_estimators = self.num_estimators + 1
        # self.model.set_params(n_estimators=self.num_estimators)
        self.model = self.model_fit(df_train, labels)

        # only save if this is the first time training
        if not self.is_trained:
            self.save()

        self.is_trained = True

        return

    # evaluate model using the supplied (normalised) dataframe as test data.
    def evaluate(self, df_norm: DataFrame):
        return

    # 'recosnstruct' a dataframe by passing it through the classifier
    def reconstruct(self, df_norm: DataFrame) -> DataFrame:
        return df_norm

    # transform supplied (normalised) dataframe into a lower dimension version
    def transform(self, df_norm: DataFrame) -> DataFrame:
        return df_norm

    # run the model prediction against the entire data buffer
    def backtest(self, data):
        # for sklearn-based models, this is the same thing as running predict(). Here for compatibility with other types
        return self.predict(data)

    # only need to override/define the predict function
    def predict(self, df_norm: DataFrame):

        if self.model is None:
            print("    ERR: no classifier")
            return np.zeros(np.shape(df_norm)[0])

        has_predict = getattr(self.model, "predict", None)
        if callable(has_predict):
            pred = self.model_predict(df_norm)

        else:
            # some sklearn classifiers have fit_predict instead
            has_fit_predict = getattr(self.model, "fit_predict", None)
            if callable(has_fit_predict):
                pred = self.model.fit_predict(df_norm)
            else:
                print("    ERR: classifier does not have a predict() or fit_predict() method")
                return np.zeros(np.shape(df_norm)[0])


        # not all sklearn algorithms support score_samples method, so double-check
        has_scores = getattr(self.model, "score_samples", None)
        if not callable(has_scores):
            self.use_scores = False

        if self.use_scores:
            scores = self.model.score_samples(df_norm)
            # thresh = np.quantile(scores, self.contamination)
            thresh = scores.mean() - 2.0 * scores.std()
            # print("thresh:{:.3f} min:{:.3f} max:{:.3f} mean:{:.3f} std:{:.3f}".format(thresh,
            #                                                                           scores.min(), scores.max(),
            #                                                                           scores.mean(), scores.std()))
            index = np.where(scores <= thresh)
            predictions = np.zeros(np.shape(pred)[0])
            predictions[index] = 1.0
        else:
            predictions = pd.Series(pred).replace([-1, 1], [1.0, 0.0])

        return predictions

    # returns path to the root directory used for storing models
    def get_model_root_dir(self):
        # set as subdirectory of location of this file (so that it can be included in the repository)
        file_dir = os.path.dirname(str(Path(__file__)))
        root_dir = file_dir + "/models/"
        if not os.path.exists(root_dir):
            os.makedirs(root_dir)
        return root_dir

    # returns path to 'full' model file
    def get_model_path(self):
        root_dir = self.get_model_root_dir()
        save_dir = root_dir + self.category + '/'
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        model_path = save_dir + self.name + ".sav"
        return model_path

    def get_checkpoint_path(self):
        checkpoint_dir = '/tmp' + "/" + self.name + "/"
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
        model_path = checkpoint_dir + "checkpoint.sav"
        return model_path

    def save(self, path=""):

        if len(path) == 0:
            self.model_path = self.get_model_path()
            path = self.model_path
        else:
            self.model_path = path

        # only save if this is a new model
        if self.new_model:
            save_dir = os.path.dirname(path)
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)

            # use joblib to save model state
            print("    saving to: ", self.model_path)
            joblib.dump(self.model, self.model_path)
        return

    def load(self, path=""):

        if len(path) == 0:
            self.model_path = self.get_model_path()
            path = self.model_path
        else:
            self.model_path = path

        if self.use_saved_model:
            if os.path.exists(path):
                # use joblib to reload model state
                print("    loading from: ", self.model_path)
                self.model = joblib.load(self.model_path)
                self.loaded_from_file = True
                # self.is_trained = True # training is NOT cumulative for sklearn classifiers
            else:
                self.new_model = True
        return self.model

    def model_exists(self) -> bool:
        path = self.get_model_path()
        return os.path.exists(path)

    def model_is_trained(self) -> bool:
        return self.is_trained

    def needs_clean_data(self) -> bool:
        # print("    clean_data_required: ", self.clean_data_required)
        return self.clean_data_required

    def needs_dataframes(self) -> bool:
        return self.requires_dataframes

    def prescale_data(self) -> bool:
        return self.prescale_dataframe

    def returns_single_prediction(self) -> bool:
        return self.single_prediction

    def new_model_created(self) -> bool:
        return ClassifierSklearn.new_model  # note use of class-level variable
