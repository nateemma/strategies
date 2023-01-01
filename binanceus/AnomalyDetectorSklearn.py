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

import tensorflow as tf

seed = 42
os.environ['PYTHONHASHSEED'] = str(seed)
random.seed(seed)
tf.random.set_seed(seed)
np.random.seed(seed)

tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.WARN)

import keras
from keras import layers
from sklearn.ensemble import IsolationForest

import h5py
import joblib

from numpy import quantile

class AnomalyDetectorSklearn():

    classifier = None
    is_trained = False
    clean_data_required = False # training data should not contain anomalies
    num_estimators = 10
    name = ""
    model_path = ""
    use_saved_model = True
    loaded_from_file = False
    contamination = 0.01 # ratio of signals to samples. Used in several algorithms, so saved

    def __init__(self, pair, tag=""):
        super().__init__()

        self.loaded_from_file = False

        #TODO: add num_features, use to modify model_path
        pname = pair.split("/")[0]

        self.name = self.__class__.__name__ + "_" + pname + "_" + tag
        self.model_path = self.get_model_path()

        # load saved model if present
        if self.use_saved_model & os.path.exists(self.model_path):
            self.classifier = self.load()
            if self.classifier == None:
                print("    Failed to load model ({})".format(self.model_path))


    # create classifier - subclasses should overide this
    def create_classifier(self):

        classifier = None

        print("    ERR: create_classifier() should be defined by the subclass")
        classifier = IsolationForest() # just so that there is something viable to use

        return classifier

    # update training using the suplied (normalised) dataframe. Training is cumulative
    # the 'labels' args should contain 0.0 for normal results, '1.0' for anomalies (buy or sell)
    def train(self, df_train_norm: DataFrame, df_test_norm: DataFrame, train_labels, test_labels, force_train=False):


        # NOTE: sklearn algorithms are *not* cumulative or reusable, so need to re-train each time
        # # already trained? Just return
        # if self.is_trained and (force_train == False):
        #     return

        if not self.clean_data_required:
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
        self.contamination = round((train_labels.sum() / np.shape(train_labels)[0]), 3)

        # if classifier is not yet defined, create it
        if self.classifier == None:
            self.classifier = self.create_classifier()

        # TODO: create class each time, with actual ratio of 'anomalies' ?!
        print("    fitting classifier: ", self.__class__.__name__)
        # self.num_estimators = self.num_estimators + 1
        # self.classifier.set_params(n_estimators=self.num_estimators)
        self.classifier = self.classifier.fit(df_train)

        # only save if this is the first time training
        if not self.is_trained:
            self.save()

        self.is_trained = True



        return


    # evaluate model using the supplied (normalised) dataframe as test data.
    def evaluate(self, df_norm: DataFrame):
        return

    # 'recosnstruct' a dataframe by passing it through the classifier
    def reconstruct(self, df_norm:DataFrame) -> DataFrame:
        return df_norm

    # transform supplied (normalised) dataframe into a lower dimension version
    def transform(self, df_norm: DataFrame) -> DataFrame:
        return df_norm


    # only need to override/define the predict function
    def predict(self, df_norm: DataFrame):

        if self.classifier is None:
            print("    ERR: no classifier")
            return np.zeros(np.shape(df_norm)[0])

        pred = self.classifier.predict(df_norm)

        use_scores = True

        # not all sklearn algorithms support score_samples method, so double-check
        has_scores = getattr(self.classifier, "score_samples", None)
        if not callable(has_scores):
            use_scores = False

        if use_scores:
            scores = self.classifier.score_samples(df_norm)
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

    # returns path to 'full' model file
    def get_model_path(self):
        # set as subdirectory of location of this file (so that it can be included in the repository)
        file_dir = os.path.dirname(str(Path(__file__)))
        save_dir = file_dir + "/models/" + self.__class__.__name__ + '/'
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        model_path = save_dir + self.name + ".sav"
        return model_path

    def save(self, path=""):
        if self.use_saved_model and (not self.loaded_from_file):
            # use joblib to save classifier state
            print("    saving to: ", self.model_path)
            joblib.dump(self.classifier, self.model_path)
        return

    def load(self, path=""):
        if self.use_saved_model:
            # use joblib to reload classifier state
            print("    loading from: ", self.model_path)
            self.classifier = joblib.load(self.model_path)
            self.loaded_from_file = True
            # self.is_trained = True
        return self.classifier

    def model_is_trained(self) -> bool:
        return self.is_trained

    def needs_clean_data(self) -> bool:
        # print("    clean_data_required: ", self.clean_data_required)
        return self.clean_data_required