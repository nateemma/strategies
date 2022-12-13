# class that implements an Anomaly detector using the Isolation Forest (IFOR) algorithm


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

class AnomalyDetector_IFOR():

    classifier = None
    is_trained = True
    clean_data_required = True # training data should not contain anomalies
    num_estimators = 10
    name = ""
    model_path = ""
    use_saved_model = False

    def __init__(self, tag=""):
        super().__init__()

        #TODO: add nium_features, use to modify model_path
        self.name = tag + self.__class__.__name__
        self.model_path = self.get_model_path()

        # load saved model if present
        if self.use_saved_model & os.path.exists(self.model_path):
            self.classifier = self.load()
            if self.classifier == None:
                print("    Failed to load model ({})".format(self.model_path))
        else:
            # self.classifier = IsolationForest(contamination=.01)
            # self.classifier = IsolationForest(n_estimators=self.num_estimators, warm_start=True)
            self.classifier = IsolationForest(warm_start=True)  # produces a warning, but seems to work better

    # update training using the suplied (normalised) dataframe. Training is cumulative
    # the 'labels' args should contain 0.0 for normal results, '1.0' for anomalies (buy or sell)
    def train(self, df_train_norm: DataFrame, df_test_norm: DataFrame, train_labels, test_labels, force_train=False):

        if self.model_is_trained() and (force_train == False):
            return

        if not self.clean_data_required:
            df1 = df_train_norm.copy()
            df1['%labels'] = train_labels
            df1 = df1[(df1['%labels'] < 0.1)]
            df_train = df1.drop('%labels', axis=1)
        else:
            df_train = df_train_norm.copy()

        # TODO: create class each time, with actual ratio of 'anomalies' ?!
        print("    fitting classifier: ", self.__class__.__name__)
        # self.num_estimators = self.num_estimators + 1
        # self.classifier.set_params(n_estimators=self.num_estimators)
        self.classifier = self.classifier.fit(df_train)
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

        pred = self.classifier.predict(df_norm)
        predictions = pd.Series(pred).replace([-1, 1], [1.0, 0.0])

        return predictions

    def get_model_path(self):
        # path to 'full' model file
        save_dir = './'
        model_path = save_dir + self.name + ".sav"
        return model_path

    def save(self, path=""):
        if self.use_saved_model:
            # use joblib to save classifier state
            print("    saving to: ", self.model_path)
            joblib.dump(self.classifier, self.model_path)
        return

    def load(self, path=""):
        if self.use_saved_model:
            # use joblib to reload classifier state
            print("    loading from: ", self.model_path)
            self.classifier = joblib.load(self.model_path)
        return self.classifier

    def model_is_trained(self) -> bool:
        return False

    def needs_clean_data(self) -> bool:
        # print("    clean_data_required: ", self.clean_data_required)
        return self.clean_data_required