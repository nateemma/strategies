# class that implements an Anomaly detector using the K Means clustering algorithm


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
from sklearn.cluster import KMeans

import h5py

class AnomalyDetector_KMeans():

    classifier = None
    clean_data_required = False # training data should not contain anomalies

    def __init__(self, tag=""):
        super().__init__()
        self.classifier = KMeans(n_clusters=2)


    # update training using the suplied (normalised) dataframe. Training is cumulative
    # the 'labels' args should contain 0.0 for normal results, '1.0' for anomalies (buy or sell)
    def train(self, df_train_norm: DataFrame, df_test_norm: DataFrame, train_labels, test_labels, force_train=False):

        if self.is_trained and not force_train:
            return

        print("    fitting classifier: ", self.__class__.__name__)
        self.classifier = self.classifier.fit(df_train_norm)
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

        labels = self.classifier.predict(df_norm)
        predictions = pd.Series(np.where((labels > 0.0), 1.0, 0.0))

        # print("df_norm: {} labels: {} predictions: {}".format(df_norm.shape, np.shape(labels), np.shape(predictions)))
        # print(labels)

        return predictions

    def save(self, path=""):
        return

    def load(self, path=""):
        return self.classifier

    def model_is_trained(self) -> bool:
        return False

    def needs_clean_data(self) -> bool:
        # print("    clean_data_required: ", self.clean_data_required)
        return self.clean_data_required