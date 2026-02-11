# class that implements an Anomaly detector using the DBSCAN algorithm


import numpy as np
from pandas import DataFrame, Series
import pandas as pd
from sklearn.cluster import DBSCAN

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
from sklearn.svm import OneClassSVM

from ClassifierSklearn import ClassifierSklearn


import h5py

class AnomalyDetector_DBSCAN(ClassifierSklearn):

    classifier = None
    clean_data_required = False # training data can contain anomalies

    def create_classifier(self):
        classifier = DBSCAN(eps=1.0)
        return classifier

    # DBSCAN is different in that it doesn't really match the usual fit/predict model
    # So, need to override the predict() method of the base class
    def predict(self, df_norm: DataFrame):

        if self.model is None:
            print("    ERR: no classifier")
            return np.zeros(np.shape(df_norm)[0])

        num_samples = np.shape(df_norm)[0]
        if self.contamination > 0:
            min_samples = int(self.contamination * num_samples / 4)
        else:
            min_samples = int(num_samples * 0.05)
        print(f'num_samples: {num_samples} self.contamination:{self.contamination} min_samples: {min_samples}')
        db = DBSCAN(eps=0.00001, min_samples=min_samples).fit(df_norm)
        labels = db.labels_

        no_clusters = len(np.unique(labels))
        no_noise = np.sum(np.array(labels) == -1, axis=0)

        print('Estimated no. of clusters: %d' % no_clusters)
        print('Estimated no. of noise points: %d' % no_noise)

        predictions = pd.Series(labels).replace([-1, 1], [1.0, 0.0])
        print(f"    dbscan: found {predictions.sum()} anomalies")

        return predictions

