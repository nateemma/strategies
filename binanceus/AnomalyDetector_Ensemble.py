# class that implements an Anomaly detector using a stacked ensemble of various anomaly detection techniques

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent))

import logging
import warnings
import random
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
os.environ['TF_DETERMINISTIC_OPS'] = '1'

import numpy as np
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.mixture import GaussianMixture
from sklearn.neighbors import LocalOutlierFactor
from sklearn.svm import OneClassSVM

pd.options.mode.chained_assignment = None  # default='warn'

# Strategy specific imports, files must reside in same folder as strategy

import tensorflow as tf

seed = 42
os.environ['PYTHONHASHSEED'] = str(seed)
random.seed(seed)
tf.random.set_seed(seed)
np.random.seed(seed)

tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.WARN)

log = logging.getLogger(__name__)
# log.setLevel(logging.DEBUG)
warnings.simplefilter(action='ignore', category=pd.errors.PerformanceWarning)

from sklearn.ensemble import IsolationForest, StackingClassifier, RandomForestClassifier
from ClassifierSklearn import ClassifierSklearn



class AnomalyDetector_Ensemble(ClassifierSklearn):
    classifier = None
    clean_data_required = True  # training data should not contain anomalies
    use_scores = False

    c1 = None
    c2 = None
    c3 = None
    c4 = None
    c_ensemble = None

    def create_classifier(self):
        self.c1 = IsolationForest(contamination=self.contamination)
        # self.c2 = GaussianMixture(reg_covar=1e-5, n_components=2)
        self.c3 = LocalOutlierFactor(n_neighbors=30, novelty=True, contamination=self.contamination)
        self.c4 = OneClassSVM(gamma='scale', nu=self.contamination)
        self.c_ensemble = IsolationForest(contamination=self.contamination)
        return self.c_ensemble

    def model_fit(self, df, labels):

        # fit all of the classifiers
        self.c1.fit(df, labels)
        # self.c2.fit(df, labels)
        self.c3.fit(df, labels)
        self.c4.fit(df, labels)

        # get predictions from each algorithm
        y1 = self.c1.predict(df)
        # y2 = self.c2.predict(df)
        y3 = self.c3.predict(df)
        y4 = self.c4.predict(df)

        # Fit ensemble classifier using predictions from each algorithm
        # X_new = np.column_stack((y1, y2, y3, y4))
        X_new = np.column_stack((y1, y3, y4))
        # X_new = np.column_stack((y1, y2, y3))
        self.c_ensemble.fit(X_new, labels)

        return self.c_ensemble

    def model_predict(self, df):

        # get predictions from each algorithm
        y1 = self.c1.predict(df)
        # y2 = self.c2.predict(df)
        y3 = self.c3.predict(df)
        y4 = self.c4.predict(df)

        # run ensemble classifier using predictions from each algorithm
        # X_new = np.column_stack((y1, y2, y3, y4))
        X_new = np.column_stack((y1, y3, y4))
        # X_new = np.column_stack((y1, y2, y3))
        return self.c_ensemble.predict(X_new)