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

from sklearn.ensemble import IsolationForest, StackingClassifier
from ClassifierSklearn import ClassifierSklearn



class AnomalyDetector_Ensemble(ClassifierSklearn):
    classifier = None
    clean_data_required = True  # training data should not contain anomalies

    def create_classifier(self):
        # Stacked 'ensemble' of classifiers
        c1 = IsolationForest(contamination=self.contamination)
        c2 = GaussianMixture()
        c3 = LocalOutlierFactor(n_neighbors=30, novelty=True, contamination=self.contamination)
        c4 = OneClassSVM(gamma='scale', nu=self.contamination)
        estimators = [('c1', c1), ('c2', c2), ('c3', c3), ('c4', c4)]
        # estimators = [('c2', c2), ('c3', c3), ('c4', c4)]
        classifier = StackingClassifier(estimators=estimators,
                                        final_estimator=LogisticRegression())
        return classifier
