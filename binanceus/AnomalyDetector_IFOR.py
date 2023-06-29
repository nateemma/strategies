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

#import keras
from keras import layers
from sklearn.ensemble import IsolationForest
from ClassifierSklearn import ClassifierSklearn

import h5py
import joblib

class AnomalyDetector_IFOR(ClassifierSklearn):

    classifier = None
    clean_data_required = False # training data should not contain anomalies


    def create_classifier(self):
        classifier = IsolationForest(contamination=self.contamination)
        return classifier
