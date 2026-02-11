# Neural Network Binary Classifier: this subclass uses a simple TFT model (keras implementation)


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
from ClassifierKerasLinear import ClassifierKerasLinear
from DataframeUtils import ScalerType
from tft_model import TemporalFusionTransformer

import h5py


class NNPredictor_kTFT(ClassifierKerasLinear):
    is_trained = False
    clean_data_required = False  # training data can contain anomalies
    model_per_pair = False # separate model per pair

    scaler_type = ScalerType.Robust

    kTFT = None

    # override the build_model function in subclasses
    def create_model(self, seq_len, num_features):


        params = {}
        params["name"] = "TFT"
        params["model_folder"] = self.get_model_root_dir()
        params["input_size"] = num_features
        params["output_size"] = 1
        params["total_time_steps"] = seq_len
        self.kTFT = TemporalFusionTransformer(raw_params=params)

        return self.kTFT.model
