# Neural Network Binary Classifier: this subclass uses an NLinear model

# testing Ray distributed server: see  https://docs.ray.io/en/master/ray-overview/installation.html#m1-mac-apple-silicon-support
# NOTE: currently not working, waiting for package updates for M1

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
np.random.seed(seed)

tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.WARN)

import keras
from keras import layers
from ClassifierDarts import ClassifierDarts
from darts.models import NLinearModel

import multiprocessing
import ray
from ray_lightning import RayStrategy

#------------------------

class NNPredictor_Ray(ClassifierDarts):
    is_trained = False
    clean_data_required = False  # training data can contain anomalies
    model_per_pair = False  # separate model per pair

    # ------------------------

    def __init__(self, pair, seq_len, num_features, tag="", use_gpu=True):
        super().__init__(pair, seq_len, num_features, tag, use_gpu)
        ray.init()

    # ------------------------

    # override the build_model function in subclasses
    def create_model(self, seq_len, num_features):

        ray_strategy = RayStrategy(
            num_workers=multiprocessing.cpu_count(),
            num_cpus_per_worker=1,
            use_gpu=True
        )

        # this model type has a tendency to exit early, so set min no. of epochs
        train_args = self.trainer_args
        train_args["min_epochs"] = 16

        # add in ray callback
        train_args["strategy"] = ray_strategy

        model = NLinearModel(input_chunk_length=seq_len,
                             output_chunk_length=self.lookahead,
                             pl_trainer_kwargs=train_args
                             )
        return model

    # ------------------------

    # class-specific load
    def load_from_file(self, model_path):
        return NLinearModel.load(model_path)

