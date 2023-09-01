# Neural Network Binary Classifier: this subclass uses dTransformer model


import numpy as np
import torch
from darts.utils.utils import SeasonalityMode
from pandas import DataFrame, Series
import pandas as pd

pd.options.mode.chained_assignment = None  # default='warn'

# Strategy specific imports, files must reside in same folder as strategy
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent))
sys.path.append(str(Path(__file__).parent.parent))

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

from utils.ClassifierDarts import ClassifierDarts
from darts.models import TransformerModel


class NNPredictor_dTransformer(ClassifierDarts):
    is_trained = False
    clean_data_required = False  # training data can contain anomalies
    model_per_pair = False  # separate model per pair
    lookahead = 4
    model_name = ""

    # override the build_model function in subclasses
    def create_model(self, seq_len, num_features):
        self.model_name = self.__class__.__name__

        model = TransformerModel(input_chunk_length=seq_len,
                           output_chunk_length=self.get_lookahead(),
                           batch_size=self.batch_size,
                           pl_trainer_kwargs=self.get_trainer_args(),
                           model_name=self.model_name
                           )
        return model

    # class-specific load
    def load_from_file(self, model_path, use_gpu=True):
        return TransformerModel.load(model_path)

    # class-specific load
    def load_from_checkpoint(self, path):
        return TransformerModel.load_from_checkpoint(self.model_name, best=True)

    def get_loss_metric(self):
        return 'train_loss'
