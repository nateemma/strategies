# A subclass of ClassifierKerasLinear that provides a Classifier for use with Google's Temporal Fusion Transformer (TFT)
# The TFT usage model does not match that of other keras classifiers particularly well, so this class essentially
# provides a facade/wrapper that makes it look like the other classifiers


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

import h5py

import json

from DataframeUtils import DataframeUtils
from ClassifierKerasLinear import ClassifierKerasLinear
from tft_model import TemporalFusionTransformer


class ClassifierKerasTFT(ClassifierKerasLinear):

    clean_data_required = False
    requires_dataframes = True
    tft_params = {}
    lookahead = 4

    # ---------------------------

    # create model - required by framework
    def create_model(self, seq_len, num_features):

        # just return None. Cannot create until we have collected enough data to provide required paramters
        model = None
        return model

    # ---------------------------

    def set_lookahead(self, lookahead: int):
        self.lookahead = lookahead

    # ---------------------------

    # this is where most of the TFT-specific setup happens. Collects paramters from available data and sets up TFT
    # All preliminary setup must be done before calling this routine
    def create_tft_model(self, dataframe: DataFrame):

        '''
        The following parameters have to be added to an associative array and passed to the TFT constructor.
        This list was derived by looking at the code, so could potentially change
        
            name: Name of model
            time_steps: Total number of input time steps per forecast date (i.e. Width of Temporal fusion decoder N)
            input_size: Total number of inputs
            output_size: Total number of outputs
            category_counts: Number of categories per categorical variable
            n_multiprocessing_workers: Number of workers to use for parallel computations
            column_definition: List of tuples of (string, DataType, InputType) that define each column
            quantiles: Quantiles to forecast for TFT
            use_cudnn: Whether to use Keras CuDNNLSTM or standard LSTM layers
            hidden_layer_size: Internal state size of TFT
            dropout_rate: Dropout discard rate
            max_gradient_norm: Maximum norm for gradient clipping
            learning_rate: Initial learning rate of ADAM optimizer
            minibatch_size: Size of minibatches for training
            num_epochs: Maximum number of epochs for training
            early_stopping_patience: Maximum number of iterations of non-improvement before early stopping kicks in
            num_encoder_steps: Size of LSTM encoder -- i.e. number of past time steps before forecast date to use
            num_stacks: Number of self-attention layers to apply (default is 1 for basic TFT)
            num_heads: Number of heads for interpretable mulit-head attention
        '''

        self.tft_params["name"] = self.name

        # Data parameters
        self.tft_params['input_size'] = self.num_features
        self.tft_params['time_steps'] = self.seq_len
        self.tft_params['output_size'] = 1
        self.tft_params['multiprocessing_workers'] = 6

        all_cols = dataframe.columns.values
        tgt_cols = ['dwt']
        observed_cols = ['date', 'open', 'close', 'high', 'low', 'volume']
        static_cols = ['temp', 'predict']
        id_cols = ['date']
        time_cols = ['date', 'days_from_start']
        categorical_cols = ['day_of_week', 'day_of_month', 'month', 'week_of_year']
        known_cols = list(set(all_cols) - set(tgt_cols) - set(observed_cols) - set(static_cols) -
                          set(id_cols) - set(time_cols) - set(categorical_cols))



        # Relevant indices for TFT
        self.tft_params['input_obs_loc'] = [dataframe.columns.get_loc(c) for c in tgt_cols if c in dataframe]
        self.tft_params['static_input_loc'] = [dataframe.columns.get_loc(c) for c in static_cols if c in dataframe]

        # columns
        # TARGET = 0
        # OBSERVED_INPUT = 1
        # KNOWN_INPUT = 2
        # STATIC_INPUT = 3
        # ID = 4  # Single column used as an entity identifier
        # TIME = 5
        # self.tft_params['category_counts'] = [len(categorical_cols)]

        self.tft_params['category_counts'] = json.dumps([len(tgt_cols), len(observed_cols),
                                              len(known_cols), len(static_cols),
                                              len(id_cols), len(time_cols)])
        print(f"self.tft_params['category_counts']: {self.tft_params['category_counts']}")

        self.tft_params['known_regular_inputs'] = json.dumps(known_cols)
        self.tft_params['known_categorical_inputs'] = json.dumps(categorical_cols)

        self.tft_params['column_definition'] = all_cols

        # Network params
        self.tft_params['hidden_layer_size'] = 128
        self.tft_params['dropout_rate'] = 0.1
        self.tft_params['max_gradient_norm'] = 0.01
        self.tft_params['learning_rate'] = 0.001
        self.tft_params['minibatch_size'] = self.batch_size
        self.tft_params['num_epochs'] = 128
        self.tft_params['early_stopping_patience'] = 5

        self.tft_params['num_encoder_steps'] = 256 # ???
        self.tft_params['total_time_steps'] = self.tft_params['num_encoder_steps'] + self.seq_len

        self.tft_params['stack_size'] = 2
        self.tft_params['num_heads'] = 4

        # Serialisation options
        self.tft_params['model_folder'] = self.get_model_root_dir()
        

        # dataframe-derived parameters 

        
        model = TemporalFusionTransformer(raw_params=self.tft_params)

        return model

    # ---------------------------

    # Need to override train() method so that we can set up the TFT class (need dataframe info for this)
    def train(self, df_train_norm, df_test_norm, train_results, test_results, force_train=False):


        if not self.dataframeUtils.is_dataframe(df_train_norm):
            print("    ERR: df_train_norm is not a DataFrame. Training aborted")
            return

        self.num_features = np.shape(df_train_norm)[1]

        # lazy loading because params can change up to this point
        if self.model is None:
            # load saved model if present
            self.model = self.load()

        # print(f'is_trained:{self.is_trained} force_train:{force_train}')

        # if model is already trained, and caller is not requesting a re-train, then just return
        if (self.model is not None) \
                and self.model_is_trained() \
                and (not force_train) \
                and (not self.new_model_created()):

            # print(f"    Not training. is_trained:{self.is_trained} force_train:{force_train} new_model:{self.new_model}")
            print("    Model is already trained")
            return

        # if model still doesn't exist, create it (lazy initialisation)
        if self.model is None:
            self.model = self.create_tft_model(df_train_norm)
            self.model = self.compile_model(self.model)
            self.model.summary()

        # from here, we can just call the parent train() routine
        super(ClassifierKerasTFT, self).train(df_train_norm, df_test_norm, train_results, test_results, force_train)
        
        return

    # ---------------------------
