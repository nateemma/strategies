# base class that implements an Anomaly detector using sklearn algorithms
# subclasses should override the create_classifier() method


import numpy as np
from pandas import DataFrame
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
warnings.simplefilter(action="ignore", category=pd.errors.PerformanceWarning)

import random

import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "1"
os.environ["TF_DETERMINISTIC_OPS"] = "1"

# import tensorflow as tf

seed = 42
os.environ["PYTHONHASHSEED"] = str(seed)
random.seed(seed)
# tf.random.set_seed(seed)
np.random.seed(seed)

# tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.WARN)

# import keras
from sklearn.ensemble import IsolationForest

import joblib

from utils.DataframeUtils import DataframeUtils
from Predictors.BaseClassifier import BaseClassifier


class SklearnBaseClassifier(BaseClassifier):
    model = None
    is_trained = False
    category = ""
    name = ""
    model_path = ""
    checkpoint_path = ""
    root_dir = ""  # Root directory for model storage
    batch_size = (
        None  # For compatibility with ClassifierKeras API (not used by sklearn)
    )

    use_saved_model = True
    loaded_from_file = False
    contamination = (
        0.01  # ratio of signals to samples. Used in several algorithms, so saved
    )

    clean_data_required = False  # train with positive rows removed
    model_per_pair = True  # set to False to combine across all pairs
    new_model = False  # True if a new model was created this run

    dataframeUtils = None
    requires_dataframes = (
        True  # set to True if classifier takes dataframes rather than tensors
    )
    prescale_dataframe = (
        False  # set to True if algorithms need dataframes to be pre-scaled
    )
    single_prediction = (
        False  # True if alogorithm only produces 1 prediction (not entire data array)
    )
    use_scores = True  # True if model supports scoring of results (ensembles do not)

    def __init__(self, pair, tag=""):
        super().__init__()

        self.loaded_from_file = False

        if self.model_per_pair:
            pair_suffix = "_" + pair.split("/")[0]
        else:
            pair_suffix = ""

        if tag == "":
            tag_suffix = ""
        else:
            tag_suffix = "_" + tag

        self.category = self.__class__.__name__
        self.name = self.category + pair_suffix + tag_suffix

        # self.model_path = self.get_model_path()
        self.set_model_name(self.category, self.name)

        if self.dataframeUtils is None:
            self.dataframeUtils = DataframeUtils()

    # set model name - this overrides the default naming. This allows the strategy to set the naming convention
    # directory and extension are handled, just need to supply the category (e.g. the strat name) and main file name
    # caller will have to take care of adding pair names, tag etc.
    def set_model_name(self, category, model_name):
        root_dir = self.get_model_root_dir()
        save_dir = root_dir + category + "/"
        file_path = save_dir + model_name + ".sav"

        # update tracking vars (need to override defaults)
        self.category = category
        self.model_path = file_path
        self.name = model_name
        # print(f"    Set model path:{self.model_path}")

        return self.model_path

    def set_model_path(self, path):
        """
        Set the model path (compatibility with ClassifierKeras API).
        This method is called by NNStrategy to set the model save/load path.
        Converts .keras extension to .sav for sklearn models.
        """
        # Convert .keras extension to .sav for sklearn models
        if path.endswith(".keras"):
            path = path[:-6] + ".sav"

        self.model_path = path
        filename = path.split("/")[-1]
        self.name = filename.split(".")[0]
        self.root_dir = os.path.dirname(path)
        if not os.path.exists(self.root_dir):
            os.makedirs(self.root_dir)

        # Update category from path if possible
        path_parts = path.split("/")
        if len(path_parts) >= 2:
            # Category is typically the second-to-last directory
            self.category = path_parts[-2] if len(path_parts) >= 2 else self.category

        return

    def set_batch_size(self, batch_size):
        """
        Set batch size (compatibility with ClassifierKeras API).
        Sklearn classifiers don't use batch_size, so this is a no-op.
        """
        # Sklearn classifiers don't use batch_size, but we store it for compatibility
        self.batch_size = batch_size
        return

    # create classifier - subclasses should overide this
    def create_classifier(self):

        classifier = None

        print("    ERR: create_classifier() should be defined by the subclass")
        classifier = IsolationForest()  # just so that there is something viable to use

        return classifier

    # fit the model (can be overridden)
    def model_fit(self, df, labels):
        return self.model.fit(df, labels)

    # run predciction (can be overridden)
    def model_predict(self, df):
        return self.model.predict(df)

    # update training using the suplied (normalised) dataframe. Training is cumulative
    # the 'labels' args should contain 0.0 for normal results, '1.0' for anomalies (buy or sell)
    def train(
        self,
        df_train_norm: DataFrame,
        df_test_norm: DataFrame,
        train_labels,
        test_labels,
        force_train=False,
        class_weights=None,
    ):

        # NOTE: sklearn algorithms are *not* cumulative or reusable, so need to re-train each time
        # # already trained? Just return
        # if self.is_trained and (force_train == False):
        #     return

        if self.clean_data_required:
            # only use entries that do not have buy/sell signal
            df1 = df_train_norm.copy()
            df1["%labels"] = train_labels
            df1 = df1[(df1["%labels"] < 0.1)]
            labels = df1["%labels"]
            df_train = df1.drop("%labels", axis=1)
        else:
            df_train = df_train_norm.copy()
            labels = train_labels

        # calculate contamination rate (using original data, not cleaned)
        # self.contamination = round((train_labels.sum() / np.shape(train_labels)[0]), 3)
        self.contamination = train_labels.sum() / np.shape(train_labels)[0]

        # if classifier is not yet defined, create it
        if self.model == None:
            self.model = self.create_classifier()

        # TODO: create class each time, with actual ratio of 'anomalies' ?!
        print(
            f"    fitting classifier: {self.__class__.__name__} contamination: {self.contamination}"
        )
        # self.num_estimators = self.num_estimators + 1
        # self.model.set_params(n_estimators=self.num_estimators)
        self.model = self.model_fit(df_train, labels)

        # Save the model after training (if not already trained)
        if not self.is_trained:
            # Set new_model flag to ensure save() actually saves
            if not self.model_exists():
                self.new_model = True
            self.save()

        self.is_trained = True

        return

    # evaluate model using the supplied (normalised) dataframe as test data.
    def evaluate(self, df_norm: DataFrame):
        return

    # 'recosnstruct' a dataframe by passing it through the classifier
    def reconstruct(self, df_norm: DataFrame) -> DataFrame:
        return df_norm

    # transform supplied (normalised) dataframe into a lower dimension version
    def transform(self, df_norm: DataFrame) -> DataFrame:
        return df_norm

    # run the model prediction against the entire data buffer
    def backtest(self, data):
        # for sklearn-based models, this is the same thing as running predict(). Here for compatibility with other types
        return self.predict(data)

    # only need to override/define the predict function
    def predict(self, df_norm: DataFrame):
        """
        Predict class probabilities for the given dataframe.
        Returns probabilities in shape (n_samples, n_classes) for compatibility with NNStrategy.
        """

        # Lazy loading: if model is None, try to load saved model
        if self.model is None:
            # Load saved model if present (compatibility with ClassifierKeras lazy loading)
            self.model = self.load()

        if self.model is None:
            print("    ERR: no classifier and no saved model found")
            # Return zero probabilities for all classes
            n_samples = np.shape(df_norm)[0]
            return np.zeros((n_samples, 3))  # 3 classes: Sell/Hold/Buy

        # Try to get probabilities using predict_proba (preferred for classifiers)
        has_predict_proba = getattr(self.model, "predict_proba", None)
        if callable(has_predict_proba):
            # Most sklearn classifiers support predict_proba
            proba = self.model.predict_proba(df_norm)
            # Ensure we have the right number of classes (should be 3 for Sell/Hold/Buy)
            if proba.shape[1] < 3:
                # Pad with zeros if fewer classes
                padded = np.zeros((proba.shape[0], 3))
                padded[:, : proba.shape[1]] = proba
                return padded
            elif proba.shape[1] > 3:
                # Truncate if more classes (shouldn't happen, but be safe)
                return proba[:, :3]
            return proba

        # Fallback: use predict() and convert class indices to one-hot probabilities
        has_predict = getattr(self.model, "predict", None)
        if callable(has_predict):
            pred = self.model_predict(df_norm)
            # Convert class indices to one-hot probabilities
            n_samples = len(pred)
            n_classes = 3  # Sell/Hold/Buy
            proba = np.zeros((n_samples, n_classes))

            # Handle different prediction formats
            if isinstance(pred, pd.Series):
                pred = pred.values

            # Convert to integer class indices
            pred = np.asarray(pred, dtype=int)

            # Ensure class indices are in valid range [0, n_classes-1]
            pred = np.clip(pred, 0, n_classes - 1)

            # Create one-hot encoding (probability = 1.0 for predicted class)
            for i, class_idx in enumerate(pred):
                proba[i, class_idx] = 1.0

            return proba

        # Last resort: some sklearn algorithms have fit_predict instead
        has_fit_predict = getattr(self.model, "fit_predict", None)
        if callable(has_fit_predict):
            pred = self.model.fit_predict(df_norm)
            # Convert to one-hot probabilities
            n_samples = len(pred)
            n_classes = 3
            proba = np.zeros((n_samples, n_classes))
            pred = np.asarray(pred, dtype=int)
            pred = np.clip(pred, 0, n_classes - 1)
            for i, class_idx in enumerate(pred):
                proba[i, class_idx] = 1.0
            return proba

        print(
            "    ERR: classifier does not have a predict(), predict_proba(), or fit_predict() method"
        )
        n_samples = np.shape(df_norm)[0]
        return np.zeros((n_samples, 3))

    # returns path to the root directory used for storing models

    # returns path to 'full' model file
    def get_model_path(self):
        root_dir = self.get_model_root_dir()
        save_dir = root_dir + self.category + "/"
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        model_path = save_dir + self.name + ".sav"
        return model_path

    def get_checkpoint_path(self):
        checkpoint_dir = "/tmp" + "/" + self.name + "/"
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
        model_path = checkpoint_dir + "checkpoint.sav"
        return model_path

    def save(self, path=""):
        """
        Save the model to disk.
        If called from train() with new_model=True, will save.
        Can also be called directly with force=True to always save.
        """
        if len(path) == 0:
            if not self.model_path or self.model_path == "":
                self.model_path = self.get_model_path()
            path = self.model_path
        else:
            self.model_path = path

        # Save if this is a new model, or if model doesn't exist yet
        if self.new_model or not os.path.exists(path):
            save_dir = os.path.dirname(path)
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)

            # use joblib to save model state
            print("    saving sklearn model to: ", self.model_path)
            joblib.dump(self.model, self.model_path)
            self.new_model = False  # Reset flag after saving
        return

    def load(self, path=""):
        """
        Load saved model from disk.
        Returns the loaded model, or None if not found.
        """

        if len(path) == 0:
            # Use model_path if set (from set_model_path), otherwise generate path
            if not self.model_path or self.model_path == "":
                self.model_path = self.get_model_path()
            path = self.model_path
        else:
            self.model_path = path

        if self.use_saved_model:
            if os.path.exists(path):
                # use joblib to reload model state
                print("    loading sklearn model from: ", self.model_path)
                self.model = joblib.load(self.model_path)
                self.loaded_from_file = True
                self.is_trained = True  # Model is trained if loaded from file
                return self.model
            else:
                # Model file doesn't exist, mark as new model
                self.new_model = True
                print(f"    No saved model found at: {self.model_path}")
        else:
            print("    Model loading disabled (use_saved_model=False)")

        return None

    def model_exists(self) -> bool:
        """
        Check if model exists on disk.
        Uses self.model_path if set (from set_model_path()), otherwise generates path via get_model_path().
        """
        # If model_path was set via set_model_path(), use it (this is what NNStrategy does)
        if self.model_path and self.model_path != "":
            return os.path.exists(self.model_path)
        # Otherwise, generate path using get_model_path()
        path = self.get_model_path()
        return os.path.exists(path)

    def new_model_created(self) -> bool:
        return SklearnBaseClassifier.new_model  # note use of class-level variable
