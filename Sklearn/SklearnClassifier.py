# pragma pylint: disable=C0103, C0114, C0115, C0116, C0301, C0302, C0303, C0325, C0411, C0413
# pragma pylint: disable=E1101
# pragma pylint: disable=W0105, W1203, W1309, W1514, W0613, W0621,
# type: ignore
# pylint: disable=import-error

# Sklearn Classifier: collection of sklearn classifiers, types and access functions

# usage: classifier, name = SklearnClassifier.create_classifier(classifier_type, pair, nfeatures, seq_len, tag="")

import numpy as np
from pandas import DataFrame, Series
import pandas as pd

pd.options.mode.chained_assignment = None  # default='warn'

# Strategy specific imports, files must reside in same folder as strategy
import sys
from pathlib import Path
from enum import Enum, auto

sys.path.append(str(Path(__file__).parent))
sys.path.append(str(Path(__file__).parent.parent))

import logging
import warnings

warnings.simplefilter(action="ignore", category=FutureWarning)

from sklearn.ensemble import (
    RandomForestClassifier,
    GradientBoostingClassifier,
    AdaBoostClassifier,
    VotingClassifier,
    StackingClassifier,
)
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.svm import SVC, LinearSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.discriminant_analysis import (
    LinearDiscriminantAnalysis,
    QuadraticDiscriminantAnalysis,
)
from xgboost import XGBClassifier

from utils.ClassifierSklearn import ClassifierSklearn
from utils.DataframeUtils import DataframeUtils


# --------------------------------------------------------------
# Define classes for each type of classifier
# --------------------------------------------------------------


class SklearnClassifier_LogisticRegression(ClassifierSklearn):
    is_trained = False
    clean_data_required = False

    def create_classifier(self):
        return LogisticRegression(max_iter=10000)


class SklearnClassifier_DecisionTree(ClassifierSklearn):
    is_trained = False
    clean_data_required = False

    def create_classifier(self):
        return DecisionTreeClassifier()


class SklearnClassifier_RandomForest(ClassifierSklearn):
    is_trained = False
    clean_data_required = False

    def create_classifier(self):
        return RandomForestClassifier()


class SklearnClassifier_GaussianNB(ClassifierSklearn):
    is_trained = False
    clean_data_required = False

    def create_classifier(self):
        return GaussianNB()


class SklearnClassifier_MLP(ClassifierSklearn):
    is_trained = False
    clean_data_required = False

    def create_classifier(self):
        return MLPClassifier(
            hidden_layer_sizes=(64, 32, 1),
            max_iter=50,
            activation="relu",
            learning_rate="adaptive",
            alpha=1e-5,
            solver="adam",
            verbose=0,
        )


class SklearnClassifier_KNeighbors(ClassifierSklearn):
    is_trained = False
    clean_data_required = False

    def create_classifier(self):
        return KNeighborsClassifier(n_neighbors=3)


class SklearnClassifier_StochasticGradientDescent(ClassifierSklearn):
    is_trained = False
    clean_data_required = False

    def create_classifier(self):
        return SGDClassifier()


class SklearnClassifier_GradientBoosting(ClassifierSklearn):
    is_trained = False
    clean_data_required = False

    def create_classifier(self):
        return GradientBoostingClassifier()


class SklearnClassifier_AdaBoost(ClassifierSklearn):
    is_trained = False
    clean_data_required = False

    def create_classifier(self):
        return AdaBoostClassifier()


class SklearnClassifier_LinearSVC(ClassifierSklearn):
    is_trained = False
    clean_data_required = False

    def create_classifier(self):
        return LinearSVC(dual=False)


class SklearnClassifier_GaussianSVC(ClassifierSklearn):
    is_trained = False
    clean_data_required = False

    def create_classifier(self):
        return SVC(kernel="rbf")


class SklearnClassifier_PolySVC(ClassifierSklearn):
    is_trained = False
    clean_data_required = False

    def create_classifier(self):
        return SVC(kernel="poly")


class SklearnClassifier_SigmoidSVC(ClassifierSklearn):
    is_trained = False
    clean_data_required = False

    def create_classifier(self):
        return SVC(kernel="sigmoid")


class SklearnClassifier_LinearDiscriminantAnalysis(ClassifierSklearn):
    is_trained = False
    clean_data_required = False

    def create_classifier(self):
        return LinearDiscriminantAnalysis()


class SklearnClassifier_QuadraticDiscriminantAnalysis(ClassifierSklearn):
    is_trained = False
    clean_data_required = False

    def create_classifier(self):
        return QuadraticDiscriminantAnalysis()


class SklearnClassifier_XGBoost(ClassifierSklearn):
    is_trained = False
    clean_data_required = False

    def create_classifier(self):
        return XGBClassifier()


class SklearnClassifier_Voting(ClassifierSklearn):
    is_trained = False
    clean_data_required = False

    def create_classifier(self):
        # Create base classifiers
        c1 = LinearDiscriminantAnalysis()
        c2 = SVC(kernel="sigmoid")
        c3 = XGBClassifier()
        c4 = AdaBoostClassifier()
        return VotingClassifier(
            estimators=[("c1", c1), ("c2", c2), ("c3", c3), ("c4", c4)], voting="hard"
        )


class SklearnClassifier_Stacking(ClassifierSklearn):
    is_trained = False
    clean_data_required = False

    def create_classifier(self):
        # Create base classifiers for stacking
        c1 = LinearDiscriminantAnalysis()
        c2 = LinearSVC(dual=False)
        c3 = SGDClassifier()
        c4 = LogisticRegression(max_iter=10000)
        c5 = AdaBoostClassifier()
        estimators = [("c1", c1), ("c2", c2), ("c3", c3), ("c4", c4), ("c5", c5)]
        return StackingClassifier(
            estimators=estimators, final_estimator=LogisticRegression(max_iter=10000)
        )


# --------------------------------------------------------------
# Types of Classifier
# --------------------------------------------------------------


class ClassifierType(Enum):
    LogisticRegression = SklearnClassifier_LogisticRegression
    DecisionTree = SklearnClassifier_DecisionTree
    RandomForest = SklearnClassifier_RandomForest
    GaussianNB = SklearnClassifier_GaussianNB
    MLP = SklearnClassifier_MLP
    KNeighbors = SklearnClassifier_KNeighbors
    StochasticGradientDescent = SklearnClassifier_StochasticGradientDescent
    GradientBoosting = SklearnClassifier_GradientBoosting
    AdaBoost = SklearnClassifier_AdaBoost
    LinearSVC = SklearnClassifier_LinearSVC
    GaussianSVC = SklearnClassifier_GaussianSVC
    PolySVC = SklearnClassifier_PolySVC
    SigmoidSVC = SklearnClassifier_SigmoidSVC
    LinearDiscriminantAnalysis = SklearnClassifier_LinearDiscriminantAnalysis
    QuadraticDiscriminantAnalysis = SklearnClassifier_QuadraticDiscriminantAnalysis
    XGBoost = SklearnClassifier_XGBoost
    Voting = SklearnClassifier_Voting
    Stacking = SklearnClassifier_Stacking


# --------------------------------------------------------------
# Factory function to create classifier based on type
# --------------------------------------------------------------


def create_classifier(clf_type: ClassifierType, pair, nfeatures, seq_len, nclasses, tag=""):
    """
    Factory function to create a sklearn classifier instance.

    Args:
        clf_type: ClassifierType enum value specifying the classifier type
        pair: Trading pair (e.g., "BTC/USDT")
        nfeatures: Number of features (sklearn works with 2D DataFrames, not sequences)
        seq_len: Sequence length (ignored - sklearn classifiers work with single timesteps)
        nclasses: Number of output classes
        tag: Optional tag for model naming

    Returns:
        tuple: (classifier instance, classifier name string)
    
    Note:
        ClassifierSklearn expects DataFrames (2D: samples x features), not tensors.
        No sequence or tensor conversion is needed.
    """
    clf_name = str(clf_type).split(".")[-1]
    clf = clf_type.value(pair, tag=tag)

    # Sklearn classifiers work with 2D DataFrames (samples, features), not sequences or tensors
    # Store metadata for reference (but seq_len is effectively 1 for sklearn)
    clf.seq_len = 1
    clf.nfeatures = nfeatures
    clf.nclasses = nclasses

    return clf, clf_name



