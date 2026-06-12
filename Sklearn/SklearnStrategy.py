# pragma pylint: disable=C0103, C0114, C0115, C0116, C0301, C0302, C0303, C0325, C0411, C0413
# pragma pylint: disable=W0105, W1203, W1309, W1514, W0613, W0621,
# type: ignore
# pylint: disable=import-error

"""
SklearnStrategy - Base class for sklearn classifier strategies
"""

import sys
from pathlib import Path
from pandas import DataFrame
import pandas as pd
import numpy as np
from typing import Any, List
from sklearn.utils import shuffle

# Add Sklearn folder + parent strategies root to sys.path for sibling imports.
group_dir = str(Path(__file__).parent)
strat_dir = str(Path(__file__).parent.parent)
sys.path.append(group_dir)
sys.path.append(strat_dir)

from Framework.BaseNNStrategy import BaseNNStrategy  # noqa: E402
from Framework.BaseStrategy import TradingAction  # noqa: E402
from Predictors.SklearnBaseClassifier import SklearnBaseClassifier  # noqa: E402
from utils.ClassifierKeras import ClassifierKeras  # noqa: E402
import SklearnClassifier  # noqa: E402


class SklearnStrategy(BaseNNStrategy):
    """
    Base strategy for sklearn classifiers.
    Inherits from BaseNNStrategy but uses sklearn classifiers instead of
    neural networks.

    Sklearn classifiers work with 2D DataFrames (samples, features), not 3D
    tensors.  This class overrides training and prediction methods to handle
    the DataFrame conversion.
    """

    # Sklearn classifiers work with DataFrames directly, not tensors
    # seq_len is effectively 1 for sklearn (single timestep per sample)
    seq_len = 1

    def get_classifier_type(self):
        """Return the type of sklearn classifier used for training/predicting"""
        return SklearnClassifier.ClassifierType.RandomForest

    def get_classifier(
        self, classifier_type, pair, seq_len, num_features
    ) -> SklearnBaseClassifier:
        """Return the sklearn classifier used for training/predicting"""
        # seq_len is ignored for sklearn (set to 1), but we pass it for compatibility
        clf, _ = SklearnClassifier.create_classifier(
            classifier_type, pair, num_features, seq_len, 3  # 3 classes: Sell/Hold/Buy
        )
        # Sklearn classifiers don't use batch_size, but set_model_path is handled by SklearnBaseClassifier
        return clf

    def train_model(
        self,
        dataframes: List[DataFrame],
        labels: List[Any],
        classifier: SklearnBaseClassifier,
    ):
        """
        Train the sklearn model.
        Overrides NNStrategy.train_model() to work with DataFrames instead of tensors.
        """
        # Prepare normalized DataFrames (sklearn works with DataFrames, not tensors)
        df_train_list = []
        df_test_list = []
        train_labels_list = []
        test_labels_list = []

        for i, df in enumerate(dataframes):
            # Normalize the dataframe
            df_norm = self.scale_dataframe(df)
            pair_labels = np.asarray(labels[i])

            # Split data with proper temporal separation
            split_idx = int(0.8 * len(df_norm))
            df_train = df_norm.iloc[:split_idx].copy()
            df_test = df_norm.iloc[split_idx:].copy()
            train_labels = pair_labels[:split_idx]
            test_labels = pair_labels[split_idx:]

            df_train_list.append(df_train)
            df_test_list.append(df_test)
            train_labels_list.append(train_labels)
            test_labels_list.append(test_labels)

        # Combine all pairs into single DataFrames
        df_train = pd.concat(df_train_list, ignore_index=True)
        df_test = pd.concat(df_test_list, ignore_index=True)
        train_labels = np.concatenate(train_labels_list, axis=0)
        test_labels = np.concatenate(test_labels_list, axis=0)

        # Allow subclasses to enhance the training data (e.g., WGAN augmentation)
        # This is called after combining all pairs but before shuffling
        df_train, train_labels = self.enhance_training_data(df_train, train_labels)

        # Shuffle training data
        df_train, train_labels = shuffle(df_train, train_labels, random_state=42)

        # Train the sklearn classifier (no class_weights parameter for sklearn)
        classifier.train(
            df_train, df_test, train_labels, test_labels, force_train=False
        )

        return None

    def get_predictions(self, dataframe: DataFrame, classifier: SklearnBaseClassifier):
        """
        Get predictions from the sklearn model.
        Overrides NNStrategy.get_predictions() to work with DataFrames instead of tensors.
        """
        pred_threshold = self.prediction_threshold.value

        # Normalize dataframe (sklearn works with DataFrames directly)
        df_norm = self.scale_dataframe(dataframe)

        # Get predictions from sklearn classifier
        pred_probs = classifier.predict(df_norm)

        # Convert predictions to probabilities if needed (sklearn returns class indices)
        # For sklearn, we need to handle the prediction format
        # Most sklearn classifiers return class indices, not probabilities
        # If we need probabilities, we'd need to use predict_proba() instead

        # For now, assume classifier.predict() returns probabilities or class indices
        # This may need adjustment based on the specific sklearn classifier used
        predictions = self.argmax_with_threshold(
            pred_probs,
            threshold=pred_threshold,
            default_class=TradingAction.HOLD,
        )

        return predictions
