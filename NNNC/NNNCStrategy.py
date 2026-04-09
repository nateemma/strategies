# pragma pylint: disable=C0103, C0114, C0115, C0116, C0301, C0302, C0303, C0325, C0411, C0413
# pragma pylint: disable=W0105, W1203, W1309, W1514, W0613, W0621,
# type: ignore
# pylint: disable=import-error

"""
NNNCStrategy - Base class for Neural Network N-ary Classification strategies
"""

import sys
from pathlib import Path
from pandas import DataFrame
import numpy as np
import traceback

# Add parent directory to path to import NNNC
group_dir = str(Path(__file__).parent)
sys.path.append(group_dir)

from Framework.BaseNNStrategy import BaseNNStrategy, HAS_MLX, StrategyConfig
from utils.Environment import Environment
from Framework.BaseStrategy import BaseStrategy, ScalerType, MarketRegime, TradingAction, FlowDirection, MomentumDirection, RiskLevel, GANType
from typing import Any
import NNNClassifier
from Framework.BaseStrategy import ModelType, NormalizationType


class NNNCStrategy(BaseNNStrategy):

    # Strategy configuration
    strategy_config = StrategyConfig(
        normalization=NormalizationType.ROLLING_ROBUST,
        model_type=ModelType.KERAS,
        needs_training=True,
        seq_len=16,
    )

    augment_training_data = True  # no GAn, so augment signals

    def get_classifier_type(self):
        """Return the type of classifier used for training/predicting"""
        return NNNClassifier.ClassifierType.LSTM

    def get_classifier(
        self, classifier_type, pair, seq_len, num_features
    ) -> Any:
        """Return the classifier used for training/predicting"""
        clf, _ = NNNClassifier.create_classifier(
            classifier_type, pair, num_features, seq_len, 3
        )
        return clf
