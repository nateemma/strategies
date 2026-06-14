# pragma pylint: disable=C0103, C0114, C0115, C0116, C0301, C0302, C0303, C0325, C0411, C0413
# pragma pylint: disable=W0105, W1203, W1309, W1514, W0613, W0621,
# type: ignore
# pylint: disable=import-error
# flake8: noqa: F401, E402, F541, W0718, W0719

"""
NNGANomalyStrategy - GANomaly-based strategy using NNStrategy framework
"""


import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from pandas import DataFrame
import numpy as np
import pandas as pd
import traceback

from freqtrade.strategy import DecimalParameter, IntParameter

import logging

log = logging.getLogger(__name__)

# Add parent directory to path to import NNNC
group_dir = str(Path(__file__).parent)
sys.path.append(group_dir)

from Framework.BaseNNStrategy import BaseNNStrategy, HAS_MLX, StrategyConfig
from utils.Environment import Environment
from Framework.BaseStrategy import BaseStrategy, ScalerType, MarketRegime, TradingAction, FlowDirection, MomentumDirection, RiskLevel, GANType
from Framework.BaseNNStrategy import BaseNNStrategy, HAS_MLX, StrategyConfig
from utils.Environment import Environment
from Framework.BaseStrategy import BaseStrategy, ScalerType, MarketRegime, TradingAction, FlowDirection, MomentumDirection, RiskLevel, GANType

# from utils.ClassifierKerasAnomaly import ClassifierKerasAnomaly
from NNAnomalyStrategy import NNAnomalyStrategy
import NNGANomalyClassifier


# -----------


class NNGANomalyStrategy(NNAnomalyStrategy):
    """
    GANomaly-based strategy that uses GAN training for anomaly detection.
    Inherits from NNAnomalyStrategy to reuse the anomaly detection framework.
    """

    plot_config = {
        "main_plot": {
            "close": {"color": "lightsteelblue"},
        },
        "subplots": {
            "Diff": {
                "predict_buy": {"color": "green"},
                "predict_sell": {"color": "red"},
                "%train_trading": {"color": "green"},
                "%trading": {"color": "cyan"},
                "anomaly_threshold": {"color": "purple"},
                "reconstruction_error": {"color": "orange"},
            },
        },
    }

    # -----------
    # Hyperopt parameters
    # -----------

    # Buy hyperspace params:
    buy_params = { **NNAnomalyStrategy.buy_params,
        "prediction_threshold": 0.6,
        "anomaly_threshold_multiplier": 1.8,
        "min_anomaly_duration": 2,
        "entry_error_threshold": 0.01,
    }


    # -----------
    # Class level parameters
    # -----------

    augment_training_data = False  # GANomaly doesn't use external GAN augmentation

    # -----------
    # Override functions
    # -----------

    def get_classifier_type(self):
        """Return the type of classifier used for training/predicting"""
        return NNGANomalyClassifier.ClassifierType.LSTM

    # -----------

    def get_classifier(
        self, classifier_type, pair, seq_len, num_features
    ) -> NNGANomalyClassifier:
        """Return the classifier used for training/predicting"""

        classifier, _ = NNGANomalyClassifier.create_classifier(
            classifier_type,
            pair,
            num_features,
            seq_len
        )

        return classifier
