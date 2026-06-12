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

    # Model architecture is selected declaratively: subclasses set
    # ``classifier_type`` to a ``NNNClassifier.ClassifierType`` (Keras) or a
    # ``ClassifierTypeMLX`` (MLX, via MLXClassifierMixin) value instead of
    # overriding ``get_classifier_type()``. The framework reads it through
    # ``get_classifier_type()`` below (see BaseNNStrategy: it caches
    # ``self.classifier_type = self.get_classifier_type()`` before building).
    classifier_type = NNNClassifier.ClassifierType.LSTM

    # NNNC-family ATR-adaptive stoploss tuning. NNNC has no model-driven
    # soft exit (unlike NNMT's strategy_custom_exit on the profit/regime
    # heads), so the stop is the only safety net for losing trades.
    # BaseNNStrategy defaults to multiplier=2.5, floor=-0.04, cap=-0.02
    # (range [-2%, -4%]); on NNNC that fires too aggressively and kills
    # trades that would have recovered (backtest 2026-05-26: 167 stops at
    # -2.53% avg vs static -5% baseline at 75 stops, net -8pp).
    # Loosen to range [-4%, -6%] — adaptive but with enough room for
    # the unclog / model_exit winners to materialise.
    atr_stoploss_multiplier = 4.0
    atr_stoploss_floor = -0.06   # loosest stop allowed (most negative)
    atr_stoploss_cap = -0.04     # tightest stop allowed (closest to zero)

    # HORIZON is now inherited from BaseNNStrategy → TrainingConfig.HORIZON.
    # Production = 3. The 2026-05-28 univariate H sweep landed at H=3 at
    # threshold=0.005 (+21.68% vs +15.97%); the 2026-05-30 learnability
    # sweep then showed (H=3, threshold=0.003) is the right joint setting.
    # Threshold lives in TrainingConfig too — change them together, not
    # in isolation. Don't reintroduce the override here.

    def get_classifier_type(self):
        """Return the type of classifier used for training/predicting"""
        return self.classifier_type

    def get_classifier(
        self, classifier_type, pair, seq_len, num_features
    ) -> Any:
        """Return the classifier used for training/predicting"""
        clf, _ = NNNClassifier.create_classifier(
            classifier_type, pair, num_features, seq_len, 3
        )
        return clf
