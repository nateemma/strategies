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
from freqtrade.strategy import DecimalParameter

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

    # First-hour stop-loss grace (BaseStrategy): wide stop for the first 3h,
    # then tighten to the ATR-adaptive stop. Restores the pre-confirm_trade_exit
    # "don't stop out on entry noise" behaviour, safely. See NNMT sweep (3h/-0.15).
    stoploss_grace_hours = DecimalParameter(
        0.0, 6.0, default=3.0, decimals=1, space="sell", load=True, optimize=True
    )
    stoploss_grace_level = DecimalParameter(
        -0.30, -0.05, default=-0.15, decimals=2, space="sell", load=True, optimize=True
    )

    # HORIZON is now inherited from BaseNNStrategy → TrainingConfig.HORIZON.
    # Production = 3. The 2026-05-28 univariate H sweep landed at H=3 at
    # threshold=0.005 (+21.68% vs +15.97%); the 2026-05-30 learnability
    # sweep then showed (H=3, threshold=0.003) is the right joint setting.
    # Threshold lives in TrainingConfig too — change them together, not
    # in isolation. Don't reintroduce the override here.

    # Inherit the shared entry/exit controls; override only what differs.
    #
    # Values validated 2026-09-02 by a 4x4 rvol x guard_metric grid at
    # thr=0.69 across three windows (W1 2025-09..2026-08, W2 2024-06..2025-05,
    # W3 2023). See regime/README.md.
    #
    #   entry_rvol_threshold 3.1  The dominant guard: it alone moves trade count
    #       49 -> 160 and separates profitable from unprofitable. Its optimum on
    #       RETURN wanders by era (W1 wants 2.5, W2 and W3 want 1.5 - no two
    #       windows agree), while its optimum on CALMAR is 3.1 in ALL THREE.
    #       Was 0.5 here, which measured -4.73% on W1. Do not lower it on
    #       single-window evidence: rvol 2.5 looked like a free win on W1
    #       (+1.00pp, more trades, higher win rate) and lost on both others.
    #
    #   entry_guard_threshold 0.3  Uniformly dominant: 0.3 >= 0.0 >= -0.4 on
    #       both return and Calmar in every cell of all three windows, and 0.6
    #       is byte-identical to 0.3, so this is a saturating plateau rather
    #       than a peak. Against the previous -0.4: return +0.50/+0.19/+0.77pp,
    #       Calmar +3.31/+1.27/+5.56, and W3 goes 13 -> 20 trades.
    #
    #   adx 38 / atr_pct 0.006 / bb_width 0.06 are LOAD-BEARING, not slack.
    #       Loosening adx to 30 costs -3.92%; atr_pct to 0.003 costs -7.10%.
    #       Removing bb_width is invisible on W1 (-0.05pp) but turns W3 from
    #       +3.01% into -0.23% - it is the guard that carries the distant era.
    buy_params = { **BaseNNStrategy.buy_params,
        "entry_adx_threshold": 38.0,
        "entry_atr_pct": 0.006,
        "entry_bb_width_threshold": 0.06,
        "entry_close_norm_threshold": -0.1,
        "entry_enable_guards": True,
        "entry_guard_threshold": 0.3,
        "entry_rvol_threshold": 3.1,
        "prediction_threshold": 0.69,
    }
    
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
