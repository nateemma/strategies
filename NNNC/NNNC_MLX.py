# pragma pylint: disable=C0103, C0114, C0115, C0116, C0301, C0302, C0303, C0325, C0411, C0413
# pragma pylint: disable=W0105, W1203, W1309, W1514, W0613, W0621,
# type: ignore
# pylint: disable=import-error
# flake8: noqa: F401, E402, F541, W0718, W0719

"""
NNNC_MLX - Subclass of NNNCStrategy using MLX variants (Apple's native metal layers). Should be much faster
"""

import sys
from pathlib import Path
import os

import mlx.core as mx

group_dir = str(Path(__file__).parent)
sys.path.append(group_dir)

from NNNCStrategy import NNNCStrategy
from NNNClassifierMLX import ClassifierTypeMLX, create_classifier_mlx
from ClassifierKeras import ClassifierKeras


class NNNC_MLX(NNNCStrategy):

    # Use only real signals as the basis; the GAN provides synthetic
    # samples below, so layered signal augmentation would double-count.
    augment_training_data = True

    # Trend filter on entry (2026-05-30 experiment). Per
    # project_gbb_labeler_exhausted.md: the gbb labeler's signal can't
    # clear stop_loss without help. Restricting model_entry to confirmed
    # uptrends removes bleed-pair losses (LTC/NEAR/AAVE/AVAX during
    # range-bound periods) without naming them.
    #
    # EMA(200) filter (50h lookback on 15m) was too aggressive — cut 84%
    # of trades to 163 / +4.32%. EMA(100) (25h) loosens the gate while
    # still excluding clear bear/range regimes.
    entry_trend_filter_enable = False
    entry_trend_ema_period = 100


    # default is LSTM type. Override get_classifier_type() in subclass
    def get_classifier_type(self):
        """Return the type of classifier used for training/predicting"""
        return ClassifierTypeMLX.LSTM

    def get_classifier(
        self, classifier_type, pair, seq_len, num_features
    ) -> ClassifierKeras:
        if hasattr(mx, "metal") and mx.metal.is_available():
            """Return the classifier used for training/predicting"""
            clf, _ = create_classifier_mlx(
                classifier_type, pair, num_features, seq_len, 3
            )
        else:
            print(
                "ERROR: This strategy requires Apple's MLX package, and only runs on native Apple hardware"
            )
            clf = None
        return clf
