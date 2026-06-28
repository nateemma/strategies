# pragma pylint: disable=C0103, C0114, C0115, C0116, C0301, C0302, C0303, C0325, C0411, C0413
# pragma pylint: disable=W0105, W1203, W1309, W1514, W0613, W0621,
# type: ignore
# pylint: disable=import-error
# flake8: noqa: E402

"""
NNGANomaly_MLX - MLX (Apple mlx) GANomaly strategy.

Mirrors NNGANomalyStrategy but selects the MLX classifier port
(NNGANomalyClassifierMLX). Requires MLX / Metal to be available.
"""

import logging
import sys
from pathlib import Path


log = logging.getLogger(__name__)

group_dir = str(Path(__file__).parent)
sys.path.append(group_dir)

import NNGANomalyClassifierMLX
from Framework.BaseNNStrategy import HAS_MLX
from NNGANomalyStrategy import NNGANomalyStrategy


# -----------


class NNGANomaly_MLX(NNGANomalyStrategy):
    """
    GANomaly strategy using the MLX classifier port. Inherits all params and
    plot config from NNGANomalyStrategy; only swaps the classifier backend.
    """

    # Buy hyperspace params:
    # buy_params = {
    #     **NNGANomalyStrategy.buy_params,
    #     "prediction_threshold": 0.65,
    #     "anomaly_threshold_multiplier": 1.1,
    #     "min_anomaly_duration": 3,
    #     "entry_error_threshold": 0.3,
    # }

    def get_classifier_type(self):
        if not HAS_MLX:
            raise RuntimeError("NNGANomaly_MLX requires MLX (Apple mlx) — not available")
        return NNGANomalyClassifierMLX.ClassifierTypeMLX.LSTM

    def get_classifier(self, classifier_type, pair, seq_len, num_features):
        classifier, _ = NNGANomalyClassifierMLX.create_classifier_mlx(
            classifier_type,
            pair,
            num_features,
            seq_len,
        )
        return classifier
