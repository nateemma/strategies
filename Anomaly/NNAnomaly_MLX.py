# pragma pylint: disable=C0103, C0114, C0115, C0116, C0301, C0302, C0303, C0325, C0411, C0413
# pragma pylint: disable=W0105, W1203, W1309, W1514, W0613, W0621,
# type: ignore
# pylint: disable=import-error
# flake8: noqa: E402

"""
NNAnomaly_MLX - MLX (Apple mlx) plain-autoencoder anomaly strategy.

The MLX-backend counterpart of NNAnomalyStrategy (a plain autoencoder + trading
head, no adversarial GAN). There is no standalone MLX autoencoder detector — the
MLX GANomaly detector becomes a pure autoencoder when its adversarial Phase 1 is
skipped (gan_epochs=0). Reusing it (rather than duplicating the encoder/decoder/
head) keeps the architecture, training, and scoring byte-identical to
NNGANomalyStrategyMLX *minus* the GAN, so the two form a clean GAN vs no-GAN pair
on the same MLX path. Requires MLX / Metal to be available.
"""

import logging
import sys
from pathlib import Path


log = logging.getLogger(__name__)

group_dir = str(Path(__file__).parent)
sys.path.append(group_dir)

import NNGANomalyClassifierMLX
from Framework.BaseNNStrategy import HAS_MLX
from NNAnomalyStrategy import NNAnomalyStrategy


# -----------


class NNAnomaly_MLX(NNAnomalyStrategy):
    """
    Plain autoencoder anomaly strategy on the MLX backend. Inherits all entry/
    scoring logic, params, and plot config from NNAnomalyStrategy; swaps the
    Keras autoencoder for the MLX detector with the adversarial phase disabled.
    """

    def get_classifier_type(self):
        if not HAS_MLX:
            raise RuntimeError("NNAnomaly_MLX requires MLX (Apple mlx) — not available")
        return NNGANomalyClassifierMLX.ClassifierTypeMLX.LSTM

    def get_classifier(self, classifier_type, pair, seq_len, num_features):
        classifier, _ = NNGANomalyClassifierMLX.create_classifier_mlx(
            classifier_type,
            pair,
            num_features,
            seq_len,
        )
        # adv_weight=0 removes the adversarial term from the joint GANomaly
        # objective, leaving the same encoder-decoder-encoder bow-tie trained on
        # contextual (L1) + encoder (L2) loss only — i.e. the no-GAN ablation of
        # the exact same architecture/score. Instance-scoped, so
        # NNGANomalyStrategyMLX's adv_weight is untouched.
        classifier.adv_weight = 0.0
        return classifier
