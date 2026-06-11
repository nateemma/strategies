# pragma pylint: disable=C0103, C0114, C0115, C0116, C0301, C0302, C0303, C0325, C0411, C0413
# pragma pylint: disable=W0105, W1203, W1309, W1514, W0613, W0621,
# type: ignore
# pylint: disable=import-error
# flake8: noqa: F401, E402, F541, W0718, W0719

"""
NNNC_CGP - Subclass of NNNCStrategy using CTAB-GAN+ for high-fidelity augmentation
MLX variants use Apple's native metal layers. Should be much faster
"""

import sys
from pathlib import Path
import os

group_dir = str(Path(__file__).parent)
sys.path.append(group_dir)

from NNNC_CGP import NNNC_CGP
from NNNClassifierMLX import MLXClassifierMixin, ClassifierTypeMLX
from GANs.GANType import GANType  # noqa: E402


class NNNC_CGP_MLX(MLXClassifierMixin, NNNC_CGP):

    buy_params = { **NNNC_CGP.buy_params,
        "prediction_threshold": 0.7
        }

    # GAN augmentation — single-task CTAB-GAN+.
    gan_type = GANType.CTAB_GAN

    # Use only real signals as the basis; the GAN provides synthetic
    # samples below, so layered signal augmentation would double-count.
    augment_training_data = True

    # Don't push above ~1.0 — the model starts overfitting to synthetic.
    gan_target_ratio = 0.4

    # Per-class autoencoder filter — manifold-aware rejection of off-real
    # synth samples. Same setting used on NNNC_DDPM_MLX (the AE filter is
    # GAN-type-agnostic; see project_ae_filter_win.md). Requires
    # CreateAutoencoderFilter to have been trained for this strategy's
    # storage location.
    gan_synth_autoencoder_threshold = 0.005

    # turn on diagnostics for the GAN (class-level override — the version
    # consumed by BaseNNStrategy at training time is the class attribute,
    # NOT the duplicate field on StrategyConfig).
    gan_run_diagnostics = True

    # v2 pipeline: CTAB-GAN+ handles its own normalization via VGM; strategy
    # passes raw features and the tensor scaler runs after augmentation.
    # Reads model from saved_data/GANs_PostScale/ctab_gan/.
    use_post_gan_scaling = True

    # Architecture: MLX LSTM (default from MLXClassifierMixin). Subclasses
    # set ``classifier_type`` to a different ClassifierTypeMLX value.
    classifier_type = ClassifierTypeMLX.LSTM
