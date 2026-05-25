# pragma pylint: disable=C0103, C0114, C0115, C0116, C0301, C0302, C0303, C0325, C0411, C0413
# pragma pylint: disable=W0105, W1203, W1309, W1514, W0613, W0621,
# type: ignore
# pylint: disable=import-error
# flake8: noqa: F401, E402, F541, W0718, W0719

"""
NNNC_WGAN — NNNCStrategy with WGAN-GP augmentation.

Just declares ``gan_type = GANType.WGAN``; everything else (loading the
GAN, validating its metadata, balancing classes, denormalising) is
dispatched by ``BaseNNStrategy.enhance_training_data`` — see
``GANs.balance.balance_single_task`` for the actual augmentation.
"""

from NNNCStrategy import NNNCStrategy
from GANs.GANType import GANType

import dataclasses                                                                                                                                               
from Framework.BaseStrategy import StrategyConfig  

class NNNC_WGAN(NNNCStrategy):

    # GAN augmentation — single-task WGAN-GP (MLX-accelerated when available).
    gan_type = GANType.WGAN

    # Use only real signals as the basis; the GAN provides the synthetic
    # samples below, so layered signal augmentation would double-count.
    augment_training_data = False

    # Single-task ratio: each minority class is brought up to 80% of the
    # majority class size.  See balance_single_task for semantics.
    gan_target_ratio = 0.8

    # v2 pipeline: WGAN MLX generator outputs raw-space features (after
    # internal z-score + inverse), strategy applies tensor scaling after
    # augmentation. Reads model from saved_data/GANs_PostScale/wgan/.
    use_post_gan_scaling = True

    # Note: dataclasses.replace on strategy_config doesn't actually flip
    # gan_run_diagnostics — that's a class-attribute lookup (see memory
    # note reference_gan_run_diagnostics_quirk). Set as plain attr too.
    gan_run_diagnostics = True
    strategy_config = dataclasses.replace(NNNCStrategy.strategy_config, gan_run_diagnostics=True)
