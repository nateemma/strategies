# pragma pylint: disable=C0103, C0114, C0115, C0116
# type: ignore
# pylint: disable=import-error
# flake8: noqa: F401, E402
"""
CreateMTWGAN — backwards-compatibility shim.

Kept so existing freqtrade configs (``--strategy CreateMTWGAN``) and
backtest metadata continue to work.  The actual implementation lives in
``CreateMTGAN`` and is selected via ``gan_type = GANType.MT_WGAN``.
"""

from __future__ import annotations

import sys
from pathlib import Path

group_dir = str(Path(__file__).parent)
sys.path.append(group_dir)

from CreateMTGAN import CreateMTGAN  # noqa: E402
from Framework.BaseStrategy import GANType  # noqa: E402


class CreateMTWGAN(CreateMTGAN):
    gan_type = GANType.MT_WGAN
    # v2 pipeline: GAN trains on raw features; tensor scaler runs after
    # augmentation. Saves/loads under saved_data/GANs_PostScale/mt_wgan/
    # rather than the pre-normalized GANs/mt_wgan/.
    use_post_gan_scaling = True
    # After training, generate ~50K samples and print the same fidelity
    # report the strategy emits (mean/σ shifts, joint correlations, lag-1
    # autocorrelation). Set False to skip if you only want loss output.
    gan_run_diagnostics = True
