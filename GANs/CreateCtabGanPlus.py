# pragma pylint: disable=C0103, C0114, C0115, C0116
# type: ignore
# pylint: disable=import-error
"""
CreateCtabGanPlus — backwards-compatibility shim.

Kept so existing freqtrade configs (``--strategy CreateCtabGanPlus``) and
backtest metadata continue to work.  The actual implementation lives in
``CreateGAN`` and is selected via ``gan_type = GANType.CTAB_GAN``.
"""

from __future__ import annotations

import sys
from pathlib import Path

group_dir = str(Path(__file__).parent)
sys.path.append(group_dir)

from CreateGAN import CreateGAN  # noqa: E402
from Framework.BaseStrategy import GANType  # noqa: E402


class CreateCtabGanPlus(CreateGAN):
    gan_type = GANType.CTAB_GAN

    # Keep the local strategy defaults aligned with MASTER values
    MIN_BUY_GAIN_THRESHOLD = CreateGAN.MASTER_MIN_BUY_GAIN_THRESHOLD
    MIN_SELL_LOSS_THRESHOLD = CreateGAN.MASTER_MIN_SELL_LOSS_THRESHOLD
    TRAINING_TYPE = CreateGAN.MASTER_TRAINING_TYPE
