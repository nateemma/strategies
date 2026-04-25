# pragma pylint: disable=C0103, C0114, C0115, C0116
# type: ignore
# pylint: disable=import-error
"""
CreateMTCtabGanPlus — backwards-compatibility shim.

Kept so existing freqtrade configs (``--strategy CreateMTCtabGanPlus``)
continue to work.  The actual implementation lives in ``CreateMTGAN``
and is selected via ``gan_type = GANType.MT_CTAB_GAN``.
"""

from __future__ import annotations

import sys
from pathlib import Path

group_dir = str(Path(__file__).parent)
sys.path.append(group_dir)

from CreateMTGAN import CreateMTGAN  # noqa: E402
from Framework.BaseStrategy import GANType  # noqa: E402


class CreateMTCtabGanPlus(CreateMTGAN):
    gan_type = GANType.MT_CTAB_GAN

    # Keep the local strategy defaults aligned with MASTER values
    MIN_BUY_GAIN_THRESHOLD = CreateMTGAN.MASTER_MIN_BUY_GAIN_THRESHOLD
    MIN_SELL_LOSS_THRESHOLD = CreateMTGAN.MASTER_MIN_SELL_LOSS_THRESHOLD
    TRAINING_TYPE = CreateMTGAN.MASTER_TRAINING_TYPE
