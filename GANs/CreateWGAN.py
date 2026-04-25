# pragma pylint: disable=C0103, C0114, C0115, C0116
# type: ignore
# pylint: disable=import-error
"""
CreateWGAN — backwards-compatibility shim.

Kept so existing freqtrade configs (``--strategy CreateWGAN``) and
backtest metadata continue to work.  The actual implementation lives in
``CreateGAN`` and is selected via ``gan_type = GANType.WGAN``.
"""

from __future__ import annotations

import sys
from pathlib import Path

group_dir = str(Path(__file__).parent)
sys.path.append(group_dir)

from CreateGAN import CreateGAN  # noqa: E402
from Framework.BaseStrategy import GANType  # noqa: E402


class CreateWGAN(CreateGAN):
    gan_type = GANType.WGAN
