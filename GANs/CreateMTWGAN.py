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
