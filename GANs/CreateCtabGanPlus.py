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
    use_post_gan_scaling = True
    gan_run_diagnostics = True
