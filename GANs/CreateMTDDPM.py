# pragma pylint: disable=C0103, C0114, C0115, C0116
# type: ignore
# pylint: disable=import-error
# flake8: noqa: F401, E402
"""
CreateMTDDPM — multi-task DDPM creator strategy.

Thin subclass of CreateMTGAN that selects the MT_DDPM backend via the
``gan_type`` class attribute. Mirrors CreateMTWGAN's shim pattern.
"""

from __future__ import annotations

import sys
from pathlib import Path

group_dir = str(Path(__file__).parent)
sys.path.append(group_dir)

from CreateMTGAN import CreateMTGAN  # noqa: E402
from Framework.BaseStrategy import GANType  # noqa: E402


class CreateMTDDPM(CreateMTGAN):
    gan_type = GANType.MT_DDPM
