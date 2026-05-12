# pragma pylint: disable=C0103, C0114, C0115, C0116
# type: ignore
# pylint: disable=import-error
"""
CreateTabDDPM — builder strategy for the TabDDPM GAN.

Run under freqtrade backtesting to train + save a TabDDPM model.  The
actual training implementation lives in ``CreateGAN``; this class just
selects the backend via ``gan_type = GANType.TAB_DDPM``.

Usage:
    zsh user_data/strategies/scripts/test_strat.sh GANs CreateTabDDPM \\
        --timerange=20220101-

Saves to: ``user_data/strategies/saved_data/CreateTabDDPM/GANs/tab_ddpm/``
"""

from __future__ import annotations

import sys
from pathlib import Path

group_dir = str(Path(__file__).parent)
sys.path.append(group_dir)

from CreateGAN import CreateGAN  # noqa: E402
from Framework.BaseStrategy import GANType  # noqa: E402


class CreateTabDDPM(CreateGAN):
    gan_type = GANType.TAB_DDPM
