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
from typing import Any, Dict

group_dir = str(Path(__file__).parent)
sys.path.append(group_dir)

from CreateGAN import CreateGAN  # noqa: E402
from Framework.BaseStrategy import GANType  # noqa: E402


class CreateTabDDPM(CreateGAN):
    gan_type = GANType.TAB_DDPM
    # v2 pipeline: writes to saved_data/GANs_PostScale/tab_ddpm/. Must match
    # the same flag on NNNC_DDPM_MLX_LSTM (strategy side).
    use_post_gan_scaling = True
    gan_run_diagnostics = True

    def get_default_gan_config(self) -> Dict[str, Any]:
        # Enable balanced per-class batch sampling so minority classes (0/2)
        # get equal gradient signal during training. Without this, class 1
        # (Hold, ~75% of rows) dominates gradient and the model under-trains
        # the volatility-feature tails on c0/c2 — exactly the joint-corr
        # collapse signature observed on the g=1.0 clean-data baseline.
        base = dict(super().get_default_gan_config())
        base["class_balanced_sampling"] = True
        return base
