# pragma pylint: disable=C0103, C0114, C0115, C0116, C0301, C0302, C0303, C0325, C0411, C0413
# type: ignore
"""
NNNC_DDPM_MLX_LSTM_MT_v2 — single-task NNNC LSTM with MT_DDPM augmentation
using the post-GAN scaling pipeline. Pairs with CreateMTDDPM_v2.

This is the OPT-IN counterpart to NNNC_DDPM_MLX_LSTM_MT, used to evaluate
whether the post-GAN normalisation pipeline is equal-to or better-than
the existing pre-GAN normalisation pipeline.
"""

import sys
from pathlib import Path

group_dir = str(Path(__file__).parent)
sys.path.append(group_dir)

from NNNC_DDPM_MLX_LSTM_MT import NNNC_DDPM_MLX_LSTM_MT


class NNNC_DDPM_MLX_LSTM_MT_v2(NNNC_DDPM_MLX_LSTM_MT):
    use_post_gan_scaling = True
