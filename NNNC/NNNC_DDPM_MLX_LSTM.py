"""
NNNC_DDPM_MLX_LSTM — NNNC MLX-LSTM classifier with TabDDPM augmentation.

Same architecture as NNNC_CGP_MLX_LSTM (MLX LSTM backbone, NNNC family);
swaps the augmentation backend from CTAB-GAN+ to TabDDPM by setting
``gan_type = GANType.TAB_DDPM``.

Train the GAN first with CreateTabDDPM, then train this strategy — the
saved TabDDPM model will be loaded from
``saved_data/NNNC_DDPM_MLX_LSTM/GANs/tab_ddpm/`` 
"""

import sys
from pathlib import Path

group_dir = str(Path(__file__).parent)
sys.path.append(group_dir)

from NNNC_CGP_MLX_LSTM import NNNC_CGP_MLX_LSTM
from Framework.BaseStrategy import GANType


class NNNC_DDPM_MLX_LSTM(NNNC_CGP_MLX_LSTM):
    gan_type = GANType.TAB_DDPM
