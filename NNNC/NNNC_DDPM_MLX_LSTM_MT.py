"""
NNNC_DDPM_MLX_LSTM_MT — Single-task NNNC LSTM classifier with MT_DDPM
(tensor-aware diffusion) augmentation.

Wraps the existing single-task one-hot labels as {"trading": one_hot}
before calling the multi-task GAN. The classifier itself is unchanged
from NNNC_DDPM_MLX_LSTM — only the augmentation backend differs.

The point of this strategy: keep the single-task classifier (no
auxiliary-task dilution) but feed it temporally-coherent 3D synthetic
windows from MT_DDPM instead of iid rows from TAB_DDPM.

GAN save path: saved_data/GANs_PostScale/mt_ddpm/ — uses the v2 pipeline
(post-augmentation tensor scaling). The MT_DDPM model trained by
CreateMTDDPM with the same flag is found here automatically.
"""

import sys
from pathlib import Path

group_dir = str(Path(__file__).parent)
sys.path.append(group_dir)

from NNNC_DDPM_MLX_LSTM import NNNC_DDPM_MLX_LSTM
from Framework.BaseStrategy import GANType


class NNNC_DDPM_MLX_LSTM_MT(NNNC_DDPM_MLX_LSTM):
    gan_type = GANType.MT_DDPM
    gan_target_ratio = 0.8
    augment_training_data = True
    use_post_gan_scaling = True
