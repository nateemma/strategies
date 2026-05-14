"""
NNNC_DDPM_MLX_LSTM_MT — Single-task NNNC LSTM classifier with MT_DDPM
(tensor-aware diffusion) augmentation.

Wraps the existing single-task one-hot labels as {"trading": one_hot}
before calling the multi-task GAN. The classifier itself is unchanged
from NNNC_DDPM_MLX_LSTM — only the augmentation backend differs.

The point of this strategy: keep the single-task classifier (no
auxiliary-task dilution) but feed it temporally-coherent 3D synthetic
windows from MT_DDPM instead of iid rows from TAB_DDPM.

GAN save path: saved_data/GANs/mt_ddpm/ (shared across all strategies;
the existing MT_DDPM model trained by a CreateMTDDPM strategy is found
here automatically — no separate retrain is required).
"""

import sys
from pathlib import Path

group_dir = str(Path(__file__).parent)
sys.path.append(group_dir)

from NNNC_DDPM_MLX_LSTM import NNNC_DDPM_MLX_LSTM
from Framework.BaseStrategy import GANType


class NNNC_DDPM_MLX_LSTM_MT(NNNC_DDPM_MLX_LSTM):
    gan_type = GANType.MT_DDPM
