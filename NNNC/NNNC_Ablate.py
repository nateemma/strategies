"""NNNC_Ablate — throwaway no-GAN ablation copy of NNNC_DDPM_MLX.

Same MLX-LSTM classifier and H=48 / 0.007 / gbb label regime (inherited
from TrainingConfig defaults — the GAN-saved params only bind GAN-based
models, so the no-GAN path reads the class-attribute defaults), but the
GAN is disabled and include_list is selected per rung via ABLATION_RUNG.

Isolated scaler names so the production global main_scaler is never
touched. NOT for production use.
"""

import sys
from pathlib import Path

group_dir = str(Path(__file__).parent)
sys.path.append(group_dir)

from NNNC_MLX import NNNC_MLX
from Framework.BaseStrategy import GANType
from _ablation_config import current_include_list, SkipColumnCheck


class NNNC_Ablate(SkipColumnCheck, NNNC_MLX):
    # No GAN — isolate the classifier's dependence on each feature.
    # Everything else (signal augmentation, labels, architecture) matches
    # production so the only varied factor is the feature set.
    gan_type = GANType.NONE

    # Per-rung feature set (selected by the ABLATION_RUNG env var at import).
    include_list = current_include_list()

    # Isolated scalers — never clobber production's global main_scaler.
    main_scaler_name = "exp_scaler"
    main_tensor_scaler_name = "exp_tensor_scaler"
