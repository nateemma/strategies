# pragma pylint: disable=C0103, C0114, C0115, C0116, C0301
# type: ignore

"""
NNNC_DDPM_MLX_P1_ctrl — Phase-1 CONTROL (unchanged sampler) for the GAN
sample-quality plan (docs/GAN_TODO.md #5).

Trains a fresh TabDDPM-augmented classifier at the current sampler settings
(num_sample_steps=50, _ZSCORE_CLIP=4) so its fidelity report is the baseline the
Phase-1 sampler tweaks are measured against, and confirms it reproduces the ~20.72%
Phase-0 GAN P&L at prediction_threshold=0.45. Own class name -> own model dir, so
production NNNC_DDPM_MLX is untouched.
"""

import sys
from pathlib import Path

group_dir = str(Path(__file__).parent)
sys.path.append(group_dir)

from NNNC_DDPM_MLX import NNNC_DDPM_MLX


class NNNC_DDPM_MLX_P1_ctrl(NNNC_DDPM_MLX):

    # Powered operating point (matches Phase-0).
    buy_params = {**NNNC_DDPM_MLX.buy_params, "prediction_threshold": 0.45}

    # Current sampler settings, stated explicitly (this is the control).
    gan_inference_sample_steps = 50
    gan_inference_zscore_clip = 4.0
