# pragma pylint: disable=C0103, C0114, C0115, C0116, C0301
# type: ignore

"""
NNNC_DDPM_MLX_P1_steps250 — Phase-1 more DDIM sampling steps (250) for the GAN
sample-quality plan (docs/GAN_TODO.md #5).

More reverse-diffusion steps = a better approximation of the true reverse
distribution, which can reduce the over-dispersion at its source. Inference-time
only (GAN not retrained; num_sample_steps pushed onto the loaded model via
_apply_gan_inference_overrides). Trains a fresh classifier on the resulting synth.
Compare fidelity + P&L to NNNC_DDPM_MLX_P1_ctrl.
"""

import sys
from pathlib import Path

group_dir = str(Path(__file__).parent)
sys.path.append(group_dir)

from NNNC_DDPM_MLX import NNNC_DDPM_MLX


class NNNC_DDPM_MLX_P1_steps250(NNNC_DDPM_MLX):

    buy_params = {**NNNC_DDPM_MLX.buy_params, "prediction_threshold": 0.45}

    gan_inference_sample_steps = 250
    gan_inference_zscore_clip = 4.0
