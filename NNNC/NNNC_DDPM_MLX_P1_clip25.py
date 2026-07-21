# pragma pylint: disable=C0103, C0114, C0115, C0116, C0301
# type: ignore

"""
NNNC_DDPM_MLX_P1_clip25 — Phase-1 tighter z-score clip (2.5) for the GAN
sample-quality plan (docs/GAN_TODO.md #5).

Most DIRECT lever on the diagnosed over-dispersion: the sampler emits z-scores that
saturate the ±4σ band, so tightening the generate-time clip to ±2.5σ caps the
over-dispersed tails. Inference-time only (the GAN is NOT retrained; sampling clip
is pushed onto the loaded model via _apply_gan_inference_overrides). Trains a fresh
classifier on the tighter synth. Compare fidelity (σ_syn/σ_real -> ~1) + P&L to
NNNC_DDPM_MLX_P1_ctrl.
"""

import sys
from pathlib import Path

group_dir = str(Path(__file__).parent)
sys.path.append(group_dir)

from NNNC_DDPM_MLX import NNNC_DDPM_MLX


class NNNC_DDPM_MLX_P1_clip25(NNNC_DDPM_MLX):

    buy_params = {**NNNC_DDPM_MLX.buy_params, "prediction_threshold": 0.45}

    gan_inference_sample_steps = 50
    gan_inference_zscore_clip = 2.5
