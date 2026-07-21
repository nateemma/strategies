# pragma pylint: disable=C0103, C0114, C0115, C0116, C0301
# type: ignore

"""
NNNC_DDPM_MLX_P2_disp20 — as NNNC_DDPM_MLX_P2_disp15 but 2.0x output dispersion,
which raises σ_ratio ~0.48 -> ~0.96 (matching real variance ~0.9). GAN_TODO #5.

Brackets the dispersion sweep: disp15 (partial correction) vs disp20 (full match to
real σ). Correlations preserved exactly (validated). If widening the synth to match
real dispersion recovers the GAN toward/past non-GAN (21.24%), the underperformance
was a fixable under-dispersion artifact; if P&L stays flat as σ->1, it's the
information ceiling. Compare fidelity (σ_ratio, joints) + trades + P&L to control.
"""

import sys
from pathlib import Path

group_dir = str(Path(__file__).parent)
sys.path.append(group_dir)

from NNNC_DDPM_MLX import NNNC_DDPM_MLX


class NNNC_DDPM_MLX_P2_disp20(NNNC_DDPM_MLX):

    buy_params = {**NNNC_DDPM_MLX.buy_params, "prediction_threshold": 0.45}

    gan_inference_dispersion_scale = 2.0
