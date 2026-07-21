# pragma pylint: disable=C0103, C0114, C0115, C0116, C0301
# type: ignore

"""
NNNC_DDPM_MLX_P2_disp15 — diagnosis-driven fix (joint-preserving), GAN_TODO #5.

TabDDPM under-disperses (σ_syn/σ_real ~0.5) because its denoiser is mode-seeking;
this teaches the classifier a too-tight Buy/Sell region so it rejects net-winning
marginal entries (24 fewer trades, ~+0.76pp of missed net-winners vs non-GAN).
Sampler stochasticity (η/churn) can't fix it — the model denoises noise back to
the modes (validated). So we widen the FINAL de-z-scored output 1.5x around the
per-class mean: this raises σ_ratio ~0.48 -> ~0.72 while leaving feature
correlations EXACTLY unchanged (validated standalone — the earlier failure scaled
in z-space before the clip, which truncated the tails). Inference-time, no GAN
retrain; fresh classifier on the wider synth. Expect: LESS conservative (trades ->
599), P&L -> toward/past non-GAN 21.24%. Compare to control (20.48%).
"""

import sys
from pathlib import Path

group_dir = str(Path(__file__).parent)
sys.path.append(group_dir)

from NNNC_DDPM_MLX import NNNC_DDPM_MLX


class NNNC_DDPM_MLX_P2_disp15(NNNC_DDPM_MLX):

    buy_params = {**NNNC_DDPM_MLX.buy_params, "prediction_threshold": 0.45}

    gan_inference_dispersion_scale = 1.5
