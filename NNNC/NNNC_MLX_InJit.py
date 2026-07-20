# pragma pylint: disable=C0103, C0114, C0115, C0116, C0301
# type: ignore

"""
NNNC_MLX_InJit — Stage-0 NoisyCoconut (input-space jitter) on the NON-GAN base.

Same mechanism as NNNC_DDPM_MLX_InJit but parented off the plain NNNC_MLX
(no GAN augmentation, no post_gan_scaling) — the CORRECT base for isolating a
classifier-head mechanism. NO retraining — reuses the trained NNNC_MLX weights.

A/B against the NNNC_MLX base over the sigma sweep {0.01, 0.02, 0.05, 0.1};
sigma=0 must reproduce NNNC_MLX exactly. See
docs/superpowers/specs/2026-07-18-noisycoconut-nnnc-design.md.
"""

import sys
from pathlib import Path

group_dir = str(Path(__file__).parent)
sys.path.append(group_dir)

from NNNC_MLX import NNNC_MLX
from NNNClassifierMLX import ClassifierTypeMLX
from NoisyCoconutStrategyMixin import NoisyCoconutStrategyMixin


class NNNC_MLX_InJit(NoisyCoconutStrategyMixin, NNNC_MLX):

    # Reuse the plain non-GAN NNNC_MLX weights (not the DDPM ones).
    reuse_model_from = "NNNC_MLX"

    classifier_type = ClassifierTypeMLX.LSTM_INJIT

    # Stage-0 sweep knob (input tensor is already normalised → scalar sigma).
    noisy_sigma = 0.05
    noisy_k = 16
