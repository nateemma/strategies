# pragma pylint: disable=C0103, C0114, C0115, C0116, C0301
# type: ignore

"""
NNNC_MLX_Noisy — Stage-1 NoisyCoconut (latent-space jitter) on the NON-GAN base.

Same COCONUT mechanism as NNNC_DDPM_MLX_Noisy but parented off the plain
NNNC_MLX (no GAN augmentation, no post_gan_scaling) — the CORRECT base for
isolating a classifier-head mechanism. NO retraining — reuses the trained
NNNC_MLX weights via the byte-identical encode()/decode() split.

A/B against the NNNC_MLX base over the sigma sweep {0.05, 0.1, 0.2, 0.4};
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


class NNNC_MLX_Noisy(NoisyCoconutStrategyMixin, NNNC_MLX):

    # Reuse the plain non-GAN NNNC_MLX weights (not the DDPM ones).
    reuse_model_from = "NNNC_MLX"

    classifier_type = ClassifierTypeMLX.LSTM_NOISY

    # Stage-1 sweep knob (scaled per latent-dim batch std inside the predictor).
    noisy_sigma = 0.1
    noisy_k = 16
