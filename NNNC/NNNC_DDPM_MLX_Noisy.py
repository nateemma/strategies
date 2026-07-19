# pragma pylint: disable=C0103, C0114, C0115, C0116, C0301
# type: ignore

"""
NNNC_DDPM_MLX_Noisy — Stage-1 NoisyCoconut (latent-space) experiment.

Inherits production NNNC_DDPM_MLX verbatim (same gbb labels, guards, threshold,
TabDDPM chain) and only swaps the predictor for latent-space jitter multi-path
voting — the actual COCONUT mechanism. NO retraining — reuses the trained
NNNC_DDPM_MLX weights via the byte-identical encode()/decode() split.

A/B against production over the sigma sweep {0.05, 0.1, 0.2, 0.4}; sigma=0 must
reproduce production exactly. See
docs/superpowers/specs/2026-07-18-noisycoconut-nnnc-design.md.
"""

import sys
from pathlib import Path

group_dir = str(Path(__file__).parent)
sys.path.append(group_dir)

from NNNC_DDPM_MLX import NNNC_DDPM_MLX
from NNNClassifierMLX import ClassifierTypeMLX
from NoisyCoconutStrategyMixin import NoisyCoconutStrategyMixin


class NNNC_DDPM_MLX_Noisy(NoisyCoconutStrategyMixin, NNNC_DDPM_MLX):

    classifier_type = ClassifierTypeMLX.LSTM_NOISY

    # Stage-1 sweep knob (scaled per latent-dim batch std inside the predictor).
    noisy_sigma = 0.1
    noisy_k = 16
