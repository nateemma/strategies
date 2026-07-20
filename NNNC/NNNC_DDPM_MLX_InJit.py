# pragma pylint: disable=C0103, C0114, C0115, C0116, C0301
# type: ignore

"""
NNNC_DDPM_MLX_InJit — Stage-0 pre-gate for the NoisyCoconut experiment.

Inherits production NNNC_DDPM_MLX verbatim (same gbb labels, guards, threshold,
TabDDPM chain) and only swaps the predictor for input-space jitter multi-path
voting. NO retraining — reuses the trained NNNC_DDPM_MLX weights.

A/B against production over the sigma sweep {0.01, 0.02, 0.05, 0.1}; sigma=0
must reproduce production exactly. See
docs/superpowers/specs/2026-07-18-noisycoconut-nnnc-design.md.
"""

import sys
from pathlib import Path

group_dir = str(Path(__file__).parent)
sys.path.append(group_dir)

from NNNC_DDPM_MLX import NNNC_DDPM_MLX
from NNNClassifierMLX import ClassifierTypeMLX
from NoisyCoconutStrategyMixin import NoisyCoconutStrategyMixin


class NNNC_DDPM_MLX_InJit(NoisyCoconutStrategyMixin, NNNC_DDPM_MLX):

    classifier_type = ClassifierTypeMLX.LSTM_INJIT

    # Stage-0 sweep knob (input tensor is already normalised → scalar sigma).
    noisy_sigma = 0.05
    noisy_k = 16
