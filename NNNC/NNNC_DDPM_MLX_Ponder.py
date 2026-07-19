# pragma pylint: disable=C0103, C0114, C0115, C0116, C0301
# type: ignore

"""
NNNC_DDPM_MLX_Ponder — Stage-2 looped "pondering" (COCONUT) experiment.

Inherits production NNNC_DDPM_MLX verbatim (same gbb labels, guards, threshold,
TabDDPM chain) and swaps the classifier for the recurrent-refinement head:
N shared Ponder steps on the LSTM latent before decode. REQUIRES a retrain (new
params — cannot reuse production weights); auto-reuses the shared TabDDPM GAN +
scalers, so the only variable vs production is the ponder head.

A/B: train + backtest ponder_steps in {0, 2, 4} (0 == matched production arch) on
one PINNED timerange, seed=42. See
docs/superpowers/specs/2026-07-18-noisycoconut-nnnc-design.md.
"""

import sys
from pathlib import Path

group_dir = str(Path(__file__).parent)
sys.path.append(group_dir)

from NNNC_DDPM_MLX import NNNC_DDPM_MLX
from NNNClassifierMLX import ClassifierTypeMLX
from PonderStrategyMixin import PonderStrategyMixin


class NNNC_DDPM_MLX_Ponder(PonderStrategyMixin, NNNC_DDPM_MLX):

    classifier_type = ClassifierTypeMLX.LSTM_PONDER

    # Swept via distinct-name subclasses (retrain per N). ponder_steps=0 is the
    # matched control (identical forward to production _LSTMModel).
    ponder_steps = 3
