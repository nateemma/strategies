# pragma pylint: disable=C0103, C0114, C0115, C0116, C0301
# type: ignore

"""
NNNC_DDPM_MLX_Ponder — Stage-2 looped "pondering" (COCONUT) experiment.

Inherits production NNNC_DDPM_MLX's labels/guards/threshold and swaps the
classifier for the recurrent-refinement head: N shared Ponder steps on the LSTM
latent before decode. REQUIRES a retrain (new params — cannot reuse production
weights).

**Non-GAN** (``gan_type = NONE``): the shared Jul-4 gan_scaler_a predates the
di_diff_scaled/spread_ma feature additions, so any GAN-augmented retrain crashes
in normalise_for_gan. Dropping the GAN sidesteps that entirely (the main tensor
scaler is consistent — production predicts fine with it) and touches no shared
production artifacts. Per the gan_ratio_sweep finding (no-GAN baseline wins /
augmentation is net-noise-adding), no-GAN is a legitimate regime, not a
downgrade. ponder_steps=0 is then the non-GAN production-arch control.

A/B: train + backtest ponder_steps in {0, 2, 4} on one PINNED timerange,
seed=42. See docs/superpowers/specs/2026-07-18-noisycoconut-nnnc-design.md.
"""

import sys
from pathlib import Path

group_dir = str(Path(__file__).parent)
sys.path.append(group_dir)

from NNNC_DDPM_MLX import NNNC_DDPM_MLX
from NNNClassifierMLX import ClassifierTypeMLX
from PonderStrategyMixin import PonderStrategyMixin
from Framework.BaseStrategy import GANType


class NNNC_DDPM_MLX_Ponder(PonderStrategyMixin, NNNC_DDPM_MLX):

    classifier_type = ClassifierTypeMLX.LSTM_PONDER

    # Non-GAN: sidesteps the stale gan_scaler crash; no-GAN is a valid regime here.
    gan_type = GANType.NONE

    # N=2 = confirmed robust optimum of the inverted-U (peak at all seeds; N2 beats
    # the N0 control 6/6 across seeds+eras, ~+0.9pp). ponder_steps=0 = matched
    # (non-GAN) control, byte-identical forward to production _LSTMModel.
    ponder_steps = 2
