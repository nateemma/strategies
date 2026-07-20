# pragma pylint: disable=C0103, C0114, C0115, C0116, C0301
# type: ignore

"""
NNNC_MLX_PnlLoss — P&L-magnitude-weighted loss probe (non-GAN, gbb).

Inherits the plain non-GAN NNNC_MLX base (gan_type=NONE, post_gan_scaling=False)
and weights each training sample's focal loss by its realised forward-excursion
magnitude (blend alpha). Isolates the LOSS effect on the plain LSTM (no ponder).

A/B: train + backtest pnl_loss_alpha in {0, 0.5, 1.0} (alpha=0 == control) on a
PINNED window, seed-robust. See
docs/superpowers/specs/2026-07-19-pnl-weighted-loss-probe-design.md.
"""

import sys
from pathlib import Path

group_dir = str(Path(__file__).parent)
sys.path.append(group_dir)

from NNNC_MLX import NNNC_MLX
from PnlLossStrategyMixin import PnlLossStrategyMixin


class NNNC_MLX_PnlLoss(PnlLossStrategyMixin, NNNC_MLX):

    # Blend strength; swept via distinct-name subclasses. alpha=0 = control.
    pnl_loss_alpha = 0.5
