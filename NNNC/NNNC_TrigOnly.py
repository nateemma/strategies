import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent))

from NNNC_MLX_Lbl02 import NNNC_MLX_Lbl02
from TriggerOnlyMixin import TriggerOnlyMixin


class NNNC_TrigOnly(TriggerOnlyMixin, NNNC_MLX_Lbl02):
    """THE ARM: loss masked to trigger bars, trigger applied at inference.

    All model capacity goes to P(future_gain >= 0.02 | trigger) — the only part
    of the label that is not already a threshold rule on two supplied features.

    Compare against:
      r02t08    NNNC_MLX_Lbl02, no trigger anywhere          (the base control)
      triggate  same weights, trigger at inference only      (isolates the gate)
      this arm  trigger at inference AND masked in training  (the package)
    """

    pnl_loss_alpha = 1.0
