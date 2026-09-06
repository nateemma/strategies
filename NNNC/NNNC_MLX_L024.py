import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent))

from NNNC_MLX import NNNC_MLX


class NNNC_MLX_L024(NNNC_MLX):
    """REAL model at label 2.4% — the untested cell just above the current 0.02.

    Ceiling gains over the 0.02 label (oracle, guards open, summed over three
    windows): 0.024 = +44.20pp, 0.026 = +61.10pp. The question is whether a real
    model realises any of it, since raising the label to 0.03 previously gained
    +71pp of ceiling and LOST realised return -- learnability falls with the
    threshold (MCC 0.78 -> 0.54 -> 0.41 across 0.007 -> 0.03).

    These settings sit close enough to 0.02 that learnability should barely
    move, which is what makes them the plausible cell.

    Guards-open / threshold operating point is pinned in NNNC_MLX_L024.json. The
    threshold is RE-SWEPT per label rather than inherited: a different label
    shifts the confidence distribution, so a fixed threshold is confounded.

    Trained at MLX_SEED=42.
    """

    MIN_BUY_GAIN_THRESHOLD = 0.024
    MIN_SELL_LOSS_THRESHOLD = 0.024
