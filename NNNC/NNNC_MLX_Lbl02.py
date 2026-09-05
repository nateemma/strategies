import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent))

from NNNC_MLX import NNNC_MLX


class NNNC_MLX_Lbl02(NNNC_MLX):
    """REAL model trained on the label the oracle gate passed (feature 004 follow-up).

    The gate (regime/ORACLE_GATE.md) showed that at the CEILING — perfect
    prediction, guards neutralised — raising the gain threshold from 0.007 to
    0.02 turns W1 from -18.89% to +26.54%, and flips all three windows. A 0.7%
    target against a -4%..-6% stop plus fees cannot pay however perfectly it is
    predicted; a 2% target can.

    The gate says only that a model trained on this label COULD pay. This arm
    tests whether a REAL model realises any of it.

    That distinction is the whole point. Feature 003 raised the breakout label's
    threshold on the same economic reasoning, improved reward:risk 0.57 -> 1.28,
    and realised return did NOT follow — but 003 never measured its ceiling, so
    it could not tell a wrong label from an unlearnable one. Here the ceiling is
    known, so a null result means "the model cannot learn it", not "the label
    was wrong".

    HORIZON stays at 48 and every guard, stop and exit is inherited unchanged,
    so the label threshold is the only difference from production NNNC_MLX
    (Constitution VIII).
    """

    MIN_BUY_GAIN_THRESHOLD = 0.02
    MIN_SELL_LOSS_THRESHOLD = 0.02
