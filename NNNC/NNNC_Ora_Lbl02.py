import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent))

from NNNC_Oracle import NNNC_Oracle


class NNNC_Ora_Lbl02(NNNC_Oracle):
    """T021 — LABEL-side variant, measured SEPARATELY from every exit change
    (FR-015, Constitution VIII). LOOKAHEAD CEILING arm, never promotable.

    NNNC's label is TRAINING_TYPE 17 (gbb), thresholds 0.007, HORIZON 48. These
    are class attributes from Framework/TrainingConfig, unreachable via --set
    (research.md R6). This arm raises the gain threshold to 0.02 with the
    horizon unchanged, so the exit stack is untouched.
    """

    MIN_BUY_GAIN_THRESHOLD = 0.02
    MIN_SELL_LOSS_THRESHOLD = 0.02
