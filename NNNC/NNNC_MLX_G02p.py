import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent))

from LabelParamsMixin import LabelParamsMixin
from NNNC_MLX_Lbl02 import NNNC_MLX_Lbl02


class NNNC_MLX_G02p(LabelParamsMixin, NNNC_MLX_Lbl02):
    """REAL model on a relaxed-trigger label — guard_metric < +0.2 (was -0.2)

    Oracle gate PASSED on all three windows (W1 +43.60%, W2 +63.32%, W3 +43.28%). Free MCC from the trigger
    is 0.700 against the control's 0.701, so this costs almost nothing in
    learnability head start (regime/tradeoff.py).

    Control: `r02t08` (stock trigger, gain 0.02), guards off, thr 0.80 --
    W1 431tr +6.94%, W2 763tr +16.91%, W3 295tr +10.25%.
    """

    LABEL_PARAMS = {"guard_threshold": 0.2}
