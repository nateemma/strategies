import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent))

from LabelParamsMixin import LabelParamsMixin
from NNNC_Oracle import NNNC_Oracle


class NNNC_Ora_G10(LabelParamsMixin, NNNC_Oracle):
    """Oracle gate arm — guard term REMOVED (guard_metric is bounded in [-1,1])

    LOOKAHEAD CEILING arm, never promotable. Control is `ora_lbl02`
    (stock trigger, gain 0.02): W1 1333tr +26.54%.
    """

    MIN_BUY_GAIN_THRESHOLD = 0.02
    MIN_SELL_LOSS_THRESHOLD = 0.02
    LABEL_PARAMS = {"guard_threshold": 1.0}
