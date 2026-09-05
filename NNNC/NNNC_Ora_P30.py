import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent))

from NNNC_Ora_Prec import NNNC_Ora_Prec


class NNNC_Ora_P30(NNNC_Ora_Prec):
    """CONTROL: precision 0.3 — 30% true label, 70% random

    LOOKAHEAD-FREE at precision 0.0 (it uses the label only to fix the COUNT),
    so this is the random-selection control the precision sweep needs. If random
    selection of the same number of bars also returns positively, the sweep is
    measuring trade COUNT and slot allocation, not label quality.
    """

    ORACLE_PRECISION = 0.3
    ORACLE_SEED = 17
