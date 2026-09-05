import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent))

from NNNC_Ora_Prec import NNNC_Ora_Prec


class NNNC_Ora_V1(NNNC_Ora_Prec):
    """Volume-matched comparator for the headroom re-measurement (task 2).

    precision 1.0, subsample 0.3 — PERFECT selection at the model's volume

    The real model at threshold 0.80 takes ~431 trades guards-off. Comparing it
    against a 1333-trade oracle would confound SELECTION QUALITY with how many
    concurrency slots each arm consumes, so both the perfect and the random
    comparator are subsampled to approximately the model's volume.
    """

    ORACLE_PRECISION = 1.0
    ORACLE_SUBSAMPLE = 0.3
    ORACLE_SEED = 17
