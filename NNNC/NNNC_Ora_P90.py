import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent))

from NNNC_Ora_Prec import NNNC_Ora_Prec


class NNNC_Ora_P90(NNNC_Ora_Prec):
    """Partial oracle at precision 0.9. LOOKAHEAD arm, never promotable."""

    ORACLE_PRECISION = 0.9
