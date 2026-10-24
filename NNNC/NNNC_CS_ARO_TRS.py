# pragma pylint: disable=C0103, C0114, C0115, C0116, C0301
# type: ignore
"""cs_aro_tr with the transition stand-in scored INSIDE the strategy (TR_SOURCE "standin"; the live-capable
path, NNNC/_transition_standin.py) instead of the per-fold tables. Parity arm: on the 15 folds it must
reproduce cs_aro_tr's trades before the live-capable path becomes the reference. Same weights."""

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent))

from NNNC_CS_ARO_TR import NNNC_CS_ARO_TR


class NNNC_CS_ARO_TRS(NNNC_CS_ARO_TR):
    TR_SOURCE = "standin"
