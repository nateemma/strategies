# pragma pylint: disable=C0103, C0114, C0115, C0116, C0301
# type: ignore
"""v2 parity arm cs_aro_tr_v2: cs_aro + tr_scale with the transition stand-in v2 (inputs from the history store's OHLCV,
saved_data/transition_v2/, unrounded = the tested stand-in). Must reproduce cs_aro_tr exactly. Same weights."""

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent))

from NNNC_CS_ARO_TR import NNNC_CS_ARO_TR


class NNNC_CS_ARO_TR_V2(NNNC_CS_ARO_TR):
    TR_SOURCE = "standin"
    TR_STANDIN_ROOT = "transition_v2"
