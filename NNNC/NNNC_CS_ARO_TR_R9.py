# pragma pylint: disable=C0103, C0114, C0115, C0116, C0301
# type: ignore
"""040 rounding amendment arm cs_aro_tr_r9 (research/nnmt/preregistration_transition_round.json): cs_aro + tr_scale on the ROUNDED
transition stand-in (saved_data/transition_r9/, inputs rounded to 9 decimals), scored in-strategy. Same weights."""

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent))

from NNNC_CS_ARO_TR import NNNC_CS_ARO_TR


class NNNC_CS_ARO_TR_R9(NNNC_CS_ARO_TR):
    TR_SOURCE = "standin"
    TR_STANDIN_ROOT = "transition_r9"
