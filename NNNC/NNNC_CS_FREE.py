# pragma pylint: disable=C0103, C0114, C0115, C0116, C0301
# type: ignore
"""cs_aro on the model-free w_free book (research/train_window/cs_free.py; preregistration_cs_free.json):
the ARO p_buy sizer exactly as NNNC_CS_ARO, entries = NNNC_W_FREE's (trailing-90-day width quantile, no model)."""

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent))

from NNNC_CS_ARO import NNNC_CS_ARO
from NNNC_W_FREE import NNNC_W_FREE


class NNNC_CS_FREE(NNNC_CS_ARO):
    WIDTH_CUTS = NNNC_W_FREE.WIDTH_CUTS
