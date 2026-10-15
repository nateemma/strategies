# pragma pylint: disable=C0103, C0114, C0115, C0116, C0301
# type: ignore
"""Control-sizer study (research/control_sizer/): the CONTROL `NNNC_A_Raw_Open` weights as a
capital-neutral stake sizer on the width book, agree = p_buy above the fold's 0.5-share threshold.
Identical to 034's `NNNC_LS_CONTROL_SZ` (built for the S1 pick only); its own name gives the arm its
own fold classes and records."""

import sys
from pathlib import Path


sys.path.append(str(Path(__file__).parent))

from NNNC_LS_CONTROL_SZ import NNNC_LS_CONTROL_SZ


class NNNC_CS_ARO(NNNC_LS_CONTROL_SZ):
    pass
