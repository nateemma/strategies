# pragma pylint: disable=C0103, C0114, C0115, C0116, C0301
# type: ignore
"""035 follow-up both32 (research/new_inputs/preregistration_b32.json): `NNNC_CS_ARO` plus the 8 raw8
columns, scaled by the composite main_scaler_both32. The width-book sizer (the control sizer with only the weights changed; same model key as NNNC_NI_B32)."""

import sys
from pathlib import Path


sys.path.append(str(Path(__file__).parent))
sys.path.append(str(Path(__file__).parent.parent))

from _both32 import Both32Mixin
from _raw8_features import RAW8_COLUMNS
from NNNC_CS_ARO import NNNC_CS_ARO


class NNNC_NI_B32_SZ(Both32Mixin, NNNC_CS_ARO):
    include_list = list(NNNC_CS_ARO.include_list) + list(RAW8_COLUMNS)
