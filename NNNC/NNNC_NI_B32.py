# pragma pylint: disable=C0103, C0114, C0115, C0116, C0301
# type: ignore
"""035 follow-up both32 (research/new_inputs/preregistration_b32.json): `NNNC_A_Raw_Open` plus the 8 raw8
columns, scaled by the composite main_scaler_both32. The training class."""

import sys
from pathlib import Path


sys.path.append(str(Path(__file__).parent))
sys.path.append(str(Path(__file__).parent.parent))

from _both32 import Both32Mixin
from _raw8_features import RAW8_COLUMNS
from NNNC_A_Raw_Open import NNNC_A_Raw_Open


class NNNC_NI_B32(Both32Mixin, NNNC_A_Raw_Open):
    include_list = list(NNNC_A_Raw_Open.include_list) + list(RAW8_COLUMNS)
