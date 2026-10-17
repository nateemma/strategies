# pragma pylint: disable=C0103, C0114, C0115, C0116, C0301
# type: ignore
"""Composite sizer study (research/cmp_sizer/): `NNNC_CS_HGB`'s score lookup pointed at the 037 'dip'
composite tables (user_data/data/cmp_sizer/HGB_fNN.parquet, research/cmp_sizer/signals.py).
Entries, weights (model key = cs_aro's), stakes and the 0.5-share threshold rule are cs_aro's."""

import sys
from pathlib import Path


sys.path.append(str(Path(__file__).parent))

from NNNC_CS_HGB import NNNC_CS_HGB


class NNNC_CS_CMP(NNNC_CS_HGB):
    HGB_SIGNALS = Path(__file__).resolve().parents[2] / "data" / "cmp_sizer"
