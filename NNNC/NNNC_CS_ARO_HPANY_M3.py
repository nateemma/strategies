# pragma pylint: disable=C0103, C0114, C0115, C0116, C0301
# type: ignore
"""`NNNC_CS_ARO_HPANY` (any-signal held-pair add) allowing up to 3 adds per trade instead of 1: each
fresh width-book signal on the held pair adds one 1x stake at that candle's own liquidity cap (last fill
>= 1 h old, clocks restart from each add). Pre-registered in research/tb_model/preregistration_cap.json."""

import sys
from pathlib import Path


sys.path.append(str(Path(__file__).parent))

from NNNC_CS_ARO_HPANY import NNNC_CS_ARO_HPANY


class NNNC_CS_ARO_HPANY_M3(NNNC_CS_ARO_HPANY):
    HELD_PAIR_ADD = {"max_adds": 3}
