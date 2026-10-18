# pragma pylint: disable=C0103, C0114, C0115, C0116, C0301
# type: ignore
"""Leverage control v2 for cs_aro_hp: `NNNC_CS_ARO_LEV` with STAKE_SCALE re-calibrated so committed
capital matches cs_aro_hp's (1.7432: stake = min(k x intended, entry candle's fillable cap), solved on
cs_aro's trades for 1.3187x its stake-hours). v1 (x1.3187) failed its instrument because 68% of entries
are already liquidity-capped. Pre-registered in research/tb_model/preregistration_lev2.json."""

import sys
from pathlib import Path


sys.path.append(str(Path(__file__).parent))

from NNNC_CS_ARO_LEV import NNNC_CS_ARO_LEV


class NNNC_CS_ARO_LEV2(NNNC_CS_ARO_LEV):
    STAKE_SCALE = 1.7432
