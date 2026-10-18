# pragma pylint: disable=C0103, C0114, C0115, C0116, C0301
# type: ignore
"""All-available entry: `NNNC_CS_ARO` (add off) staking min(2,000 x sizer weight, the entry candle's
fillable cap, freqtrade's max_stake = available wallet). STAKE_SCALE = 2000 / 900 on the 900 proposed
stake, so a thin candle takes whatever is fillable and a liquid one takes up to the limit.
Pre-registered in research/tb_model/preregistration_all.json."""

import sys
from pathlib import Path


sys.path.append(str(Path(__file__).parent))

from NNNC_CS_ARO_LEV import NNNC_CS_ARO_LEV


class NNNC_CS_ARO_ALL2K(NNNC_CS_ARO_LEV):
    STAKE_SCALE = 2000 / 900
