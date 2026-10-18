# pragma pylint: disable=C0103, C0114, C0115, C0116, C0301
# type: ignore
"""All-available entry at half the participation: `NNNC_CS_ARO_ALL3K` with the fillable cap at 1/20 of
the entry candle's quote volume (5%) instead of 1/10. The entry gate in confirm_trade_entry uses the
same multiple. Pre-registered in research/tb_model/preregistration_p5.json."""

import sys
from pathlib import Path


sys.path.append(str(Path(__file__).parent))

from NNNC_CS_ARO_ALL3K import NNNC_CS_ARO_ALL3K


class NNNC_CS_ARO_ALL3K_P5(NNNC_CS_ARO_ALL3K):
    QUOTE_VOLUME_HEADROOM_MULT = 20.0
