# pragma pylint: disable=C0103, C0114, C0115, C0116, C0301
# type: ignore
# pylint: disable=import-error
"""SimpleTrigger grace study (research/st_grace, 2026-10-13): the gated-reversion reference st_prom_c2 (028 C2 promoted) with NNNC's adopted
24 h grace window (ed6049b) in place of 6.5 h, same -27% depth. Pre-registered arm."""

import sys
from pathlib import Path


sys.path.append(str(Path(__file__).parent.parent))

from SimpleTrigger.SimpleTrigger_RevOnlyVolreg2_P2 import SimpleTrigger_RevOnlyVolreg2_P2


class SimpleTrigger_RevOnlyVolreg2_P2_G24(SimpleTrigger_RevOnlyVolreg2_P2):
    SELL_OVERRIDES = {**SimpleTrigger_RevOnlyVolreg2_P2.SELL_OVERRIDES, "stoploss_grace_hours": 24.0}
