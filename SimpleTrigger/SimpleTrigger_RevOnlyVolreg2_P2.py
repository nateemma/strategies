# pragma pylint: disable=C0103, C0114, C0115, C0116, C0301
# type: ignore
# pylint: disable=import-error
"""028 US3: gated reversion with C2 -- the ATR-scaled take-profit ADOPTED on the 2022 window
(specs/028-recovery-aware-exits/verdicts.md) -- switched on through the SHARED exit
(`ExitParams.take_profit_atr_mult`, resolved in `BaseStrategy`), not by method override. Must
reproduce `SimpleTrigger_RevOnlyVolreg2_XC2`'s trades on every fold (FR-011)."""

import sys
from pathlib import Path


sys.path.append(str(Path(__file__).parent.parent))

from SimpleTrigger.SimpleTrigger_RevOnlyVolreg2 import SimpleTrigger_RevOnlyVolreg2


class SimpleTrigger_RevOnlyVolreg2_P2(SimpleTrigger_RevOnlyVolreg2):
    FAMILY_EXIT_OVERRIDES = {"reversion": {"take_profit_atr_mult": 2.8970711921574543}}
