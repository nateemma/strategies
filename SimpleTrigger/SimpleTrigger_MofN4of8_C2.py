# pragma pylint: disable=C0103, C0114, C0115, C0116, C0301
# type: ignore
"""`mofn_4of8` with C2, the ATR-scaled take-profit adopted in 028 (2.897 x ATR, [1%, 5%]), switched on
through the shared exit (`ExitParams.take_profit_atr_mult`) as in SimpleTrigger_RevOnlyVolreg2_P2
(research/mofn_trigger/preregistration_c2.json)."""

import sys
from pathlib import Path


sys.path.append(str(Path(__file__).parent.parent))

from SimpleTrigger.SimpleTrigger_MofN4of8 import SimpleTrigger_MofN4of8


class SimpleTrigger_MofN4of8_C2(SimpleTrigger_MofN4of8):
    FAMILY_EXIT_OVERRIDES = {"reversion": {"take_profit_atr_mult": 2.8970711921574543}}
