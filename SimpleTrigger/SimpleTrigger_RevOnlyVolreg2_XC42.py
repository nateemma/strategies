# pragma pylint: disable=C0103, C0114, C0115, C0116, C0301
# type: ignore
# pylint: disable=import-error
"""028 EXPLORATORY setup (post-hoc on 2023-26): gated reversion with BOTH 027 couplings C4 and C2 --
the grace window extended to 12 h (XC4's override) on top of the ATR-scaled take-profit (XC2's
`_exit_params`). Frozen values; judged on 2022 under specs/028-recovery-aware-exits/preregistration.json.
The framework fingerprint cannot see XC2's method override, so this class hashes like XC4."""

import sys
from pathlib import Path


sys.path.append(str(Path(__file__).parent.parent))

from SimpleTrigger.SimpleTrigger_RevOnlyVolreg2_XC2 import SimpleTrigger_RevOnlyVolreg2_XC2


class SimpleTrigger_RevOnlyVolreg2_XC42(SimpleTrigger_RevOnlyVolreg2_XC2):
    SELL_OVERRIDES = {**SimpleTrigger_RevOnlyVolreg2_XC2.SELL_OVERRIDES, "stoploss_grace_hours": 12.0}
