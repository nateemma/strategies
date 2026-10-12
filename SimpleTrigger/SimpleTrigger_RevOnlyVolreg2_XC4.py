# pragma pylint: disable=C0103, C0114, C0115, C0116, C0301
# type: ignore
# pylint: disable=import-error
"""027 EXPLORATORY (post-hoc, not out-of-sample): gated reversion with the grace window extended to
the label horizon (12 h), same -27% depth -- coupling C4 at its early-chosen value
(specs/027-target-aware-exits/feasibility.json). Chosen after late-half replay results were seen, so
its late verdict is exploratory only (027 report, option 1)."""

import sys
from pathlib import Path


sys.path.append(str(Path(__file__).parent.parent))

from SimpleTrigger.SimpleTrigger_RevOnlyVolreg2 import SimpleTrigger_RevOnlyVolreg2


class SimpleTrigger_RevOnlyVolreg2_XC4(SimpleTrigger_RevOnlyVolreg2):
    SELL_OVERRIDES = {**SimpleTrigger_RevOnlyVolreg2.SELL_OVERRIDES, "stoploss_grace_hours": 12.0}
