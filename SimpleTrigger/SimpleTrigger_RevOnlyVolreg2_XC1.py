# pragma pylint: disable=C0103, C0114, C0115, C0116, C0301
# type: ignore
# pylint: disable=import-error
"""027 EXPLORATORY (post-hoc, not out-of-sample): gated reversion whose post-grace stop is never
tighter than -3.71%, the early p90 drawdown winners take before the +2% target -- coupling C1 at its
early-chosen value (specs/027-target-aware-exits/feasibility.json). With the ATR stop already in
[-6%, -4%], this binds only where volume confirmation tightens the stop towards -2%.

Implemented in the arm, not Framework/: exploratory, and 027 built no framework switch."""

import sys
from pathlib import Path


sys.path.append(str(Path(__file__).parent.parent))

from SimpleTrigger.SimpleTrigger_RevOnlyVolreg2 import SimpleTrigger_RevOnlyVolreg2


class SimpleTrigger_RevOnlyVolreg2_XC1(SimpleTrigger_RevOnlyVolreg2):
    STOP_MIN_DEPTH = -0.03710764139923586

    def _resolve_normal_stop(self, pair, trade=None):
        return min(super()._resolve_normal_stop(pair, trade), self.STOP_MIN_DEPTH)
