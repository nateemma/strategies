# pragma pylint: disable=C0103, C0114, C0115, C0116, C0301
# type: ignore
# pylint: disable=import-error
"""Momentum-first (026 addendum 2) on the tighter volreg cut."""

import sys
from pathlib import Path


sys.path.append(str(Path(__file__).parent.parent))

from SimpleTrigger.SimpleTrigger_RegGateVolreg2 import SimpleTrigger_RegGateVolreg2


class SimpleTrigger_PreemptVolreg2(SimpleTrigger_RegGateVolreg2):
    MOMENTUM_PREEMPT = True
