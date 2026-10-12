# pragma pylint: disable=C0103, C0114, C0115, C0116, C0301
# type: ignore
# pylint: disable=import-error
"""Momentum-first (026 addendum 2): the volreg-gated combined arm with MOMENTUM_PREEMPT."""

import sys
from pathlib import Path


sys.path.append(str(Path(__file__).parent.parent))

from SimpleTrigger.SimpleTrigger_RegGateVolreg import SimpleTrigger_RegGateVolreg


class SimpleTrigger_PreemptVolreg(SimpleTrigger_RegGateVolreg):
    MOMENTUM_PREEMPT = True
