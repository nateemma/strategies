# pragma pylint: disable=C0103, C0114, C0115, C0116, C0301
# type: ignore
# pylint: disable=import-error
"""Separate books (026 addendum 2): the gated reversion book alone; momentum's book is st_bgo_mx."""

import sys
from pathlib import Path


sys.path.append(str(Path(__file__).parent.parent))

from SimpleTrigger.SimpleTrigger_RegGateVolreg import SimpleTrigger_RegGateVolreg


class SimpleTrigger_RevOnlyVolreg(SimpleTrigger_RegGateVolreg):
    DISABLED_TRIGGERS = ("breakout_gbb",)
