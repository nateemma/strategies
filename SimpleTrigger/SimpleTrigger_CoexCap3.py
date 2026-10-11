# pragma pylint: disable=C0103, C0114, C0115, C0116, C0301
# type: ignore
# pylint: disable=import-error
"""Coexistence candidate: FAMILY_PAIR_CAP = 3 (feature 025 US1)."""

import sys
from pathlib import Path


sys.path.append(str(Path(__file__).parent.parent))

from SimpleTrigger.SimpleTrigger_FamCombined import SimpleTrigger_FamCombined


class SimpleTrigger_CoexCap3(SimpleTrigger_FamCombined):
    FAMILY_PAIR_CAP = 3
